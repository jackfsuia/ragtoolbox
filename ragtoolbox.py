# Operators of this lib for now:
#
# html_to_text
# extract_txt_from_pdf
# recursive_chunk
# chunk_file
# delete_small_chunks
# overlap_chunks
# chunk_to_limited_tokens
# openai_embedding
# baichuan_embedding
# qwen_embedding
# ernie_embedding



import os

def html_to_text(filename='CVX.html'):
    import html2text
    with open(filename, 'r') as file:
        content = file.read()
        
        text_content = html2text.html2text(content)
        with open('htmltotxt', 'w') as f2:
            f2.write(text_content)

def extract_txt_from_pdf(fn='CVX.pdf', tgt_path='./'):
    import pdfplumber
    with pdfplumber.open(fn) as pdf:
        text = []
        for page in pdf.pages:
            # remove tables from each page extracted by pdfplumber
            tables = page.find_tables()
            for table in tables:
                page = page.outside_bbox(table.bbox)
            # remove page number from the end of each page
            page_text = page.extract_text()
            page_num = str(page.page_number)
            if page_text.rstrip().endswith(page_num):
                page_text = page_text.rstrip()[:-len(page_num)]
            if page_text.strip():
                text.append(page_text)
        base_fn = os.path.basename(fn).lower().replace('.pdf', '.txt')
        with open(os.path.join(tgt_path, base_fn), 'w', encoding='utf-8') as f:
            f.write('\n'.join(text))


def delete_Table_of_Contents(text):
    return text
    
            
def recursive_chunk(text:str, split_symbols:list[str]=[r'\n[0-9][0-9]?\.[0-9][^a-zA-Z]',], max_len = None)->list[str]:
    import re
    res=[text]

    if max_len:
        if len(text) < max_len:
            return res
        #apply split_symbols until chunks under max_len
        while split_symbols:
            op=split_symbols.pop(0)
            res_temp=[]
            while res:
                chunk = res.pop(0)
                if len(chunk) < max_len:
                    res_temp+=[chunk]
                else:
                    chunks=re.split(op, chunk)
                    res_temp+=chunks
            res = res_temp

        def constlen_chunk(s):
            if max_len:
                return [s[i:i+max_len] for i in range(0, len(s), max_len)]
            return s
        
        #keep them under max_len
        res_temp=[]
        while res:
            chunk = res.pop(0)
            if len(chunk) < max_len:
                res_temp+=[chunk]
            else:
                chunks=constlen_chunk(chunk)
                res_temp+=chunks
        res = res_temp


    if not max_len:
        #apply split_symbols
        while split_symbols:
            op=split_symbols.pop(0)
            res_temp=[]
            while res:
                chunk = res.pop(0)
                chunks=re.split(op, chunk)
                res_temp+=chunks
            res = res_temp
    res_temp = res
    # delete all the ''
    res=[]
    for c in res_temp:
        if c:
            res.append(c)
    return res


def chunk_file(file='cvx_text.txt', split_symbols=[r'\n[0-9][0-9]?\.[0-9][^a-zA-Z]',],max_len = None):
    
    with open(file, 'r', encoding='utf-8') as f:  
        text = f.read()
        chunks=recursive_chunk(text, split_symbols,max_len)
        return chunks

def delete_small_chunks(chunks:list[str], mini_len):

    res=[]
    while chunks:
        chunk = chunks.pop(0)
        if len(chunk) >= mini_len:
            res+=[chunk]
    return res

def overlap_chunks(chunks:list[str], left_overlap_len=0, right_overlap_len=0):
    if len(chunks) <= 1 or (left_overlap_len==0 and right_overlap_len==0):
        return chunks
    res=[]
    first_chunk=chunks[0]
    first_chunk+=chunks[1][0:right_overlap_len]
    res.append(first_chunk)
    for i in range(1, len(chunks)-1):
        lstr=chunks[i-1][-left_overlap_len:]
        rstr=chunks[i+1][0:right_overlap_len]
        res.append(lstr+chunks[i]+rstr)
    last_chunk=chunks[len(chunks)-1]
    last_chunk+=chunks[len(chunks)-2][-left_overlap_len:]
    res.append(last_chunk)
    return res


def chunk_to_limited_tokens(chunks:list[str], tokenizer =None, max_len = 500, )->list[str]:

    from transformers import AutoTokenizer
    def constlen_chunk(t):
        if max_len:
            return [t[i:i+max_len] for i in range(0, len(t), max_len)]
        return t


    tokenized_chunks = tokenizer(chunks)['input_ids']

    res=[]
    while tokenized_chunks:
        chunk = tokenized_chunks.pop(0)
        if len(chunk) < max_len:
            res+=[chunk]
        else:
            chunks=constlen_chunk(chunk)
            res+=chunks

    res = tokenizer.batch_decode(res, skip_special_tokens=False)
    return res




def baichuan_embedding(chunks:list[str], batch_limit=None, max_retry=6, URL="http://api.baichuan-ai.com/v1/embeddings", KEY="abcd"):
    import json
    import requests
    from tqdm import tqdm
    import time

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}

    embeddings = []
    if len(chunks) == 1:
        data = {
            "model": "Baichuan-Text-Embedding",
            "input": chunks[0],
        }
        response = requests.post(
            URL,
            headers=headers,
            json=data,
        )
        content = json.loads(response.text)
        embeddings = [content["data"][0]["embedding"]]
    else:
        for i in tqdm(range(0, len(chunks), batch_limit)):
            data = {
                "model": "Baichuan-Text-Embedding",
                "input": chunks[i : i + batch_limit],
            }

            for retry in range(0, max_retry):
                response = requests.post(
                    URL,
                    headers=headers,
                    json=data,
                )

                if response.status_code == 200:
                    content = json.loads(response.text)
                    embeddings += [emb["embedding"] for emb in content["data"]]
                    break
                else:
                    time.sleep(3 ** retry)
                    if retry == max_retry - 1:
                        raise Exception(f"http status code:{response.status_code} ")

    return embeddings


def ernie_embedding(chunks:list[str], batch_limit=None, max_retry=6, URL="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token=", APIKEY="cde",Skey="abc"):
    import json
    import requests
    def get_access_token():
     
            
        url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={APIKEY}&client_secret={Skey}"
        
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")

    url = URL + get_access_token()
    
    payload = json.dumps({
        "input": chunks
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    content = json.loads(response.text)
    return [emb["embedding"] for emb in content["data"]]


def qwen_embedding(chunks:list[str], batch_limit=25, max_retry=6, KEY="abc"):
    import dashscope
    from tqdm import tqdm
    import time
    dashscope.api_key = KEY

    embeddings = []
    for i in tqdm(range(0, len(chunks), batch_limit)):
     

        for retry in range(0, max_retry):
         
            response = dashscope.TextEmbedding.call(model=dashscope.TextEmbedding.Models.text_embedding_v1, input=chunks[i : i + batch_limit])

            if response.status_code == 200:
                embeddings += [emb['embedding'] for emb in response.output['embeddings']]
                break
            else:
                time.sleep(3 ** retry)
                if retry == max_retry - 1:
                    raise Exception(f"http status code:{response.status_code} ")


    return embeddings


def oepnai_embedding(chunks:list[str],KEY="abc", model="text-embedding-3-small", batch_limit=16, max_retry=6, ):
    from openai import OpenAI
    os.environ["OPENAI_API_KEY"] = KEY
    client = OpenAI()
    data = client.embeddings.create(input = chunks, model=model).data
    
    return [d.embedding for d in data]
