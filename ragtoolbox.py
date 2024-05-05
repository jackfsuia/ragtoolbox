# Operators of this lib for now:
#
# html_to_text
# extract_txt_from_pdf
# recursive_chunk
# chunk_file
# delete_small_chunks
# overlap_chunks
# chunk_to_limited_tokens

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

    return res


def chunk_file(file='cvx_text.txt', split_symbols=[r'\n[0-9][0-9]?\.[0-9][^a-zA-Z]',],max_len = None):
    
    with open(file, 'r', encoding='utf-8') as f:  
        text = f.read()
        chunks=recursive_chunk(text, split_symbols,max_len)
        return chunks

def delete_small_chunks(chunks, mini_len):

    res=[]
    while chunks:
        chunk = chunks.pop(0)
        if len(chunk) >= mini_len:
            res+=[chunk]
    return res

def overlap_chunks(chunks, left_overlap_len=0, right_overlap_len=0):
    if len(chunks) <= 1 or (left_overlap_len==0 and right_overlap_len==0):
        return chunks
    res=[]
    first_chunk=chunks[0]
    first_chunk+=chunks[1][0:right_overlap_len]
    res.append(first_chunk)
    for i in enumerate(1, len(chunks)-1):
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