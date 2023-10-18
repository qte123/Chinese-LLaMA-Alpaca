
def find_tokens_and_changes(old_tokens, new_tokens):
    old_len=len(old_tokens)
    changes=new_tokens[old_len:]
    return changes

# 示例用法
old_tokens = "你好很高兴"
new_tokens = "你好很高兴认识"

changes = find_tokens_and_changes(old_tokens, new_tokens)
print("Old Tokens:", old_tokens)
print("Changes:", changes)
print("New Tokens:", new_tokens)
