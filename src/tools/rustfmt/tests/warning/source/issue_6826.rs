pub fn check(c: char) -> bool {
        c.is_ascii_alphanumeric() // 123
        || c == '。' 
        || c == '、'
}
