use unicode_xid::UnicodeXID;

pub fn is_ident_start(c: char) -> bool {
    (c >= 'a' && c <= 'z')
        || (c >= 'A' && c <= 'Z')
        || c == '_'
        || (c > '\x7f' && UnicodeXID::is_xid_start(c))
}

pub fn is_ident_continue(c: char) -> bool {
    (c >= 'a' && c <= 'z')
        || (c >= 'A' && c <= 'Z')
        || (c >= '0' && c <= '9')
        || c == '_'
        || (c > '\x7f' && UnicodeXID::is_xid_continue(c))
}

pub fn is_whitespace(c: char) -> bool {
    //FIXME: use is_pattern_whitespace
    //https://github.com/behnam/rust-unic/issues/192
    c.is_whitespace()
}

pub fn is_dec_digit(c: char) -> bool {
    '0' <= c && c <= '9'
}
