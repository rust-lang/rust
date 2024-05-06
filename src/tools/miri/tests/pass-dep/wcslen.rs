fn to_c_wchar_t_str(s: &str) -> Vec<libc::wchar_t> {
    let mut r = Vec::<libc::wchar_t>::new();
    for c in s.bytes() {
        if c == 0 {
            panic!("can't contain a null character");
        }
        if c >= 128 {
            panic!("only ASCII supported");
        }
        r.push(c.into());
    }
    r.push(0);
    r
}

pub fn main() {
    let s = to_c_wchar_t_str("Rust");
    let len = unsafe { libc::wcslen(s.as_ptr()) };
    assert_eq!(len, 4);
}
