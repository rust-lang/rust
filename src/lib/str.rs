export unsafe_from_bytes;

native "rust" mod rustrt {
    type sbuf;
    fn str_buf(s: str) -> sbuf;
    fn str_byte_len(s: str) -> uint;
    fn str_alloc(n_bytes: uint) -> str;
    fn str_from_vec(b: &[mutable? u8]) -> str;
    fn str_from_cstr(cstr: sbuf) -> str;
    fn str_from_buf(buf: sbuf, len: uint) -> str;
    fn str_push_byte(s: str, byte: uint) -> str;
    fn str_slice(s: str, begin: uint, end: uint) -> str;
    fn refcount<T>(s: str) -> uint;
}

fn unsafe_from_bytes(v: &[mutable? u8]) -> str {
    ret rustrt::str_from_vec(v);
}
