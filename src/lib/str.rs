export unsafe_from_bytes;

native "rust" mod rustrt {
    fn str_from_vec(b: &[mutable? u8]) -> str;
    fn refcount<T>(s: str) -> uint;
}

fn unsafe_from_bytes(v: &[mutable? u8]) -> str {
    ret rustrt::str_from_vec(v);
}
