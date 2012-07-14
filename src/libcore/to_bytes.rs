iface to_bytes {
    fn to_bytes() -> ~[u8];
}

impl of to_bytes for ~[u8] {
    fn to_bytes() -> ~[u8] { copy self }
}

impl of to_bytes for @~[u8] {
    fn to_bytes() -> ~[u8] { copy *self }
}

impl of to_bytes for ~str {
    fn to_bytes() -> ~[u8] { str::bytes(self) }
}

impl of to_bytes for @(~str) {
    fn to_bytes() -> ~[u8] { str::bytes(*self) }
}
