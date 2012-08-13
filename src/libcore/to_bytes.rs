trait ToBytes {
    fn to_bytes() -> ~[u8];
}

impl ~[u8]: ToBytes {
    fn to_bytes() -> ~[u8] { copy self }
}

impl @~[u8]: ToBytes {
    fn to_bytes() -> ~[u8] { copy *self }
}

impl ~str: ToBytes {
    fn to_bytes() -> ~[u8] { str::bytes(self) }
}

impl @(~str): ToBytes {
    fn to_bytes() -> ~[u8] { str::bytes(*self) }
}
