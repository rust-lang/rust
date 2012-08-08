trait to_bytes {
    fn to_bytes() -> ~[u8];
}

impl ~[u8]: to_bytes {
    fn to_bytes() -> ~[u8] { copy self }
}

impl @~[u8]: to_bytes {
    fn to_bytes() -> ~[u8] { copy *self }
}

impl ~str: to_bytes {
    fn to_bytes() -> ~[u8] { str::bytes(self) }
}

impl @(~str): to_bytes {
    fn to_bytes() -> ~[u8] { str::bytes(*self) }
}
