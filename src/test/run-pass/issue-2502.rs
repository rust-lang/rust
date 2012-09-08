struct font {
    fontbuf: &self/~[u8],
}

impl font {
    fn buf() -> &self/~[u8] {
        self.fontbuf
    }
}

fn font(fontbuf: &r/~[u8]) -> font/&r {
    font {
        fontbuf: fontbuf
    }
}

fn main() { }
