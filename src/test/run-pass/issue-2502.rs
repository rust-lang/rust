struct font {
    let fontbuf: &self/~[u8];

    new(fontbuf: &self/~[u8]) {
        self.fontbuf = fontbuf;
    }

    fn buf() -> &self/~[u8] {
        self.fontbuf
    }
}

fn main() { }
