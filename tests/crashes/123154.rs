//@ known-bug: #123154
struct AA {
    pub data: [&usize]
}

impl AA {
    const fn new() -> Self { }
}

static AA = AA::new();

fn main() { }
