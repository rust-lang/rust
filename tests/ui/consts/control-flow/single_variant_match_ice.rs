//@ check-pass

enum Foo {
    Prob,
}

const FOO: u32 = match Foo::Prob {
    Foo::Prob => 42,
};

const BAR: u32 = match Foo::Prob {
    x => 42,
};

impl Foo {
    pub const fn as_val(&self) -> u8 {
        use self::Foo::*;

        match *self {
            Prob => 0x1,
        }
    }
}

fn main() {}
