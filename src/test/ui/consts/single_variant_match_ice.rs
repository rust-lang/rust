enum Foo {
    Prob,
}

const FOO: u32 = match Foo::Prob { //~ ERROR `match` is not allowed in a `const`
    Foo::Prob => 42,
};

const BAR: u32 = match Foo::Prob { //~ ERROR `match` is not allowed in a `const`
    x => 42,
};

impl Foo {
    pub const fn as_val(&self) -> u8 {
        use self::Foo::*;

        match *self { //~ ERROR `match` is not allowed in a `const fn`
            Prob => 0x1,
        }
    }
}

fn main() {}
