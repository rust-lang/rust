enum Foo {
    Prob,
}

const FOO: u32 = match Foo::Prob {
    Foo::Prob => 42, //~ ERROR unimplemented expression type
};

const BAR: u32 = match Foo::Prob {
    x => 42, //~ ERROR unimplemented expression type
};

impl Foo {
    pub const fn as_val(&self) -> u8 {
        use self::Foo::*;

        match *self {
            Prob => 0x1, //~ ERROR loops and conditional expressions are not stable in const fn
        }
    }
}

fn main() {}
