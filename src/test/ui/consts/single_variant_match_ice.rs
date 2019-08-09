enum Foo {
    Prob,
}

const FOO: u32 = match Foo::Prob { //~ ERROR unimplemented expression type
    Foo::Prob => 42,
};

const BAR: u32 = match Foo::Prob { //~ ERROR unimplemented expression type
    x => 42,
};

impl Foo {
    pub const fn as_val(&self) -> u8 {
        use self::Foo::*;

        match *self {
            //~^ ERROR loops and conditional expressions are not stable in const fn
            Prob => 0x1,
        }
    }
}

fn main() {}
