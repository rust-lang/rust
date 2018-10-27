enum Foo {
    Prob,
}

impl Foo {
    pub const fn as_val(&self) -> u8 {
        use self::Foo::*;

        match *self {
            Prob => 0x1, //~ ERROR `if`, `match`, `&&` and `||` are not stable in const fn
        }
    }
}

fn main() {}
