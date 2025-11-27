// https://github.com/rust-lang/rust/issues/56593#issuecomment-1133174514

use thing::*;

#[derive(Debug)]
pub enum Thing {
    Foo
}

mod tests {
    use super::*;

    fn test_thing() {
        let thing: crate::Thing = Thing::Foo;
        //~^ ERROR `Thing` is ambiguous
        //~| ERROR no variant or associated item named `Foo` found for enum `thing::Thing`
    }
}

mod thing {
    pub enum Thing {
        Bar
    }
}

fn main() { }
