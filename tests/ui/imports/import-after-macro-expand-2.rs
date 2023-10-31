// check-pass
// https://github.com/rust-lang/rust/issues/56593#issuecomment-1133174514

use thing::*;

#[derive(Debug)]
pub enum Thing {
    Foo
}

mod tests {
    use super::*;

    fn test_thing() {
        let thing: crate::thing::Thing = Thing::Bar;
        // FIXME: `thing` should refer to `crate::Thing`,
        // FIXME: but doesn't currently refer to it due to backward compatibility
    }
}

mod thing {
    pub enum Thing {
        Bar
    }
}

fn main() { }
