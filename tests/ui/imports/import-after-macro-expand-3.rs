//@ check-pass
// similar with `import-after-macro-expand-2.rs`

use thing::*;

pub enum Thing {
    Foo
}

mod tests {
    use super::*;

    fn test_thing() {
        let thing: crate::Thing = Thing::Foo;
    }
}

mod thing {
    pub enum Thing {
        Bar
    }
}

fn main() {}
