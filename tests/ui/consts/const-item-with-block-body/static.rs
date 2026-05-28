//@ run-pass
#![allow(unused_imports)]

mod foo {
    pub trait Value { //~ WARN trait `Value` is never used
        fn value(&self) -> usize;
    }
}

static BLOCK_USE: usize = {
    use foo::Value;
    100
};

static BLOCK_STRUCT_DEF: usize = {
    struct Foo {
        a: usize
    }
    Foo{ a: 300 }.a
};

static BLOCK_FN_DEF: fn(usize) -> usize = {
    fn foo(a: usize) -> usize {
        a + 10
    }
    foo
};

static BLOCK_MACRO_RULES: usize = {
    macro_rules! baz {
        () => (412)
    }
    baz!()
};

pub fn main() {
    assert_eq!(BLOCK_USE, 100);
    assert_eq!(BLOCK_STRUCT_DEF, 300);
    assert_eq!(BLOCK_FN_DEF(390), 400);
    assert_eq!(BLOCK_MACRO_RULES, 412);
}
