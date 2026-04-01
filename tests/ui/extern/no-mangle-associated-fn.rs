//@ aux-build: no-mangle-associated-fn.rs
//@ run-pass

extern crate no_mangle_associated_fn;

struct Foo;

impl Foo {
    #[no_mangle]
    fn foo() -> u8 {
        1
    }
}

trait Bar { //~ WARN trait `Bar` is never used
    fn qux() -> u8;
}

impl Bar for Foo {
    #[no_mangle]
    fn qux() -> u8 {
        4
    }
}

fn main() {
    extern "Rust" {
        fn foo() -> u8;
        fn bar() -> u8;
        fn baz() -> u8;
        fn qux() -> u8;
    }
    assert_eq!(unsafe { foo() }, 1);
    assert_eq!(unsafe { bar() }, 2);
    assert_eq!(unsafe { baz() }, 3);
    assert_eq!(unsafe { qux() }, 4);
}
