// aux-build: no-mangle-associated-fn.rs
// run-pass

extern crate no_mangle_associated_fn;

struct Foo;

impl Foo {
    #[no_mangle]
    fn foo() -> u8 {
        1
    }
}

fn main() {
    extern "Rust" {
        fn foo() -> u8;
        fn bar() -> u8;
    }
    assert_eq!(unsafe { foo() }, 1);
    assert_eq!(unsafe { bar() }, 2);
}
