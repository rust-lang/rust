//@ aux-build:attr-from-macro.rs
//@ run-pass

extern crate attr_from_macro;

attr_from_macro::creator! {
    struct Foo;
    enum Bar;
    enum FooBar;
}

fn main() {
    // Checking the `repr(u32)` on the enum.
    assert_eq!(4, std::mem::size_of::<Bar>());
    // Checking the `repr(u16)` on the enum.
    assert_eq!(2, std::mem::size_of::<FooBar>());

    // Checking the Debug impl on the types.
    eprintln!("{:?} {:?} {:?}", Foo, Bar::A, FooBar::A);
}
