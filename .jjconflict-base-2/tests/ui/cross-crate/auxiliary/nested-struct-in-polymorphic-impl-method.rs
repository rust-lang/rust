//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/3136
#![crate_type = "lib"]

trait x {
    fn use_x<T>(&self);
}
struct y(());
impl x for y {
    fn use_x<T>(&self) {
        struct foo {
            //~ ERROR quux
            i: (),
        }
        fn new_foo<T>(i: ()) -> foo {
            foo { i: i }
        }
    }
}
