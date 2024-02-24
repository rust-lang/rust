//@ compile-flags:-Zprint-mono-items=eager

#![deny(dead_code)]
#![feature(start)]

trait SomeTrait {
    fn foo(&self);
}

// This function is never instantiated but the contained impl must still be
// discovered.
pub fn generic_function<T>(x: T) -> (T, i32) {
    impl SomeTrait for i64 {
        //~ MONO_ITEM fn generic_function::<impl SomeTrait for i64>::foo
        fn foo(&self) {}
    }

    (x, 0)
}

//~ MONO_ITEM fn start
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    0i64.foo();

    0
}
