// NOTE: We always compile this test with -Copt-level=0 because higher opt-levels
//       prevent drop-glue from participating in share-generics.
//@ compile-flags: -Zshare-generics=yes -Copt-level=0

#![crate_type = "rlib"]

pub fn generic_fn<T>(x: T, y: T) -> (T, T) {
    (x, y)
}

pub fn use_generic_fn_f32() -> (f32, f32) {
    // This line causes drop glue for Foo to be instantiated. We want to make
    // sure that this crate exports an instance to be re-used by share-generics.
    let _ = Foo(0);

    generic_fn(0.0f32, 1.0f32)
}

pub struct Foo(pub u32);

impl Drop for Foo {
    fn drop(&mut self) {
        println!("foo");
    }
}
