// Regression test for #130310
// Tests that we do not fall into infinite
// recursion while checking FFI safety of
// recursive types like `A<T>` below

//@ build-pass
use std::marker::PhantomData;

#[repr(C)]
struct A<T> {
    a: *const A<A<T>>, // Recursive because of this field
    p: PhantomData<T>,
}

extern "C" {
    fn f(a: *const A<()>);
    //~^ WARN `extern` block uses type `*const A<()>`, which is not FFI-safe
}

fn main() {}
