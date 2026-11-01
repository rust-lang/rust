//! Auxiliary crate for regression test of https://github.com/rust-lang/rust/issues/34796
#![crate_type = "lib"]
pub trait Future {
    type Item;
    type Error;
}

impl Future for u32 {
    type Item = ();
    type Error = Box<()>;
}

fn foo() -> Box<dyn Future<Item=(), Error=Box<()>>> {
    Box::new(0u32)
}

pub fn bar<F, A, B>(_s: F)
    where F: Fn(A) -> B,
{
    foo();
}
