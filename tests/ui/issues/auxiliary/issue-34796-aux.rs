#![crate_type = "lib"]
pub trait Future {
    type Item;
    type Error;
}

impl Future for u32 {
    type Item = ();
    type Error = Box<()>;
}

fn foo() -> Box<Future<Item=(), Error=Box<()>>> {
    Box::new(0u32)
}

pub fn bar<F, A, B>(_s: F)
    where F: Fn(A) -> B,
{
    foo();
}
