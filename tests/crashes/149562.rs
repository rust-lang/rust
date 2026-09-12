//@ known-bug: #149562
//@ needs-rustc-debug-assertions
fn a<T>() -> T
where
    T: ?Sized,
    T: ?Sized,
{
}

fn main() {}
