// run-pass
// compile-flags: -Zsymbol-mangling-version=v0

pub fn f<T: ?Sized>() {}
pub trait Frob<T: ?Sized> {}
fn main() {
    f::<dyn Frob<str>>();
    f::<dyn for<'a> Frob<str>>();
}
