// run-pass
use std::iter::{Fuse, Cloned};
use std::slice::Iter;

struct Foo<'a, T: 'a>(&'a T);
impl<'a, T: 'a> Copy for Foo<'a, T> {}
impl<'a, T: 'a> Clone for Foo<'a, T> {
    fn clone(&self) -> Self { *self }
}

fn copy_ex() {
    let s = 2;
    let k1 = || s;
    let upvar = Foo(&k1);
    let k = || upvar;
    k();
}

fn main() {
    let _f: *mut <Fuse<Cloned<Iter<u8>>> as Iterator>::Item = std::ptr::null_mut();

    copy_ex();
}
