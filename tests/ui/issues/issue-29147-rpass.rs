//@ run-pass
#![recursion_limit="1024"]
#![allow(dead_code)]

use std::mem;

pub struct S0<T>(T,T);
pub struct S1<T>(Option<Box<S0<S0<T>>>>,Option<Box<S0<S0<T>>>>);
pub struct S2<T>(Option<Box<S1<S1<T>>>>,Option<Box<S1<S1<T>>>>);
pub struct S3<T>(Option<Box<S2<S2<T>>>>,Option<Box<S2<S2<T>>>>);
pub struct S4<T>(Option<Box<S3<S3<T>>>>,Option<Box<S3<S3<T>>>>);
pub struct S5<T>(Option<Box<S4<S4<T>>>>,Option<Box<S4<S4<T>>>>,Option<T>);

trait Foo { fn xxx(&self); }
/// some local of #[fundamental] trait
trait Bar {}

impl<T> Foo for T where T: Bar, T: Sync {
    fn xxx(&self) {}
}

impl Foo for S5<u8> { fn xxx(&self) {} }

fn main() {
    let s = S5(None,None,None);
    s.xxx();
    assert_eq!(mem::size_of_val(&s.2), mem::size_of::<Option<u8>>());
}
