#![recursion_limit="1024"]
#![allow(dead_code)]

pub struct S0<T>(T,T);
pub struct S1<T>(Option<Box<S0<S0<T>>>>,Option<Box<S0<S0<T>>>>);
pub struct S2<T>(Option<Box<S1<S1<T>>>>,Option<Box<S1<S1<T>>>>);
pub struct S3<T>(Option<Box<S2<S2<T>>>>,Option<Box<S2<S2<T>>>>);
pub struct S4<T>(Option<Box<S3<S3<T>>>>,Option<Box<S3<S3<T>>>>);
pub struct S5<T>(Option<Box<S4<S4<T>>>>,Option<Box<S4<S4<T>>>>,Option<T>);

trait Foo { fn xxx(&self); }
trait Bar {} // anything local or #[fundamental]

impl<T> Foo for T where T: Bar, T: Sync {
    fn xxx(&self) {}
}

impl Foo for S5<u32> { fn xxx(&self) {} }
impl Foo for S5<u64> { fn xxx(&self) {} }

fn main() {
    let _ = <S5<_>>::xxx; //~ ERROR type annotations needed
}
