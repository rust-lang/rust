#![allow(type_alias_bounds)]
#![allow(dead_code)]
// Test syntax checks for `?Sized` syntax.

use std::marker::PhantomData;

trait T1  { }
pub trait T2  { }
trait T3<X: T1> : T2 { }
trait T4<X: ?Sized> { }
trait T5<X: ?Sized, Y> { }
trait T6<Y, X: ?Sized> { }
trait T7<X: ?Sized, Y: ?Sized> { }
trait T8<X: ?Sized+T2> { }
trait T9<X: T2 + ?Sized> { }
struct S1<X: ?Sized>(PhantomData<X>);
enum E<X: ?Sized> { E1(PhantomData<X>) }
impl <X: ?Sized> T1 for S1<X> {}
fn f<X: ?Sized>() {}
type TT<T: ?Sized> = T;

pub fn main() {
}
