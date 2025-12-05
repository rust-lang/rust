//@ known-bug: #140303
//@compile-flags: -Zvalidate-mir
use std::future::Future;
async fn a() -> impl Sized {
    b(c)
}
async fn c(); // kaboom
fn b<d>(e: d) -> impl Sized
where
    d: f,
{
    || -> <d>::h { panic!() }
}
trait f {
    type h;
}
impl<d, g> f for d
where
    d: Fn() -> g,
    g: Future,
{
}
