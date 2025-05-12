//! This test checks that the param env canonicalization cache
//! does not end up with inconsistent values.

//@ check-pass

pub fn poison1() -> impl Sized
where
    (): 'static,
{
}
pub fn poison2() -> impl Sized
where
    (): 'static,
{
    define_by_query((poison2, ()));
}
pub fn poison3() -> impl Sized
where
    (): 'static,
{
}

trait Query {}
impl<Out, F: Fn() -> Out> Query for (F, Out) {}
fn define_by_query(_: impl Query) {}

fn main() {}
