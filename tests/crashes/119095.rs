//@ known-bug: #119095
//@ edition: 2021

fn any<T>() -> T {
    loop {}
}

trait Acquire {
    type Connection;
}

impl Acquire for &'static () {
    type Connection = ();
}

trait Unit {}
impl Unit for () {}

fn get_connection<T>() -> impl Unit
where
    T: Acquire,
    T::Connection: Unit,
{
    any::<T::Connection>()
}

fn main() {
    let future = async { async { get_connection::<&'static ()>() }.await };

    future.resolve_me();
}

trait ResolveMe {
    fn resolve_me(self);
}

impl<S> ResolveMe for S
where
    (): CheckSend<S>,
{
    fn resolve_me(self) {}
}

trait CheckSend<F> {}
impl<F> CheckSend<F> for () where F: Send {}

trait NeverImplemented {}
impl<E, F> CheckSend<F> for E where E: NeverImplemented {}
