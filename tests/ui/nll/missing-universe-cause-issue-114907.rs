// This is a non-regression test for issue #114907 where an ICE happened because of missing
// `UniverseInfo`s accessed during diagnostics.
//
// A couple notes:
// - the `FnOnce` bounds need an arg that is a reference
// - a custom `Drop` is needed somewhere in the type that `accept` returns, to create universes
//   during liveness and dropck outlives computation

//@ check-fail

trait Role {
    type Inner;
}

struct HandshakeCallback<C>(C);
impl<C: FnOnce(&())> Role for HandshakeCallback<C> {
    type Inner = ();
}

struct Handshake<R: Role> {
    _inner: Option<R::Inner>,
}
impl<R: Role> Drop for Handshake<R> {
    fn drop(&mut self) {}
}

fn accept<C: FnOnce(&())>(_: C) -> Handshake<HandshakeCallback<C>> {
    todo!()
}

fn main() {
    let callback = |_| {};
    accept(callback);
    //~^ ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `FnOnce` is not general enough
    //~| ERROR higher-ranked subtype error
    //~| ERROR higher-ranked subtype error
}
