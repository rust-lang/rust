// Regression test for #143481, where we were calling `predicates_of` on
// a Crate HIR node because we were using a dummy obligation cause's body id
// without checking that it was meaningful first.

trait Role {
    type Inner;
}
struct HandshakeCallback<C>(C);
impl<C: Clone> Role for HandshakeCallback {
    //~^ ERROR missing generics
    type Inner = usize;
}
struct Handshake<R: Role>(R::Inner);
fn accept() -> Handshake<HandshakeCallback<()>> {
    todo!()
}

fn main() {}
