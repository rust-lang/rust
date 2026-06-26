//@ run-pass
struct S<T: 'static>(#[allow(dead_code)] Option<&'static T>);

trait Tr { type Out; }
impl<T> Tr for T { type Out = T; }

impl<T: 'static> Copy for S<T> where S<T>: Tr<Out=T> {}
impl<T: 'static> Clone for S<T> where S<T>: Tr<Out=T> {
    fn clone(&self) -> Self { *self }
}
fn main() {
    S::<()>(None);
}
