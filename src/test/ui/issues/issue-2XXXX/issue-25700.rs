struct S<T: 'static>(Option<&'static T>);

trait Tr { type Out; }
impl<T> Tr for T { type Out = T; }

impl<T: 'static> Copy for S<T> where S<T>: Tr<Out=T> {}
impl<T: 'static> Clone for S<T> where S<T>: Tr<Out=T> {
    fn clone(&self) -> Self { *self }
}
fn main() {
    let t = S::<()>(None);
    drop(t);
    drop(t); //~ ERROR use of moved value
}
