//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/153849>

#[expect(dead_code)]
// Must be invariant
pub struct Server<T>(*mut T);
impl<T> Server<T> {
    fn new(_: T) -> Self
    where
        // Must be higher-ranked
        T: Fn(&mut i32),
    {
        todo!()
    }
}

fn main() {
    // Must have a type annotation
    let _: Server<_> = Server::new(|_| ());
}
