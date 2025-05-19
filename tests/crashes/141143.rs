//@ known-bug: #141143
trait TypedClient {
    fn publish_typed<F>(&self) -> impl Sized
    where
        F: Clone;
}
impl TypedClient for () {
    fn publish_typed<F>(&self) -> impl Sized {}
}

fn main() {
    ().publish_typed();
}
