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
    //~^ ERROR type annotations needed [E0283]
}
