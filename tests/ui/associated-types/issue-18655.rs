//@ run-pass
trait Factory {
    type Product;
    fn create(&self) -> <Self as Factory>::Product;
}

impl Factory for f64 {
    type Product = f64;
    fn create(&self) -> f64 { *self * *self }
}

impl<A: Factory, B: Factory> Factory for (A, B) {
    type Product = (<A as Factory>::Product, <B as Factory>::Product);
    fn create(&self) -> (<A as Factory>::Product, <B as Factory>::Product) {
        let (ref a, ref b) = *self;
        (a.create(), b.create())
    }
}

fn main() {
    assert_eq!((16., 25.), (4., 5.).create());
}
