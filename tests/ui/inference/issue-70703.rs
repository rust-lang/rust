//@ check-pass

trait Factory {
    type Product;
}

impl Factory for () {
    type Product = ();
}

trait ProductConsumer<P> {
    fn consume(self, product: P);
}

impl<P> ProductConsumer<P> for () {
    fn consume(self, _: P) {}
}

fn make_product_consumer<F: Factory>(_: F) -> impl ProductConsumer<F::Product> {
    ()
}

fn main() {
    let consumer = make_product_consumer(());
    consumer.consume(());
}
