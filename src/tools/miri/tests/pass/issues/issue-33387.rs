use std::sync::Arc;

trait Foo {}

impl Foo for [u8; 2] {}

fn main() {
    let _val: Arc<dyn Foo + Send> = Arc::new([3, 4]);
}
