#![allow(order_dependent_trait_objects)]
trait Trait {}

impl Trait for dyn Send + Sync {}
impl Trait for dyn Sync + Send {}
fn assert_trait<T: Trait + ?Sized>() {}

fn main() {
    assert_trait::<dyn Send + Sync>();
    //~^ ERROR type annotations needed: cannot satisfy `dyn Send + Sync: Trait`
}
