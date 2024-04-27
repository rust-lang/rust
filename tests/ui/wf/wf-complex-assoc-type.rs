trait MyTrait {}
struct AssertMyTrait<T: MyTrait>(T);

trait HelperTrait {
    type MyItem;
}

impl HelperTrait for () {
    type MyItem = Option<((AssertMyTrait<bool>, u8))>; //~ ERROR the trait bound
}

fn main() {}
