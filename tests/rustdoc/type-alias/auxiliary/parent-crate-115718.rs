pub struct MyStruct<T>(T);

pub trait MyTrait1 {
    fn method_trait_1();
}

impl MyTrait1 for MyStruct<u16> {
    fn method_trait_1() {}
}
