pub trait MyTrait {
    /// docs for my_trait_method
    fn my_trait_method() {}
}

pub struct MyStruct;

impl MyTrait for MyStruct {}
