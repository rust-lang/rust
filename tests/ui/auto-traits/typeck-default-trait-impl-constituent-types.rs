#![feature(auto_traits)]
#![feature(negative_impls)]

auto trait MyTrait {}

impl<T> !MyTrait for *mut T {}

struct MyS;

struct MyS2;

impl !MyTrait for MyS2 {}

struct MyS3;

fn is_mytrait<T: MyTrait>() {}

fn main() {
    is_mytrait::<MyS>();

    is_mytrait::<MyS2>();
    //~^ ERROR `MyS2: MyTrait` is not satisfied
}
