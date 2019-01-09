#![feature(optin_builtin_traits)]

auto trait MyTrait {}

struct MyS;

struct MyS2;

impl !MyTrait for MyS2 {}

fn is_mytrait<T: MyTrait>() {}

fn main() {
    is_mytrait::<MyS>();

    is_mytrait::<(MyS2, MyS)>();
    //~^ ERROR `MyS2: MyTrait` is not satisfied
}
