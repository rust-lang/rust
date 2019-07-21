// ignore-tidy-linelength
#![feature(const_in_array_repeat_expressions)]

#[derive(Copy, Clone)]
struct Foo<T>(T);

fn main() {
    [Foo(String::new()); 4];
    //~^ ERROR the trait bound `Foo<std::string::String>: std::marker::Copy` is not satisfied [E0277]
}
