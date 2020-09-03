#![feature(const_in_array_repeat_expressions)]

#[derive(Copy, Clone)]
struct Foo<T>(T);

fn main() {
    [Foo(String::new()); 4];
    //~^ ERROR the trait bound `Foo<String>: Copy` is not satisfied [E0277]
}
