#![feature(const_in_array_repeat_expressions)]

// check-pass

#[derive(Copy, Clone)]
struct Foo<T>(T);

fn main() {
    [Foo(String::new()); 4];
}
