// This test checks that the span of c-variadics is handled correctly when creating obligations

struct NeedsCopy<T: Copy>(T);

extern "C" {
    fn f1(...) -> NeedsCopy<String>;
    //~^ ERROR the trait bound `String: Copy` is not satisfied
    fn f2(x: i32, ...) -> NeedsCopy<String>;
    //~^ ERROR the trait bound `String: Copy` is not satisfied
    fn f3(..., x: i32) -> NeedsCopy<String>;
    //~^ ERROR `...` must be the last argument of a C-variadic function
    //~| ERROR the trait bound `String: Copy` is not satisfied
}

fn main() { }
