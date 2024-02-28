#[derive(Copy, Clone)]
struct Foo<T>(T);

fn main() {
    [Foo(String::new()); 4];
    //~^ ERROR trait `Copy` is not implemented for `String`
}
