#[derive(Copy, Clone)]
struct Foo<T>(T);

fn main() {
    [Foo(String::new()); 4];
    //~^ ERROR the trait bound `String: Copy` is not satisfied [E0277]
}
