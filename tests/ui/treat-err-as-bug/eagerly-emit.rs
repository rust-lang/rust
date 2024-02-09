// compile-flags: -Zeagerly-emit-delayed-bugs

trait Foo {}

fn main() {}

fn f() -> impl Foo {
    //~^ ERROR the trait bound `i32: Foo` is not satisfied
    1i32
}
