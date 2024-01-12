// compile-flags: -Zeagerly-emit-delayed-bugs

trait Foo {}

fn main() {}

fn f() -> impl Foo {
    //~^ ERROR the trait bound `i32: Foo` is not satisfied
    //~| ERROR `report_selection_error` did not emit an error
    1i32
}
