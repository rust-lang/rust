//@ run-rustfix

//issue #106496

struct S;

trait X {}
impl X for S {}

fn foo<T: X>(_: &T) {}
fn test_foo() {
    let hello = &S;
    foo(*hello);
    //~^ ERROR mismatched types
}

fn bar(_: &String) {}
fn test_bar() {
    let v = String::from("hello");
    let s = &v;
    bar(*s);
    //~^ ERROR mismatched types
}

fn main() {
    test_foo();
    test_bar();
}
