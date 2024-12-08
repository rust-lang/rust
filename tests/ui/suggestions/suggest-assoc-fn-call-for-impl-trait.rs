//@ run-rustfix

struct A {

}

trait M {
    fn foo(_a: Self);
    fn bar(_a: Self);
    fn baz(_a: i32);
}

impl M for A {
    fn foo(_a: Self) {}
    fn bar(_a: A) {}
    fn baz(_a: i32) {}
}

fn main() {
    let _a = A {};
    _a.foo();
    //~^ ERROR no method named `foo` found
    _a.baz(0);
    //~^ ERROR no method named `baz` found

    let _b = A {};
    _b.bar();
    //~^ ERROR no method named `bar` found
}
