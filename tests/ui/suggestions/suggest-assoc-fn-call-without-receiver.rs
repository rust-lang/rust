// run-rustfix

struct A {}

impl A {
    fn hello(_a: i32) {}
    fn test(_a: Self, _b: i32) {}
}

fn main() {
    let _a = A {};
    _a.hello(1);
    //~^ ERROR no method named `hello` found
    _a.test(1);
    //~^ ERROR no method named `test` found
}
