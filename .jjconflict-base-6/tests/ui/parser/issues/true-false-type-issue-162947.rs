//@ run-rustfix

struct A;

impl A {
    fn _a() -> true { //~ ERROR: expected type, found keyword `true`
        false
    }
    fn b(&self) {}
}

fn main() {
    let a = A;
    a.b();

    let _b: true = true; //~ ERROR: expected type, found keyword `true`
}
