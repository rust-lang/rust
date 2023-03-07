#[derive(Clone)]
struct S;

// without Clone
struct T;

fn foo(_: S) {}

fn test1() {
    let s = &S;
    foo(s); //~ ERROR mismatched types
}

fn bar(_: T) {}
fn test2() {
    let t = &T;
    bar(t); //~ ERROR mismatched types
}

fn main() {
    test1();
    test2();
}
