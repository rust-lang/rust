// xfail-test
fn foo<T>() {
    struct foo {
        mut x: T, //~ ERROR quux
        drop { }
    }
}
fn main() { }
