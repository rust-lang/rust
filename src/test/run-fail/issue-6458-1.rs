// error-pattern:explicit panic

fn foo<T>(t: T) {}
fn main() {
    foo(panic!())
}
