#[derive(Clone)]
struct S;

trait X {}

impl X for S {}

fn foo<T: X>(_: T) {}
fn bar<T: X>(s: &T) {
    foo(s); //~ ERROR trait `X` is not implemented for `&T`
}

fn bar_with_clone<T: X + Clone>(s: &T) {
    foo(s); //~ ERROR trait `X` is not implemented for `&T`
}

fn main() {
    let s = &S;
    bar(s);
}
