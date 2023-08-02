// compile-flags: -Ztrait-solver=next
// run-pass

trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

struct Foo;
impl Foo {
    fn new() -> Self { Foo }
}

fn main() {
    <Foo as Mirror>::Assoc::new();
}
