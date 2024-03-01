// Check alternate receivers work

//@ needs-sanitizer-cfi
//@ compile-flags: --crate-type=bin -Cprefer-dynamic=off -Clto -Zsanitizer=cfi
//@ compile-flags: -C codegen-units=1 -C opt-level=0
//@ run-pass

use std::sync::Arc;

trait Fooable {
    fn foo(self: Arc<Self>);
}

struct Bar;

impl Fooable for Bar {
    fn foo(self: Arc<Self>) {}
}

fn main() {
    let bar: Arc<dyn Fooable> = Arc::new(Bar);
    bar.foo();
}
