// Validate that objects that might have custom drop can be dropped with CFI on. See #118761

//@ needs-sanitizer-cfi
//@ compile-flags: --crate-type=bin -Cprefer-dynamic=off -Clto -Zsanitizer=cfi -C codegen-units=1
//@ compile-flags: -C opt-level=0
//@ run-pass

struct Bar;
trait Fooable {}
impl Fooable for Bar {}

fn main() {
   let _: Box<dyn Fooable> = Box::new(Bar);
}
