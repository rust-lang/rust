// Check trait objects run correctly

//@ needs-sanitizer-cfi
//@ compile-flags: --crate-type=bin -Cprefer-dynamic=off -Clto -Zsanitizer=cfi
//@ compile-flags: -C codegen-units=1 -C opt-level=0
//@ run-pass

struct Bar;
trait Fooable {
    fn foo(&self) -> i32;
}

impl Fooable for Bar {
    fn foo(&self) -> i32 {
        3
    }
}

fn main() {
   let bar: Box<dyn Fooable> = Box::new(Bar);
   bar.foo();
}
