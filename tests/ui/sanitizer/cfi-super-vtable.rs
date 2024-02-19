// Check that super-traits with vptrs have their shims generated

//@ needs-sanitizer-cfi
//@ compile-flags: --crate-type=bin -Cprefer-dynamic=off -Clto -Zsanitizer=cfi
//@ compile-flags: -C codegen-units=1 -C opt-level=0
//@ build-pass

trait Parent1 {
    fn p1(&self);
}

trait Parent2 {
    fn p2(&self);
}

// We need two parent traits to force the vtable upcasting code to choose to add a pointer to
// another vtable to the child. This vtable is generated even if trait upcasting is not in use.
trait Child : Parent1 + Parent2 {
    fn c(&self);
}

struct Foo;

impl Parent1 for Foo {
    fn p1(&self) {}
}

impl Parent2 for Foo {
    fn p2(&self) {}
}

impl Child for Foo {
    fn c(&self) {}
}

fn main() {
    let x = &Foo as &dyn Child;
    x.c();
}
