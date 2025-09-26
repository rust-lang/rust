//@ revisions: current next
//@[next] compile-flags: -Znext-solver

// If we don't treat `impl Sized` as rigid, the first call would
// resolve to the trait method, constraining the opaque, while the
// second call would resolve to the inherent method.
//
// We avoid cases like this by rejecting candidates which constrain
// opaque types encountered in the autoderef chain.
//
// FIXME(-Znext-solver): ideally we would note that the inference variable
// is an opaque type in the error message and change this to a type annotations
// needed error.

trait Trait: Sized {
    fn method(self) {}
}
impl Trait for &Foo {}

struct Foo;
impl Foo {
    fn method(&self) {}
}

fn define_opaque(b: bool) -> impl Sized {
    if b {
        let x = &define_opaque(false);
        x.method();
        //~^ ERROR no method named `method` found for reference
        x.method();
        //~^ ERROR no method named `method` found for reference
    }

    Foo
}

fn main() {
    define_opaque(true);
}
