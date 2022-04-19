// Check that explicit region bounds are allowed on the various
// nominal types (but not on other types) and that they are type
// checked.

// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

struct Foo;

impl Foo {
    fn some_method<A:'static>(self) { }
}

fn caller<'a>(x: &isize) {
    Foo.some_method::<&'a isize>();
    //[base]~^ ERROR does not fulfill the required lifetime
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() { }
