// Test that dyn Foo<Bar = T> is invariant with respect to T.
// Failure to enforce invariance here can be weaponized, see #71550 for details.

trait Foo {
    type Bar;
}

fn make() -> Box<dyn Foo<Bar = &'static u32>> {
    panic!()
}

fn take<'a>(_: &'a u32) {
    let _: Box<dyn Foo<Bar = &'a u32>> = make();
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
