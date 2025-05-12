// Regression test for #48071. This test used to ICE because -- in
// the leak-check -- it would pass since we knew that the return type
// was `'static`, and hence `'static: 'a` was legal even for a
// placeholder region, but in NLL land it would fail because we had
// rewritten `'static` to a region variable.
//
//@ check-pass

trait Foo {
    fn foo(&self) { }
}

impl Foo for () {
}

type MakeFooFn = for<'a> fn(&'a u8) -> Box<dyn Foo + 'a>;

fn make_foo(x: &u8) -> Box<dyn Foo + 'static> {
    Box::new(())
}

fn main() {
    let x: MakeFooFn = make_foo as MakeFooFn;
}
