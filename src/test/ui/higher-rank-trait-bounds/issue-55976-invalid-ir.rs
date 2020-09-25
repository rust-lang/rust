// Regression test for issue #55976
// Tests that we don't generate invalid LLVM IR when certain
// higher-ranked trait bounds are involved.

// run-pass

pub struct Foo<T>(T, [u8; 64]);

pub fn abc<'a>(x: &Foo<Box<dyn for<'b> Fn(&'b u8)>>) -> &Foo<Box<dyn Fn(&'a u8)>> { x }

fn main() {
    abc(&Foo(Box::new(|_x| ()), [0; 64]));
}
