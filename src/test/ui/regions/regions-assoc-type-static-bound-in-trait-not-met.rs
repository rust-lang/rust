// Test that the compiler checks that the 'static bound declared in
// the trait must be satisfied on the impl. Issue #20890.

trait Foo {
    type Value: 'static;
    fn dummy(&self) { }
}

impl<'a> Foo for &'a i32 {
    //~^ ERROR cannot infer
    type Value = &'a i32;
}

impl<'a> Foo for i32 {
    // OK.
    type Value = i32;
}

fn main() { }
