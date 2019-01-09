// Test that the compiler checks that arbitrary region bounds declared
// in the trait must be satisfied on the impl. Issue #20890.

trait Foo<'a> {
    type Value: 'a;
    fn dummy(&'a self) { }
}

impl<'a> Foo<'a> for &'a i16 {
    // OK.
    type Value = &'a i32;
}

impl<'a> Foo<'static> for &'a i32 {
    //~^ ERROR cannot infer
    type Value = &'a i32;
}

impl<'a,'b> Foo<'b> for &'a i64 {
    //~^ ERROR cannot infer
    type Value = &'a i32;
}

fn main() { }
