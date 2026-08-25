//! <https://github.com/rust-lang/rust/issues/21974>.
//!
//! Test that (for now) we report an ambiguity error here, because
//! specific trait relationships are ignored for the purposes of trait
//! matching. This behavior should likely be improved such that this
//! test passes.

trait Foo {
    fn foo(self);
}

fn foo<'a,'b,T>(x: &'a T, y: &'b T)
    where &'a T : Foo, //~ ERROR type annotations needed
          &'b T : Foo
{
    x.foo(); //~ ERROR type annotations needed
    y.foo();
}

fn main() { }
