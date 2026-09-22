//@ compile-flags: -Znext-solver

// The impl GAT must satisfy the trait declaration's where clauses even when
// its item bounds are normalized using an environment equality.

trait Foo {
    type T<'a>: Sized where Self: 'a;
}

impl Foo for &() {
    type T<'a> = (); //~ ERROR lifetime bound not satisfied
}

fn main() {}
