//@ compile-flags: -Znext-solver

// Make sure that, like the old trait solver, we end up requiring that the WC of
// impl GAT matches that of the trait. This is not a restriction that we *need*,
// but is a side-effect of registering the where clauses when normalizing the GAT
// when proving it satisfies its item bounds.

trait Foo {
    type T<'a>: Sized where Self: 'a;
}

impl Foo for &() {
    type T<'a> = (); //~ ERROR the type `&()` does not fulfill the required lifetime
}

fn main() {}
