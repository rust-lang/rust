//@ check-pass

// Another minimized regression test for #112832.
trait Trait {
    type Assoc;
}

trait Sub<'a>: Trait<Assoc = <Self as Sub<'a>>::SubAssoc> {
    type SubAssoc;
}

// By using the where-clause we normalize `<T as Trait>::Assoc` to
// `<T as Sub<'a>>::SubAssoc` where `'a` is an unconstrained region
// variable.
fn foo<T>(x: <T as Trait>::Assoc)
where
    for<'a> T: Sub<'a>,
{}

fn main() {}
