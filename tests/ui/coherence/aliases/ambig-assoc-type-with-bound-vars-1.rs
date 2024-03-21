// Regression test for #114061. We previously did
// not consider `for<'a> <T as WithAssoc<'a>>::Assoc: IsUnit`
// to be unknowable, even though the projection is
// ambiguous.
trait IsUnit {}
impl IsUnit for () {}


pub trait WithAssoc<'a> {
    type Assoc;
}

// The two impls of `Trait` overlap
pub trait Trait {}
impl<T> Trait for T
where
    T: 'static,
    for<'a> T: WithAssoc<'a>,
    for<'a> <T as WithAssoc<'a>>::Assoc: IsUnit,
{
}
impl<T> Trait for Box<T> {}
//~^ ERROR conflicting implementations of trait `Trait`

fn main() {}
