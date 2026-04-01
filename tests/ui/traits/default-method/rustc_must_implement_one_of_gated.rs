#[rustc_must_implement_one_of(eq, neq)]
//~^ ERROR use of an internal attribute [E0658]
//~| NOTE the `#[rustc_must_implement_one_of]` attribute is an internal implementation detail that will never be stable
//~| NOTE the `#[rustc_must_implement_one_of]` attribute is used to change minimal complete definition of a trait. Its syntax and semantics are highly experimental and will be subject to change before stabilization
trait Equal {
    fn eq(&self, other: &Self) -> bool {
        !self.neq(other)
    }

    fn neq(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

fn main() {}
