//@ compile-flags: -Znext-solver
//@ check-pass

trait Trait {
    type Assoc;
}

// We have param env candidate for the trait goal but not the projection.
// Under such circumstance, consider object candidate if the self_ty is trait object.
fn foo<T>(x: <dyn Trait<Assoc = T> as Trait>::Assoc) -> T
where
    dyn Trait<Assoc = T>: Trait,
{
    x
}

fn main() {}
