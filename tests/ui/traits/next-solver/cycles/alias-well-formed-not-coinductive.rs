//@ compile-flags: -Znext-solver
//@ check-fail

trait Bound {}

trait Needs {}

impl Needs for () {}

trait Trait {
    type Assoc<T>
    where
        T: Bound;
}

impl Trait for () {
    type Assoc<T>
        = ()
    where
        T: Bound;
}

// Normalizing `Assoc<()>` also requires its own `(): Bound` clause.
// Treating `AliasWellFormed` as coinductive would incorrectly make
// this cycle productive.
impl Bound for ()
//~^ ERROR overflow evaluating the requirement `<() as Trait>::Assoc<()> == _`
where
    <() as Trait>::Assoc<()>: Needs,
{}

fn require_bound<T: Bound>() {}

fn main() {
    require_bound::<()>();
    //~^ ERROR overflow evaluating the requirement `(): Bound`
}
