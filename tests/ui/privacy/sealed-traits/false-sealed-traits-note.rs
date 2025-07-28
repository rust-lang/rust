// We should not emit sealed traits note, see issue #143392 and #143121

/// Reported in #143392
mod inner {
    pub trait TraitA {}

    pub trait TraitB: TraitA {}
}

struct Struct;

impl inner::TraitB for Struct {} //~ ERROR the trait bound `Struct: TraitA` is not satisfied [E0277]

/// Reported in #143121
mod x {
    pub trait A {}
    pub trait B: A {}

    pub struct C;
    impl B for C {} //~ ERROR the trait bound `C: A` is not satisfied [E0277]
}

fn main(){}
