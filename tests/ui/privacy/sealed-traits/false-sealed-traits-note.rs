// We should not emit sealed traits note, see issue #143392

mod inner {
    pub trait TraitA {}

    pub trait TraitB: TraitA {}
}

struct Struct;

impl inner::TraitB for Struct {} //~ ERROR the trait bound `Struct: TraitA` is not satisfied [E0277]

fn main(){}
