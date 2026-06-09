trait TraitA {
    type TypeA;
}

trait TraitD {
    type TypeD;
}

pub trait TraitB {
    type TypeB: TraitD;

    fn f(_: &<Self::TypeB as TraitD>::TypeD);
}

pub trait TraitC<E> {
    type TypeC<'a>: TraitB;

    fn g<'a>(_: &<<Self::TypeC<'a> as TraitB>::TypeB as TraitA>::TypeA);
    //~^ ERROR the trait bound `<<Self as TraitC<E>>::TypeC<'a> as TraitB>::TypeB: TraitA` is not satisfied
}

fn main() {}
