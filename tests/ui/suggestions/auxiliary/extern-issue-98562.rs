pub trait TraitE {
    type I3;
}

pub trait TraitD {
    type I3;
}

pub trait TraitC {
    type I1;
    type I2;
}

pub trait TraitB {
    type Item;
}

pub trait TraitA<G1, G2, G3> {
    fn baz<
        U: TraitC<I1 = G1, I2 = G2> + TraitD<I3 = G3> + TraitE,
        V: TraitD<I3 = G1>
    >(_: U, _: V) -> Self
    where
        U: TraitB,
        <U as TraitB>::Item: Copy;
}
