pub trait Bar: Super<SuperAssoc: Bound> {}

pub trait Super {
    type SuperAssoc;
}

pub trait Bound {}
