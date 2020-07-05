pub struct Abc<A> {
    pub a: A,
}

pub struct Def<A: Clone> {
    pub d: A,
}

pub fn abc<A>(_: A) {}

pub fn def<A: Clone>(_: A) {}
