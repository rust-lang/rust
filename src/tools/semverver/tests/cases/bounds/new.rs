pub struct Abc<A: Clone> {
    pub a: A,
}

pub struct Def<A> {
    pub d: A,
}

pub fn abc<A: Clone>(_: A) {}

pub fn def<A>(_: A) {}
