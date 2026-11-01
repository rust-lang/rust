//@ known-bug: #150263
//@ compile-flags: --crate-type lib -C opt-level=3

pub trait Scope {
    type Timestamp;
}
impl<G> Scope for G {
    type Timestamp = ();
}

pub fn create<G: Scope>() {
    enter::<G>();
}

fn enter<G>() {
    unary::<G>(|_: <G as Scope>::Timestamp| {});
}

fn unary<G: Scope>(constructor: impl FnOnce(G::Timestamp)) {
    constructor(None.unwrap());
}
