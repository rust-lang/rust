// This test seems to involve a "stashed diagnostic" (or at least it used to at the time of
// writing). Ensure we handle that correctly.

pub trait Trait {
    type Assoc: Assoc;
}

pub trait Assoc {}

fn main() {}
