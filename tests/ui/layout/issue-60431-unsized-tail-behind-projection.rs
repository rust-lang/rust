// rust-lang/rust#60431: This is a scenario where to determine the size of
// `&Ref<Obstack>`, we need to know the concrete type of the last field in
// `Ref<Obstack>` (i.e. its "struct tail"), and determining that concrete type
// requires normalizing `Obstack::Dyn`.
//
// The old "struct tail" computation did not perform such normalization, and so
// the compiler would ICE when trying to figure out if `Ref<Obstack>` is a
// dynamically-sized type (DST).

//@ run-pass

use std::mem;

pub trait Arena {
    type Dyn : ?Sized;
}

pub struct DynRef {
    _dummy: [()],
}

pub struct Ref<A: Arena> {
    _value: u8,
    _dyn_arena: A::Dyn,
}

pub struct Obstack;

impl Arena for Obstack {
    type Dyn = DynRef;
}

fn main() {
    assert_eq!(mem::size_of::<&Ref<Obstack>>(), mem::size_of::<&[()]>());
}
