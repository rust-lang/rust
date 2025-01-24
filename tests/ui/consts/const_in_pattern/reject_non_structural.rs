//@ compile-flags: -Zdeduplicate-diagnostics=yes

// This test of structural match checking enumerates the different kinds of
// const definitions, collecting cases where the const pattern is rejected.
//
// Note: Even if a non-structural-match type is part of an expression in a
// const's definition, that does not necessarily disqualify the const from being
// a match pattern: in principle, we just need the types involved in the final
// value to be structurally matchable.

// See also RFC 1445

#![feature(type_ascription)]

#[derive(Copy, Clone, Debug)]
struct NoPartialEq;

#[derive(Copy, Clone, Debug)]
struct NoDerive;
//~^ NOTE must be annotated with `#[derive(PartialEq)]`
//~| NOTE must be annotated with `#[derive(PartialEq)]`
//~| NOTE must be annotated with `#[derive(PartialEq)]`
//~| NOTE must be annotated with `#[derive(PartialEq)]`
//~| NOTE must be annotated with `#[derive(PartialEq)]`
//~| NOTE must be annotated with `#[derive(PartialEq)]`
//~| NOTE must be annotated with `#[derive(PartialEq)]`
//~| NOTE must be annotated with `#[derive(PartialEq)]`
//~| NOTE must be annotated with `#[derive(PartialEq)]`
//~| NOTE must be annotated with `#[derive(PartialEq)]`

// This impl makes `NoDerive` irreflexive.
impl PartialEq for NoDerive { fn eq(&self, _: &Self) -> bool { false } }
//~^ NOTE StructuralPartialEq.html for details
//~| NOTE StructuralPartialEq.html for details
//~| NOTE StructuralPartialEq.html for details
//~| NOTE StructuralPartialEq.html for details
//~| NOTE StructuralPartialEq.html for details
//~| NOTE StructuralPartialEq.html for details
//~| NOTE StructuralPartialEq.html for details
//~| NOTE StructuralPartialEq.html for details
//~| NOTE StructuralPartialEq.html for details
//~| NOTE StructuralPartialEq.html for details

impl Eq for NoDerive { }

type OND = Option<NoDerive>;

struct TrivialEq(OND);

// This impl makes `TrivialEq` trivial.
impl PartialEq for TrivialEq { fn eq(&self, _: &Self) -> bool { true } }

impl Eq for TrivialEq { }

fn main() {
    #[derive(PartialEq, Eq, Debug)]
    enum Derive<X> { Some(X), None, }

    const ENUM: Derive<NoDerive> = Derive::Some(NoDerive); //~ NOTE constant defined here
    match Derive::Some(NoDerive) { ENUM => dbg!(ENUM), _ => panic!("whoops"), };
    //~^ ERROR constant of non-structural type `NoDerive` in a pattern
    //~| NOTE constant of non-structural type

    const FIELD: OND = TrivialEq(Some(NoDerive)).0; //~ NOTE constant defined here
    match Some(NoDerive) { FIELD => dbg!(FIELD), _ => panic!("whoops"), };
    //~^ ERROR constant of non-structural type `NoDerive` in a pattern
    //~| NOTE constant of non-structural type

    const NO_DERIVE_SOME: OND = Some(NoDerive);
    const INDIRECT: OND = NO_DERIVE_SOME; //~ NOTE constant defined here
    match Some(NoDerive) {INDIRECT => dbg!(INDIRECT), _ => panic!("whoops"), };
    //~^ ERROR constant of non-structural type `NoDerive` in a pattern
    //~| NOTE constant of non-structural type

    const TUPLE: (OND, OND) = (None, Some(NoDerive)); //~ NOTE constant defined here
    match (None, Some(NoDerive)) { TUPLE => dbg!(TUPLE), _ => panic!("whoops"), };
    //~^ ERROR constant of non-structural type `NoDerive` in a pattern
    //~| NOTE constant of non-structural type

    const TYPE_ASCRIPTION: OND = type_ascribe!(Some(NoDerive), OND); //~ NOTE constant defined here
    match Some(NoDerive) { TYPE_ASCRIPTION => dbg!(TYPE_ASCRIPTION), _ => panic!("whoops"), };
    //~^ ERROR constant of non-structural type `NoDerive` in a pattern
    //~| NOTE constant of non-structural type

    const ARRAY: [OND; 2] = [None, Some(NoDerive)]; //~ NOTE constant defined here
    match [None, Some(NoDerive)] { ARRAY => dbg!(ARRAY), _ => panic!("whoops"), };
    //~^ ERROR constant of non-structural type `NoDerive` in a pattern
    //~| NOTE constant of non-structural type

    const REPEAT: [OND; 2] = [Some(NoDerive); 2]; //~ NOTE constant defined here
    match [Some(NoDerive); 2] { REPEAT => dbg!(REPEAT), _ => panic!("whoops"), };
    //~^ ERROR constant of non-structural type `NoDerive` in a pattern
    //~| NOTE constant of non-structural type

    trait Trait: Sized { const ASSOC: Option<Self>; } //~ NOTE constant defined here
    impl Trait for NoDerive { const ASSOC: Option<NoDerive> = Some(NoDerive); }
    match Some(NoDerive) { NoDerive::ASSOC => dbg!(NoDerive::ASSOC), _ => panic!("whoops"), };
    //~^ ERROR constant of non-structural type `NoDerive` in a pattern
    //~| NOTE constant of non-structural type

    const BLOCK: OND = { NoDerive; Some(NoDerive) }; //~ NOTE constant defined here
    match Some(NoDerive) { BLOCK => dbg!(BLOCK), _ => panic!("whoops"), };
    //~^ ERROR constant of non-structural type `NoDerive` in a pattern
    //~| NOTE constant of non-structural type

    const ADDR_OF: &OND = &Some(NoDerive); //~ NOTE constant defined here
    match &Some(NoDerive) { ADDR_OF => dbg!(ADDR_OF), _ => panic!("whoops"), };
    //~^ ERROR constant of non-structural type `NoDerive` in a pattern
    //~| NOTE constant of non-structural type
}
