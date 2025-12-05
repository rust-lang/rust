//@ run-pass

#![allow(non_local_definitions)]

// This test is checking our logic for structural match checking by enumerating
// the different kinds of const expressions. This test is collecting cases where
// we have accepted the const expression as a pattern in the past and wish to
// continue doing so.
//
// Even if a non-structural-match type is part of an expression in a const's
// definition, that does not necessarily disqualify the const from being a match
// pattern: in principle, we just need the types involved in the final value to
// be structurally matchable.

// See also RFC 1445

#![feature(type_ascription)]

#[derive(Copy, Clone, Debug)]
struct NoPartialEq(u32);

#[derive(Copy, Clone, Debug)]
struct NoDerive(u32);

// This impl makes `NoDerive` irreflexive.
impl PartialEq for NoDerive { fn eq(&self, _: &Self) -> bool { false } }
impl Eq for NoDerive { }

type OND = Option<NoDerive>;

fn main() {
    const FIELD1: u32 = NoPartialEq(1).0;
    match 1 { FIELD1 => dbg!(FIELD1), _ => panic!("whoops"), };
    const FIELD2: u32 = NoDerive(1).0;
    match 1 { FIELD2 => dbg!(FIELD2), _ => panic!("whoops"), };

    enum CLike { One = 1, #[allow(dead_code)] Two = 2, }
    const ONE_CAST: u32 = CLike::One as u32;
    match 1 { ONE_CAST => dbg!(ONE_CAST), _ => panic!("whoops"), };

    const NO_DERIVE_NONE: OND = None;
    const INDIRECT: OND = NO_DERIVE_NONE;
    match None { INDIRECT => dbg!(INDIRECT), _ => panic!("whoops"), };

    const TUPLE: (OND, OND) = (None, None);
    match (None, None) { TUPLE => dbg!(TUPLE), _ => panic!("whoops"), };

    const TYPE_ASCRIPTION: OND = type_ascribe!(None, OND);
    match None { TYPE_ASCRIPTION => dbg!(TYPE_ASCRIPTION), _ => panic!("whoops"), };

    const ARRAY: [OND; 2] = [None, None];
    match [None; 2] { ARRAY => dbg!(ARRAY), _ => panic!("whoops"), };

    const REPEAT: [OND; 2] = [None; 2];
    match [None, None] { REPEAT => dbg!(REPEAT), _ => panic!("whoops"), };

    trait Trait: Sized { const ASSOC: Option<Self>; }
    impl Trait for NoDerive { const ASSOC: Option<NoDerive> = None; }
    match None { NoDerive::ASSOC => dbg!(NoDerive::ASSOC), _ => panic!("whoops"), };

    const BLOCK: OND = { NoDerive(10); None };
    match None { BLOCK => dbg!(BLOCK), _ => panic!("whoops"), };

    const ADDR_OF: &OND = &None;
    match &None { ADDR_OF => dbg!(ADDR_OF),  _ => panic!("whoops"), };

    // These ones are more subtle: the final value is fine, but statically analyzing the expression
    // that computes the value would likely (incorrectly) have us conclude that this may match on
    // values that do not have structural equality.
    const INDEX: Option<NoDerive> = [None, Some(NoDerive(10))][0];
    match None { Some(_) => panic!("whoops"), INDEX => dbg!(INDEX), };

    const fn build() -> Option<NoDerive> { None }
    const CALL: Option<NoDerive> = build();
    match None { Some(_) => panic!("whoops"), CALL => dbg!(CALL), };

    impl NoDerive { const fn none() -> Option<NoDerive> { None } }
    const METHOD_CALL: Option<NoDerive> = NoDerive::none();
    match None { Some(_) => panic!("whoops"), METHOD_CALL => dbg!(METHOD_CALL), };
}
