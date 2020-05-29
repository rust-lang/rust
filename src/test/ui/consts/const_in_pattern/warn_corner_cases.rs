// run-pass

// This test is checking our logic for structural match checking by enumerating
// the different kinds of const expressions. This test is collecting cases where
// we have accepted the const expression as a pattern in the past but we want
// to begin warning the user that a future version of Rust may start rejecting
// such const expressions.

// The specific corner cases we are exploring here are instances where the
// const-evaluator computes a value that *does* meet the conditions for
// structural-match, but the const expression itself has abstractions (like
// calls to const functions) that may fit better with a type-based analysis
// rather than a committment to a specific value.

#![warn(indirect_structural_match)]

#[derive(Copy, Clone, Debug)]
struct NoDerive(u32);

// This impl makes `NoDerive` irreflexive.
impl PartialEq for NoDerive { fn eq(&self, _: &Self) -> bool { false } }
impl Eq for NoDerive { }

fn main() {
    const INDEX: Option<NoDerive> = [None, Some(NoDerive(10))][0];
    match None { Some(_) => panic!("whoops"), INDEX => dbg!(INDEX), };
    //~^ WARN must be annotated with `#[derive(PartialEq, Eq)]`
    //~| WARN this was previously accepted

    const fn build() -> Option<NoDerive> { None }
    const CALL: Option<NoDerive> = build();
    match None { Some(_) => panic!("whoops"), CALL => dbg!(CALL), };
    //~^ WARN must be annotated with `#[derive(PartialEq, Eq)]`
    //~| WARN this was previously accepted

    impl NoDerive { const fn none() -> Option<NoDerive> { None } }
    const METHOD_CALL: Option<NoDerive> = NoDerive::none();
    match None { Some(_) => panic!("whoops"), METHOD_CALL => dbg!(METHOD_CALL), };
    //~^ WARN must be annotated with `#[derive(PartialEq, Eq)]`
    //~| WARN this was previously accepted
}
