// RFC 1445 introduced `#[structural_match]`; this attribute must
// appear on the `struct`/`enum` definition for any `const` used in a
// pattern.
//
// This is our (forever-unstable) way to mark a datatype as having a
// `PartialEq` implementation that is equivalent to recursion over its
// substructure. This avoids (at least in the short term) any need to
// resolve the question of what semantics is used for such matching.
// (See RFC 1445 for more details and discussion.)

// Issue 62307 pointed out a case where the structural-match checking
// was too shallow.
#![warn(indirect_structural_match, nontrivial_structural_match)]
// run-pass

#[derive(Debug)]
struct B(i32);

// Overriding `PartialEq` to use this strange notion of "equality" exposes
// whether `match` is using structural-equality or method-dispatch
// under the hood, which is the antithesis of rust-lang/rfcs#1445
impl PartialEq for B {
    fn eq(&self, other: &B) -> bool { std::cmp::min(self.0, other.0) == 0 }
}

fn main() {
    const RR_B0: & & B = & & B(0);
    const RR_B1: & & B = & & B(1);

    match RR_B0 {
        RR_B1 => { println!("CLAIM RR0: {:?} matches {:?}", RR_B1, RR_B0); }
        //~^ WARN must be annotated with `#[derive(PartialEq, Eq)]`
        //~| WARN this was previously accepted
        _ => { }
    }

    match RR_B1 {
        RR_B1 => { println!("CLAIM RR1: {:?} matches {:?}", RR_B1, RR_B1); }
        //~^ WARN must be annotated with `#[derive(PartialEq, Eq)]`
        //~| WARN this was previously accepted
        _ => { }
    }
}
