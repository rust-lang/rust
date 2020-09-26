// This is part of a set of tests exploring the different ways a
// structural-match ADT might try to hold a
// non-structural-match in hidden manner that lets matches
// through that we had intended to reject.
//
// See discussion on rust-lang/rust#62307 and rust-lang/rust#62339
#![warn(indirect_structural_match)]
// run-pass

struct NoDerive(i32);

// This impl makes NoDerive irreflexive.
impl PartialEq for NoDerive { fn eq(&self, _: &Self) -> bool { false } }

impl Eq for NoDerive { }

#[derive(PartialEq, Eq)]
struct WrapInline(NoDerive);

const WRAP_INDIRECT_INLINE: & &WrapInline = & &WrapInline(NoDerive(0));

fn main() {
    match WRAP_INDIRECT_INLINE {
        WRAP_INDIRECT_INLINE => { panic!("WRAP_INDIRECT_INLINE matched itself"); }
        //~^ WARN must be annotated with `#[derive(PartialEq, Eq)]`
        //~| WARN this was previously accepted
        _ => { println!("WRAP_INDIRECT_INLINE did not match itself"); }
    }
}
