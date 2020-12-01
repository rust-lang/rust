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
struct WrapParam<'a, T>(&'a &'a T);

const WRAP_DOUBLY_INDIRECT_PARAM: & &WrapParam<NoDerive> = & &WrapParam(& & NoDerive(0));

fn main() {
    match WRAP_DOUBLY_INDIRECT_PARAM {
        WRAP_DOUBLY_INDIRECT_PARAM => { panic!("WRAP_DOUBLY_INDIRECT_PARAM matched itself"); }
        //~^ WARN must be annotated with `#[derive(PartialEq, Eq)]`
        //~| WARN this was previously accepted
        _ => { println!("WRAP_DOUBLY_INDIRECT_PARAM correctly did not match itself"); }
    }
}
