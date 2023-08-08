// This is part of a set of tests exploring the different ways a
// structural-match ADT might try to hold a
// non-structural-match in hidden manner that lets matches
// through that we had intended to reject.
//
// See discussion on rust-lang/rust#62307 and rust-lang/rust#62339
#![warn(indirect_structural_match)]
// run-pass

struct NoDerive(#[allow(unused_tuple_struct_fields)] i32);

// This impl makes NoDerive irreflexive.
impl PartialEq for NoDerive { fn eq(&self, _: &Self) -> bool { false } }

impl Eq for NoDerive { }

#[derive(PartialEq, Eq)]
struct WrapInline<'a>(&'a &'a NoDerive);

const WRAP_DOUBLY_INDIRECT_INLINE: & &WrapInline = & &WrapInline(& & NoDerive(0));

fn main() {
    match WRAP_DOUBLY_INDIRECT_INLINE {
        WRAP_DOUBLY_INDIRECT_INLINE => { panic!("WRAP_DOUBLY_INDIRECT_INLINE matched itself"); }
        //~^ WARN must be annotated with `#[derive(PartialEq, Eq)]`
        //~| WARN this was previously accepted
        _ => { println!("WRAP_DOUBLY_INDIRECT_INLINE correctly did not match itself"); }
    }
}
