// This is part of a set of tests exploring the different ways a
// structural-match ADT might try to hold a
// non-structural-match in hidden manner that lets matches
// through that we had intended to reject.
//
// See discussion on rust-lang/rust#62307 and rust-lang/rust#62339

struct NoDerive(#[allow(dead_code)] i32);

// This impl makes NoDerive irreflexive.
impl PartialEq for NoDerive { fn eq(&self, _: &Self) -> bool { false } }

impl Eq for NoDerive { }

#[derive(PartialEq, Eq)]
struct WrapInline(NoDerive);

const WRAP_DIRECT_INLINE: WrapInline = WrapInline(NoDerive(0));

fn main() {
    match WRAP_DIRECT_INLINE {
        WRAP_DIRECT_INLINE => { panic!("WRAP_DIRECT_INLINE matched itself"); }
        //~^ ERROR constant of non-structural type `NoDerive` in a pattern
        _ => { println!("WRAP_DIRECT_INLINE did not match itself"); }
    }
}
