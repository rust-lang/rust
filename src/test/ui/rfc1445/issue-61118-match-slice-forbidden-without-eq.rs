// Issue 61118 pointed out a case where we hit an ICE during code gen:
// the compiler assumed that `PartialEq` was always implemented on any
// use of a `const` item in a pattern context, but the pre-existing
// checking for the presence of `#[structural_match]` was too shallow
// (see rust-lang/rust#62307), and so we hit cases where we were
// trying to dispatch to `PartialEq` on types that did not implement
// that trait.

struct B(i32);

const A: &[B] = &[];

pub fn main() {
    match &[][..] {
        A => (),
        //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]`
        _ => (),
    }
}
