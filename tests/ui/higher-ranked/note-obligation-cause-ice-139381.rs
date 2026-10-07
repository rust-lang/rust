//! regression test for a failed debug assertion that was caused by improper
//! handling of bound clauses in `note_obligation_cause_code_inner`.
//! See https://github.com/rust-lang/rust/issues/139381#event-32605696260.

//@ check-fail
//@ needs-rustc-debug-assertions

trait A<'a> {
    type Assoc: ?Sized;
}

impl<'a> A<'a> for () {
    type Assoc = &'a ();
}

fn hello() -> impl for<'a> A<'a, Assoc: Into<u8> + 'static + Copy> {
    //~^ ERROR not satisfied
    //~| NOTE not implemented for `u8`
    //~| NOTE in this expansion of desugaring
    //~| NOTE in this expansion of desugaring
    //~| NOTE required for `&'a ()` to implement `for<'a> Into<u8>`
    //~| NOTE in this expansion of desugaring
    ()
}

fn main() {}
