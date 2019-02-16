// This is testing whether `#[inline]` signals an error or warning
// when put in "weird" places.
//
// (This file sits on its own because it actually signals an error,
// which would mess up the treatment of other cases in
// issue-43106-gating-of-builtin-attrs.rs)

// Crate-level is accepted, though it is almost certainly unused?
#![inline]

#[inline]
//~^ ERROR attribute should be applied to function or closure
mod inline {
    mod inner { #![inline] }
    //~^ ERROR attribute should be applied to function or closure

    #[inline = "2100"] fn f() { }
    //~^ WARN attribute must be of the form
    //~| WARN this was previously accepted

    #[inline] struct S;
    //~^ ERROR attribute should be applied to function or closure

    #[inline] type T = S;
    //~^ ERROR attribute should be applied to function or closure

    #[inline] impl S { }
    //~^ ERROR attribute should be applied to function or closure
}

fn main() {}
