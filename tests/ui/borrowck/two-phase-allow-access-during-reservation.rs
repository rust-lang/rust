//@ revisions: nll_target

// The following revisions are disabled due to missing support for two_phase_beyond_autoref
//@ unused-revision-names: nll_beyond
//@[nll_beyond] compile-flags: -Z two_phase_beyond_autoref

// This is the second counter-example from Niko's blog post
// smallcultfollowing.com/babysteps/blog/2017/03/01/nested-method-calls-via-two-phase-borrowing/
//
// It is "artificial". It is meant to illustrate directly that we
// should allow an aliasing access during reservation, but *not* while
// the mutable borrow is active.
//
// The convention for the listed revisions: "lxl" means lexical
// lifetimes (which can be easier to reason about). "nll" means
// non-lexical lifetimes. "nll_target" means the initial conservative
// two-phase borrows that only applies to autoref-introduced borrows.
// "nll_beyond" means the generalization of two-phase borrows to all
// `&mut`-borrows (doing so makes it easier to write code for specific
// corner cases).

fn main() {
    /*0*/ let mut i = 0;

    /*1*/ let p = &mut i; // (reservation of `i` starts here)

    /*2*/ let j = i;      // OK: `i` is only reserved here
                          //[nll_target]~^  ERROR cannot use `i` because it was mutably borrowed [E0503]

    /*3*/ *p += 1;        // (mutable borrow of `i` starts here, since `p` is used)

    /*4*/ let k = i;      //[nll_beyond]~  ERROR cannot use `i` because it was mutably borrowed [E0503]
                          //[nll_target]~^ ERROR cannot use `i` because it was mutably borrowed [E0503]

    /*5*/ *p += 1;

    let _ = (j, k, p);
}
