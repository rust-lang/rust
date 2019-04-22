// ignore-tidy-linelength

// revisions: nll_target

// The following revisions are disabled due to missing support from two-phase beyond autorefs
//[nll_beyond]compile-flags: -Z borrowck=mir -Z two-phase-beyond-autoref
//[nll_beyond] should-fail

//[nll_target]compile-flags: -Z borrowck=mir

// This is a corner case that the current implementation is (probably)
// treating more conservatively than is necessary. But it also does
// not seem like a terribly important use case to cover.
//
// So this test is just making a note of the current behavior, with
// the caveat that in the future, the rules may be loosened, at which
// point this test might be thrown out.
//
// The convention for the listed revisions: "lxl" means lexical
// lifetimes (which can be easier to reason about). "nll" means
// non-lexical lifetimes. "nll_target" means the initial conservative
// two-phase borrows that only applies to autoref-introduced borrows.
// "nll_beyond" means the generalization of two-phase borrows to all
// `&mut`-borrows (doing so makes it easier to write code for specific
// corner cases).

fn main() {
    let mut vec = vec![0, 1];
    let delay: &mut Vec<_>;
    {
        let shared = &vec;

        // we reserve here, which could (on its own) be compatible
        // with the shared borrow. But in the current implementation,
        // its an error.
        delay = &mut vec;
        //[nll_beyond]~^  ERROR cannot borrow `vec` as mutable because it is also borrowed as immutable
        //[nll_target]~^^ ERROR cannot borrow `vec` as mutable because it is also borrowed as immutable

        shared[0];
    }

    // the &mut-borrow only becomes active way down here.
    //
    // (At least in theory; part of the reason this test fails is that
    // the constructed MIR throws in extra &mut reborrows which
    // flummoxes our attmpt to delay the activation point here.)
    delay.push(2);
}
