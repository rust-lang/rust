// Test that an invariant region parameter used in a contravariant way
// yields an error.
//
// Note: see variance-regions-*.rs for the tests that check that the
// variance inference works in the first place.

// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

struct Invariant<'a> {
    f: &'a mut &'a isize
}

fn use_<'short,'long>(c: Invariant<'long>,
                      s: &'short isize,
                      l: &'long isize,
                      _where:Option<&'short &'long ()>) {

    // Test whether Invariant<'long> <: Invariant<'short>.  Since
    // 'short <= 'long, this would be true if the Invariant type were
    // contravariant with respect to its parameter 'a.

    let _: Invariant<'short> = c;
    //[base]~^ ERROR E0623
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() { }
