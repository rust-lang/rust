use super::*;

use crate::{edition, Globals};

#[test]
fn interner_tests() {
    let mut i: Interner = Interner::default();
    // first one is zero:
    assert_eq!(i.intern("dog"), Symbol::new(0));
    // re-use gets the same entry:
    assert_eq!(i.intern("dog"), Symbol::new(0));
    // different string gets a different #:
    assert_eq!(i.intern("cat"), Symbol::new(1));
    assert_eq!(i.intern("cat"), Symbol::new(1));
    // dog is still at zero
    assert_eq!(i.intern("dog"), Symbol::new(0));
    let z = i.intern("zebra");
    assert_eq!(i.gensymed(z), Symbol::new(SymbolIndex::MAX_AS_U32));
    // gensym of same string gets new number:
    assert_eq!(i.gensymed(z), Symbol::new(SymbolIndex::MAX_AS_U32 - 1));
    // gensym of *existing* string gets new number:
    let d = i.intern("dog");
    assert_eq!(i.gensymed(d), Symbol::new(SymbolIndex::MAX_AS_U32 - 2));
}

#[test]
fn without_first_quote_test() {
    GLOBALS.set(&Globals::new(edition::DEFAULT_EDITION), || {
        let i = Ident::from_str("'break");
        assert_eq!(i.without_first_quote().name, kw::Break);
    });
}
