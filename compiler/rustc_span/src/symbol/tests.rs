use super::*;
use crate::create_default_session_globals_then;

#[test]
fn interner_tests() {
    let i = Interner::prefill(&[]);
    let dog = i.intern("dog");
    // first one is zero:
    assert_eq!(i.intern("dog"), dog);
    // re-use gets the same entry:
    assert_eq!(i.intern("dog"), dog);
    // different string gets a different #:
    let cat = i.intern("cat");
    assert_eq!(i.intern("cat"), cat);
    assert_eq!(i.intern("cat"), cat);
    // dog is still at zero
    assert_eq!(i.intern("dog"), dog);
}

#[test]
fn without_first_quote_test() {
    create_default_session_globals_then(|| {
        let i = Ident::from_str("'break");
        assert_eq!(i.without_first_quote().name, kw::Break);
    });
}
