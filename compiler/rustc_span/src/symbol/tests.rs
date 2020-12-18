use super::*;

use crate::{edition, SessionGlobals};

#[test]
fn interner_tests() {
    let mut i: Interner = Interner::default();
    let dog = i.intern("dog");
    // re-use gets the same entry:
    assert_eq!(i.intern("dog").as_u32(), dog.as_u32());
    // different string gets a different #:
    let cat = i.intern("cat");
    assert_ne!(dog.as_u32(), cat.as_u32());
    assert_eq!(i.intern("cat").as_u32(), cat.as_u32());
    assert_eq!(i.intern("dog").as_u32(), dog.as_u32());
}

#[test]
fn without_first_quote_test() {
    SESSION_GLOBALS.set(&SessionGlobals::new(edition::DEFAULT_EDITION), || {
        let i = Ident::from_str("'break");
        assert_eq!(i.without_first_quote().name, kw::Break);
    });
}

#[test]
fn test_static_symbols() {
    assert_eq!(Symbol::intern_static(""), Some(kw::Invalid));
    assert_eq!(Symbol::intern_static("not in the static table"), None);
    assert!(Symbol::intern_static("fn").is_some()); // don't care about exact index

    // check round-tripping
    for &string in ["as", "fn", "let", "trait", "size_of_val"].iter() {
        let sym = Symbol::intern_static(string).unwrap();
        assert_eq!(string, &*sym.as_str(), "sym #{}", sym.0.as_u32());
    }
}

#[test]
fn test_ident_is_special() {
    for &s in [kw::Invalid, kw::PathRoot, kw::DollarCrate, kw::Underscore].iter() {
        let ident = Ident::with_dummy_span(s);
        assert_eq!(ident.is_special(), true, "s = {:?}", s);
    }

    for &s in [kw::As, kw::Break, kw::UnderscoreLifetime].iter() {
        let ident = Ident::with_dummy_span(s);
        assert_eq!(ident.is_special(), false, "s = {:?}", s);
    }
}

#[test]
fn test_symbol_as_str() {
    SESSION_GLOBALS.set(&SessionGlobals::new(edition::Edition::Edition2018), || {
        for &(sym, string) in [
            (kw::Invalid, ""),
            (kw::PathRoot, "{{root}}"),
            (kw::DollarCrate, "$crate"),
            (kw::As, "as"),
            (kw::Break, "break"),
            (kw::While, "while"),
            (kw::Union, "union"),
            (sym::Alignment, "Alignment"),
            (sym::Arc, "Arc"),
            (sym::zmm_reg, "zmm_reg"),
            (sym::i64, "i64"),
        ]
        .iter()
        {
            let as_str = sym.as_str();
            assert_eq!(&*as_str, string);

            let sym2 = Symbol::intern(string);
            assert_eq!(sym, sym2, "sym={} sym2={}", sym.as_u32(), sym2.as_u32());
        }

        let colon = Symbol::intern(":");
        assert_eq!(&*colon.as_str(), ":");
    });
}

#[test]
fn test_dynamic_symbols() {
    crate::with_session_globals(crate::edition::Edition::Edition2018, || {
        let s1 = Symbol::intern("fuzzy wuzzy");
        assert!(!s1.is_static());
        assert_eq!(&*s1.as_str(), "fuzzy wuzzy");
    });
}
