use super::*;
use crate::create_default_session_globals_then;

#[test]
fn interner_tests() {
    let i = Interner::prefill(&[]);
    // first one is zero:
    assert_eq!(i.intern_str("dog"), Symbol::new(0));
    // re-use gets the same entry, even with a `ByteSymbol`
    assert_eq!(i.intern_byte_str(b"dog"), ByteSymbol::new(0));
    // different string gets a different #:
    assert_eq!(i.intern_byte_str(b"cat"), ByteSymbol::new(1));
    assert_eq!(i.intern_str("cat"), Symbol::new(1));
    // dog is still at zero
    assert_eq!(i.intern_str("dog"), Symbol::new(0));
}

#[test]
fn without_first_quote_test() {
    create_default_session_globals_then(|| {
        let i = Ident::from_str("'break");
        assert_eq!(i.without_first_quote().name, kw::Break);
    });
}

#[allow(non_upper_case_globals)]
mod extra_symbols_macro {
    use crate::{Symbol, sym};

    mod a {
        use crate::*;

        extra_symbols! {
            #[macro_export] extra_symbols_plus_a;

            Symbols {
                DEFINED_IN_A,
                std,
            }
        }
    }

    mod b {
        use crate::*;

        extra_symbols_plus_a! {
            Symbols {
                DEFINED_IN_A,
                DEFINED_IN_B,
                core,
                std,
            }
        }
    }

    #[test]
    fn extra_symbols() {
        // Extra symbols are preinterned but not considered predefined
        assert!(!Symbol::is_predefined(a::DEFINED_IN_A.as_u32()));
        assert!(!Symbol::is_predefined(b::DEFINED_IN_B.as_u32()));

        assert_eq!(sym::std, a::std);
        assert_eq!(sym::std, b::std);

        assert_eq!(a::DEFINED_IN_A, b::DEFINED_IN_A);

        assert_eq!(a::DUPLICATE_SYMBOLS, [sym::std]);
        assert_eq!(b::DUPLICATE_SYMBOLS, [a::DEFINED_IN_A, sym::core, sym::std]);
    }
}
