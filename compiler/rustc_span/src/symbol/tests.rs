use super::*;

use crate::{edition, SessionGlobals};

#[test]
fn symbol_tests() {
    let sym = |n| Some(Symbol(SymbolIndex::from_u32(n)));

    // Simple ASCII symbols.
    assert_eq!(Symbol::try_new_inlined(""), sym(0x00_00_00_00));
    assert_eq!(Symbol::try_new_inlined("a"), sym(0x00_00_00_61));
    assert_eq!(Symbol::try_new_inlined("ab"), sym(0x00_00_62_61));
    assert_eq!(Symbol::try_new_inlined("abc"), sym(0x00_63_62_61));
    assert_eq!(Symbol::try_new_inlined("abcd"), sym(0x64_63_62_61));
    assert_eq!(Symbol::try_new_inlined("abcde"), None); // too long
    assert_eq!(Symbol::try_new_inlined("abcdefghijklmnopqrstuvwxyz"), None); // too long

    // Symbols involving non-ASCII chars.
    // Note that the UTF-8 sequence for 'é' is `[0xc3, 0xa9]`.
    assert_eq!(Symbol::try_new_inlined("é"), sym(0x00_00_a9_c3));
    assert_eq!(Symbol::try_new_inlined("dé"), sym(0x00_a9_c3_64));
    assert_eq!(Symbol::try_new_inlined("édc"), sym(0x63_64_a9_c3));
    assert_eq!(Symbol::try_new_inlined("cdé"), None); // byte 3 (0xa9) is > 0x7f

    // Symbols involving NUL chars.
    assert_eq!(Symbol::try_new_inlined("\0"), None); // last byte is NUL
    assert_eq!(Symbol::try_new_inlined("a\0"), None); // last byte is NUL
    assert_eq!(Symbol::try_new_inlined("\0a"), sym(0x00_00_61_00));
    assert_eq!(Symbol::try_new_inlined("aa\0"), None); // last byte is NUL
    assert_eq!(Symbol::try_new_inlined("\0\0a"), sym(0x00_61_00_00));
    assert_eq!(Symbol::try_new_inlined("aaa\0"), None); // last byte is NUL
    assert_eq!(Symbol::try_new_inlined("\0\0\0a"), sym(0x61_00_00_00));

    // Tabled symbols.
    assert_eq!(Symbol::new_tabled(0).as_u32(), 0x80000000);
    assert_eq!(Symbol::new_tabled(5).as_u32(), 0x80000005);
    assert_eq!(Symbol::new_tabled(0x123456).as_u32(), 0x80123456);

    // Tabled symbol indices.
    assert_eq!(Symbol::new_tabled(0).as_tabled_index(), 0);
    assert_eq!(Symbol::new_tabled(5).as_tabled_index(), 5);
    assert_eq!(Symbol::new_tabled(0x123456).as_tabled_index(), 0x123456);
}

#[test]
fn symbol_interner_tests() {
    SESSION_GLOBALS.set(&SessionGlobals::new(edition::DEFAULT_EDITION), || {
        let inlined = |s, n, len| {
            // Check the symbol and the deinterned string look right.
            let sym = Symbol::intern(s);
            assert_eq!(sym.as_u32(), n);
            sym.with(|w| w == s);
            assert_eq!(sym.as_str(), s);
            assert_eq!(sym.as_str().len(), len);
        };

        let tabled = |s, len| {
            // Check the symbol and the deinterned string look right.
            let sym = Symbol::intern(s);
            assert!(sym.as_u32() & 0x80000000 != 0);
            sym.with(|w| w == s);
            assert_eq!(sym.as_str(), s);
            assert_eq!(sym.as_str().len(), len);
        };

        // Inlined symbols, lengths 1..=4.
        // Note that the UTF-8 sequence for 'é' is `[0xc3, 0xa9]`.
        inlined("", 0x00_00_00_00, 0);
        inlined("a", 0x00_00_00_61, 1);
        inlined("é", 0x00_00_a9_c3, 2);
        inlined("dé", 0x00_a9_c3_64, 3);
        inlined("édc", 0x63_64_a9_c3, 4);

        // Tabled symbols.
        tabled("abcde", 5); // tabled due to length
        tabled("cdé", 4); // tabled due to the fourth byte being > 0x7f
        tabled("a\0", 2); // tabled due to the last byte being NUL

        // Test `without_first_quote()`.
        let i = Ident::from_str("'break");
        assert_eq!(i.without_first_quote().name, kw::Break);
    });
}

#[test]
fn interner_tests() {
    let mut i: Interner = Interner::default();

    // Note that going directly through `Interner` means that no inlined
    // symbols are made.

    // First long one is zero.
    assert_eq!(i.intern("dog"), Symbol::new_tabled(0));
    // Re-use gets the same entry.
    assert_eq!(i.intern("dog"), Symbol::new_tabled(0));
    // Different string gets a different index.
    assert_eq!(i.intern("salamander"), Symbol::new_tabled(1));
    assert_eq!(i.intern("salamander"), Symbol::new_tabled(1));
    // Dog is still at zero.
    assert_eq!(i.intern("dog"), Symbol::new_tabled(0));
}
