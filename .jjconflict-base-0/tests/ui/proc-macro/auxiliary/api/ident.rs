use proc_macro::{Ident, Span};

// FIXME: `Ident` does not yet implement `PartialEq<Ident>` directly (#146553)
fn assert_eq(l: Ident, r: Ident) {
    assert_eq!(l.to_string(), r.to_string());
}

fn assert_ne(l: Ident, r: Ident) {
    assert_ne!(l.to_string(), r.to_string());
}

fn new(s: &str) -> Ident {
    Ident::new(s, Span::call_site())
}

fn new_raw(s: &str) -> Ident {
    Ident::new_raw(s, Span::call_site())
}

const LATIN_CAPITAL_LETTER_K: &str = "K";
const KELVIN_SIGN: &str = "K";

const NORMAL_MIDDLE_DOT: &str = "L·L";
const GREEK_ANO_TELEIA: &str = "L·L";

pub fn test() {
    assert_eq(new("foo"), new("foo"));
    assert_ne(new("foo"), new_raw("foo"));

    assert_ne!(LATIN_CAPITAL_LETTER_K, KELVIN_SIGN);
    assert_eq(new(LATIN_CAPITAL_LETTER_K), new(KELVIN_SIGN));
    assert_eq(new_raw(LATIN_CAPITAL_LETTER_K), new_raw(KELVIN_SIGN));

    assert_ne!(NORMAL_MIDDLE_DOT, GREEK_ANO_TELEIA);
    assert_eq(new(NORMAL_MIDDLE_DOT), new(GREEK_ANO_TELEIA));
    assert_eq(new_raw(NORMAL_MIDDLE_DOT), new_raw(GREEK_ANO_TELEIA));
}
