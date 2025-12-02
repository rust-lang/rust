//@ run-pass
//@ edition:2021
//@ proc-macro: reserved-prefixes-macro-2018.rs
//@ ignore-backends: gcc

extern crate reserved_prefixes_macro_2018 as m2018;

fn main() {
    // Ok, even though *this* crate is 2021:
    assert_eq!(m2018::number_of_tokens_in_a_prefixed_integer_literal!(), 3);
    assert_eq!(m2018::number_of_tokens_in_a_prefixed_char_literal!(), 3);
    assert_eq!(m2018::number_of_tokens_in_a_prefixed_string_literal!(), 3);
}
