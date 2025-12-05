//@ run-pass
//@ edition:2024
//@ proc-macro: reserved-guarded-strings-macro-2021.rs
//@ ignore-backends: gcc

extern crate reserved_guarded_strings_macro_2021 as m2021;

fn main() {
    // Ok, even though *this* crate is 2024:
    assert_eq!(m2021::number_of_tokens_in_a_guarded_string_literal!(), 3);
    assert_eq!(m2021::number_of_tokens_in_a_guarded_unterminated_string_literal!(), 2);
}
