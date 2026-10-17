//@ edition:2021
//@ proc-macro: reserved-guarded-strings-macro-2021.rs
//@ proc-macro: reserved-guarded-strings-macro-2024.rs

extern crate reserved_guarded_strings_macro_2021 as m2021;
extern crate reserved_guarded_strings_macro_2024 as m2024;

fn main() {
    // Ok:
    m2021::number_of_tokens_in_a_guarded_string_literal!();
    m2021::number_of_tokens_in_a_guarded_unterminated_string_literal!();

    // Error, even though *this* crate is 2021:
    m2024::number_of_tokens_in_a_guarded_string_literal!();
    //~^ ERROR invalid string literal
    m2024::number_of_tokens_in_a_guarded_unterminated_string_literal!();
    //~^ ERROR invalid string literal
}
