//@ edition:2018
//@ proc-macro: reserved-prefixes-macro-2018.rs
//@ proc-macro: reserved-prefixes-macro-2021.rs

extern crate reserved_prefixes_macro_2018 as m2018;
extern crate reserved_prefixes_macro_2021 as m2021;

fn main() {
    // Ok:
    m2018::number_of_tokens_in_a_prefixed_integer_literal!();
    m2018::number_of_tokens_in_a_prefixed_char_literal!();
    m2018::number_of_tokens_in_a_prefixed_string_literal!();

    // Error, even though *this* crate is 2018:
    m2021::number_of_tokens_in_a_prefixed_integer_literal!();
    //~^ ERROR prefix `hey` is unknown
    m2021::number_of_tokens_in_a_prefixed_char_literal!();
    //~^ ERROR prefix `hey` is unknown
    m2021::number_of_tokens_in_a_prefixed_string_literal!();
    //~^ ERROR prefix `hey` is unknown
}
