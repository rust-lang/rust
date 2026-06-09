//@ run-pass
#![allow(unused_parens)]
#![allow(non_camel_case_types)]

// Note: This test was used to demonstrate #5873 (now #23898).

enum State { ST_NULL, ST_WHITESPACE }

fn main() {
    [State::ST_NULL; (State::ST_WHITESPACE as usize)];
}
