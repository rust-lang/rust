// Ensure that you are allowed to force *any* identifier to be treated as a keyword.
//
//@ edition: 2021..
//@ check-pass
#![feature(forced_keywords)]

macro_rules! discard { ($($tt:tt)*) => {} }

discard!(k#not_a_keyword); // OK

#[cfg(false)]
invoke!(k#arbitrary_string_of_chars);

fn main() {}
