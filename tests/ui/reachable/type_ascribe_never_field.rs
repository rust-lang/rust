// Regression test for <https://github.com/rust-lang/rust/issues/149431>
//
// Checks that type ascription of a field place with type never is correctly
// checked for if it constitutes a read of type never. (it doesn't)
//
//@ check-pass

#![feature(never_type)]
#![feature(type_ascription)]
#![deny(unreachable_code)]

fn main() {
   let x: (!,);
   let _ = type_ascribe!(x.0, _);

   (); // reachable
}
