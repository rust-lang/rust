#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

//@ compile-flags: -Zvalidate-mir
//@ normalize-stderr: "\d+-byte" -> "$$BYTE-byte"

type const N: usize = "this isn't a usize";
//~^ ERROR the constant `"this isn't a usize"` is not of type `usize`

fn f() -> [u8; const { N }] {}
//~^ ERROR transmuting from

fn main() {}
