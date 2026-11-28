// Make sure `#[expect]` on an item with a derive is fulfilled when the lint is only emitted on the
// item itself, and not also on the derived impl, which reuses the spans of the input.
//
// See <https://github.com/rust-lang/rust/issues/150553> for rationale.

//@ check-fail

#![deny(non_camel_case_types, unused_parens, unused_braces, unused_qualifications)]

#[derive(Clone)]
#[expect(non_camel_case_types)]
pub struct TypeParam<t>(t); //~ ERROR type parameter `t`

#[derive(Clone, Copy, PartialEq, Eq)]
#[expect(unused_parens)]
pub struct Parens((u8)); //~ ERROR unnecessary parentheses
//~| ERROR unnecessary parentheses

#[derive(Clone, Copy, PartialEq, Eq)]
#[expect(unused_braces)]
pub struct Braces([u8; { 4 }]); //~ ERROR unnecessary braces
//~| ERROR unnecessary braces

#[derive(Clone, Copy, PartialEq, Eq)]
#[expect(unused_qualifications)]
pub struct Qualified(std::primitive::u8); //~ ERROR unnecessary qualification
//~| ERROR unnecessary qualification

fn main() {}
