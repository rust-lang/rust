// This test checks that the union keyword
// is accepted as the name of an enum variant
// when not followed by an identifier
// This special case exists because `union` is a contextual keyword.

#![allow(warnings)]

//@ check-pass

enum A { union }
enum B { union {} }
enum C { union() }
fn main(){}
