// This test checks the diplay of "/* private fields */" sentence in tuple structs.
#![crate_name = "foo"]

//@ has 'foo/struct.A.html' '//*[@class="rust item-decl"]/code' 'pub struct A(pub u8, _);'
pub struct A(pub u8, u8);
//@ has 'foo/struct.B.html' '//*[@class="rust item-decl"]/code' 'pub struct B(_, pub u8);'
pub struct B(u8, pub u8);
//@ has 'foo/struct.C.html' '//*[@class="rust item-decl"]/code' 'pub struct C(_, pub u8, _);'
pub struct C(u8, pub u8, u8);
//@ has 'foo/struct.D.html' '//*[@class="rust item-decl"]/code' 'pub struct D(pub u8, _, pub u8);'
pub struct D(pub u8, u8, pub u8);
//@ has 'foo/struct.E.html' '//*[@class="rust item-decl"]/code' 'pub struct E(/* private fields */);'
pub struct E(u8);
//@ has 'foo/struct.F.html' '//*[@class="rust item-decl"]/code' 'pub struct F(/* private fields */);'
pub struct F(u8, u8);
