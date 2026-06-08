#![deny(rustdoc::broken_intra_doc_links)]
//~^ NOTE the lint level is defined here

/// [Self::a::b]
//~^ ERROR unresolved link to `Self::a::b`
//~| NOTE the struct `MyStruct` has no field or associated item named `a`
pub struct MyStruct;
/// [Self::a::b]
//~^ ERROR unresolved link to `Self::a::b`
//~| NOTE the type alias `MyAlias` has no associated item named `a`
pub type MyAlias = MyStruct;
