#![crate_name = "foo"]

// ignore-tidy-linelength

pub struct MyStruct;

impl MyTrait for MyStruct {

// @has foo/struct.MyStruct.html '//a/@href' '../foo/struct.MyStruct.html#associatedtype.AssoType'

    /// [`AssoType`]
    ///
    /// [`AssoType`]: MyStruct::AssoType
    type AssoType = u32;

// @has foo/struct.MyStruct.html '//a/@href' '../foo/struct.MyStruct.html#associatedconstant.ASSO_CONST'

    /// [`ASSO_CONST`]
    ///
    /// [`ASSO_CONST`]: MyStruct::ASSO_CONST
    const ASSO_CONST: i32 = 10;

// @has foo/struct.MyStruct.html '//a/@href' '../foo/struct.MyStruct.html#method.trait_fn'

    /// [`trait_fn`]
    ///
    /// [`trait_fn`]: MyStruct::trait_fn
    fn trait_fn() { }
}

pub trait MyTrait {
    type AssoType;
    const ASSO_CONST: i32 = 1;
    fn trait_fn();
}
