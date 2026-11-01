#![crate_name = "foo"]


//@ has foo/index.html '//a/@href' 'struct.Foo.html#method.new'
//@ has foo/struct.Foo.html '//a/@href' 'struct.Foo.html#method.new'

/// Use [`new`] to create a new instance.
///
/// [`new`]: Self::new
pub struct Foo;

impl Foo {
    pub fn new() -> Self {
        unimplemented!()
    }
}

//@ has foo/index.html '//a/@href' 'struct.Bar.html#method.new2'
//@ has foo/struct.Bar.html '//a/@href' 'struct.Bar.html#method.new2'

/// Use [`new2`] to create a new instance.
///
/// [`new2`]: Self::new2
pub struct Bar;

impl Bar {
    pub fn new2() -> Self {
        unimplemented!()
    }
}

pub struct MyStruct {
    //@ has foo/struct.MyStruct.html '//a/@href' 'struct.MyStruct.html#structfield.struct_field'

    /// [`struct_field`]
    ///
    /// [`struct_field`]: Self::struct_field
    pub struct_field: u8,
}

pub enum MyEnum {
    //@ has foo/enum.MyEnum.html '//a/@href' 'enum.MyEnum.html#variant.EnumVariant'

    /// [`EnumVariant`]
    ///
    /// [`EnumVariant`]: Self::EnumVariant
    EnumVariant,
}

pub union MyUnion {
    //@ has foo/union.MyUnion.html '//a/@href' 'union.MyUnion.html#structfield.union_field'

    /// [`union_field`]
    ///
    /// [`union_field`]: Self::union_field
    pub union_field: f32,
}

pub trait MyTrait {
    //@ has foo/trait.MyTrait.html '//a/@href' 'trait.MyTrait.html#associatedtype.AssoType'

    /// [`AssoType`]
    ///
    /// [`AssoType`]: Self::AssoType
    type AssoType;

    //@ has foo/trait.MyTrait.html '//a/@href' 'trait.MyTrait.html#associatedconstant.ASSO_CONST'

    /// [`ASSO_CONST`]
    ///
    /// [`ASSO_CONST`]: Self::ASSO_CONST
    const ASSO_CONST: i32 = 1;

    //@ has foo/trait.MyTrait.html '//a/@href' 'trait.MyTrait.html#method.asso_fn'

    /// [`asso_fn`]
    ///
    /// [`asso_fn`]: Self::asso_fn
    fn asso_fn() {}
}

impl MyStruct {
    //@ has foo/struct.MyStruct.html '//a/@href' 'struct.MyStruct.html#method.for_impl'

    /// [`for_impl`]
    ///
    /// [`for_impl`]: Self::for_impl
    pub fn for_impl() {
        unimplemented!()
    }
}

impl MyTrait for MyStruct {
    //@ has foo/struct.MyStruct.html '//a/@href' 'struct.MyStruct.html#associatedtype.AssoType'

    /// [`AssoType`]
    ///
    /// [`AssoType`]: Self::AssoType
    type AssoType = u32;

    //@ has foo/struct.MyStruct.html '//a/@href' 'struct.MyStruct.html#associatedconstant.ASSO_CONST'

    /// [`ASSO_CONST`]
    ///
    /// [`ASSO_CONST`]: Self::ASSO_CONST
    const ASSO_CONST: i32 = 10;

    //@ has foo/struct.MyStruct.html '//a/@href' 'struct.MyStruct.html#method.asso_fn'

    /// [`asso_fn`]
    ///
    /// [`asso_fn`]: Self::asso_fn
    fn asso_fn() {
        unimplemented!()
    }
}
