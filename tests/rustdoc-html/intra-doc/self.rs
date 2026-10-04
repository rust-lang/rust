#![crate_name = "foo"]


//@ has foo/index.html '//a/@href' 'type.Foo.html#method.new'
//@ has foo/type.Foo.html '//a/@href' 'type.Foo.html#method.new'

/// Use [`new`] to create a new instance.
///
/// [`new`]: Self::new
pub struct Foo;

impl Foo {
    pub fn new() -> Self {
        unimplemented!()
    }
}

//@ has foo/index.html '//a/@href' 'type.Bar.html#method.new2'
//@ has foo/type.Bar.html '//a/@href' 'type.Bar.html#method.new2'

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
    //@ has foo/type.MyStruct.html '//a/@href' 'type.MyStruct.html#structfield.struct_field'

    /// [`struct_field`]
    ///
    /// [`struct_field`]: Self::struct_field
    pub struct_field: u8,
}

pub enum MyEnum {
    //@ has foo/type.MyEnum.html '//a/@href' 'type.MyEnum.html#variant.EnumVariant'

    /// [`EnumVariant`]
    ///
    /// [`EnumVariant`]: Self::EnumVariant
    EnumVariant,
}

pub union MyUnion {
    //@ has foo/type.MyUnion.html '//a/@href' 'type.MyUnion.html#structfield.union_field'

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
    //@ has foo/type.MyStruct.html '//a/@href' 'type.MyStruct.html#method.for_impl'

    /// [`for_impl`]
    ///
    /// [`for_impl`]: Self::for_impl
    pub fn for_impl() {
        unimplemented!()
    }
}

impl MyTrait for MyStruct {
    //@ has foo/type.MyStruct.html '//a/@href' 'type.MyStruct.html#associatedtype.AssoType'

    /// [`AssoType`]
    ///
    /// [`AssoType`]: Self::AssoType
    type AssoType = u32;

    //@ has foo/type.MyStruct.html '//a/@href' 'type.MyStruct.html#associatedconstant.ASSO_CONST'

    /// [`ASSO_CONST`]
    ///
    /// [`ASSO_CONST`]: Self::ASSO_CONST
    const ASSO_CONST: i32 = 10;

    //@ has foo/type.MyStruct.html '//a/@href' 'type.MyStruct.html#method.asso_fn'

    /// [`asso_fn`]
    ///
    /// [`asso_fn`]: Self::asso_fn
    fn asso_fn() {
        unimplemented!()
    }
}
