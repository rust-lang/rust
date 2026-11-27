#![crate_name = "foo"]

//! [`TheStructAlias::the_field`]
//! [`TheEnumAlias::TheVariant`]
//! [`TheEnumAlias::TheVariant::the_field`]
//! [`TheUnionAlias::f1`]
//!
//! [`TheStruct::trait_`]
//! [`TheStructAlias::trait_`]
//! [`TheEnum::trait_`]
//! [`TheEnumAlias::trait_`]
//!
//! [`TheStruct::inherent`]
//! [`TheStructAlias::inherent`]
//! [`TheEnum::inherent`]
//! [`TheEnumAlias::inherent`]

//@ has foo/index.html '//a[@href="type.TheStructAlias.html#structfield.the_field"]' 'TheStructAlias::the_field'
//@ has foo/index.html '//a[@href="type.TheEnumAlias.html#variant.TheVariant"]' 'TheEnumAlias::TheVariant'
//@ has foo/index.html '//a[@href="type.TheEnumAlias.html#variant.TheVariant.field.the_field"]' 'TheEnumAlias::TheVariant::the_field'
//@ has foo/index.html '//a[@href="type.TheUnionAlias.html#structfield.f1"]' 'TheUnionAlias::f1'

//@ has foo/index.html '//a[@href="struct.TheStruct.html#method.trait_"]' 'TheStruct::trait_'
//@ has foo/index.html '//a[@href="struct.TheStruct.html#method.trait_"]' 'TheStructAlias::trait_'
//@ has foo/index.html '//a[@href="enum.TheEnum.html#method.trait_"]' 'TheEnum::trait_'
// FIXME: this one should resolve to alias since it's impl Trait for TheEnumAlias
//@ has foo/index.html '//a[@href="enum.TheEnum.html#method.trait_"]' 'TheEnumAlias::trait_'

//@ has foo/index.html '//a[@href="struct.TheStruct.html#method.inherent"]' 'TheStruct::inherent'
// FIXME: this one should resolve to alias
//@ has foo/index.html '//a[@href="struct.TheStruct.html#method.inherent"]' 'TheStructAlias::inherent'
//@ has foo/index.html '//a[@href="enum.TheEnum.html#method.inherent"]' 'TheEnum::inherent'
// FIXME: this one should resolve to alias
//@ has foo/index.html '//a[@href="enum.TheEnum.html#method.inherent"]' 'TheEnumAlias::inherent'

pub struct TheStruct {
    pub the_field: i32,
}

pub type TheStructAlias = TheStruct;

pub enum TheEnum {
    TheVariant { the_field: i32 },
}

pub type TheEnumAlias = TheEnum;

pub trait Trait {
    fn trait_() {}
}

impl Trait for TheStruct {}

impl Trait for TheEnumAlias {}

impl TheStruct {
    pub fn inherent() {}
}

impl TheEnumAlias {
    pub fn inherent() {}
}

pub union TheUnion {
    pub f1: usize,
    pub f2: isize,
}

pub type TheUnionAlias = TheUnion;

fn main() {}
