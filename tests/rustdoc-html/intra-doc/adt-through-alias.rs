#![crate_name = "foo"]

//! [`TheStructAlias::the_field`]
//! [`TheEnumAlias::TheVariant`]
//! [`TheEnumAlias::TheVariant::the_field`]

// FIXME: this should resolve to the alias's version
//@ has foo/index.html '//a[@href="struct.TheStruct.html#structfield.the_field"]' 'TheStructAlias::the_field'
// FIXME: this should resolve to the alias's version
//@ has foo/index.html '//a[@href="enum.TheEnum.html#variant.TheVariant"]' 'TheEnumAlias::TheVariant'
//@ has foo/index.html '//a[@href="type.TheEnumAlias.html#variant.TheVariant.field.the_field"]' 'TheEnumAlias::TheVariant::the_field'

pub struct TheStruct {
    pub the_field: i32,
}

pub type TheStructAlias = TheStruct;

pub enum TheEnum {
    TheVariant { the_field: i32 },
}

pub type TheEnumAlias = TheEnum;

fn main() {}
