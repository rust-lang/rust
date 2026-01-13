#![crate_type = "lib"]
#![crate_name = "foo"]

//! [`TheVariant::the_field`]

//@ has 'foo/index.html' '//a[@href="enum.TheEnum.html#variant.TheVariant.field.the_field"]' 'TheVariant::the_field'

pub enum TheEnum {
    TheVariant { the_field: i32 },
}

pub use TheEnum::TheVariant;
