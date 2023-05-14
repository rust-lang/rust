// ignore-tidy-linelength
#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

// @set Parametrized = '$.index[*][?(@.name=="Parametrized")].id'
pub struct Parametrized<T>(T);

// @is '$.index[*][?(@.name=="Test")].inner.type.kind' '"qualified_path"'
// @is '$.index[*][?(@.name=="Test")].inner.type.inner.self_type.inner.id' $Parametrized
// @is '$.index[*][?(@.name=="Test")].inner.type.inner.self_type.inner.args.angle_bracketed.args[0].type' '{"inner": "i32", "kind": "primitive"}'
// @is '$.index[*][?(@.name=="Test")].inner.type.inner.name' '"Proj"'
// @is '$.index[*][?(@.name=="Test")].inner.type.inner.trait' null
pub type Test = Parametrized<i32>::Proj;

/// param_bool
impl Parametrized<bool> {
    /// param_bool_proj
    pub type Proj = ();
}

/// param_i32
impl Parametrized<i32> {
    /// param_i32_proj
    pub type Proj = String;
}

// @set param_bool = '$.index[*][?(@.docs=="param_bool")].id'
// @set param_i32 = '$.index[*][?(@.docs=="param_i32")].id'
// @set param_bool_proj = '$.index[*][?(@.docs=="param_bool_proj")].id'
// @set param_i32_proj = '$.index[*][?(@.docs=="param_i32_proj")].id'

// @is '$.index[*][?(@.docs=="param_bool")].inner.items[*]' $param_bool_proj
// @is '$.index[*][?(@.docs=="param_i32")].inner.items[*]' $param_i32_proj
