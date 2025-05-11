#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

//@ set Parametrized = '$.index[?(@.name=="Parametrized")].id'
pub struct Parametrized<T>(T);

//@ count "$.index[?(@.name=='test')].inner.function.sig.inputs[*]" 1
//@ is "$.index[?(@.name=='test')].inner.function.sig.inputs[0][0]" '"_"'
//@ is '$.index[?(@.name=="test")].inner.function.sig.inputs[0][1].qualified_path.self_type.resolved_path.id' $Parametrized
//@ is '$.index[?(@.name=="test")].inner.function.sig.inputs[0][1].qualified_path.self_type.resolved_path.args.angle_bracketed.args[0].type.primitive' \"i32\"
//@ is '$.index[?(@.name=="test")].inner.function.sig.inputs[0][1].qualified_path.name' '"Proj"'
//@ is '$.index[?(@.name=="test")].inner.function.sig.inputs[0][1].qualified_path.trait' null
pub fn test(_: Parametrized<i32>::Proj) {}

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

//@ set param_bool = '$.index[?(@.docs=="param_bool")].id'
//@ set param_i32 = '$.index[?(@.docs=="param_i32")].id'
//@ set param_bool_proj = '$.index[?(@.docs=="param_bool_proj")].id'
//@ set param_i32_proj = '$.index[?(@.docs=="param_i32_proj")].id'

//@ is '$.index[?(@.docs=="param_bool")].inner.impl.items[*]' $param_bool_proj
//@ is '$.index[?(@.docs=="param_i32")].inner.impl.items[*]' $param_i32_proj
