// ignore-tidy-linelength
#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

// @set Carrier = '$.index[*][?(@.name=="Carrier")].id'
pub struct Carrier<'a>(&'a ());

// @is '$.index[*][?(@.name=="User")].inner.type.kind' '"function_pointer"'
// @is '$.index[*][?(@.name=="User")].inner.type.inner.generic_params[*].name' \""'b"\"
// @is '$.index[*][?(@.name=="User")].inner.type.inner.decl.inputs[0][1].kind' '"qualified_path"'
// @is '$.index[*][?(@.name=="User")].inner.type.inner.decl.inputs[0][1].inner.self_type.inner.id' $Carrier
// @is '$.index[*][?(@.name=="User")].inner.type.inner.decl.inputs[0][1].inner.self_type.inner.args.angle_bracketed.args[0].lifetime' \""'b"\"
// @is '$.index[*][?(@.name=="User")].inner.type.inner.decl.inputs[0][1].inner.name' '"Focus"'
// @is '$.index[*][?(@.name=="User")].inner.type.inner.decl.inputs[0][1].inner.trait' null
// @is '$.index[*][?(@.name=="User")].inner.type.inner.decl.inputs[0][1].inner.args.angle_bracketed.args[0].type.inner' '"i32"'

pub type User = for<'b> fn(Carrier<'b>::Focus<i32>);

impl<'a> Carrier<'a> {
    pub type Focus<T> = &'a mut T;
}
