#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

//@ set Carrier = '$.index[?(@.name=="Carrier")].id'
pub struct Carrier<'a>(&'a ());

//@ count "$.index[?(@.name=='user')].inner.function.sig.inputs[*]" 1
//@ is "$.index[?(@.name=='user')].inner.function.sig.inputs[0][0]" '"_"'
//@ is '$.index[?(@.name=="user")].inner.function.sig.inputs[0][1]' 18
//@ is '$.types[18].function_pointer.generic_params[*].name' \""'b"\"
//@ is '$.types[18].function_pointer.sig.inputs[0][1]' 17
//@ is '$.types[17].qualified_path.self_type' 16
//@ is '$.types[16].resolved_path.id' $Carrier
//@ is '$.types[16].resolved_path.args.angle_bracketed.args[0].lifetime' \""'b"\"
//@ is '$.types[17].qualified_path.name' '"Focus"'
//@ is '$.types[17].qualified_path.trait' null
//@ is '$.types[17].qualified_path.args.angle_bracketed.args[0].type' 15
//@ is '$.types[15].primitive' '"i32"'
pub fn user(_: for<'b> fn(Carrier<'b>::Focus<i32>)) {}

impl<'a> Carrier<'a> {
    pub type Focus<T>
        = &'a mut T
    where
        T: 'a;
}
