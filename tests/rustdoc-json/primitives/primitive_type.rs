#![feature(never_type)]

//@ is "$.index[?(@.name=='PrimNever')].visibility" \"public\"
//@ is "$.index[?(@.name=='PrimNever')].inner.type_alias.type.primitive" \"never\"
pub type PrimNever = !;

//@ is "$.index[?(@.name=='PrimStr')].inner.type_alias.type.primitive" \"str\"
pub type PrimStr = str;

//@ is "$.index[?(@.name=='PrimBool')].inner.type_alias.type.primitive" \"bool\"
pub type PrimBool = bool;

//@ is "$.index[?(@.name=='PrimChar')].inner.type_alias.type.primitive" \"char\"
pub type PrimChar = char;

//@ is "$.index[?(@.name=='PrimU8')].inner.type_alias.type.primitive" \"u8\"
pub type PrimU8 = u8;
