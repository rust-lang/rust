#![feature(never_type)]

//@ is "$.index[?(@.name=='PrimNever')].visibility" \"public\"
//@ is "$.index[?(@.name=='PrimNever')].inner.type_alias.type" 0
//@ is "$.types[0].primitive" \"never\"
pub type PrimNever = !;

//@ is "$.index[?(@.name=='PrimStr')].inner.type_alias.type" 1
//@ is "$.types[1].primitive" \"str\"
pub type PrimStr = str;

//@ is "$.index[?(@.name=='PrimBool')].inner.type_alias.type" 2
//@ is "$.types[2].primitive" \"bool\"
pub type PrimBool = bool;

//@ is "$.index[?(@.name=='PrimChar')].inner.type_alias.type" 3
//@ is "$.types[3].primitive" \"char\"
pub type PrimChar = char;

//@ is "$.index[?(@.name=='PrimU8')].inner.type_alias.type" 4
//@ is "$.types[4].primitive" \"u8\"
pub type PrimU8 = u8;
