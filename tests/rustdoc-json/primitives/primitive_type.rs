#![feature(never_type)]

// @is "$.index[*][?(@.name=='PrimNever')].visibility" \"public\"
// @is "$.index[*][?(@.name=='PrimNever')].inner.typedef.type.primitive" \"never\"
pub type PrimNever = !;

// @is "$.index[*][?(@.name=='PrimStr')].inner.typedef.type.primitive" \"str\"
pub type PrimStr = str;

// @is "$.index[*][?(@.name=='PrimBool')].inner.typedef.type.primitive" \"bool\"
pub type PrimBool = bool;

// @is "$.index[*][?(@.name=='PrimChar')].inner.typedef.type.primitive" \"char\"
pub type PrimChar = char;

// @is "$.index[*][?(@.name=='PrimU8')].inner.typedef.type.primitive" \"u8\"
pub type PrimU8 = u8;
