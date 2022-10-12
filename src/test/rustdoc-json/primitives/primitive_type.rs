#![feature(never_type)]

// @is "$.index[*][?(@.name=='PrimNever')].visibility" \"public\"
// @is "$.index[*][?(@.name=='PrimNever')].inner.type.kind" \"primitive\"
// @is "$.index[*][?(@.name=='PrimNever')].inner.type.inner" \"never\"
pub type PrimNever = !;

// @is "$.index[*][?(@.name=='PrimStr')].inner.type.kind" \"primitive\"
// @is "$.index[*][?(@.name=='PrimStr')].inner.type.inner" \"str\"
pub type PrimStr = str;

// @is "$.index[*][?(@.name=='PrimBool')].inner.type.kind" \"primitive\"
// @is "$.index[*][?(@.name=='PrimBool')].inner.type.inner" \"bool\"
pub type PrimBool = bool;

// @is "$.index[*][?(@.name=='PrimChar')].inner.type.kind" \"primitive\"
// @is "$.index[*][?(@.name=='PrimChar')].inner.type.inner" \"char\"
pub type PrimChar = char;

// @is "$.index[*][?(@.name=='PrimU8')].inner.type.kind" \"primitive\"
// @is "$.index[*][?(@.name=='PrimU8')].inner.type.inner" \"u8\"
pub type PrimU8 = u8;
