#![feature(never_type)]

// @has primitives.json "$.index[*][?(@.name=='PrimNever')].visibility" \"public\"
// @has - "$.index[*][?(@.name=='PrimNever')].inner.type.kind" \"primitive\"
// @has - "$.index[*][?(@.name=='PrimNever')].inner.type.inner" \"never\"
pub type PrimNever = !;

// @has - "$.index[*][?(@.name=='PrimStr')].inner.type.kind" \"primitive\"
// @has - "$.index[*][?(@.name=='PrimStr')].inner.type.inner" \"str\"
pub type PrimStr = str;

// @has - "$.index[*][?(@.name=='PrimBool')].inner.type.kind" \"primitive\"
// @has - "$.index[*][?(@.name=='PrimBool')].inner.type.inner" \"bool\"
pub type PrimBool = bool;

// @has - "$.index[*][?(@.name=='PrimChar')].inner.type.kind" \"primitive\"
// @has - "$.index[*][?(@.name=='PrimChar')].inner.type.inner" \"char\"
pub type PrimChar = char;

// @has - "$.index[*][?(@.name=='PrimU8')].inner.type.kind" \"primitive\"
// @has - "$.index[*][?(@.name=='PrimU8')].inner.type.inner" \"u8\"
pub type PrimU8 = u8;
