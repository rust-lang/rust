#![no_std]

pub struct Simple;

impl Simple {
    // @has "$.index[*][?(@.name=='CONSTANT')].kind" \"assoc_const\"
    pub const CONSTANT: usize = 0;
}

pub trait EasyToImpl {
    // @has "$.index[*][?(@.name=='ToDeclare')].kind" \"assoc_type\"
    // @has "$.index[*][?(@.name=='ToDeclare')].inner.default" null
    type ToDeclare;
    // @has "$.index[*][?(@.name=='AN_ATTRIBUTE')].kind" \"assoc_const\"
    // @has "$.index[*][?(@.name=='AN_ATTRIBUTE')].inner.default" null
    const AN_ATTRIBUTE: usize;
}

impl EasyToImpl for Simple {
    // @has "$.index[*][?(@.name=='ToDeclare')].inner.default.kind" \"primitive\"
    // @has "$.index[*][?(@.name=='ToDeclare')].inner.default.inner" \"usize\"
    type ToDeclare = usize;
    // @has "$.index[*][?(@.name=='AN_ATTRIBUTE')].inner.type.kind" \"primitive\"
    // @has "$.index[*][?(@.name=='AN_ATTRIBUTE')].inner.type.inner" \"usize\"
    // @has "$.index[*][?(@.name=='AN_ATTRIBUTE')].inner.default" \"12\"
    const AN_ATTRIBUTE: usize = 12;
}
