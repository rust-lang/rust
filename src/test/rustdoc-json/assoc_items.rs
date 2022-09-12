#![no_std]

pub struct Simple;

impl Simple {
    // @is "$.index[*][?(@.name=='CONSTANT')].kind" \"assoc_const\"
    pub const CONSTANT: usize = 0;
}

pub trait EasyToImpl {
    // @is "$.index[*][?(@.docs=='ToDeclare trait')].kind" \"assoc_type\"
    // @is "$.index[*][?(@.docs=='ToDeclare trait')].inner.default" null
    // @is "$.index[*][?(@.docs=='ToDeclare trait')].inner.bounds" []
    /// ToDeclare trait
    type ToDeclare;
    // @is "$.index[*][?(@.docs=='AN_ATTRIBUTE trait')].kind" \"assoc_const\"
    // @is "$.index[*][?(@.docs=='AN_ATTRIBUTE trait')].inner.default" null
    // @is "$.index[*][?(@.docs=='AN_ATTRIBUTE trait')].inner.type.kind" '"primitive"'
    // @is "$.index[*][?(@.docs=='AN_ATTRIBUTE trait')].inner.type.inner" '"usize"'
    /// AN_ATTRIBUTE trait
    const AN_ATTRIBUTE: usize;
}

impl EasyToImpl for Simple {
    // @is "$.index[*][?(@.docs=='ToDeclare impl')].kind" '"assoc_type"'
    // @is "$.index[*][?(@.docs=='ToDeclare impl')].inner.default.kind" \"primitive\"
    // @is "$.index[*][?(@.docs=='ToDeclare impl')].inner.default.inner" \"usize\"
    /// ToDeclare impl
    type ToDeclare = usize;

    // @is "$.index[*][?(@.docs=='AN_ATTRIBUTE impl')].kind" '"assoc_const"'
    // @is "$.index[*][?(@.docs=='AN_ATTRIBUTE impl')].inner.type.kind" \"primitive\"
    // @is "$.index[*][?(@.docs=='AN_ATTRIBUTE impl')].inner.type.inner" \"usize\"
    // @is "$.index[*][?(@.docs=='AN_ATTRIBUTE impl')].inner.default" \"12\"
    /// AN_ATTRIBUTE impl
    const AN_ATTRIBUTE: usize = 12;
}
