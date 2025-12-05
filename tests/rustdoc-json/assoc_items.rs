#![no_std]

pub struct Simple;

impl Simple {
    //@ has "$.index[?(@.name=='CONSTANT')].inner.assoc_const"
    pub const CONSTANT: usize = 0;
}

pub trait EasyToImpl {
    //@ has "$.index[?(@.docs=='ToDeclare trait')].inner.assoc_type"
    //@ is "$.index[?(@.docs=='ToDeclare trait')].inner.assoc_type.type" null
    //@ is "$.index[?(@.docs=='ToDeclare trait')].inner.assoc_type.bounds" []
    /// ToDeclare trait
    type ToDeclare;
    //@ has "$.index[?(@.docs=='AN_ATTRIBUTE trait')].inner.assoc_const"
    //@ is "$.index[?(@.docs=='AN_ATTRIBUTE trait')].inner.assoc_const.value" null
    //@ is "$.index[?(@.docs=='AN_ATTRIBUTE trait')].inner.assoc_const.type.primitive" '"usize"'
    /// AN_ATTRIBUTE trait
    const AN_ATTRIBUTE: usize;
}

impl EasyToImpl for Simple {
    //@ has "$.index[?(@.docs=='ToDeclare impl')].inner.assoc_type"
    //@ is "$.index[?(@.docs=='ToDeclare impl')].inner.assoc_type.type.primitive" \"usize\"
    /// ToDeclare impl
    type ToDeclare = usize;

    //@ has "$.index[?(@.docs=='AN_ATTRIBUTE impl')].inner.assoc_const"
    //@ is "$.index[?(@.docs=='AN_ATTRIBUTE impl')].inner.assoc_const.type.primitive" \"usize\"
    //@ is "$.index[?(@.docs=='AN_ATTRIBUTE impl')].inner.assoc_const.value" \"12\"
    /// AN_ATTRIBUTE impl
    const AN_ATTRIBUTE: usize = 12;
}
