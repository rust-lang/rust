//@ set t1 = '$.index[?(@.name=="T1")].id'
pub trait T1 {}

//@ set t2 = '$.index[?(@.name=="T2")].id'
pub trait T2 {}

/// Fun impl
impl T1 for dyn T2 {}

//@ set impl = '$.index[?(@.docs=="Fun impl")].id'
//@ is '$.index[?(@.name=="T1")].inner.trait.implementations[*]' $impl
//@ is '$.index[?(@.name=="T2")].inner.trait.implementations' []

//@ is '$.index[?(@.docs=="Fun impl")].inner.impl.trait.id' $t1
//@ is '$.index[?(@.docs=="Fun impl")].inner.impl.for.dyn_trait.traits[*].trait.id' $t2
