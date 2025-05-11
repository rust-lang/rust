//@ set local = "$.index[?(@.name=='Local')]"
pub trait Local {}

//@ set impl = "$.index[?(@.docs=='local for bool')].id"
//@ is "$.index[?(@.name=='Local')].inner.trait.implementations[*]" $impl
/// local for bool
impl Local for bool {}
