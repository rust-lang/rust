//@ compile-flags: --document-private-items --document-hidden-items

pub trait TheTrait {}

#[doc(hidden)]
struct Value {}

//@ has '$.index[?(@.docs[0].text=="THE IMPL")]'
/// THE IMPL
impl TheTrait for Value {}
