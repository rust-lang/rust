#![feature(rustc_attrs)]

//@ set Local = "$.index[?(@.name=='Local')].id"
pub trait Local {}

//@ is "$.index[?(@.docs=='Local for bool')].inner.impl.trait.id" $Local
//@ is "$.index[?(@.docs=='Local for bool')].inner.impl.for.primitive" '"bool"'
/// Local for bool
impl Local for bool {}

//@ set impl =  "$.index[?(@.docs=='Local for bool')].id"
//@ is "$.index[?(@.name=='Local')].inner.trait.implementations[*]" $impl

// FIXME(#101695): Test bool's `impls` include "Local for bool"
//@ has "$.index[?(@.name=='bool')]"
#[rustc_doc_primitive = "bool"]
/// Boolean docs
mod prim_bool {}
