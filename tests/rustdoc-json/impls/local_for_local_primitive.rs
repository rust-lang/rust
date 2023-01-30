#![feature(no_core)]
#![feature(rustdoc_internals)]
#![no_core]

// @set Local = "$.index[*][?(@.name=='Local')].id"
pub trait Local {}

// @is "$.index[*][?(@.docs=='Local for bool')].inner.trait.id" $Local
// @is "$.index[*][?(@.docs=='Local for bool')].inner.for.kind" '"primitive"'
// @is "$.index[*][?(@.docs=='Local for bool')].inner.for.inner" '"bool"'
/// Local for bool
impl Local for bool {}

// @set impl =  "$.index[*][?(@.docs=='Local for bool')].id"
// @is "$.index[*][?(@.name=='Local')].inner.implementations[*]" $impl

// FIXME(#101695): Test bool's `impls` include "Local for bool"
// @has "$.index[*][?(@.name=='bool')]"
#[doc(primitive = "bool")]
/// Boolean docs
mod prim_bool {}
