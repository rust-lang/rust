//@ check-pass

#![feature(cfg_boolean_literals)]
#![feature(doc_cfg)]

#[doc(cfg(false))]
pub fn foo() {}

#[doc(cfg(true))]
pub fn bar() {}

#[doc(cfg(any(true)))]
pub fn zoo() {}

#[doc(cfg(all(true)))]
pub fn toy() {}

#[doc(cfg(not(true)))]
pub fn nay() {}
