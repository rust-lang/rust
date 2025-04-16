// #138113: rustdoc didn't gate unstable predicates inside `doc(cfg(..))`
#![feature(doc_cfg)]

// `cfg_boolean_literals`
#[doc(cfg(false))] //~ ERROR `cfg(false)` is experimental and subject to change
pub fn cfg_boolean_literals() {}

// `cfg_version`
#[doc(cfg(sanitize = "thread"))] //~ ERROR `cfg(sanitize)` is experimental and subject to change
pub fn cfg_sanitize() {}
