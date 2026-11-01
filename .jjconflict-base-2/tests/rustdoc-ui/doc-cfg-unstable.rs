// #138113: rustdoc didn't gate unstable predicates inside `doc(cfg(..))`
#![feature(doc_cfg)]

// `cfg_version`
#[doc(cfg(sanitize = "thread"))] //~ ERROR `cfg(sanitize)` is experimental and subject to change
pub fn cfg_sanitize() {}
