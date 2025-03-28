#![feature(doc_cfg)]
#[doc(cfg = "x")] //~ ERROR not followed by parentheses
#[doc(cfg(x, y))] //~ ERROR multiple `cfg` predicates
pub struct S {}
