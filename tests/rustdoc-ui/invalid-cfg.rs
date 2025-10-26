#![feature(doc_cfg)]
#[doc(cfg = "x")] //~ ERROR not followed by parentheses
#[doc(cfg(x, y))] //~ ERROR multiple `cfg` predicates
pub struct S {}

// We check it also fails on private items.
#[doc(cfg = "x")] //~ ERROR not followed by parentheses
#[doc(cfg(x, y))] //~ ERROR multiple `cfg` predicates
struct X {}

// We check it also fails on hidden items.
#[doc(cfg = "x")] //~ ERROR not followed by parentheses
#[doc(cfg(x, y))] //~ ERROR multiple `cfg` predicates
#[doc(hidden)]
pub struct Y {}

// We check it also fails on hidden AND private items.
#[doc(cfg = "x")] //~ ERROR not followed by parentheses
#[doc(cfg(x, y))] //~ ERROR multiple `cfg` predicates
#[doc(hidden)]
struct Z {}
