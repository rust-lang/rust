#![feature(doc_cfg)]
#[doc(cfg = "x")] //~ ERROR malformed `doc` attribute input
#[doc(cfg(x, y))] //~ ERROR malformed `doc` attribute input
pub struct S {}

// We check it also fails on private items.
#[doc(cfg = "x")] //~ ERROR malformed `doc` attribute input
#[doc(cfg(x, y))] //~ ERROR malformed `doc` attribute input
struct X {}

// We check it also fails on hidden items.
#[doc(cfg = "x")] //~ ERROR malformed `doc` attribute input
#[doc(cfg(x, y))] //~ ERROR malformed `doc` attribute input
#[doc(hidden)]
pub struct Y {}

// We check it also fails on hidden AND private items.
#[doc(cfg = "x")] //~ ERROR malformed `doc` attribute input
#[doc(cfg(x, y))] //~ ERROR malformed `doc` attribute input
#[doc(hidden)]
struct Z {}
