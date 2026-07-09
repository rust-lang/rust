//@ run-rustfix
#![doc(syntax="+tex_math_dollars")]
#![feature(rustdoc_texmath)]
#![deny(rustdoc::invalid_math)]

//! Let $f(x) \in \mathbb{Z}\[x\]$ be non-zero primitive squarefree polynomial
//~^ ERROR unknown
//~| HELP double-escaped
//! of degree at least $1$$
