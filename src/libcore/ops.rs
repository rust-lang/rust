// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Core operators

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

#[lang="drop"]
pub trait Drop {
    fn finalize(&self);  // XXX: Rename to "drop"? --pcwalton
}

#[cfg(stage0)]
#[lang="add"]
pub trait Add<RHS,Result> {
    pure fn add(rhs: &RHS) -> Result;
}

#[cfg(stage1)]
#[cfg(stage2)]
#[lang="add"]
pub trait Add<RHS,Result> {
    pure fn add(&self, rhs: &RHS) -> Result;
}

#[lang="sub"]
pub trait Sub<RHS,Result> {
    pure fn sub(&self, rhs: &RHS) -> Result;
}

#[lang="mul"]
pub trait Mul<RHS,Result> {
    pure fn mul(&self, rhs: &RHS) -> Result;
}

#[lang="div"]
pub trait Div<RHS,Result> {
    pure fn div(&self, rhs: &RHS) -> Result;
}

#[lang="modulo"]
pub trait Modulo<RHS,Result> {
    pure fn modulo(&self, rhs: &RHS) -> Result;
}

#[lang="neg"]
pub trait Neg<Result> {
    pure fn neg(&self) -> Result;
}

#[lang="bitand"]
pub trait BitAnd<RHS,Result> {
    pure fn bitand(&self, rhs: &RHS) -> Result;
}

#[lang="bitor"]
pub trait BitOr<RHS,Result> {
    pure fn bitor(&self, rhs: &RHS) -> Result;
}

#[lang="bitxor"]
pub trait BitXor<RHS,Result> {
    pure fn bitxor(&self, rhs: &RHS) -> Result;
}

#[lang="shl"]
pub trait Shl<RHS,Result> {
    pure fn shl(&self, rhs: &RHS) -> Result;
}

#[lang="shr"]
pub trait Shr<RHS,Result> {
    pure fn shr(&self, rhs: &RHS) -> Result;
}

#[cfg(stage0)]
#[lang="index"]
pub trait Index<Index,Result> {
    pure fn index(index: Index) -> Result;
}

#[cfg(stage1)]
#[cfg(stage2)]
#[lang="index"]
pub trait Index<Index,Result> {
    pure fn index(&self, index: Index) -> Result;
}

