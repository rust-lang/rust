// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Traits for the built-in operators

#[lang="drop"]
pub trait Drop {
    fn finalize(&self);  // FIXME(#4332): Rename to "drop"? --pcwalton
}

#[lang="add"]
pub trait Add<RHS,Result> {
    fn add(&self, rhs: &RHS) -> Result;
}

#[lang="sub"]
pub trait Sub<RHS,Result> {
    fn sub(&self, rhs: &RHS) -> Result;
}

#[lang="mul"]
pub trait Mul<RHS,Result> {
    fn mul(&self, rhs: &RHS) -> Result;
}

#[lang="div"]
#[cfg(stage0)]
pub trait Div<RHS,Result> {
    fn div(&self, rhs: &RHS) -> Result;
}
#[lang="quot"]
#[cfg(not(stage0))]
pub trait Quot<RHS,Result> {
    fn quot(&self, rhs: &RHS) -> Result;
}

#[lang="modulo"]
#[cfg(stage0)]
pub trait Modulo<RHS,Result> {
    fn modulo(&self, rhs: &RHS) -> Result;
}
#[lang="rem"]
#[cfg(not(stage0))]
pub trait Rem<RHS,Result> {
    fn rem(&self, rhs: &RHS) -> Result;
}

#[lang="neg"]
pub trait Neg<Result> {
    fn neg(&self) -> Result;
}

#[lang="not"]
pub trait Not<Result> {
    fn not(&self) -> Result;
}

#[lang="bitand"]
pub trait BitAnd<RHS,Result> {
    fn bitand(&self, rhs: &RHS) -> Result;
}

#[lang="bitor"]
pub trait BitOr<RHS,Result> {
    fn bitor(&self, rhs: &RHS) -> Result;
}

#[lang="bitxor"]
pub trait BitXor<RHS,Result> {
    fn bitxor(&self, rhs: &RHS) -> Result;
}

#[lang="shl"]
pub trait Shl<RHS,Result> {
    fn shl(&self, rhs: &RHS) -> Result;
}

#[lang="shr"]
pub trait Shr<RHS,Result> {
    fn shr(&self, rhs: &RHS) -> Result;
}

#[lang="index"]
pub trait Index<Index,Result> {
    fn index(&self, index: &Index) -> Result;
}
