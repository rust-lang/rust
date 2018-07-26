// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type A = [u8; 4]::AssocTy;
//~^ ERROR missing angle brackets in associated item path
//~| ERROR ambiguous associated type

type B = [u8]::AssocTy;
//~^ ERROR missing angle brackets in associated item path
//~| ERROR ambiguous associated type

type C = (u8)::AssocTy;
//~^ ERROR missing angle brackets in associated item path
//~| ERROR ambiguous associated type

type D = (u8, u8)::AssocTy;
//~^ ERROR missing angle brackets in associated item path
//~| ERROR ambiguous associated type

type E = _::AssocTy;
//~^ ERROR missing angle brackets in associated item path
//~| ERROR the type placeholder `_` is not allowed within types on item signatures

type F = &'static (u8)::AssocTy;
//~^ ERROR missing angle brackets in associated item path
//~| ERROR ambiguous associated type

// Qualified paths cannot appear in bounds, so the recovery
// should apply to the whole sum and not `(Send)`.
type G = 'static + (Send)::AssocTy;
//~^ ERROR missing angle brackets in associated item path
//~| ERROR ambiguous associated type

// This is actually a legal path with fn-like generic arguments in the middle!
// Recovery should not apply in this context.
type H = Fn(u8) -> (u8)::Output;
//~^ ERROR ambiguous associated type

fn main() {}
