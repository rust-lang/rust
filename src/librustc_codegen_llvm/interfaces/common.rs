// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::Backend;
use syntax::symbol::LocalInternedString;

pub trait CommonMethods : Backend {
    fn val_ty(v: Self::Value) -> Self::Type;

    // Constant constructors
    fn c_null(t: Self::Type) -> Self::Value;
    fn c_undef(t: Self::Type) -> Self::Value;
    fn c_int(t: Self::Type, i: i64) -> Self::Value;
    fn c_uint(t: Self::Type, i: u64) -> Self::Value;
    fn c_uint_big(t: Self::Type, u: u128) -> Self::Value;
    fn c_bool(&self, val: bool) -> Self::Value;
    fn c_i32(&self, i: i32) -> Self::Value;
    fn c_u32(&self, i: u32) -> Self::Value;
    fn c_u64(&self, i: u64) -> Self::Value;
    fn c_usize(&self, i: u64) -> Self::Value;
    fn c_u8(&self, i: u8) -> Self::Value;
    fn c_cstr(
        &self,
        s: LocalInternedString,
        null_terminated: bool,
    ) -> Self::Value;
    fn c_str_slice(&self, s: LocalInternedString) -> Self::Value;
    fn c_fat_ptr(
        &self,
        ptr: Self::Value,
        meta: Self::Value
    ) -> Self::Value;
    fn c_struct(
        &self,
        elts: &[Self::Value],
        packed: bool
    ) -> Self::Value;
    fn c_struct_in_context(
        llcx: Self::Context,
        elts: &[Self::Value],
        packed: bool,
    ) -> Self::Value;
    fn c_array(ty: Self::Type, elts: &[Self::Value]) -> Self::Value;
    fn c_vector(elts: &[Self::Value]) -> Self::Value;
    fn c_bytes(&self, bytes: &[u8]) -> Self::Value;
    fn c_bytes_in_context(llcx: Self::Context, bytes: &[u8]) -> Self::Value;

    fn const_get_elt(v: Self::Value, idx: u64) -> Self::Value;
    fn const_get_real(v: Self::Value) -> Option<(f64, bool)>;
    fn const_to_uint(v: Self::Value) -> u64;
    fn is_const_integral(v: Self::Value) -> bool;
    fn is_const_real(v: Self::Value) -> bool;
    fn const_to_opt_u128(v: Self::Value, sign_ext: bool) -> Option<u128>;
}
