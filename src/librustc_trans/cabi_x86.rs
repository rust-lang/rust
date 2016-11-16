// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::*;
use abi::{ArgAttribute, FnType};
use type_::Type;
use super::common::*;
use super::machine::*;

pub fn compute_abi_info(ccx: &CrateContext, fty: &mut FnType) {
    if !fty.ret.is_ignore() {
        if fty.ret.ty.kind() == Struct {
            // Returning a structure. Most often, this will use
            // a hidden first argument. On some platforms, though,
            // small structs are returned as integers.
            //
            // Some links:
            // http://www.angelcode.com/dev/callconv/callconv.html
            // Clang's ABI handling is in lib/CodeGen/TargetInfo.cpp
            let t = &ccx.sess().target.target;
            if t.options.is_like_osx || t.options.is_like_windows {
                match llsize_of_alloc(ccx, fty.ret.ty) {
                    1 => fty.ret.cast = Some(Type::i8(ccx)),
                    2 => fty.ret.cast = Some(Type::i16(ccx)),
                    4 => fty.ret.cast = Some(Type::i32(ccx)),
                    8 => fty.ret.cast = Some(Type::i64(ccx)),
                    _ => fty.ret.make_indirect(ccx)
                }
            } else {
                fty.ret.make_indirect(ccx);
            }
        } else {
            fty.ret.extend_integer_width_to(32);
        }
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        if arg.ty.kind() == Struct {
            arg.make_indirect(ccx);
            arg.attrs.set(ArgAttribute::ByVal);
        } else {
            arg.extend_integer_width_to(32);
        }
    }
}
