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
use super::common::*;
use super::machine::*;
use abi::{ArgType, FnType};
use type_::Type;

// Win64 ABI: http://msdn.microsoft.com/en-us/library/zthk2dkh.aspx

pub fn compute_abi_info(ccx: &CrateContext, fty: &mut FnType) {
    let fixup = |a: &mut ArgType| {
        match a.ty.kind() {
            Struct => match llsize_of_alloc(ccx, a.ty) {
                          1 => a.cast = Some(Type::i8(ccx)),
                          2 => a.cast = Some(Type::i16(ccx)),
                          4 => a.cast = Some(Type::i32(ccx)),
                          8 => a.cast = Some(Type::i64(ccx)),
                          _ => a.make_indirect(ccx)
                      },
            Integer => match llsize_of_alloc(ccx, a.ty) {
                           1 ... 8 => a.extend_integer_width_to(32),
                           16 => a.make_indirect(ccx),
                           _ => bug!(),
            },
            _ => (),
        }
    };

    if !fty.ret.is_ignore() {
        fixup(&mut fty.ret);
    }
    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        fixup(arg);
    }
}
