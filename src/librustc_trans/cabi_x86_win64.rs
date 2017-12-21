// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::{ArgType, FnType, LayoutExt, Reg};
use common::CrateContext;

use rustc::ty::layout::Layout;

// Win64 ABI: http://msdn.microsoft.com/en-us/library/zthk2dkh.aspx

pub fn compute_abi_info<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, fty: &mut FnType<'tcx>) {
    let fixup = |a: &mut ArgType<'tcx>| {
        let size = a.layout.size(ccx);
        if a.layout.is_aggregate() {
            match size.bits() {
                8 => a.cast_to(ccx, Reg::i8()),
                16 => a.cast_to(ccx, Reg::i16()),
                32 => a.cast_to(ccx, Reg::i32()),
                64 => a.cast_to(ccx, Reg::i64()),
                _ => a.make_indirect(ccx)
            };
        } else {
            if let Layout::Vector { .. } = *a.layout {
                // FIXME(eddyb) there should be a size cap here
                // (probably what clang calls "illegal vectors").
            } else if size.bytes() > 8 {
                a.make_indirect(ccx);
            } else {
                a.extend_integer_width_to(32);
            }
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
