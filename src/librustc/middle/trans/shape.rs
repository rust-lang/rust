// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A "shape" is a compact encoding of a type that is used by interpreted glue.
// This substitutes for the runtime tags used by e.g. MLs.


use lib::llvm::llvm;
use lib::llvm::{True, ModuleRef, ValueRef};
use middle::trans::common::*;
use middle::trans;

use core::str;

pub struct Ctxt {
    next_tag_id: u16,
    pad: u16,
    pad2: u32
}

pub fn mk_global(ccx: @CrateContext,
                 name: ~str,
                 llval: ValueRef,
                 internal: bool)
              -> ValueRef {
    unsafe {
        let llglobal = do str::as_c_str(name) |buf| {
            llvm::LLVMAddGlobal(ccx.llmod, val_ty(llval), buf)
        };
        llvm::LLVMSetInitializer(llglobal, llval);
        llvm::LLVMSetGlobalConstant(llglobal, True);

        if internal {
            ::lib::llvm::SetLinkage(llglobal,
                                    ::lib::llvm::InternalLinkage);
        }

        return llglobal;
    }
}

pub fn mk_ctxt(llmod: ModuleRef) -> Ctxt {
    unsafe {
        let llshapetablesty = trans::common::T_named_struct(~"shapes");
        let _llshapetables = str::as_c_str(~"shapes", |buf| {
            llvm::LLVMAddGlobal(llmod, llshapetablesty, buf)
        });

        return Ctxt {
            next_tag_id: 0u16,
            pad: 0u16,
            pad2: 0u32
        };
    }
}

/*
Although these two functions are never called, they are here
for a VERY GOOD REASON. See #3670
*/
pub fn add_u16(dest: &mut ~[u8], val: u16) {
    *dest += ~[(val & 0xffu16) as u8, (val >> 8u16) as u8];
}

pub fn add_substr(dest: &mut ~[u8], src: ~[u8]) {
    add_u16(&mut *dest, src.len() as u16);
    *dest += src;
}
