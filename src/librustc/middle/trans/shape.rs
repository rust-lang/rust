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

use middle::trans::type_::Type;

use core::str;

pub struct Ctxt {
    next_tag_id: u16,
    pad: u16,
    pad2: u32
}

pub fn mk_global(ccx: &CrateContext,
                 name: &str,
                 llval: ValueRef,
                 internal: bool)
              -> ValueRef {
    unsafe {
        let llglobal = do str::as_c_str(name) |buf| {
            llvm::LLVMAddGlobal(ccx.llmod, val_ty(llval).to_ref(), buf)
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
        let llshapetablesty = Type::named_struct("shapes");
        do "shapes".as_c_str |buf| {
            llvm::LLVMAddGlobal(llmod, llshapetablesty.to_ref(), buf)
        };

        Ctxt {
            next_tag_id: 0u16,
            pad: 0u16,
            pad2: 0u32
        }
    }
}
