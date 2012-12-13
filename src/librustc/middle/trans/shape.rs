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

use back::abi;
use lib::llvm::llvm;
use lib::llvm::{True, False, ModuleRef, TypeRef, ValueRef};
use middle::trans::base;
use middle::trans::common::*;
use middle::trans::machine::*;
use middle::ty::field;
use middle::ty;
use util::ppaux::ty_to_str;

use core::dvec::DVec;
use core::option::is_some;
use std::map::HashMap;
use syntax::ast;
use syntax::ast_util::dummy_sp;
use syntax::codemap::span;
use syntax::util::interner;

use ty_ctxt = middle::ty::ctxt;

type ctxt = {mut next_tag_id: u16, pad: u16, pad2: u32};

fn mk_global(ccx: @crate_ctxt, name: ~str, llval: ValueRef, internal: bool) ->
   ValueRef {
    let llglobal = do str::as_c_str(name) |buf| {
        lib::llvm::llvm::LLVMAddGlobal(ccx.llmod, val_ty(llval), buf)
    };
    lib::llvm::llvm::LLVMSetInitializer(llglobal, llval);
    lib::llvm::llvm::LLVMSetGlobalConstant(llglobal, True);

    if internal {
        lib::llvm::SetLinkage(llglobal, lib::llvm::InternalLinkage);
    }

    return llglobal;
}

fn mk_ctxt(llmod: ModuleRef) -> ctxt {
    let llshapetablesty = trans::common::T_named_struct(~"shapes");
    let _llshapetables = str::as_c_str(~"shapes", |buf| {
        lib::llvm::llvm::LLVMAddGlobal(llmod, llshapetablesty, buf)
    });

    return {mut next_tag_id: 0u16, pad: 0u16, pad2: 0u32};
}

/*
Although these two functions are never called, they are here
for a VERY GOOD REASON. See #3670
*/
fn add_u16(dest: &mut ~[u8], val: u16) {
    *dest += ~[(val & 0xffu16) as u8, (val >> 8u16) as u8];
}

fn add_substr(dest: &mut ~[u8], src: ~[u8]) {
    add_u16(dest, vec::len(src) as u16);
    *dest += src;
}

