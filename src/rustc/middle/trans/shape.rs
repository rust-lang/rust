// A "shape" is a compact encoding of a type that is used by interpreted glue.
// This substitutes for the runtime tags used by e.g. MLs.

use lib::llvm::llvm;
use lib::llvm::{True, False, ModuleRef, TypeRef, ValueRef};
use driver::session;
use driver::session::session;
use trans::base;
use middle::trans::common::*;
use middle::trans::machine::*;
use back::abi;
use middle::ty;
use middle::ty::field;
use syntax::ast;
use syntax::ast_util::dummy_sp;
use syntax::util::interner;
use util::ppaux::ty_to_str;
use syntax::codemap::span;
use dvec::DVec;

use std::map::HashMap;
use option::is_some;

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

fn add_u16(&dest: ~[u8], val: u16) {
    dest += ~[(val & 0xffu16) as u8, (val >> 8u16) as u8];
}

fn add_substr(&dest: ~[u8], src: ~[u8]) {
    add_u16(dest, vec::len(src) as u16);
    dest += src;
}

