// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// trans.rs: Translate the completed AST to the LLVM IR.
//
// Some functions here, such as trans_block and trans_expr, return a value --
// the result of the translation to LLVM -- while others, such as trans_fn,
// trans_impl, and trans_item, are called only for the side effect of adding a
// particular definition to the LLVM IR output we're producing.
//
// Hopefully useful general knowledge about trans:
//
//   * There's no way to find out the ty::t type of a ValueRef.  Doing so
//     would be "trying to get the eggs out of an omelette" (credit:
//     pcwalton).  You can, instead, find out its TypeRef by calling val_ty,
//     but many TypeRefs correspond to one ty::t; for instance, tup(int, int,
//     int) and rec(x=int, y=int, z=int) will have the same TypeRef.


use back::link::{mangle_exported_name};
use back::{link, abi};
use driver::session;
use driver::session::Session;
use driver::driver::{CrateAnalysis, CrateTranslation};
use lib::llvm::{ModuleRef, ValueRef, BasicBlockRef};
use lib::llvm::{llvm, True};
use lib;
use metadata::common::LinkMeta;
use metadata::{csearch, cstore, encoder};
use middle::astencode;
use middle::lang_items::{LangItem, ExchangeMallocFnLangItem, StartFnLangItem};
use middle::lang_items::{MallocFnLangItem, ClosureExchangeMallocFnLangItem};
use middle::trans::_match;
use middle::trans::adt;
use middle::trans::base;
use middle::trans::build::*;
use middle::trans::builder::{Builder, noname};
use middle::trans::callee;
use middle::trans::common::*;
use middle::trans::consts;
use middle::trans::controlflow;
use middle::trans::datum;
use middle::trans::debuginfo;
use middle::trans::expr;
use middle::trans::foreign;
use middle::trans::glue;
use middle::trans::inline;
use middle::trans::llrepr::LlvmRepr;
use middle::trans::machine;
use middle::trans::machine::{llalign_of_min, llsize_of};
use middle::trans::meth;
use middle::trans::monomorphize;
use middle::trans::tvec;
use middle::trans::type_of;
use middle::trans::type_of::*;
use middle::trans::value::Value;
use middle::ty;
use util::common::indenter;
use util::ppaux::{Repr, ty_to_str};

use middle::trans::type_::Type;

use std::c_str::ToCStr;
use std::hash;
use std::hashmap::HashMap;
use std::libc::c_uint;
use std::vec;
use std::local_data;
use extra::time;
use extra::sort;
use syntax::ast::Name;
use syntax::ast_map::{path, path_elt_to_str, path_name, path_pretty_name};
use syntax::ast_util::{local_def};
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::parse::token::{special_idents};
use syntax::print::pprust::stmt_to_str;
use syntax::{ast, ast_util, codemap, ast_map};
use syntax::abi::{X86, X86_64, Arm, Mips, Rust, RustIntrinsic};
use syntax::visit;
use syntax::visit::Visitor;

pub use middle::trans::context::task_llcx;

local_data_key!(task_local_insn_key: ~[&'static str])

pub fn with_insn_ctxt(blk: &fn(&[&'static str])) {
    do local_data::get(task_local_insn_key) |c| {
        match c {
            Some(ctx) => blk(*ctx),
            None => ()
        }
    }
}

pub fn init_insn_ctxt() {
    local_data::set(task_local_insn_key, ~[]);
}

pub struct _InsnCtxt { _x: () }

#[unsafe_destructor]
impl Drop for _InsnCtxt {
    fn drop(&mut self) {
        do local_data::modify(task_local_insn_key) |c| {
            do c.map |mut ctx| {
                ctx.pop();
                ctx
            }
        }
    }
}

pub fn push_ctxt(s: &'static str) -> _InsnCtxt {
    debug!("new InsnCtxt: {}", s);
    do local_data::modify(task_local_insn_key) |c| {
        do c.map |mut ctx| {
            ctx.push(s);
            ctx
        }
    }
    _InsnCtxt { _x: () }
}

struct StatRecorder<'self> {
    ccx: @mut CrateContext,
    name: &'self str,
    start: u64,
    istart: uint,
}

impl<'self> StatRecorder<'self> {
    pub fn new(ccx: @mut CrateContext,
               name: &'self str) -> StatRecorder<'self> {
        let start = if ccx.sess.trans_stats() {
            time::precise_time_ns()
        } else {
            0
        };
        let istart = ccx.stats.n_llvm_insns;
        StatRecorder {
            ccx: ccx,
            name: name,
            start: start,
            istart: istart,
        }
    }
}

#[unsafe_destructor]
impl<'self> Drop for StatRecorder<'self> {
    fn drop(&mut self) {
        if self.ccx.sess.trans_stats() {
            let end = time::precise_time_ns();
            let elapsed = ((end - self.start) / 1_000_000) as uint;
            let iend = self.ccx.stats.n_llvm_insns;
            self.ccx.stats.fn_stats.push((self.name.to_owned(),
                                          elapsed,
                                          iend - self.istart));
            self.ccx.stats.n_fns += 1;
            // Reset LLVM insn count to avoid compound costs.
            self.ccx.stats.n_llvm_insns = self.istart;
        }
    }
}

// only use this for foreign function ABIs and glue, use `decl_rust_fn` for Rust functions
pub fn decl_fn(llmod: ModuleRef, name: &str, cc: lib::llvm::CallConv, ty: Type) -> ValueRef {
    let llfn: ValueRef = do name.with_c_str |buf| {
        unsafe {
            llvm::LLVMGetOrInsertFunction(llmod, buf, ty.to_ref())
        }
    };

    lib::llvm::SetFunctionCallConv(llfn, cc);
    return llfn;
}

// only use this for foreign function ABIs and glue, use `decl_rust_fn` for Rust functions
pub fn decl_cdecl_fn(llmod: ModuleRef, name: &str, ty: Type) -> ValueRef {
    return decl_fn(llmod, name, lib::llvm::CCallConv, ty);
}

// only use this for foreign function ABIs and glue, use `get_extern_rust_fn` for Rust functions
pub fn get_extern_fn(externs: &mut ExternMap, llmod: ModuleRef, name: &str,
                     cc: lib::llvm::CallConv, ty: Type) -> ValueRef {
    match externs.find_equiv(&name) {
        Some(n) => return *n,
        None => ()
    }
    let f = decl_fn(llmod, name, cc, ty);
    externs.insert(name.to_owned(), f);
    f
}

pub fn get_extern_rust_fn(ccx: &mut CrateContext, inputs: &[ty::t], output: ty::t,
                          name: &str) -> ValueRef {
    match ccx.externs.find_equiv(&name) {
        Some(n) => return *n,
        None => ()
    }
    let f = decl_rust_fn(ccx, inputs, output, name);
    ccx.externs.insert(name.to_owned(), f);
    f
}

pub fn decl_rust_fn(ccx: &mut CrateContext, inputs: &[ty::t], output: ty::t,
                    name: &str) -> ValueRef {
    let llfty = type_of_rust_fn(ccx, inputs, output);
    let llfn = decl_cdecl_fn(ccx.llmod, name, llfty);

    match ty::get(output).sty {
        // functions returning bottom may unwind, but can never return normally
        ty::ty_bot => {
            unsafe {
                llvm::LLVMAddFunctionAttr(llfn, lib::llvm::NoReturnAttribute as c_uint)
            }
        }
        // `~` pointer return values never alias because ownership is transferred
        ty::ty_uniq(*) |
        ty::ty_evec(_, ty::vstore_uniq) => {
            unsafe {
                llvm::LLVMAddReturnAttribute(llfn, lib::llvm::NoAliasAttribute as c_uint);
            }
        }
        _ => ()
    }

    let uses_outptr = type_of::return_uses_outptr(ccx, output);
    let offset = if uses_outptr { 2 } else { 1 };

    for (i, &arg_ty) in inputs.iter().enumerate() {
        let llarg = unsafe { llvm::LLVMGetParam(llfn, (offset + i) as c_uint) };
        match ty::get(arg_ty).sty {
            // `~` pointer parameters never alias because ownership is transferred
            ty::ty_uniq(*) |
            ty::ty_evec(_, ty::vstore_uniq) |
            ty::ty_closure(ty::ClosureTy {sigil: ast::OwnedSigil, _}) => {
                unsafe {
                    llvm::LLVMAddAttribute(llarg, lib::llvm::NoAliasAttribute as c_uint);
                }
            }
            _ => ()
        }
    }

    // The out pointer will never alias with any other pointers, as the object only exists at a
    // language level after the call. It can also be tagged with SRet to indicate that it is
    // guaranteed to point to a usable block of memory for the type.
    if uses_outptr {
        unsafe {
            let outptr = llvm::LLVMGetParam(llfn, 0);
            llvm::LLVMAddAttribute(outptr, lib::llvm::StructRetAttribute as c_uint);
            llvm::LLVMAddAttribute(outptr, lib::llvm::NoAliasAttribute as c_uint);
        }
    }

    llfn
}

pub fn decl_internal_rust_fn(ccx: &mut CrateContext, inputs: &[ty::t], output: ty::t,
                             name: &str) -> ValueRef {
    let llfn = decl_rust_fn(ccx, inputs, output, name);
    lib::llvm::SetLinkage(llfn, lib::llvm::InternalLinkage);
    llfn
}

pub fn get_extern_const(externs: &mut ExternMap, llmod: ModuleRef,
                        name: &str, ty: Type) -> ValueRef {
    match externs.find_equiv(&name) {
        Some(n) => return *n,
        None => ()
    }
    unsafe {
        let c = do name.with_c_str |buf| {
            llvm::LLVMAddGlobal(llmod, ty.to_ref(), buf)
        };
        externs.insert(name.to_owned(), c);
        return c;
    }
}
pub fn umax(cx: @mut Block, a: ValueRef, b: ValueRef) -> ValueRef {
    let _icx = push_ctxt("umax");
    let cond = ICmp(cx, lib::llvm::IntULT, a, b);
    return Select(cx, cond, b, a);
}

pub fn umin(cx: @mut Block, a: ValueRef, b: ValueRef) -> ValueRef {
    let _icx = push_ctxt("umin");
    let cond = ICmp(cx, lib::llvm::IntULT, a, b);
    return Select(cx, cond, a, b);
}

// Given a pointer p, returns a pointer sz(p) (i.e., inc'd by sz bytes).
// The type of the returned pointer is always i8*.  If you care about the
// return type, use bump_ptr().
pub fn ptr_offs(bcx: @mut Block, base: ValueRef, sz: ValueRef) -> ValueRef {
    let _icx = push_ctxt("ptr_offs");
    let raw = PointerCast(bcx, base, Type::i8p());
    InBoundsGEP(bcx, raw, [sz])
}

// Increment a pointer by a given amount and then cast it to be a pointer
// to a given type.
pub fn bump_ptr(bcx: @mut Block, t: ty::t, base: ValueRef, sz: ValueRef) ->
   ValueRef {
    let _icx = push_ctxt("bump_ptr");
    let ccx = bcx.ccx();
    let bumped = ptr_offs(bcx, base, sz);
    let typ = type_of(ccx, t).ptr_to();
    PointerCast(bcx, bumped, typ)
}

// Returns a pointer to the body for the box. The box may be an opaque
// box. The result will be casted to the type of body_t, if it is statically
// known.
//
// The runtime equivalent is box_body() in "rust_internal.h".
pub fn opaque_box_body(bcx: @mut Block,
                       body_t: ty::t,
                       boxptr: ValueRef) -> ValueRef {
    let _icx = push_ctxt("opaque_box_body");
    let ccx = bcx.ccx();
    let ty = type_of(ccx, body_t);
    let ty = Type::box(ccx, &ty);
    let boxptr = PointerCast(bcx, boxptr, ty.ptr_to());
    GEPi(bcx, boxptr, [0u, abi::box_field_body])
}

// malloc_raw_dyn: allocates a box to contain a given type, but with a
// potentially dynamic size.
pub fn malloc_raw_dyn(bcx: @mut Block,
                      t: ty::t,
                      heap: heap,
                      size: ValueRef) -> Result {
    let _icx = push_ctxt("malloc_raw");
    let ccx = bcx.ccx();

    fn require_alloc_fn(bcx: @mut Block, t: ty::t, it: LangItem) -> ast::DefId {
        let li = &bcx.tcx().lang_items;
        match li.require(it) {
            Ok(id) => id,
            Err(s) => {
                bcx.tcx().sess.fatal(format!("allocation of `{}` {}",
                                          bcx.ty_to_str(t), s));
            }
        }
    }

    if heap == heap_exchange {
        let llty_value = type_of::type_of(ccx, t);


        // Allocate space:
        let r = callee::trans_lang_call(
            bcx,
            require_alloc_fn(bcx, t, ExchangeMallocFnLangItem),
            [size],
            None);
        rslt(r.bcx, PointerCast(r.bcx, r.val, llty_value.ptr_to()))
    } else {
        // we treat ~fn, @fn and @[] as @ here, which isn't ideal
        let (mk_fn, langcall) = match heap {
            heap_managed | heap_managed_unique => {
                (ty::mk_imm_box,
                 require_alloc_fn(bcx, t, MallocFnLangItem))
            }
            heap_exchange_closure => {
                (ty::mk_imm_box,
                 require_alloc_fn(bcx, t, ClosureExchangeMallocFnLangItem))
            }
            _ => fail!("heap_exchange already handled")
        };

        // Grab the TypeRef type of box_ptr_ty.
        let box_ptr_ty = mk_fn(bcx.tcx(), t);
        let llty = type_of(ccx, box_ptr_ty);

        // Get the tydesc for the body:
        let static_ti = get_tydesc(ccx, t);
        glue::lazily_emit_all_tydesc_glue(ccx, static_ti);

        // Allocate space:
        let tydesc = PointerCast(bcx, static_ti.tydesc, Type::i8p());
        let r = callee::trans_lang_call(
            bcx,
            langcall,
            [tydesc, size],
            None);
        let r = rslt(r.bcx, PointerCast(r.bcx, r.val, llty));
        maybe_set_managed_unique_rc(r.bcx, r.val, heap);
        r
    }
}

// malloc_raw: expects an unboxed type and returns a pointer to
// enough space for a box of that type.  This includes a rust_opaque_box
// header.
pub fn malloc_raw(bcx: @mut Block, t: ty::t, heap: heap) -> Result {
    let ty = type_of(bcx.ccx(), t);
    let size = llsize_of(bcx.ccx(), ty);
    malloc_raw_dyn(bcx, t, heap, size)
}

pub struct MallocResult {
    bcx: @mut Block,
    box: ValueRef,
    body: ValueRef
}

// malloc_general_dyn: usefully wraps malloc_raw_dyn; allocates a box,
// and pulls out the body
pub fn malloc_general_dyn(bcx: @mut Block, t: ty::t, heap: heap, size: ValueRef)
    -> MallocResult {
    assert!(heap != heap_exchange);
    let _icx = push_ctxt("malloc_general");
    let Result {bcx: bcx, val: llbox} = malloc_raw_dyn(bcx, t, heap, size);
    let body = GEPi(bcx, llbox, [0u, abi::box_field_body]);

    MallocResult { bcx: bcx, box: llbox, body: body }
}

pub fn malloc_general(bcx: @mut Block, t: ty::t, heap: heap) -> MallocResult {
    let ty = type_of(bcx.ccx(), t);
    assert!(heap != heap_exchange);
    malloc_general_dyn(bcx, t, heap, llsize_of(bcx.ccx(), ty))
}
pub fn malloc_boxed(bcx: @mut Block, t: ty::t)
    -> MallocResult {
    malloc_general(bcx, t, heap_managed)
}

pub fn heap_for_unique(bcx: @mut Block, t: ty::t) -> heap {
    if ty::type_contents(bcx.tcx(), t).contains_managed() {
        heap_managed_unique
    } else {
        heap_exchange
    }
}

pub fn maybe_set_managed_unique_rc(bcx: @mut Block, bx: ValueRef, heap: heap) {
    assert!(heap != heap_exchange);
    if heap == heap_managed_unique {
        // In cases where we are looking at a unique-typed allocation in the
        // managed heap (thus have refcount 1 from the managed allocator),
        // such as a ~(@foo) or such. These need to have their refcount forced
        // to -2 so the annihilator ignores them.
        let rc = GEPi(bcx, bx, [0u, abi::box_field_refcnt]);
        let rc_val = C_int(bcx.ccx(), -2);
        Store(bcx, rc_val, rc);
    }
}

// Type descriptor and type glue stuff

pub fn get_tydesc_simple(ccx: &mut CrateContext, t: ty::t) -> ValueRef {
    get_tydesc(ccx, t).tydesc
}

pub fn get_tydesc(ccx: &mut CrateContext, t: ty::t) -> @mut tydesc_info {
    match ccx.tydescs.find(&t) {
        Some(&inf) => {
            return inf;
        }
        _ => { }
    }

    ccx.stats.n_static_tydescs += 1u;
    let inf = glue::declare_tydesc(ccx, t);
    ccx.tydescs.insert(t, inf);
    return inf;
}

pub fn set_optimize_for_size(f: ValueRef) {
    lib::llvm::SetFunctionAttribute(f, lib::llvm::OptimizeForSizeAttribute)
}

pub fn set_no_inline(f: ValueRef) {
    lib::llvm::SetFunctionAttribute(f, lib::llvm::NoInlineAttribute)
}

pub fn set_no_unwind(f: ValueRef) {
    lib::llvm::SetFunctionAttribute(f, lib::llvm::NoUnwindAttribute)
}

// Tell LLVM to emit the information necessary to unwind the stack for the
// function f.
pub fn set_uwtable(f: ValueRef) {
    lib::llvm::SetFunctionAttribute(f, lib::llvm::UWTableAttribute)
}

pub fn set_inline_hint(f: ValueRef) {
    lib::llvm::SetFunctionAttribute(f, lib::llvm::InlineHintAttribute)
}

pub fn set_llvm_fn_attrs(attrs: &[ast::Attribute], llfn: ValueRef) {
    use syntax::attr::*;
    // Set the inline hint if there is one
    match find_inline_attr(attrs) {
        InlineHint   => set_inline_hint(llfn),
        InlineAlways => set_always_inline(llfn),
        InlineNever  => set_no_inline(llfn),
        InlineNone   => { /* fallthrough */ }
    }

    // Add the no-split-stack attribute if requested
    if contains_name(attrs, "no_split_stack") {
        set_no_split_stack(llfn);
    }
}

pub fn set_always_inline(f: ValueRef) {
    lib::llvm::SetFunctionAttribute(f, lib::llvm::AlwaysInlineAttribute)
}

pub fn set_fixed_stack_segment(f: ValueRef) {
    do "fixed-stack-segment".with_c_str |buf| {
        unsafe { llvm::LLVMAddFunctionAttrString(f, buf); }
    }
}

pub fn set_no_split_stack(f: ValueRef) {
    do "no-split-stack".with_c_str |buf| {
        unsafe { llvm::LLVMAddFunctionAttrString(f, buf); }
    }
}

// Double-check that we never ask LLVM to declare the same symbol twice. It
// silently mangles such symbols, breaking our linkage model.
pub fn note_unique_llvm_symbol(ccx: &mut CrateContext, sym: @str) {
    if ccx.all_llvm_symbols.contains(&sym) {
        ccx.sess.bug(~"duplicate LLVM symbol: " + sym);
    }
    ccx.all_llvm_symbols.insert(sym);
}


pub fn get_res_dtor(ccx: @mut CrateContext,
                    did: ast::DefId,
                    parent_id: ast::DefId,
                    substs: &[ty::t])
                 -> ValueRef {
    let _icx = push_ctxt("trans_res_dtor");
    if !substs.is_empty() {
        let did = if did.crate != ast::LOCAL_CRATE {
            inline::maybe_instantiate_inline(ccx, did)
        } else {
            did
        };
        assert_eq!(did.crate, ast::LOCAL_CRATE);
        let tsubsts = ty::substs {regions: ty::ErasedRegions,
                                  self_ty: None,
                                  tps: /*bad*/ substs.to_owned() };
        let (val, _) = monomorphize::monomorphic_fn(ccx,
                                                    did,
                                                    &tsubsts,
                                                    None,
                                                    None,
                                                    None);

        val
    } else if did.crate == ast::LOCAL_CRATE {
        get_item_val(ccx, did.node)
    } else {
        let tcx = ccx.tcx;
        let name = csearch::get_symbol(ccx.sess.cstore, did);
        let class_ty = ty::subst_tps(tcx,
                                     substs,
                                     None,
                                     ty::lookup_item_type(tcx, parent_id).ty);
        let llty = type_of_dtor(ccx, class_ty);
        get_extern_fn(&mut ccx.externs,
                      ccx.llmod,
                      name,
                      lib::llvm::CCallConv,
                      llty)
    }
}

// Structural comparison: a rather involved form of glue.
pub fn maybe_name_value(cx: &CrateContext, v: ValueRef, s: &str) {
    if cx.sess.opts.save_temps {
        do s.with_c_str |buf| {
            unsafe {
                llvm::LLVMSetValueName(v, buf)
            }
        }
    }
}


// Used only for creating scalar comparison glue.
pub enum scalar_type { nil_type, signed_int, unsigned_int, floating_point, }

// NB: This produces an i1, not a Rust bool (i8).
pub fn compare_scalar_types(cx: @mut Block,
                            lhs: ValueRef,
                            rhs: ValueRef,
                            t: ty::t,
                            op: ast::BinOp)
                         -> Result {
    let f = |a| compare_scalar_values(cx, lhs, rhs, a, op);

    match ty::get(t).sty {
        ty::ty_nil => rslt(cx, f(nil_type)),
        ty::ty_bool | ty::ty_ptr(_) => rslt(cx, f(unsigned_int)),
        ty::ty_char => rslt(cx, f(unsigned_int)),
        ty::ty_int(_) => rslt(cx, f(signed_int)),
        ty::ty_uint(_) => rslt(cx, f(unsigned_int)),
        ty::ty_float(_) => rslt(cx, f(floating_point)),
        ty::ty_type => {
            rslt(
                controlflow::trans_fail(
                    cx, None,
                    @"attempt to compare values of type type"),
                C_nil())
        }
        _ => {
            // Should never get here, because t is scalar.
            cx.sess().bug("non-scalar type passed to \
                           compare_scalar_types")
        }
    }
}


// A helper function to do the actual comparison of scalar values.
pub fn compare_scalar_values(cx: @mut Block,
                             lhs: ValueRef,
                             rhs: ValueRef,
                             nt: scalar_type,
                             op: ast::BinOp)
                          -> ValueRef {
    let _icx = push_ctxt("compare_scalar_values");
    fn die(cx: @mut Block) -> ! {
        cx.tcx().sess.bug("compare_scalar_values: must be a\
                           comparison operator");
    }
    match nt {
      nil_type => {
        // We don't need to do actual comparisons for nil.
        // () == () holds but () < () does not.
        match op {
          ast::BiEq | ast::BiLe | ast::BiGe => return C_i1(true),
          ast::BiNe | ast::BiLt | ast::BiGt => return C_i1(false),
          // refinements would be nice
          _ => die(cx)
        }
      }
      floating_point => {
        let cmp = match op {
          ast::BiEq => lib::llvm::RealOEQ,
          ast::BiNe => lib::llvm::RealUNE,
          ast::BiLt => lib::llvm::RealOLT,
          ast::BiLe => lib::llvm::RealOLE,
          ast::BiGt => lib::llvm::RealOGT,
          ast::BiGe => lib::llvm::RealOGE,
          _ => die(cx)
        };
        return FCmp(cx, cmp, lhs, rhs);
      }
      signed_int => {
        let cmp = match op {
          ast::BiEq => lib::llvm::IntEQ,
          ast::BiNe => lib::llvm::IntNE,
          ast::BiLt => lib::llvm::IntSLT,
          ast::BiLe => lib::llvm::IntSLE,
          ast::BiGt => lib::llvm::IntSGT,
          ast::BiGe => lib::llvm::IntSGE,
          _ => die(cx)
        };
        return ICmp(cx, cmp, lhs, rhs);
      }
      unsigned_int => {
        let cmp = match op {
          ast::BiEq => lib::llvm::IntEQ,
          ast::BiNe => lib::llvm::IntNE,
          ast::BiLt => lib::llvm::IntULT,
          ast::BiLe => lib::llvm::IntULE,
          ast::BiGt => lib::llvm::IntUGT,
          ast::BiGe => lib::llvm::IntUGE,
          _ => die(cx)
        };
        return ICmp(cx, cmp, lhs, rhs);
      }
    }
}

pub type val_and_ty_fn<'self> = &'self fn(@mut Block, ValueRef, ty::t) -> @mut Block;

pub fn load_inbounds(cx: @mut Block, p: ValueRef, idxs: &[uint]) -> ValueRef {
    return Load(cx, GEPi(cx, p, idxs));
}

pub fn store_inbounds(cx: @mut Block, v: ValueRef, p: ValueRef, idxs: &[uint]) {
    Store(cx, v, GEPi(cx, p, idxs));
}

// Iterates through the elements of a structural type.
pub fn iter_structural_ty(cx: @mut Block, av: ValueRef, t: ty::t,
                          f: val_and_ty_fn) -> @mut Block {
    let _icx = push_ctxt("iter_structural_ty");

    fn iter_variant(cx: @mut Block, repr: &adt::Repr, av: ValueRef,
                    variant: @ty::VariantInfo,
                    tps: &[ty::t], f: val_and_ty_fn) -> @mut Block {
        let _icx = push_ctxt("iter_variant");
        let tcx = cx.tcx();
        let mut cx = cx;

        for (i, &arg) in variant.args.iter().enumerate() {
            cx = f(cx,
                   adt::trans_field_ptr(cx, repr, av, variant.disr_val, i),
                   ty::subst_tps(tcx, tps, None, arg));
        }
        return cx;
    }

    let mut cx = cx;
    match ty::get(t).sty {
      ty::ty_struct(*) => {
          let repr = adt::represent_type(cx.ccx(), t);
          do expr::with_field_tys(cx.tcx(), t, None) |discr, field_tys| {
              for (i, field_ty) in field_tys.iter().enumerate() {
                  let llfld_a = adt::trans_field_ptr(cx, repr, av, discr, i);
                  cx = f(cx, llfld_a, field_ty.mt.ty);
              }
          }
      }
      ty::ty_estr(ty::vstore_fixed(_)) |
      ty::ty_evec(_, ty::vstore_fixed(_)) => {
        let (base, len) = tvec::get_base_and_byte_len(cx, av, t);
        cx = tvec::iter_vec_raw(cx, base, t, len, f);
      }
      ty::ty_tup(ref args) => {
          let repr = adt::represent_type(cx.ccx(), t);
          for (i, arg) in args.iter().enumerate() {
              let llfld_a = adt::trans_field_ptr(cx, repr, av, 0, i);
              cx = f(cx, llfld_a, *arg);
          }
      }
      ty::ty_enum(tid, ref substs) => {
          let ccx = cx.ccx();

          let repr = adt::represent_type(ccx, t);
          let variants = ty::enum_variants(ccx.tcx, tid);
          let n_variants = (*variants).len();

          // NB: we must hit the discriminant first so that structural
          // comparison know not to proceed when the discriminants differ.

          match adt::trans_switch(cx, repr, av) {
              (_match::single, None) => {
                  cx = iter_variant(cx, repr, av, variants[0],
                                    substs.tps, f);
              }
              (_match::switch, Some(lldiscrim_a)) => {
                  cx = f(cx, lldiscrim_a, ty::mk_int());
                  let unr_cx = sub_block(cx, "enum-iter-unr");
                  Unreachable(unr_cx);
                  let llswitch = Switch(cx, lldiscrim_a, unr_cx.llbb,
                                        n_variants);
                  let next_cx = sub_block(cx, "enum-iter-next");

                  for variant in (*variants).iter() {
                      let variant_cx =
                          sub_block(cx, ~"enum-iter-variant-" +
                                    variant.disr_val.to_str());
                      let variant_cx =
                          iter_variant(variant_cx, repr, av, *variant,
                                       substs.tps, |x,y,z| f(x,y,z));
                      match adt::trans_case(cx, repr, variant.disr_val) {
                          _match::single_result(r) => {
                              AddCase(llswitch, r.val, variant_cx.llbb)
                          }
                          _ => ccx.sess.unimpl("value from adt::trans_case \
                                                in iter_structural_ty")
                      }
                      Br(variant_cx, next_cx.llbb);
                  }
                  cx = next_cx;
              }
              _ => ccx.sess.unimpl("value from adt::trans_switch \
                                    in iter_structural_ty")
          }
      }
      _ => cx.sess().unimpl("type in iter_structural_ty")
    }
    return cx;
}

pub fn cast_shift_expr_rhs(cx: @mut Block, op: ast::BinOp,
                           lhs: ValueRef, rhs: ValueRef) -> ValueRef {
    cast_shift_rhs(op, lhs, rhs,
                   |a,b| Trunc(cx, a, b),
                   |a,b| ZExt(cx, a, b))
}

pub fn cast_shift_const_rhs(op: ast::BinOp,
                            lhs: ValueRef, rhs: ValueRef) -> ValueRef {
    cast_shift_rhs(op, lhs, rhs,
                   |a, b| unsafe { llvm::LLVMConstTrunc(a, b.to_ref()) },
                   |a, b| unsafe { llvm::LLVMConstZExt(a, b.to_ref()) })
}

pub fn cast_shift_rhs(op: ast::BinOp,
                      lhs: ValueRef, rhs: ValueRef,
                      trunc: &fn(ValueRef, Type) -> ValueRef,
                      zext: &fn(ValueRef, Type) -> ValueRef)
                   -> ValueRef {
    // Shifts may have any size int on the rhs
    unsafe {
        if ast_util::is_shift_binop(op) {
            let rhs_llty = val_ty(rhs);
            let lhs_llty = val_ty(lhs);
            let rhs_sz = llvm::LLVMGetIntTypeWidth(rhs_llty.to_ref());
            let lhs_sz = llvm::LLVMGetIntTypeWidth(lhs_llty.to_ref());
            if lhs_sz < rhs_sz {
                trunc(rhs, lhs_llty)
            } else if lhs_sz > rhs_sz {
                // FIXME (#1877: If shifting by negative
                // values becomes not undefined then this is wrong.
                zext(rhs, lhs_llty)
            } else {
                rhs
            }
        } else {
            rhs
        }
    }
}

pub fn fail_if_zero(cx: @mut Block, span: Span, divrem: ast::BinOp,
                    rhs: ValueRef, rhs_t: ty::t) -> @mut Block {
    let text = if divrem == ast::BiDiv {
        @"attempted to divide by zero"
    } else {
        @"attempted remainder with a divisor of zero"
    };
    let is_zero = match ty::get(rhs_t).sty {
      ty::ty_int(t) => {
        let zero = C_integral(Type::int_from_ty(cx.ccx(), t), 0u64, false);
        ICmp(cx, lib::llvm::IntEQ, rhs, zero)
      }
      ty::ty_uint(t) => {
        let zero = C_integral(Type::uint_from_ty(cx.ccx(), t), 0u64, false);
        ICmp(cx, lib::llvm::IntEQ, rhs, zero)
      }
      _ => {
        cx.tcx().sess.bug(~"fail-if-zero on unexpected type: " +
                          ty_to_str(cx.ccx().tcx, rhs_t));
      }
    };
    do with_cond(cx, is_zero) |bcx| {
        controlflow::trans_fail(bcx, Some(span), text)
    }
}

pub fn null_env_ptr(ccx: &CrateContext) -> ValueRef {
    C_null(Type::opaque_box(ccx).ptr_to())
}

pub fn trans_external_path(ccx: &mut CrateContext, did: ast::DefId, t: ty::t) -> ValueRef {
    let name = csearch::get_symbol(ccx.sess.cstore, did);
    match ty::get(t).sty {
        ty::ty_bare_fn(ref fn_ty) => {
            match fn_ty.abis.for_arch(ccx.sess.targ_cfg.arch) {
                Some(Rust) | Some(RustIntrinsic) => {
                    get_extern_rust_fn(ccx, fn_ty.sig.inputs, fn_ty.sig.output, name)
                }
                Some(*) | None => {
                    let c = foreign::llvm_calling_convention(ccx, fn_ty.abis);
                    let cconv = c.unwrap_or(lib::llvm::CCallConv);
                    let llty = type_of_fn_from_ty(ccx, t);
                    get_extern_fn(&mut ccx.externs, ccx.llmod, name, cconv, llty)
                }
            }
        }
        ty::ty_closure(ref f) => {
            get_extern_rust_fn(ccx, f.sig.inputs, f.sig.output, name)
        }
        _ => {
            let llty = type_of(ccx, t);
            get_extern_const(&mut ccx.externs, ccx.llmod, name, llty)
        }
    }
}

pub fn invoke(bcx: @mut Block, llfn: ValueRef, llargs: ~[ValueRef],
              attributes: &[(uint, lib::llvm::Attribute)])
           -> (ValueRef, @mut Block) {
    let _icx = push_ctxt("invoke_");
    if bcx.unreachable {
        return (C_null(Type::i8()), bcx);
    }

    match bcx.node_info {
        None => debug!("invoke at ???"),
        Some(node_info) => {
            debug!("invoke at {}",
                   bcx.sess().codemap.span_to_str(node_info.span));
        }
    }

    if need_invoke(bcx) {
        unsafe {
            debug!("invoking {} at {}", llfn, bcx.llbb);
            for &llarg in llargs.iter() {
                debug!("arg: {}", llarg);
            }
        }
        let normal_bcx = sub_block(bcx, "normal return");
        let llresult = Invoke(bcx,
                              llfn,
                              llargs,
                              normal_bcx.llbb,
                              get_landing_pad(bcx),
                              attributes);
        return (llresult, normal_bcx);
    } else {
        unsafe {
            debug!("calling {} at {}", llfn, bcx.llbb);
            for &llarg in llargs.iter() {
                debug!("arg: {}", llarg);
            }
        }
        let llresult = Call(bcx, llfn, llargs, attributes);
        return (llresult, bcx);
    }
}

pub fn need_invoke(bcx: @mut Block) -> bool {
    if (bcx.ccx().sess.opts.debugging_opts & session::no_landing_pads != 0) {
        return false;
    }

    // Avoid using invoke if we are already inside a landing pad.
    if bcx.is_lpad {
        return false;
    }

    if have_cached_lpad(bcx) {
        return true;
    }

    // Walk the scopes to look for cleanups
    let mut cur = bcx;
    let mut cur_scope = cur.scope;
    loop {
        cur_scope = match cur_scope {
            Some(inf) => {
                for cleanup in inf.cleanups.iter() {
                    match *cleanup {
                        clean(_, cleanup_type) | clean_temp(_, _, cleanup_type) => {
                            if cleanup_type == normal_exit_and_unwind {
                                return true;
                            }
                        }
                    }
                }
                inf.parent
            }
            None => {
                cur = match cur.parent {
                    Some(next) => next,
                    None => return false
                };
                cur.scope
            }
        }
    }
}

pub fn have_cached_lpad(bcx: @mut Block) -> bool {
    let mut res = false;
    do in_lpad_scope_cx(bcx) |inf| {
        match inf.landing_pad {
          Some(_) => res = true,
          None => res = false
        }
    }
    return res;
}

pub fn in_lpad_scope_cx(bcx: @mut Block, f: &fn(si: &mut ScopeInfo)) {
    let mut bcx = bcx;
    let mut cur_scope = bcx.scope;
    loop {
        cur_scope = match cur_scope {
            Some(inf) => {
                if !inf.empty_cleanups() || (inf.parent.is_none() && bcx.parent.is_none()) {
                    f(inf);
                    return;
                }
                inf.parent
            }
            None => {
                bcx = block_parent(bcx);
                bcx.scope
            }
        }
    }
}

pub fn get_landing_pad(bcx: @mut Block) -> BasicBlockRef {
    let _icx = push_ctxt("get_landing_pad");

    let mut cached = None;
    let mut pad_bcx = bcx; // Guaranteed to be set below
    do in_lpad_scope_cx(bcx) |inf| {
        // If there is a valid landing pad still around, use it
        match inf.landing_pad {
          Some(target) => cached = Some(target),
          None => {
            pad_bcx = lpad_block(bcx, "unwind");
            inf.landing_pad = Some(pad_bcx.llbb);
          }
        }
    }
    // Can't return from block above
    match cached { Some(b) => return b, None => () }
    // The landing pad return type (the type being propagated). Not sure what
    // this represents but it's determined by the personality function and
    // this is what the EH proposal example uses.
    let llretty = Type::struct_([Type::i8p(), Type::i32()], false);
    // The exception handling personality function. This is the C++
    // personality function __gxx_personality_v0, wrapped in our naming
    // convention.
    let personality = bcx.ccx().upcalls.rust_personality;
    // The only landing pad clause will be 'cleanup'
    let llretval = LandingPad(pad_bcx, llretty, personality, 1u);
    // The landing pad block is a cleanup
    SetCleanup(pad_bcx, llretval);

    // Because we may have unwound across a stack boundary, we must call into
    // the runtime to figure out which stack segment we are on and place the
    // stack limit back into the TLS.
    Call(pad_bcx, bcx.ccx().upcalls.reset_stack_limit, [], []);

    // We store the retval in a function-central alloca, so that calls to
    // Resume can find it.
    match bcx.fcx.personality {
      Some(addr) => Store(pad_bcx, llretval, addr),
      None => {
        let addr = alloca(pad_bcx, val_ty(llretval), "");
        bcx.fcx.personality = Some(addr);
        Store(pad_bcx, llretval, addr);
      }
    }

    // Unwind all parent scopes, and finish with a Resume instr
    cleanup_and_leave(pad_bcx, None, None);
    return pad_bcx.llbb;
}

pub fn find_bcx_for_scope(bcx: @mut Block, scope_id: ast::NodeId) -> @mut Block {
    let mut bcx_sid = bcx;
    let mut cur_scope = bcx_sid.scope;
    loop {
        cur_scope = match cur_scope {
            Some(inf) => {
                match inf.node_info {
                    Some(NodeInfo { id, _ }) if id == scope_id => {
                        return bcx_sid
                    }
                    // FIXME(#6268, #6248) hacky cleanup for nested method calls
                    Some(NodeInfo { callee_id: Some(id), _ }) if id == scope_id => {
                        return bcx_sid
                    }
                    _ => inf.parent
                }
            }
            None => {
                bcx_sid = match bcx_sid.parent {
                    None => bcx.tcx().sess.bug(format!("no enclosing scope with id {}", scope_id)),
                    Some(bcx_par) => bcx_par
                };
                bcx_sid.scope
            }
        }
    }
}


pub fn do_spill(bcx: @mut Block, v: ValueRef, t: ty::t) -> ValueRef {
    if ty::type_is_bot(t) {
        return C_null(Type::i8p());
    }
    let llptr = alloc_ty(bcx, t, "");
    Store(bcx, v, llptr);
    return llptr;
}

// Since this function does *not* root, it is the caller's responsibility to
// ensure that the referent is pointed to by a root.
pub fn do_spill_noroot(cx: @mut Block, v: ValueRef) -> ValueRef {
    let llptr = alloca(cx, val_ty(v), "");
    Store(cx, v, llptr);
    return llptr;
}

pub fn spill_if_immediate(cx: @mut Block, v: ValueRef, t: ty::t) -> ValueRef {
    let _icx = push_ctxt("spill_if_immediate");
    if type_is_immediate(cx.ccx(), t) { return do_spill(cx, v, t); }
    return v;
}

pub fn load_if_immediate(cx: @mut Block, v: ValueRef, t: ty::t) -> ValueRef {
    let _icx = push_ctxt("load_if_immediate");
    if type_is_immediate(cx.ccx(), t) { return Load(cx, v); }
    return v;
}

pub fn trans_trace(bcx: @mut Block, sp_opt: Option<Span>, trace_str: @str) {
    if !bcx.sess().trace() { return; }
    let _icx = push_ctxt("trans_trace");
    add_comment(bcx, trace_str);
    let V_trace_str = C_cstr(bcx.ccx(), trace_str);
    let (V_filename, V_line) = match sp_opt {
      Some(sp) => {
        let sess = bcx.sess();
        let loc = sess.parse_sess.cm.lookup_char_pos(sp.lo);
        (C_cstr(bcx.ccx(), loc.file.name), loc.line as int)
      }
      None => {
        (C_cstr(bcx.ccx(), @"<runtime>"), 0)
      }
    };
    let ccx = bcx.ccx();
    let V_trace_str = PointerCast(bcx, V_trace_str, Type::i8p());
    let V_filename = PointerCast(bcx, V_filename, Type::i8p());
    let args = ~[V_trace_str, V_filename, C_int(ccx, V_line)];
    Call(bcx, ccx.upcalls.trace, args, []);
}

pub fn ignore_lhs(_bcx: @mut Block, local: &ast::Local) -> bool {
    match local.pat.node {
        ast::PatWild => true, _ => false
    }
}

pub fn init_local(bcx: @mut Block, local: &ast::Local) -> @mut Block {

    debug!("init_local(bcx={}, local.id={:?})",
           bcx.to_str(), local.id);
    let _indenter = indenter();

    let _icx = push_ctxt("init_local");

    if ignore_lhs(bcx, local) {
        // Handle let _ = e; just like e;
        match local.init {
            Some(init) => {
              return expr::trans_into(bcx, init, expr::Ignore);
            }
            None => { return bcx; }
        }
    }

    _match::store_local(bcx, local.pat, local.init)
}

pub fn trans_stmt(cx: @mut Block, s: &ast::Stmt) -> @mut Block {
    let _icx = push_ctxt("trans_stmt");
    debug!("trans_stmt({})", stmt_to_str(s, cx.tcx().sess.intr()));

    if cx.sess().asm_comments() {
        add_span_comment(cx, s.span, stmt_to_str(s, cx.ccx().sess.intr()));
    }

    let mut bcx = cx;

    match s.node {
        ast::StmtExpr(e, _) | ast::StmtSemi(e, _) => {
            bcx = expr::trans_into(cx, e, expr::Ignore);
        }
        ast::StmtDecl(d, _) => {
            match d.node {
                ast::DeclLocal(ref local) => {
                    bcx = init_local(bcx, *local);
                    if cx.sess().opts.extra_debuginfo {
                        debuginfo::create_local_var_metadata(bcx, *local);
                    }
                }
                ast::DeclItem(i) => trans_item(cx.fcx.ccx, i)
            }
        }
        ast::StmtMac(*) => cx.tcx().sess.bug("unexpanded macro")
    }

    return bcx;
}

// You probably don't want to use this one. See the
// next three functions instead.
pub fn new_block(cx: @mut FunctionContext,
                 parent: Option<@mut Block>,
                 scope: Option<@mut ScopeInfo>,
                 is_lpad: bool,
                 name: &str,
                 opt_node_info: Option<NodeInfo>)
              -> @mut Block {
    unsafe {
        let llbb = do name.with_c_str |buf| {
            llvm::LLVMAppendBasicBlockInContext(cx.ccx.llcx, cx.llfn, buf)
        };
        let bcx = @mut Block::new(llbb,
                                  parent,
                                  is_lpad,
                                  opt_node_info,
                                  cx);
        bcx.scope = scope;
        for cx in parent.iter() {
            if cx.unreachable {
                Unreachable(bcx);
                break;
            }
        }
        bcx
    }
}

pub fn simple_block_scope(parent: Option<@mut ScopeInfo>,
                          node_info: Option<NodeInfo>) -> @mut ScopeInfo {
    @mut ScopeInfo {
        parent: parent,
        loop_break: None,
        loop_label: None,
        cleanups: ~[],
        cleanup_paths: ~[],
        landing_pad: None,
        node_info: node_info,
    }
}

// Use this when you're at the top block of a function or the like.
pub fn top_scope_block(fcx: @mut FunctionContext, opt_node_info: Option<NodeInfo>)
                    -> @mut Block {
    return new_block(fcx, None, Some(simple_block_scope(None, opt_node_info)), false,
                  "function top level", opt_node_info);
}

pub fn scope_block(bcx: @mut Block,
                   opt_node_info: Option<NodeInfo>,
                   n: &str) -> @mut Block {
    return new_block(bcx.fcx, Some(bcx), Some(simple_block_scope(None, opt_node_info)), bcx.is_lpad,
                  n, opt_node_info);
}

pub fn loop_scope_block(bcx: @mut Block,
                        loop_break: @mut Block,
                        loop_label: Option<Name>,
                        n: &str,
                        opt_node_info: Option<NodeInfo>) -> @mut Block {
    return new_block(bcx.fcx, Some(bcx), Some(@mut ScopeInfo {
        parent: None,
        loop_break: Some(loop_break),
        loop_label: loop_label,
        cleanups: ~[],
        cleanup_paths: ~[],
        landing_pad: None,
        node_info: opt_node_info,
    }), bcx.is_lpad, n, opt_node_info);
}

// Use this when creating a block for the inside of a landing pad.
pub fn lpad_block(bcx: @mut Block, n: &str) -> @mut Block {
    new_block(bcx.fcx, Some(bcx), None, true, n, None)
}

// Use this when you're making a general CFG BB within a scope.
pub fn sub_block(bcx: @mut Block, n: &str) -> @mut Block {
    new_block(bcx.fcx, Some(bcx), None, bcx.is_lpad, n, None)
}

pub fn raw_block(fcx: @mut FunctionContext, is_lpad: bool, llbb: BasicBlockRef) -> @mut Block {
    @mut Block::new(llbb, None, is_lpad, None, fcx)
}


// trans_block_cleanups: Go through all the cleanups attached to this
// block and execute them.
//
// When translating a block that introduces new variables during its scope, we
// need to make sure those variables go out of scope when the block ends.  We
// do that by running a 'cleanup' function for each variable.
// trans_block_cleanups runs all the cleanup functions for the block.
pub fn trans_block_cleanups(bcx: @mut Block, cleanups: ~[cleanup]) -> @mut Block {
    trans_block_cleanups_(bcx, cleanups, false)
}

pub fn trans_block_cleanups_(bcx: @mut Block,
                             cleanups: &[cleanup],
                             /* cleanup_cx: block, */
                             is_lpad: bool) -> @mut Block {
    let _icx = push_ctxt("trans_block_cleanups");
    // NB: Don't short-circuit even if this block is unreachable because
    // GC-based cleanup needs to the see that the roots are live.
    let no_lpads =
        bcx.ccx().sess.opts.debugging_opts & session::no_landing_pads != 0;
    if bcx.unreachable && !no_lpads { return bcx; }
    let mut bcx = bcx;
    for cu in cleanups.rev_iter() {
        match *cu {
            clean(cfn, cleanup_type) | clean_temp(_, cfn, cleanup_type) => {
                // Some types don't need to be cleaned up during
                // landing pads because they can be freed en mass later
                if cleanup_type == normal_exit_and_unwind || !is_lpad {
                    bcx = cfn.clean(bcx);
                }
            }
        }
    }
    return bcx;
}

// In the last argument, Some(block) mean jump to this block, and none means
// this is a landing pad and leaving should be accomplished with a resume
// instruction.
pub fn cleanup_and_leave(bcx: @mut Block,
                         upto: Option<BasicBlockRef>,
                         leave: Option<BasicBlockRef>) {
    let _icx = push_ctxt("cleanup_and_leave");
    let mut cur = bcx;
    let mut bcx = bcx;
    let is_lpad = leave == None;
    loop {
        debug!("cleanup_and_leave: leaving {}", cur.to_str());

        if bcx.sess().trace() {
            trans_trace(
                bcx, None,
                (format!("cleanup_and_leave({})", cur.to_str())).to_managed());
        }

        let mut cur_scope = cur.scope;
        loop {
            cur_scope = match cur_scope {
                Some (inf) if !inf.empty_cleanups() => {
                    let (sub_cx, dest, inf_cleanups) = {
                        let inf = &mut *inf;
                        let mut skip = 0;
                        let mut dest = None;
                        {
                            let r = (*inf).cleanup_paths.rev_iter().find(|cp| cp.target == leave);
                            for cp in r.iter() {
                                if cp.size == inf.cleanups.len() {
                                    Br(bcx, cp.dest);
                                    return;
                                }

                                skip = cp.size;
                                dest = Some(cp.dest);
                            }
                        }
                        let sub_cx = sub_block(bcx, "cleanup");
                        Br(bcx, sub_cx.llbb);
                        inf.cleanup_paths.push(cleanup_path {
                            target: leave,
                            size: inf.cleanups.len(),
                            dest: sub_cx.llbb
                        });
                        (sub_cx, dest, inf.cleanups.tailn(skip).to_owned())
                    };
                    bcx = trans_block_cleanups_(sub_cx,
                                                inf_cleanups,
                                                is_lpad);
                    for &dest in dest.iter() {
                        Br(bcx, dest);
                        return;
                    }
                    inf.parent
                }
                Some(inf) => inf.parent,
                None => break
            }
        }

        match upto {
          Some(bb) => { if cur.llbb == bb { break; } }
          _ => ()
        }
        cur = match cur.parent {
          Some(next) => next,
          None => { assert!(upto.is_none()); break; }
        };
    }
    match leave {
      Some(target) => Br(bcx, target),
      None => {
          let ll_load = Load(bcx, bcx.fcx.personality.unwrap());
          Resume(bcx, ll_load);
      }
    }
}

pub fn cleanup_block(bcx: @mut Block, upto: Option<BasicBlockRef>) -> @mut Block{
    let _icx = push_ctxt("cleanup_block");
    let mut cur = bcx;
    let mut bcx = bcx;
    loop {
        debug!("cleanup_block: {}", cur.to_str());

        if bcx.sess().trace() {
            trans_trace(
                bcx, None,
                (format!("cleanup_block({})", cur.to_str())).to_managed());
        }

        let mut cur_scope = cur.scope;
        loop {
            cur_scope = match cur_scope {
                Some (inf) => {
                    bcx = trans_block_cleanups_(bcx, inf.cleanups.to_owned(), false);
                    inf.parent
                }
                None => break
            }
        }

        match upto {
          Some(bb) => { if cur.llbb == bb { break; } }
          _ => ()
        }
        cur = match cur.parent {
          Some(next) => next,
          None => { assert!(upto.is_none()); break; }
        };
    }
    bcx
}

pub fn cleanup_and_Br(bcx: @mut Block, upto: @mut Block, target: BasicBlockRef) {
    let _icx = push_ctxt("cleanup_and_Br");
    cleanup_and_leave(bcx, Some(upto.llbb), Some(target));
}

pub fn leave_block(bcx: @mut Block, out_of: @mut Block) -> @mut Block {
    let _icx = push_ctxt("leave_block");
    let next_cx = sub_block(block_parent(out_of), "next");
    if bcx.unreachable { Unreachable(next_cx); }
    cleanup_and_Br(bcx, out_of, next_cx.llbb);
    next_cx
}

pub fn with_scope(bcx: @mut Block,
                  opt_node_info: Option<NodeInfo>,
                  name: &str,
                  f: &fn(@mut Block) -> @mut Block) -> @mut Block {
    let _icx = push_ctxt("with_scope");

    debug!("with_scope(bcx={}, opt_node_info={:?}, name={})",
           bcx.to_str(), opt_node_info, name);
    let _indenter = indenter();

    let scope = simple_block_scope(bcx.scope, opt_node_info);
    bcx.scope = Some(scope);
    let ret = f(bcx);
    let ret = trans_block_cleanups_(ret, (scope.cleanups).clone(), false);
    bcx.scope = scope.parent;
    ret
}

pub fn with_scope_result(bcx: @mut Block,
                         opt_node_info: Option<NodeInfo>,
                         _name: &str,
                         f: &fn(@mut Block) -> Result) -> Result {
    let _icx = push_ctxt("with_scope_result");

    let scope = simple_block_scope(bcx.scope, opt_node_info);
    bcx.scope = Some(scope);
    let Result { bcx: out_bcx, val } = f(bcx);
    let out_bcx = trans_block_cleanups_(out_bcx,
                                        (scope.cleanups).clone(),
                                        false);
    bcx.scope = scope.parent;

    rslt(out_bcx, val)
}

pub fn with_scope_datumblock(bcx: @mut Block, opt_node_info: Option<NodeInfo>,
                             name: &str, f: &fn(@mut Block) -> datum::DatumBlock)
                          -> datum::DatumBlock {
    use middle::trans::datum::DatumBlock;

    let _icx = push_ctxt("with_scope_result");
    let scope_cx = scope_block(bcx, opt_node_info, name);
    Br(bcx, scope_cx.llbb);
    let DatumBlock {bcx, datum} = f(scope_cx);
    DatumBlock {bcx: leave_block(bcx, scope_cx), datum: datum}
}

pub fn block_locals(b: &ast::Block, it: &fn(@ast::Local)) {
    for s in b.stmts.iter() {
        match s.node {
          ast::StmtDecl(d, _) => {
            match d.node {
              ast::DeclLocal(ref local) => it(*local),
              _ => {} /* fall through */
            }
          }
          _ => {} /* fall through */
        }
    }
}

pub fn with_cond(bcx: @mut Block, val: ValueRef, f: &fn(@mut Block) -> @mut Block) -> @mut Block {
    let _icx = push_ctxt("with_cond");
    let next_cx = base::sub_block(bcx, "next");
    let cond_cx = base::sub_block(bcx, "cond");
    CondBr(bcx, val, cond_cx.llbb, next_cx.llbb);
    let after_cx = f(cond_cx);
    if !after_cx.terminated { Br(after_cx, next_cx.llbb); }
    next_cx
}

pub fn call_memcpy(cx: @mut Block, dst: ValueRef, src: ValueRef, n_bytes: ValueRef, align: u32) {
    let _icx = push_ctxt("call_memcpy");
    let ccx = cx.ccx();
    let key = match ccx.sess.targ_cfg.arch {
        X86 | Arm | Mips => "llvm.memcpy.p0i8.p0i8.i32",
        X86_64 => "llvm.memcpy.p0i8.p0i8.i64"
    };
    let memcpy = ccx.intrinsics.get_copy(&key);
    let src_ptr = PointerCast(cx, src, Type::i8p());
    let dst_ptr = PointerCast(cx, dst, Type::i8p());
    let size = IntCast(cx, n_bytes, ccx.int_type);
    let align = C_i32(align as i32);
    let volatile = C_i1(false);
    Call(cx, memcpy, [dst_ptr, src_ptr, size, align, volatile], []);
}

pub fn memcpy_ty(bcx: @mut Block, dst: ValueRef, src: ValueRef, t: ty::t) {
    let _icx = push_ctxt("memcpy_ty");
    let ccx = bcx.ccx();
    if ty::type_is_structural(t) {
        let llty = type_of::type_of(ccx, t);
        let llsz = llsize_of(ccx, llty);
        let llalign = llalign_of_min(ccx, llty);
        call_memcpy(bcx, dst, src, llsz, llalign as u32);
    } else {
        Store(bcx, Load(bcx, src), dst);
    }
}

pub fn zero_mem(cx: @mut Block, llptr: ValueRef, t: ty::t) {
    if cx.unreachable { return; }
    let _icx = push_ctxt("zero_mem");
    let bcx = cx;
    let ccx = cx.ccx();
    let llty = type_of::type_of(ccx, t);
    memzero(&B(bcx), llptr, llty);
}

// Always use this function instead of storing a zero constant to the memory
// in question. If you store a zero constant, LLVM will drown in vreg
// allocation for large data structures, and the generated code will be
// awful. (A telltale sign of this is large quantities of
// `mov [byte ptr foo],0` in the generated code.)
pub fn memzero(b: &Builder, llptr: ValueRef, ty: Type) {
    let _icx = push_ctxt("memzero");
    let ccx = b.ccx;

    let intrinsic_key = match ccx.sess.targ_cfg.arch {
        X86 | Arm | Mips => "llvm.memset.p0i8.i32",
        X86_64 => "llvm.memset.p0i8.i64"
    };

    let llintrinsicfn = ccx.intrinsics.get_copy(&intrinsic_key);
    let llptr = b.pointercast(llptr, Type::i8().ptr_to());
    let llzeroval = C_u8(0);
    let size = machine::llsize_of(ccx, ty);
    let align = C_i32(llalign_of_min(ccx, ty) as i32);
    let volatile = C_i1(false);
    b.call(llintrinsicfn, [llptr, llzeroval, size, align, volatile], []);
}

pub fn alloc_ty(bcx: @mut Block, t: ty::t, name: &str) -> ValueRef {
    let _icx = push_ctxt("alloc_ty");
    let ccx = bcx.ccx();
    let ty = type_of::type_of(ccx, t);
    assert!(!ty::type_has_params(t));
    let val = alloca(bcx, ty, name);
    return val;
}

pub fn alloca(cx: @mut Block, ty: Type, name: &str) -> ValueRef {
    alloca_maybe_zeroed(cx, ty, name, false)
}

pub fn alloca_maybe_zeroed(cx: @mut Block, ty: Type, name: &str, zero: bool) -> ValueRef {
    let _icx = push_ctxt("alloca");
    if cx.unreachable {
        unsafe {
            return llvm::LLVMGetUndef(ty.ptr_to().to_ref());
        }
    }
    let p = Alloca(cx, ty, name);
    if zero {
        let b = cx.fcx.ccx.builder();
        b.position_before(cx.fcx.alloca_insert_pt.unwrap());
        memzero(&b, p, ty);
    }
    p
}

pub fn arrayalloca(cx: @mut Block, ty: Type, v: ValueRef) -> ValueRef {
    let _icx = push_ctxt("arrayalloca");
    if cx.unreachable {
        unsafe {
            return llvm::LLVMGetUndef(ty.to_ref());
        }
    }
    return ArrayAlloca(cx, ty, v);
}

pub struct BasicBlocks {
    sa: BasicBlockRef,
}

pub fn mk_staticallocas_basic_block(llfn: ValueRef) -> BasicBlockRef {
    unsafe {
        let cx = task_llcx();
        do "static_allocas".with_c_str | buf| {
            llvm::LLVMAppendBasicBlockInContext(cx, llfn, buf)
        }
    }
}

pub fn mk_return_basic_block(llfn: ValueRef) -> BasicBlockRef {
    unsafe {
        let cx = task_llcx();
        do "return".with_c_str |buf| {
            llvm::LLVMAppendBasicBlockInContext(cx, llfn, buf)
        }
    }
}

// Creates and returns space for, or returns the argument representing, the
// slot where the return value of the function must go.
pub fn make_return_pointer(fcx: @mut FunctionContext, output_type: ty::t) -> ValueRef {
    unsafe {
        if type_of::return_uses_outptr(fcx.ccx, output_type) {
            llvm::LLVMGetParam(fcx.llfn, 0)
        } else {
            let lloutputtype = type_of::type_of(fcx.ccx, output_type);
            let bcx = fcx.entry_bcx.unwrap();
            Alloca(bcx, lloutputtype, "__make_return_pointer")
        }
    }
}

// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args
pub fn new_fn_ctxt_w_id(ccx: @mut CrateContext,
                        path: path,
                        llfndecl: ValueRef,
                        id: ast::NodeId,
                        output_type: ty::t,
                        skip_retptr: bool,
                        param_substs: Option<@param_substs>,
                        opt_node_info: Option<NodeInfo>,
                        sp: Option<Span>)
                     -> @mut FunctionContext {
    for p in param_substs.iter() { p.validate(); }

    debug!("new_fn_ctxt_w_id(path={}, id={:?}, \
            param_substs={})",
           path_str(ccx.sess, path),
           id,
           param_substs.repr(ccx.tcx));

    let substd_output_type = match param_substs {
        None => output_type,
        Some(substs) => {
            ty::subst_tps(ccx.tcx, substs.tys, substs.self_ty, output_type)
        }
    };
    let uses_outptr = type_of::return_uses_outptr(ccx, substd_output_type);
    let debug_context = debuginfo::create_function_debug_context(ccx, id, param_substs, llfndecl);

    let fcx = @mut FunctionContext {
          llfn: llfndecl,
          llenv: unsafe {
              llvm::LLVMGetUndef(Type::i8p().to_ref())
          },
          llretptr: None,
          entry_bcx: None,
          alloca_insert_pt: None,
          llreturn: None,
          llself: None,
          personality: None,
          caller_expects_out_pointer: uses_outptr,
          llargs: @mut HashMap::new(),
          lllocals: @mut HashMap::new(),
          llupvars: @mut HashMap::new(),
          id: id,
          param_substs: param_substs,
          span: sp,
          path: path,
          ccx: ccx,
          debug_context: debug_context,
    };
    fcx.llenv = unsafe {
          llvm::LLVMGetParam(llfndecl, fcx.env_arg_pos() as c_uint)
    };

    unsafe {
        let entry_bcx = top_scope_block(fcx, opt_node_info);
        Load(entry_bcx, C_null(Type::i8p()));

        fcx.entry_bcx = Some(entry_bcx);
        fcx.alloca_insert_pt = Some(llvm::LLVMGetFirstInstruction(entry_bcx.llbb));
    }

    if !ty::type_is_voidish(ccx.tcx, substd_output_type) {
        // If the function returns nil/bot, there is no real return
        // value, so do not set `llretptr`.
        if !skip_retptr || uses_outptr {
            // Otherwise, we normally allocate the llretptr, unless we
            // have been instructed to skip it for immediate return
            // values.
            fcx.llretptr = Some(make_return_pointer(fcx, substd_output_type));
        }
    }
    fcx
}

pub fn new_fn_ctxt(ccx: @mut CrateContext,
                   path: path,
                   llfndecl: ValueRef,
                   output_type: ty::t,
                   sp: Option<Span>)
                -> @mut FunctionContext {
    new_fn_ctxt_w_id(ccx, path, llfndecl, -1, output_type, false, None, None, sp)
}

// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_llargs_for_fn_args.
//  - new_fn_ctxt
//  - trans_args

// create_llargs_for_fn_args: Creates a mapping from incoming arguments to
// allocas created for them.
//
// When we translate a function, we need to map its incoming arguments to the
// spaces that have been created for them (by code in the llallocas field of
// the function's fn_ctxt).  create_llargs_for_fn_args populates the llargs
// field of the fn_ctxt with
pub fn create_llargs_for_fn_args(cx: @mut FunctionContext,
                                 self_arg: self_arg,
                                 args: &[ast::arg])
                              -> ~[ValueRef] {
    let _icx = push_ctxt("create_llargs_for_fn_args");

    match self_arg {
      impl_self(tt, self_mode) => {
        cx.llself = Some(ValSelfData {
            v: cx.llenv,
            t: tt,
            is_copy: self_mode == ty::ByCopy
        });
      }
      no_self => ()
    }

    // Return an array containing the ValueRefs that we get from
    // llvm::LLVMGetParam for each argument.
    do vec::from_fn(args.len()) |i| {
        unsafe { llvm::LLVMGetParam(cx.llfn, cx.arg_pos(i) as c_uint) }
    }
}

pub fn copy_args_to_allocas(fcx: @mut FunctionContext,
                            bcx: @mut Block,
                            args: &[ast::arg],
                            raw_llargs: &[ValueRef],
                            arg_tys: &[ty::t]) -> @mut Block {
    debug!("copy_args_to_allocas: raw_llargs={} arg_tys={}",
           raw_llargs.llrepr(fcx.ccx),
           arg_tys.repr(fcx.ccx.tcx));

    let _icx = push_ctxt("copy_args_to_allocas");
    let mut bcx = bcx;

    match fcx.llself {
        Some(slf) => {
            let self_val = if slf.is_copy
                    && datum::appropriate_mode(bcx.ccx(), slf.t).is_by_value() {
                let tmp = BitCast(bcx, slf.v, type_of(bcx.ccx(), slf.t));
                let alloc = alloc_ty(bcx, slf.t, "__self");
                Store(bcx, tmp, alloc);
                alloc
            } else {
                PointerCast(bcx, slf.v, type_of(bcx.ccx(), slf.t).ptr_to())
            };

            fcx.llself = Some(ValSelfData {v: self_val, ..slf});
            add_clean(bcx, self_val, slf.t);

            if fcx.ccx.sess.opts.extra_debuginfo {
                debuginfo::create_self_argument_metadata(bcx, slf.t, self_val);
            }
        }
        _ => {}
    }

    for (arg_n, &arg_ty) in arg_tys.iter().enumerate() {
        let raw_llarg = raw_llargs[arg_n];

        // For certain mode/type combinations, the raw llarg values are passed
        // by value.  However, within the fn body itself, we want to always
        // have all locals and arguments be by-ref so that we can cancel the
        // cleanup and for better interaction with LLVM's debug info.  So, if
        // the argument would be passed by value, we store it into an alloca.
        // This alloca should be optimized away by LLVM's mem-to-reg pass in
        // the event it's not truly needed.
        // only by value if immediate:
        let llarg = if datum::appropriate_mode(bcx.ccx(), arg_ty).is_by_value() {
            let alloc = alloc_ty(bcx, arg_ty, "__arg");
            Store(bcx, raw_llarg, alloc);
            alloc
        } else {
            raw_llarg
        };
        bcx = _match::store_arg(bcx, args[arg_n].pat, llarg);

        if fcx.ccx.sess.opts.extra_debuginfo {
            debuginfo::create_argument_metadata(bcx, &args[arg_n]);
        }
    }

    return bcx;
}

// Ties up the llstaticallocas -> llloadenv -> lltop edges,
// and builds the return block.
pub fn finish_fn(fcx: @mut FunctionContext, last_bcx: @mut Block) {
    let _icx = push_ctxt("finish_fn");

    let ret_cx = match fcx.llreturn {
        Some(llreturn) => {
            if !last_bcx.terminated {
                Br(last_bcx, llreturn);
            }
            raw_block(fcx, false, llreturn)
        }
        None => last_bcx
    };
    build_return_block(fcx, ret_cx);
    fcx.cleanup();
}

// Builds the return block for a function.
pub fn build_return_block(fcx: &FunctionContext, ret_cx: @mut Block) {
    // Return the value if this function immediate; otherwise, return void.
    if fcx.llretptr.is_none() || fcx.caller_expects_out_pointer {
        return RetVoid(ret_cx);
    }

    let retptr = Value(fcx.llretptr.unwrap());
    let retval = match retptr.get_dominating_store(ret_cx) {
        // If there's only a single store to the ret slot, we can directly return
        // the value that was stored and omit the store and the alloca
        Some(s) => {
            let retval = *s.get_operand(0).unwrap();
            s.erase_from_parent();

            if retptr.has_no_uses() {
                retptr.erase_from_parent();
            }

            retval
        }
        // Otherwise, load the return value from the ret slot
        None => Load(ret_cx, fcx.llretptr.unwrap())
    };


    Ret(ret_cx, retval);
}

pub enum self_arg { impl_self(ty::t, ty::SelfMode), no_self, }

// trans_closure: Builds an LLVM function out of a source function.
// If the function closes over its environment a closure will be
// returned.
pub fn trans_closure(ccx: @mut CrateContext,
                     path: path,
                     decl: &ast::fn_decl,
                     body: &ast::Block,
                     llfndecl: ValueRef,
                     self_arg: self_arg,
                     param_substs: Option<@param_substs>,
                     id: ast::NodeId,
                     attributes: &[ast::Attribute],
                     output_type: ty::t,
                     maybe_load_env: &fn(@mut FunctionContext)) {
    ccx.stats.n_closures += 1;
    let _icx = push_ctxt("trans_closure");
    set_uwtable(llfndecl);

    debug!("trans_closure(..., param_substs={})",
           param_substs.repr(ccx.tcx));

    let fcx = new_fn_ctxt_w_id(ccx,
                               path,
                               llfndecl,
                               id,
                               output_type,
                               false,
                               param_substs,
                               body.info(),
                               Some(body.span));

    // Create the first basic block in the function and keep a handle on it to
    //  pass to finish_fn later.
    let bcx_top = fcx.entry_bcx.unwrap();
    let mut bcx = bcx_top;
    let block_ty = node_id_type(bcx, body.id);

    // Set up arguments to the function.
    let arg_tys = ty::ty_fn_args(node_id_type(bcx, id));
    let raw_llargs = create_llargs_for_fn_args(fcx, self_arg, decl.inputs);

    // Set the fixed stack segment flag if necessary.
    if attr::contains_name(attributes, "fixed_stack_segment") {
        set_no_inline(fcx.llfn);
        set_fixed_stack_segment(fcx.llfn);
    }

    bcx = copy_args_to_allocas(fcx, bcx, decl.inputs, raw_llargs, arg_tys);

    maybe_load_env(fcx);

    // Up until here, IR instructions for this function have explicitly not been annotated with
    // source code location, so we don't step into call setup code. From here on, source location
    // emitting should be enabled.
    debuginfo::start_emitting_source_locations(fcx);

    // This call to trans_block is the place where we bridge between
    // translation calls that don't have a return value (trans_crate,
    // trans_mod, trans_item, et cetera) and those that do
    // (trans_block, trans_expr, et cetera).
    if body.expr.is_none() || ty::type_is_voidish(bcx.tcx(), block_ty) {
        bcx = controlflow::trans_block(bcx, body, expr::Ignore);
    } else {
        let dest = expr::SaveIn(fcx.llretptr.unwrap());
        bcx = controlflow::trans_block(bcx, body, dest);
    }

    match fcx.llreturn {
        Some(llreturn) => cleanup_and_Br(bcx, bcx_top, llreturn),
        None => bcx = cleanup_block(bcx, Some(bcx_top.llbb))
    };

    // Put return block after all other blocks.
    // This somewhat improves single-stepping experience in debugger.
    unsafe {
        for &llreturn in fcx.llreturn.iter() {
            llvm::LLVMMoveBasicBlockAfter(llreturn, bcx.llbb);
        }
    }

    // Insert the mandatory first few basic blocks before lltop.
    finish_fn(fcx, bcx);
}

// trans_fn: creates an LLVM function corresponding to a source language
// function.
pub fn trans_fn(ccx: @mut CrateContext,
                path: path,
                decl: &ast::fn_decl,
                body: &ast::Block,
                llfndecl: ValueRef,
                self_arg: self_arg,
                param_substs: Option<@param_substs>,
                id: ast::NodeId,
                attrs: &[ast::Attribute]) {

    let the_path_str = path_str(ccx.sess, path);
    let _s = StatRecorder::new(ccx, the_path_str);
    debug!("trans_fn(self_arg={:?}, param_substs={})",
           self_arg,
           param_substs.repr(ccx.tcx));
    let _icx = push_ctxt("trans_fn");
    let output_type = ty::ty_fn_ret(ty::node_id_to_type(ccx.tcx, id));
    trans_closure(ccx,
                  path.clone(),
                  decl,
                  body,
                  llfndecl,
                  self_arg,
                  param_substs,
                  id,
                  attrs,
                  output_type,
                  |_fcx| { });
}

fn insert_synthetic_type_entries(bcx: @mut Block,
                                 fn_args: &[ast::arg],
                                 arg_tys: &[ty::t])
{
    /*!
     * For tuple-like structs and enum-variants, we generate
     * synthetic AST nodes for the arguments.  These have no types
     * in the type table and no entries in the moves table,
     * so the code in `copy_args_to_allocas` and `bind_irrefutable_pat`
     * gets upset. This hack of a function bridges the gap by inserting types.
     *
     * This feels horrible. I think we should just have a special path
     * for these functions and not try to use the generic code, but
     * that's not the problem I'm trying to solve right now. - nmatsakis
     */

    let tcx = bcx.tcx();
    for i in range(0u, fn_args.len()) {
        debug!("setting type of argument {} (pat node {}) to {}",
               i, fn_args[i].pat.id, bcx.ty_to_str(arg_tys[i]));

        let pat_id = fn_args[i].pat.id;
        let arg_ty = arg_tys[i];
        tcx.node_types.insert(pat_id as uint, arg_ty);
    }
}

pub fn trans_enum_variant(ccx: @mut CrateContext,
                          _enum_id: ast::NodeId,
                          variant: &ast::variant,
                          args: &[ast::variant_arg],
                          disr: ty::Disr,
                          param_substs: Option<@param_substs>,
                          llfndecl: ValueRef) {
    let _icx = push_ctxt("trans_enum_variant");

    trans_enum_variant_or_tuple_like_struct(
        ccx,
        variant.node.id,
        args,
        disr,
        param_substs,
        llfndecl);
}

pub fn trans_tuple_struct(ccx: @mut CrateContext,
                          fields: &[@ast::struct_field],
                          ctor_id: ast::NodeId,
                          param_substs: Option<@param_substs>,
                          llfndecl: ValueRef) {
    let _icx = push_ctxt("trans_tuple_struct");

    trans_enum_variant_or_tuple_like_struct(
        ccx,
        ctor_id,
        fields,
        0,
        param_substs,
        llfndecl);
}

trait IdAndTy {
    fn id(&self) -> ast::NodeId;
    fn ty<'a>(&'a self) -> &'a ast::Ty;
}

impl IdAndTy for ast::variant_arg {
    fn id(&self) -> ast::NodeId { self.id }
    fn ty<'a>(&'a self) -> &'a ast::Ty { &self.ty }
}

impl IdAndTy for @ast::struct_field {
    fn id(&self) -> ast::NodeId { self.node.id }
    fn ty<'a>(&'a self) -> &'a ast::Ty { &self.node.ty }
}

pub fn trans_enum_variant_or_tuple_like_struct<A:IdAndTy>(
    ccx: @mut CrateContext,
    ctor_id: ast::NodeId,
    args: &[A],
    disr: ty::Disr,
    param_substs: Option<@param_substs>,
    llfndecl: ValueRef)
{
    // Translate variant arguments to function arguments.
    let fn_args = do args.map |varg| {
        ast::arg {
            is_mutbl: false,
            ty: (*varg.ty()).clone(),
            pat: ast_util::ident_to_pat(
                ccx.tcx.sess.next_node_id(),
                codemap::dummy_sp(),
                special_idents::arg),
            id: varg.id(),
        }
    };

    let no_substs: &[ty::t] = [];
    let ty_param_substs = match param_substs {
        Some(ref substs) => {
            let v: &[ty::t] = substs.tys;
            v
        }
        None => {
            let v: &[ty::t] = no_substs;
            v
        }
    };

    let ctor_ty = ty::subst_tps(ccx.tcx,
                                ty_param_substs,
                                None,
                                ty::node_id_to_type(ccx.tcx, ctor_id));

    let result_ty = match ty::get(ctor_ty).sty {
        ty::ty_bare_fn(ref bft) => bft.sig.output,
        _ => ccx.sess.bug(
            format!("trans_enum_variant_or_tuple_like_struct: \
                  unexpected ctor return type {}",
                 ty_to_str(ccx.tcx, ctor_ty)))
    };

    let fcx = new_fn_ctxt_w_id(ccx,
                               ~[],
                               llfndecl,
                               ctor_id,
                               result_ty,
                               false,
                               param_substs,
                               None,
                               None);

    let arg_tys = ty::ty_fn_args(ctor_ty);

    let raw_llargs = create_llargs_for_fn_args(fcx, no_self, fn_args);

    let bcx = fcx.entry_bcx.unwrap();

    insert_synthetic_type_entries(bcx, fn_args, arg_tys);
    let bcx = copy_args_to_allocas(fcx, bcx, fn_args, raw_llargs, arg_tys);

    let repr = adt::represent_type(ccx, result_ty);
    adt::trans_start_init(bcx, repr, fcx.llretptr.unwrap(), disr);
    for (i, fn_arg) in fn_args.iter().enumerate() {
        let lldestptr = adt::trans_field_ptr(bcx,
                                             repr,
                                             fcx.llretptr.unwrap(),
                                             disr,
                                             i);
        let llarg = fcx.llargs.get_copy(&fn_arg.pat.id);
        let arg_ty = arg_tys[i];
        memcpy_ty(bcx, lldestptr, llarg, arg_ty);
    }
    finish_fn(fcx, bcx);
}

pub fn trans_enum_def(ccx: @mut CrateContext, enum_definition: &ast::enum_def,
                      id: ast::NodeId, vi: @~[@ty::VariantInfo],
                      i: &mut uint) {
    for variant in enum_definition.variants.iter() {
        let disr_val = vi[*i].disr_val;
        *i += 1;

        match variant.node.kind {
            ast::tuple_variant_kind(ref args) if args.len() > 0 => {
                let llfn = get_item_val(ccx, variant.node.id);
                trans_enum_variant(ccx, id, variant, *args,
                                   disr_val, None, llfn);
            }
            ast::tuple_variant_kind(_) => {
                // Nothing to do.
            }
            ast::struct_variant_kind(struct_def) => {
                trans_struct_def(ccx, struct_def);
            }
        }
    }
}

pub struct TransItemVisitor {
    ccx: @mut CrateContext,
}

impl Visitor<()> for TransItemVisitor {
    fn visit_item(&mut self, i: @ast::item, _:()) {
        trans_item(self.ccx, i);
    }
}

pub fn trans_item(ccx: @mut CrateContext, item: &ast::item) {
    let _icx = push_ctxt("trans_item");
    let path = match ccx.tcx.items.get_copy(&item.id) {
        ast_map::node_item(_, p) => p,
        // tjc: ?
        _ => fail!("trans_item"),
    };
    match item.node {
      ast::item_fn(ref decl, purity, _abis, ref generics, ref body) => {
        if purity == ast::extern_fn  {
            let llfndecl = get_item_val(ccx, item.id);
            foreign::trans_rust_fn_with_foreign_abi(
                ccx,
                &vec::append((*path).clone(),
                             [path_name(item.ident)]),
                decl,
                body,
                item.attrs,
                llfndecl,
                item.id);
        } else if !generics.is_type_parameterized() {
            let llfndecl = get_item_val(ccx, item.id);
            trans_fn(ccx,
                     vec::append((*path).clone(), [path_name(item.ident)]),
                     decl,
                     body,
                     llfndecl,
                     no_self,
                     None,
                     item.id,
                     item.attrs);
        } else {
            // Be sure to travel more than just one layer deep to catch nested
            // items in blocks and such.
            let mut v = TransItemVisitor{ ccx: ccx };
            v.visit_block(body, ());
        }
      }
      ast::item_impl(ref generics, _, _, ref ms) => {
        meth::trans_impl(ccx,
                         (*path).clone(),
                         item.ident,
                         *ms,
                         generics,
                         item.id);
      }
      ast::item_mod(ref m) => {
        trans_mod(ccx, m);
      }
      ast::item_enum(ref enum_definition, ref generics) => {
        if !generics.is_type_parameterized() {
            let vi = ty::enum_variants(ccx.tcx, local_def(item.id));
            let mut i = 0;
            trans_enum_def(ccx, enum_definition, item.id, vi, &mut i);
        }
      }
      ast::item_static(_, m, expr) => {
          consts::trans_const(ccx, m, item.id);
          // Do static_assert checking. It can't really be done much earlier
          // because we need to get the value of the bool out of LLVM
          if attr::contains_name(item.attrs, "static_assert") {
              if m == ast::MutMutable {
                  ccx.sess.span_fatal(expr.span,
                                      "cannot have static_assert on a mutable \
                                       static");
              }
              let v = ccx.const_values.get_copy(&item.id);
              unsafe {
                  if !(llvm::LLVMConstIntGetZExtValue(v) != 0) {
                      ccx.sess.span_fatal(expr.span, "static assertion failed");
                  }
              }
          }
      },
      ast::item_foreign_mod(ref foreign_mod) => {
        foreign::trans_foreign_mod(ccx, foreign_mod);
      }
      ast::item_struct(struct_def, ref generics) => {
        if !generics.is_type_parameterized() {
            trans_struct_def(ccx, struct_def);
        }
      }
      ast::item_trait(*) => {
        // Inside of this trait definition, we won't be actually translating any
        // functions, but the trait still needs to be walked. Otherwise default
        // methods with items will not get translated and will cause ICE's when
        // metadata time comes around.
        let mut v = TransItemVisitor{ ccx: ccx };
        visit::walk_item(&mut v, item, ());
      }
      _ => {/* fall through */ }
    }
}

pub fn trans_struct_def(ccx: @mut CrateContext, struct_def: @ast::struct_def) {
    // If this is a tuple-like struct, translate the constructor.
    match struct_def.ctor_id {
        // We only need to translate a constructor if there are fields;
        // otherwise this is a unit-like struct.
        Some(ctor_id) if struct_def.fields.len() > 0 => {
            let llfndecl = get_item_val(ccx, ctor_id);
            trans_tuple_struct(ccx, struct_def.fields,
                               ctor_id, None, llfndecl);
        }
        Some(_) | None => {}
    }
}

// Translate a module. Doing this amounts to translating the items in the
// module; there ends up being no artifact (aside from linkage names) of
// separate modules in the compiled program.  That's because modules exist
// only as a convenience for humans working with the code, to organize names
// and control visibility.
pub fn trans_mod(ccx: @mut CrateContext, m: &ast::_mod) {
    let _icx = push_ctxt("trans_mod");
    for item in m.items.iter() {
        trans_item(ccx, *item);
    }
}

fn finish_register_fn(ccx: @mut CrateContext, sp: Span, sym: ~str, node_id: ast::NodeId,
                      llfn: ValueRef) {
    ccx.item_symbols.insert(node_id, sym);

    if !*ccx.sess.building_library {
        lib::llvm::SetLinkage(llfn, lib::llvm::InternalLinkage);
    }

    // FIXME #4404 android JNI hacks
    let is_entry = is_entry_fn(&ccx.sess, node_id) && (!*ccx.sess.building_library ||
                      (*ccx.sess.building_library &&
                       ccx.sess.targ_cfg.os == session::OsAndroid));
    if is_entry {
        create_entry_wrapper(ccx, sp, llfn);
    }
}

pub fn register_fn(ccx: @mut CrateContext,
                   sp: Span,
                   sym: ~str,
                   node_id: ast::NodeId,
                   node_type: ty::t)
                   -> ValueRef {
    let f = match ty::get(node_type).sty {
        ty::ty_bare_fn(ref f) => {
            assert!(f.abis.is_rust() || f.abis.is_intrinsic());
            f
        }
        _ => fail!("expected bare rust fn or an intrinsic")
    };

    let llfn = decl_rust_fn(ccx, f.sig.inputs, f.sig.output, sym);
    finish_register_fn(ccx, sp, sym, node_id, llfn);
    llfn
}

// only use this for foreign function ABIs and glue, use `register_fn` for Rust functions
pub fn register_fn_llvmty(ccx: @mut CrateContext,
                          sp: Span,
                          sym: ~str,
                          node_id: ast::NodeId,
                          cc: lib::llvm::CallConv,
                          fn_ty: Type)
                          -> ValueRef {
    debug!("register_fn_fuller creating fn for item {} with path {}",
           node_id,
           ast_map::path_to_str(item_path(ccx, &node_id), token::get_ident_interner()));

    let llfn = decl_fn(ccx.llmod, sym, cc, fn_ty);
    finish_register_fn(ccx, sp, sym, node_id, llfn);
    llfn
}

pub fn is_entry_fn(sess: &Session, node_id: ast::NodeId) -> bool {
    match *sess.entry_fn {
        Some((entry_id, _)) => node_id == entry_id,
        None => false
    }
}

// Create a _rust_main(args: ~[str]) function which will be called from the
// runtime rust_start function
pub fn create_entry_wrapper(ccx: @mut CrateContext,
                           _sp: Span,
                           main_llfn: ValueRef) {
    let et = ccx.sess.entry_type.unwrap();
    match et {
        session::EntryMain => {
            create_entry_fn(ccx, main_llfn, true);
        }
        session::EntryStart => create_entry_fn(ccx, main_llfn, false),
        session::EntryNone => {}    // Do nothing.
    }

    fn create_entry_fn(ccx: @mut CrateContext,
                       rust_main: ValueRef,
                       use_start_lang_item: bool) {
        let llfty = Type::func([ccx.int_type, Type::i8().ptr_to().ptr_to()],
                               &ccx.int_type);

        // FIXME #4404 android JNI hacks
        let main_name = if *ccx.sess.building_library {
            "amain"
        } else {
            "main"
        };
        let llfn = decl_cdecl_fn(ccx.llmod, main_name, llfty);
        let llbb = do "top".with_c_str |buf| {
            unsafe {
                llvm::LLVMAppendBasicBlockInContext(ccx.llcx, llfn, buf)
            }
        };
        let bld = ccx.builder.B;
        unsafe {
            llvm::LLVMPositionBuilderAtEnd(bld, llbb);

            let (start_fn, args) = if use_start_lang_item {
                let start_def_id = match ccx.tcx.lang_items.require(StartFnLangItem) {
                    Ok(id) => id,
                    Err(s) => { ccx.tcx.sess.fatal(s); }
                };
                let start_fn = if start_def_id.crate == ast::LOCAL_CRATE {
                    get_item_val(ccx, start_def_id.node)
                } else {
                    let start_fn_type = csearch::get_type(ccx.tcx,
                                                          start_def_id).ty;
                    trans_external_path(ccx, start_def_id, start_fn_type)
                };

                let args = {
                    let opaque_rust_main = do "rust_main".with_c_str |buf| {
                        llvm::LLVMBuildPointerCast(bld, rust_main, Type::i8p().to_ref(), buf)
                    };

                    ~[
                        C_null(Type::opaque_box(ccx).ptr_to()),
                        opaque_rust_main,
                        llvm::LLVMGetParam(llfn, 0),
                        llvm::LLVMGetParam(llfn, 1)
                     ]
                };
                (start_fn, args)
            } else {
                debug!("using user-defined start fn");
                let args = ~[
                    C_null(Type::opaque_box(ccx).ptr_to()),
                    llvm::LLVMGetParam(llfn, 0 as c_uint),
                    llvm::LLVMGetParam(llfn, 1 as c_uint)
                ];

                (rust_main, args)
            };

            let result = do args.as_imm_buf |buf, len| {
                llvm::LLVMBuildCall(bld, start_fn, buf, len as c_uint, noname())
            };

            llvm::LLVMBuildRet(bld, result);
        }
    }
}

pub fn fill_fn_pair(bcx: @mut Block, pair: ValueRef, llfn: ValueRef,
                    llenvptr: ValueRef) {
    let ccx = bcx.ccx();
    let code_cell = GEPi(bcx, pair, [0u, abi::fn_field_code]);
    Store(bcx, llfn, code_cell);
    let env_cell = GEPi(bcx, pair, [0u, abi::fn_field_box]);
    let llenvblobptr = PointerCast(bcx, llenvptr, Type::opaque_box(ccx).ptr_to());
    Store(bcx, llenvblobptr, env_cell);
}

pub fn item_path(ccx: &CrateContext, id: &ast::NodeId) -> path {
    ty::item_path(ccx.tcx, ast_util::local_def(*id))
}

fn exported_name(ccx: &mut CrateContext, path: path, ty: ty::t, attrs: &[ast::Attribute]) -> ~str {
    match attr::first_attr_value_str_by_name(attrs, "export_name") {
        // Use provided name
        Some(name) => name.to_owned(),

        // Don't mangle
        _ if attr::contains_name(attrs, "no_mangle")
            => path_elt_to_str(*path.last(), token::get_ident_interner()),

        // Usual name mangling
        _ => mangle_exported_name(ccx, path, ty)
    }
}

pub fn get_item_val(ccx: @mut CrateContext, id: ast::NodeId) -> ValueRef {
    debug!("get_item_val(id=`{:?}`)", id);

    let val = ccx.item_vals.find_copy(&id);
    match val {
        Some(v) => v,
        None => {
            let mut exprt = false;
            let item = ccx.tcx.items.get_copy(&id);
            let val = match item {
                ast_map::node_item(i, pth) => {

                    let elt = path_pretty_name(i.ident, id as u64);
                    let my_path = vec::append_one((*pth).clone(), elt);
                    let ty = ty::node_id_to_type(ccx.tcx, i.id);
                    let sym = exported_name(ccx, my_path, ty, i.attrs);

                    let v = match i.node {
                        ast::item_static(_, _, expr) => {
                            // If this static came from an external crate, then
                            // we need to get the symbol from csearch instead of
                            // using the current crate's name/version
                            // information in the hash of the symbol
                            debug!("making {}", sym);
                            let sym = match ccx.external_srcs.find(&i.id) {
                                Some(&did) => {
                                    debug!("but found in other crate...");
                                    csearch::get_symbol(ccx.sess.cstore, did)
                                }
                                None => sym
                            };

                            // We need the translated value here, because for enums the
                            // LLVM type is not fully determined by the Rust type.
                            let (v, inlineable) = consts::const_expr(ccx, expr);
                            ccx.const_values.insert(id, v);
                            let mut inlineable = inlineable;

                            unsafe {
                                let llty = llvm::LLVMTypeOf(v);
                                let g = do sym.with_c_str |buf| {
                                    llvm::LLVMAddGlobal(ccx.llmod, llty, buf)
                                };

                                if !*ccx.sess.building_library {
                                    lib::llvm::SetLinkage(g, lib::llvm::InternalLinkage);
                                }

                                // Apply the `unnamed_addr` attribute if
                                // requested
                                if attr::contains_name(i.attrs,
                                                       "address_insignificant"){
                                    lib::llvm::SetUnnamedAddr(g, true);
                                    lib::llvm::SetLinkage(g,
                                        lib::llvm::InternalLinkage);

                                    // This is a curious case where we must make
                                    // all of these statics inlineable. If a
                                    // global is tagged as
                                    // address_insignificant, then LLVM won't
                                    // coalesce globals unless they have an
                                    // internal linkage type. This means that
                                    // external crates cannot use this global.
                                    // This is a problem for things like inner
                                    // statics in generic functions, because the
                                    // function will be inlined into another
                                    // crate and then attempt to link to the
                                    // static in the original crate, only to
                                    // find that it's not there. On the other
                                    // side of inlininig, the crates knows to
                                    // not declare this static as
                                    // available_externally (because it isn't)
                                    inlineable = true;
                                }

                                if !inlineable {
                                    debug!("{} not inlined", sym);
                                    ccx.non_inlineable_statics.insert(id);
                                }
                                ccx.item_symbols.insert(i.id, sym);
                                g
                            }
                        }

                        ast::item_fn(_, purity, _, _, _) => {
                            let llfn = if purity != ast::extern_fn {
                                register_fn(ccx, i.span, sym, i.id, ty)
                            } else {
                                foreign::register_rust_fn_with_foreign_abi(ccx,
                                                                           i.span,
                                                                           sym,
                                                                           i.id)
                            };
                            set_llvm_fn_attrs(i.attrs, llfn);
                            llfn
                        }

                        _ => fail!("get_item_val: weird result in table")
                    };

                    match (attr::first_attr_value_str_by_name(i.attrs, "link_section")) {
                        Some(sect) => unsafe {
                            do sect.with_c_str |buf| {
                                llvm::LLVMSetSection(v, buf);
                            }
                        },
                        None => ()
                    }

                    v
                }

                ast_map::node_trait_method(trait_method, _, pth) => {
                    debug!("get_item_val(): processing a node_trait_method");
                    match *trait_method {
                        ast::required(_) => {
                            ccx.sess.bug("unexpected variant: required trait method in \
                                         get_item_val()");
                        }
                        ast::provided(m) => {
                            exprt = true;
                            register_method(ccx, id, pth, m)
                        }
                    }
                }

                ast_map::node_method(m, _, pth) => {
                    register_method(ccx, id, pth, m)
                }

                ast_map::node_foreign_item(ni, abis, _, pth) => {
                    let ty = ty::node_id_to_type(ccx.tcx, ni.id);
                    exprt = true;

                    match ni.node {
                        ast::foreign_item_fn(*) => {
                            let path = vec::append((*pth).clone(), [path_name(ni.ident)]);
                            foreign::register_foreign_item_fn(ccx, abis, &path, ni)
                        }
                        ast::foreign_item_static(*) => {
                            let ident = foreign::link_name(ccx, ni);
                            unsafe {
                                let g = do ident.with_c_str |buf| {
                                    let ty = type_of(ccx, ty);
                                    llvm::LLVMAddGlobal(ccx.llmod, ty.to_ref(), buf)
                                };
                                if attr::contains_name(ni.attrs, "weak_linkage") {
                                    lib::llvm::SetLinkage(g, lib::llvm::ExternalWeakLinkage);
                                }
                                g
                            }
                        }
                    }
                }

                ast_map::node_variant(ref v, enm, pth) => {
                    let llfn;
                    match v.node.kind {
                        ast::tuple_variant_kind(ref args) => {
                            assert!(args.len() != 0u);
                            let pth = vec::append((*pth).clone(),
                                                  [path_name(enm.ident),
                                                   path_name((*v).node.name)]);
                            let ty = ty::node_id_to_type(ccx.tcx, id);
                            let sym = exported_name(ccx, pth, ty, enm.attrs);

                            llfn = match enm.node {
                                ast::item_enum(_, _) => {
                                    register_fn(ccx, (*v).span, sym, id, ty)
                                }
                                _ => fail!("node_variant, shouldn't happen")
                            };
                        }
                        ast::struct_variant_kind(_) => {
                            fail!("struct variant kind unexpected in get_item_val")
                        }
                    }
                    set_inline_hint(llfn);
                    llfn
                }

                ast_map::node_struct_ctor(struct_def, struct_item, struct_path) => {
                    // Only register the constructor if this is a tuple-like struct.
                    match struct_def.ctor_id {
                        None => {
                            ccx.tcx.sess.bug("attempt to register a constructor of \
                                              a non-tuple-like struct")
                        }
                        Some(ctor_id) => {
                            let ty = ty::node_id_to_type(ccx.tcx, ctor_id);
                            let sym = exported_name(ccx, (*struct_path).clone(), ty,
                                                    struct_item.attrs);
                            let llfn = register_fn(ccx, struct_item.span,
                                                   sym, ctor_id, ty);
                            set_inline_hint(llfn);
                            llfn
                        }
                    }
                }

                ref variant => {
                    ccx.sess.bug(format!("get_item_val(): unexpected variant: {:?}",
                                 variant))
                }
            };

            if !exprt && !ccx.reachable.contains(&id) {
                lib::llvm::SetLinkage(val, lib::llvm::InternalLinkage);
            }

            ccx.item_vals.insert(id, val);
            val
        }
    }
}

pub fn register_method(ccx: @mut CrateContext,
                       id: ast::NodeId,
                       path: @ast_map::path,
                       m: @ast::method) -> ValueRef {
    let mty = ty::node_id_to_type(ccx.tcx, id);

    let mut path = (*path).clone();
    path.push(path_pretty_name(m.ident, token::gensym("meth") as u64));

    let sym = exported_name(ccx, path, mty, m.attrs);

    let llfn = register_fn(ccx, m.span, sym, id, mty);
    set_llvm_fn_attrs(m.attrs, llfn);
    llfn
}

pub fn vp2i(cx: @mut Block, v: ValueRef) -> ValueRef {
    let ccx = cx.ccx();
    return PtrToInt(cx, v, ccx.int_type);
}

pub fn p2i(ccx: &CrateContext, v: ValueRef) -> ValueRef {
    unsafe {
        return llvm::LLVMConstPtrToInt(v, ccx.int_type.to_ref());
    }
}

macro_rules! ifn (
    ($intrinsics:ident, $name:expr, $args:expr, $ret:expr) => ({
        let name = $name;
        let f = decl_cdecl_fn(llmod, name, Type::func($args, &$ret));
        $intrinsics.insert(name, f);
    })
)

pub fn declare_intrinsics(llmod: ModuleRef) -> HashMap<&'static str, ValueRef> {
    let i8p = Type::i8p();
    let mut intrinsics = HashMap::new();

    ifn!(intrinsics, "llvm.memcpy.p0i8.p0i8.i32",
         [i8p, i8p, Type::i32(), Type::i32(), Type::i1()], Type::void());
    ifn!(intrinsics, "llvm.memcpy.p0i8.p0i8.i64",
         [i8p, i8p, Type::i64(), Type::i32(), Type::i1()], Type::void());
    ifn!(intrinsics, "llvm.memmove.p0i8.p0i8.i32",
         [i8p, i8p, Type::i32(), Type::i32(), Type::i1()], Type::void());
    ifn!(intrinsics, "llvm.memmove.p0i8.p0i8.i64",
         [i8p, i8p, Type::i64(), Type::i32(), Type::i1()], Type::void());
    ifn!(intrinsics, "llvm.memset.p0i8.i32",
         [i8p, Type::i8(), Type::i32(), Type::i32(), Type::i1()], Type::void());
    ifn!(intrinsics, "llvm.memset.p0i8.i64",
         [i8p, Type::i8(), Type::i64(), Type::i32(), Type::i1()], Type::void());

    ifn!(intrinsics, "llvm.trap", [], Type::void());
    ifn!(intrinsics, "llvm.frameaddress", [Type::i32()], i8p);

    ifn!(intrinsics, "llvm.powi.f32", [Type::f32(), Type::i32()], Type::f32());
    ifn!(intrinsics, "llvm.powi.f64", [Type::f64(), Type::i32()], Type::f64());
    ifn!(intrinsics, "llvm.pow.f32",  [Type::f32(), Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.pow.f64",  [Type::f64(), Type::f64()], Type::f64());

    ifn!(intrinsics, "llvm.sqrt.f32", [Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.sqrt.f64", [Type::f64()], Type::f64());
    ifn!(intrinsics, "llvm.sin.f32",  [Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.sin.f64",  [Type::f64()], Type::f64());
    ifn!(intrinsics, "llvm.cos.f32",  [Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.cos.f64",  [Type::f64()], Type::f64());
    ifn!(intrinsics, "llvm.exp.f32",  [Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.exp.f64",  [Type::f64()], Type::f64());
    ifn!(intrinsics, "llvm.exp2.f32", [Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.exp2.f64", [Type::f64()], Type::f64());
    ifn!(intrinsics, "llvm.log.f32",  [Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.log.f64",  [Type::f64()], Type::f64());
    ifn!(intrinsics, "llvm.log10.f32",[Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.log10.f64",[Type::f64()], Type::f64());
    ifn!(intrinsics, "llvm.log2.f32", [Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.log2.f64", [Type::f64()], Type::f64());

    ifn!(intrinsics, "llvm.fma.f32",  [Type::f32(), Type::f32(), Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.fma.f64",  [Type::f64(), Type::f64(), Type::f64()], Type::f64());

    ifn!(intrinsics, "llvm.fabs.f32", [Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.fabs.f64", [Type::f64()], Type::f64());
    ifn!(intrinsics, "llvm.copysign.f32", [Type::f32(), Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.copysign.f64", [Type::f64(), Type::f64()], Type::f64());

    ifn!(intrinsics, "llvm.floor.f32",[Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.floor.f64",[Type::f64()], Type::f64());
    ifn!(intrinsics, "llvm.ceil.f32", [Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.ceil.f64", [Type::f64()], Type::f64());
    ifn!(intrinsics, "llvm.trunc.f32",[Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.trunc.f64",[Type::f64()], Type::f64());

    ifn!(intrinsics, "llvm.rint.f32", [Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.rint.f64", [Type::f64()], Type::f64());
    ifn!(intrinsics, "llvm.nearbyint.f32", [Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.nearbyint.f64", [Type::f64()], Type::f64());
    ifn!(intrinsics, "llvm.round.f32", [Type::f32()], Type::f32());
    ifn!(intrinsics, "llvm.round.f64", [Type::f64()], Type::f64());

    ifn!(intrinsics, "llvm.ctpop.i8", [Type::i8()], Type::i8());
    ifn!(intrinsics, "llvm.ctpop.i16",[Type::i16()], Type::i16());
    ifn!(intrinsics, "llvm.ctpop.i32",[Type::i32()], Type::i32());
    ifn!(intrinsics, "llvm.ctpop.i64",[Type::i64()], Type::i64());

    ifn!(intrinsics, "llvm.ctlz.i8",  [Type::i8() , Type::i1()], Type::i8());
    ifn!(intrinsics, "llvm.ctlz.i16", [Type::i16(), Type::i1()], Type::i16());
    ifn!(intrinsics, "llvm.ctlz.i32", [Type::i32(), Type::i1()], Type::i32());
    ifn!(intrinsics, "llvm.ctlz.i64", [Type::i64(), Type::i1()], Type::i64());

    ifn!(intrinsics, "llvm.cttz.i8",  [Type::i8() , Type::i1()], Type::i8());
    ifn!(intrinsics, "llvm.cttz.i16", [Type::i16(), Type::i1()], Type::i16());
    ifn!(intrinsics, "llvm.cttz.i32", [Type::i32(), Type::i1()], Type::i32());
    ifn!(intrinsics, "llvm.cttz.i64", [Type::i64(), Type::i1()], Type::i64());

    ifn!(intrinsics, "llvm.bswap.i16",[Type::i16()], Type::i16());
    ifn!(intrinsics, "llvm.bswap.i32",[Type::i32()], Type::i32());
    ifn!(intrinsics, "llvm.bswap.i64",[Type::i64()], Type::i64());

    ifn!(intrinsics, "llvm.sadd.with.overflow.i8",
        [Type::i8(), Type::i8()], Type::struct_([Type::i8(), Type::i1()], false));
    ifn!(intrinsics, "llvm.sadd.with.overflow.i16",
        [Type::i16(), Type::i16()], Type::struct_([Type::i16(), Type::i1()], false));
    ifn!(intrinsics, "llvm.sadd.with.overflow.i32",
        [Type::i32(), Type::i32()], Type::struct_([Type::i32(), Type::i1()], false));
    ifn!(intrinsics, "llvm.sadd.with.overflow.i64",
        [Type::i64(), Type::i64()], Type::struct_([Type::i64(), Type::i1()], false));

    ifn!(intrinsics, "llvm.uadd.with.overflow.i8",
        [Type::i8(), Type::i8()], Type::struct_([Type::i8(), Type::i1()], false));
    ifn!(intrinsics, "llvm.uadd.with.overflow.i16",
        [Type::i16(), Type::i16()], Type::struct_([Type::i16(), Type::i1()], false));
    ifn!(intrinsics, "llvm.uadd.with.overflow.i32",
        [Type::i32(), Type::i32()], Type::struct_([Type::i32(), Type::i1()], false));
    ifn!(intrinsics, "llvm.uadd.with.overflow.i64",
        [Type::i64(), Type::i64()], Type::struct_([Type::i64(), Type::i1()], false));

    ifn!(intrinsics, "llvm.ssub.with.overflow.i8",
        [Type::i8(), Type::i8()], Type::struct_([Type::i8(), Type::i1()], false));
    ifn!(intrinsics, "llvm.ssub.with.overflow.i16",
        [Type::i16(), Type::i16()], Type::struct_([Type::i16(), Type::i1()], false));
    ifn!(intrinsics, "llvm.ssub.with.overflow.i32",
        [Type::i32(), Type::i32()], Type::struct_([Type::i32(), Type::i1()], false));
    ifn!(intrinsics, "llvm.ssub.with.overflow.i64",
        [Type::i64(), Type::i64()], Type::struct_([Type::i64(), Type::i1()], false));

    ifn!(intrinsics, "llvm.usub.with.overflow.i8",
        [Type::i8(), Type::i8()], Type::struct_([Type::i8(), Type::i1()], false));
    ifn!(intrinsics, "llvm.usub.with.overflow.i16",
        [Type::i16(), Type::i16()], Type::struct_([Type::i16(), Type::i1()], false));
    ifn!(intrinsics, "llvm.usub.with.overflow.i32",
        [Type::i32(), Type::i32()], Type::struct_([Type::i32(), Type::i1()], false));
    ifn!(intrinsics, "llvm.usub.with.overflow.i64",
        [Type::i64(), Type::i64()], Type::struct_([Type::i64(), Type::i1()], false));

    ifn!(intrinsics, "llvm.smul.with.overflow.i8",
        [Type::i8(), Type::i8()], Type::struct_([Type::i8(), Type::i1()], false));
    ifn!(intrinsics, "llvm.smul.with.overflow.i16",
        [Type::i16(), Type::i16()], Type::struct_([Type::i16(), Type::i1()], false));
    ifn!(intrinsics, "llvm.smul.with.overflow.i32",
        [Type::i32(), Type::i32()], Type::struct_([Type::i32(), Type::i1()], false));
    ifn!(intrinsics, "llvm.smul.with.overflow.i64",
        [Type::i64(), Type::i64()], Type::struct_([Type::i64(), Type::i1()], false));

    ifn!(intrinsics, "llvm.umul.with.overflow.i8",
        [Type::i8(), Type::i8()], Type::struct_([Type::i8(), Type::i1()], false));
    ifn!(intrinsics, "llvm.umul.with.overflow.i16",
        [Type::i16(), Type::i16()], Type::struct_([Type::i16(), Type::i1()], false));
    ifn!(intrinsics, "llvm.umul.with.overflow.i32",
        [Type::i32(), Type::i32()], Type::struct_([Type::i32(), Type::i1()], false));
    ifn!(intrinsics, "llvm.umul.with.overflow.i64",
        [Type::i64(), Type::i64()], Type::struct_([Type::i64(), Type::i1()], false));

    return intrinsics;
}

pub fn declare_dbg_intrinsics(llmod: ModuleRef, intrinsics: &mut HashMap<&'static str, ValueRef>) {
    ifn!(intrinsics, "llvm.dbg.declare", [Type::metadata(), Type::metadata()], Type::void());
    ifn!(intrinsics,
         "llvm.dbg.value",   [Type::metadata(), Type::i64(), Type::metadata()], Type::void());
}

pub fn trap(bcx: @mut Block) {
    match bcx.ccx().intrinsics.find_equiv(& &"llvm.trap") {
      Some(&x) => { Call(bcx, x, [], []); },
      _ => bcx.sess().bug("unbound llvm.trap in trap")
    }
}

pub fn decl_gc_metadata(ccx: &mut CrateContext, llmod_id: &str) {
    if !ccx.sess.opts.gc || !ccx.uses_gc {
        return;
    }

    let gc_metadata_name = ~"_gc_module_metadata_" + llmod_id;
    let gc_metadata = do gc_metadata_name.with_c_str |buf| {
        unsafe {
            llvm::LLVMAddGlobal(ccx.llmod, Type::i32().to_ref(), buf)
        }
    };
    unsafe {
        llvm::LLVMSetGlobalConstant(gc_metadata, True);
        lib::llvm::SetLinkage(gc_metadata, lib::llvm::ExternalLinkage);
        ccx.module_data.insert(~"_gc_module_metadata", gc_metadata);
    }
}

pub fn create_module_map(ccx: &mut CrateContext) -> (ValueRef, uint) {
    let str_slice_type = Type::struct_([Type::i8p(), ccx.int_type], false);
    let elttype = Type::struct_([str_slice_type, ccx.int_type], false);
    let maptype = Type::array(&elttype, ccx.module_data.len() as u64);
    let map = do "_rust_mod_map".with_c_str |buf| {
        unsafe {
            llvm::LLVMAddGlobal(ccx.llmod, maptype.to_ref(), buf)
        }
    };
    lib::llvm::SetLinkage(map, lib::llvm::InternalLinkage);
    let mut elts: ~[ValueRef] = ~[];

    // This is not ideal, but the borrow checker doesn't
    // like the multiple borrows. At least, it doesn't
    // like them on the current snapshot. (2013-06-14)
    let mut keys = ~[];
    for (k, _) in ccx.module_data.iter() {
        keys.push(k.to_managed());
    }

    for key in keys.iter() {
            let val = *ccx.module_data.find_equiv(key).unwrap();
            let v_ptr = p2i(ccx, val);
            let elt = C_struct([
                C_estr_slice(ccx, *key),
                v_ptr
            ], false);
            elts.push(elt);
    }
    unsafe {
        llvm::LLVMSetInitializer(map, C_array(elttype, elts));
    }
    return (map, keys.len())
}


pub fn decl_crate_map(sess: session::Session, mapmeta: LinkMeta,
                      llmod: ModuleRef) -> ValueRef {
    let targ_cfg = sess.targ_cfg;
    let int_type = Type::int(targ_cfg.arch);
    let mut n_subcrates = 1;
    let cstore = sess.cstore;
    while cstore::have_crate_data(cstore, n_subcrates) { n_subcrates += 1; }
    let mapname = if *sess.building_library {
        format!("{}_{}_{}", mapmeta.name, mapmeta.vers, mapmeta.extras_hash)
    } else {
        ~"toplevel"
    };

    let sym_name = ~"_rust_crate_map_" + mapname;
    let slicetype = Type::struct_([int_type, int_type], false);
    let maptype = Type::struct_([Type::i32(), slicetype, slicetype], false);
    let map = do sym_name.with_c_str |buf| {
        unsafe {
            llvm::LLVMAddGlobal(llmod, maptype.to_ref(), buf)
        }
    };
    // On windows we'd like to export the toplevel cratemap
    // such that we can find it from libstd.
    if targ_cfg.os == session::OsWin32 && "toplevel" == mapname {
        lib::llvm::SetLinkage(map, lib::llvm::DLLExportLinkage);
    } else {
        lib::llvm::SetLinkage(map, lib::llvm::ExternalLinkage);
    }

    return map;
}

pub fn fill_crate_map(ccx: &mut CrateContext, map: ValueRef) {
    let mut subcrates: ~[ValueRef] = ~[];
    let mut i = 1;
    let cstore = ccx.sess.cstore;
    while cstore::have_crate_data(cstore, i) {
        let cdata = cstore::get_crate_data(cstore, i);
        let nm = format!("_rust_crate_map_{}_{}_{}",
                      cdata.name,
                      cstore::get_crate_vers(cstore, i),
                      cstore::get_crate_hash(cstore, i));
        let cr = do nm.with_c_str |buf| {
            unsafe {
                llvm::LLVMAddGlobal(ccx.llmod, ccx.int_type.to_ref(), buf)
            }
        };
        subcrates.push(p2i(ccx, cr));
        i += 1;
    }
    unsafe {
        let maptype = Type::array(&ccx.int_type, subcrates.len() as u64);
        let vec_elements = do "_crate_map_child_vectors".with_c_str |buf| {
            llvm::LLVMAddGlobal(ccx.llmod, maptype.to_ref(), buf)
        };
        lib::llvm::SetLinkage(vec_elements, lib::llvm::InternalLinkage);

        llvm::LLVMSetInitializer(vec_elements, C_array(ccx.int_type, subcrates));
        let (mod_map, mod_count) = create_module_map(ccx);

        llvm::LLVMSetInitializer(map, C_struct(
            [C_i32(2),
             C_struct([
                p2i(ccx, mod_map),
                C_uint(ccx, mod_count)
             ], false),
             C_struct([
                p2i(ccx, vec_elements),
                C_uint(ccx, subcrates.len())
             ], false)
        ], false));
    }
}

pub fn crate_ctxt_to_encode_parms<'r>(cx: &'r CrateContext, ie: encoder::encode_inlined_item<'r>)
    -> encoder::EncodeParams<'r> {

        let diag = cx.sess.diagnostic();
        let item_symbols = &cx.item_symbols;
        let discrim_symbols = &cx.discrim_symbols;
        let link_meta = &cx.link_meta;
        encoder::EncodeParams {
            diag: diag,
            tcx: cx.tcx,
            reexports2: cx.exp_map2,
            item_symbols: item_symbols,
            discrim_symbols: discrim_symbols,
            non_inlineable_statics: &cx.non_inlineable_statics,
            link_meta: link_meta,
            cstore: cx.sess.cstore,
            encode_inlined_item: ie,
            reachable: cx.reachable,
        }
}

pub fn write_metadata(cx: &CrateContext, crate: &ast::Crate) {
    if !*cx.sess.building_library { return; }

    let encode_inlined_item: encoder::encode_inlined_item =
        |ecx, ebml_w, path, ii|
        astencode::encode_inlined_item(ecx, ebml_w, path, ii, cx.maps);

    let encode_parms = crate_ctxt_to_encode_parms(cx, encode_inlined_item);
    let llmeta = C_bytes(encoder::encode_metadata(encode_parms, crate));
    let llconst = C_struct([llmeta], false);
    let mut llglobal = do "rust_metadata".with_c_str |buf| {
        unsafe {
            llvm::LLVMAddGlobal(cx.llmod, val_ty(llconst).to_ref(), buf)
        }
    };
    unsafe {
        llvm::LLVMSetInitializer(llglobal, llconst);
        do cx.sess.targ_cfg.target_strs.meta_sect_name.with_c_str |buf| {
            llvm::LLVMSetSection(llglobal, buf)
        };
        lib::llvm::SetLinkage(llglobal, lib::llvm::InternalLinkage);

        let t_ptr_i8 = Type::i8p();
        llglobal = llvm::LLVMConstBitCast(llglobal, t_ptr_i8.to_ref());
        let llvm_used = do "llvm.used".with_c_str |buf| {
            llvm::LLVMAddGlobal(cx.llmod, Type::array(&t_ptr_i8, 1).to_ref(), buf)
        };
        lib::llvm::SetLinkage(llvm_used, lib::llvm::AppendingLinkage);
        llvm::LLVMSetInitializer(llvm_used, C_array(t_ptr_i8, [llglobal]));
    }
}

// Writes the current ABI version into the crate.
pub fn write_abi_version(ccx: &mut CrateContext) {
    unsafe {
        let llval = C_uint(ccx, abi::abi_version);
        let llglobal = do "rust_abi_version".with_c_str |buf| {
            llvm::LLVMAddGlobal(ccx.llmod, val_ty(llval).to_ref(), buf)
        };
        llvm::LLVMSetInitializer(llglobal, llval);
        llvm::LLVMSetGlobalConstant(llglobal, True);
    }
}

pub fn trans_crate(sess: session::Session,
                   crate: ast::Crate,
                   analysis: &CrateAnalysis,
                   output: &Path) -> CrateTranslation {
    // Before we touch LLVM, make sure that multithreading is enabled.
    if unsafe { !llvm::LLVMRustStartMultithreading() } {
        sess.bug("couldn't enable multi-threaded LLVM");
    }

    let mut symbol_hasher = hash::default_state();
    let link_meta = link::build_link_meta(sess, &crate, output,
                                          &mut symbol_hasher);

    // Append ".rc" to crate name as LLVM module identifier.
    //
    // LLVM code generator emits a ".file filename" directive
    // for ELF backends. Value of the "filename" is set as the
    // LLVM module identifier.  Due to a LLVM MC bug[1], LLVM
    // crashes if the module identifer is same as other symbols
    // such as a function name in the module.
    // 1. http://llvm.org/bugs/show_bug.cgi?id=11479
    let llmod_id = link_meta.name.to_owned() + ".rc";

    let ccx = @mut CrateContext::new(sess,
                                     llmod_id,
                                     analysis.ty_cx,
                                     analysis.exp_map2,
                                     analysis.maps,
                                     symbol_hasher,
                                     link_meta,
                                     analysis.reachable);
    {
        let _icx = push_ctxt("text");
        trans_mod(ccx, &crate.module);
    }

    decl_gc_metadata(ccx, llmod_id);
    fill_crate_map(ccx, ccx.crate_map);

    // NOTE win32: wart with exporting crate_map symbol
    // We set the crate map (_rust_crate_map_toplevel) to use dll_export
    // linkage but that ends up causing the linker to look for a
    // __rust_crate_map_toplevel symbol (extra underscore) which it will
    // subsequently fail to find. So to mitigate that we just introduce
    // an alias from the symbol it expects to the one that actually exists.
    if ccx.sess.targ_cfg.os == session::OsWin32 &&
       !*ccx.sess.building_library {

        let maptype = val_ty(ccx.crate_map).to_ref();

        do "__rust_crate_map_toplevel".with_c_str |buf| {
            unsafe {
                llvm::LLVMAddAlias(ccx.llmod, maptype,
                                   ccx.crate_map, buf);
            }
        }
    }

    glue::emit_tydescs(ccx);
    write_abi_version(ccx);
    if ccx.sess.opts.debuginfo {
        debuginfo::finalize(ccx);
    }

    // Translate the metadata.
    write_metadata(ccx, &crate);
    if ccx.sess.trans_stats() {
        println("--- trans stats ---");
        println!("n_static_tydescs: {}", ccx.stats.n_static_tydescs);
        println!("n_glues_created: {}", ccx.stats.n_glues_created);
        println!("n_null_glues: {}", ccx.stats.n_null_glues);
        println!("n_real_glues: {}", ccx.stats.n_real_glues);

        println!("n_fns: {}", ccx.stats.n_fns);
        println!("n_monos: {}", ccx.stats.n_monos);
        println!("n_inlines: {}", ccx.stats.n_inlines);
        println!("n_closures: {}", ccx.stats.n_closures);
        println("fn stats:");
        do sort::quick_sort(ccx.stats.fn_stats) |&(_, _, insns_a), &(_, _, insns_b)| {
            insns_a > insns_b
        }
        for tuple in ccx.stats.fn_stats.iter() {
            match *tuple {
                (ref name, ms, insns) => {
                    println!("{} insns, {} ms, {}", insns, ms, *name);
                }
            }
        }
    }
    if ccx.sess.count_llvm_insns() {
        for (k, v) in ccx.stats.llvm_insns.iter() {
            println!("{:7u} {}", *v, *k);
        }
    }

    let llcx = ccx.llcx;
    let link_meta = ccx.link_meta;
    let llmod = ccx.llmod;

    return CrateTranslation {
        context: llcx,
        module: llmod,
        link: link_meta
    };
}
