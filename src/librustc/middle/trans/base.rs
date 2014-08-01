// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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
//     but one TypeRef corresponds to many `ty::t`s; for instance, tup(int, int,
//     int) and rec(x=int, y=int, z=int) will have the same TypeRef.

#![allow(non_camel_case_types)]

use back::link::{mangle_exported_name};
use back::{link, abi};
use driver::config;
use driver::config::{NoDebugInfo, FullDebugInfo};
use driver::driver::{CrateAnalysis, CrateTranslation, ModuleTranslation};
use driver::session::Session;
use lint;
use llvm::{BasicBlockRef, ModuleRef, ValueRef, Vector, get_param};
use llvm;
use metadata::{csearch, encoder, loader};
use middle::astencode;
use middle::lang_items::{LangItem, ExchangeMallocFnLangItem, StartFnLangItem};
use middle::subst;
use middle::weak_lang_items;
use middle::subst::Subst;
use middle::trans::_match;
use middle::trans::adt;
use middle::trans::build::*;
use middle::trans::builder::{Builder, noname};
use middle::trans::callee;
use middle::trans::cleanup::{CleanupMethods, ScopeId};
use middle::trans::cleanup;
use middle::trans::common::{Block, C_bool, C_bytes_in_context, C_i32, C_integral, C_nil};
use middle::trans::common::{C_null, C_struct_in_context, C_u64, C_u8, C_uint, C_undef};
use middle::trans::common::{CrateContext, ExternMap, FunctionContext};
use middle::trans::common::{NodeInfo, Result, SubstP, monomorphize_type};
use middle::trans::common::{node_id_type, param_substs, return_type_is_void};
use middle::trans::common::{tydesc_info, type_is_immediate};
use middle::trans::common::{type_is_zero_size, val_ty};
use middle::trans::common;
use middle::trans::consts;
use middle::trans::context::SharedCrateContext;
use middle::trans::controlflow;
use middle::trans::datum;
use middle::trans::debuginfo;
use middle::trans::expr;
use middle::trans::foreign;
use middle::trans::glue;
use middle::trans::inline;
use middle::trans::intrinsic;
use middle::trans::machine;
use middle::trans::machine::{llsize_of, llsize_of_real};
use middle::trans::meth;
use middle::trans::monomorphize;
use middle::trans::tvec;
use middle::trans::type_::Type;
use middle::trans::type_of;
use middle::trans::type_of::*;
use middle::trans::value::Value;
use middle::ty;
use middle::typeck;
use util::common::indenter;
use util::ppaux::{Repr, ty_to_string};
use util::sha2::Sha256;
use util::nodemap::NodeMap;

use arena::TypedArena;
use libc::{c_uint, uint64_t};
use std::c_str::ToCStr;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::rc::Rc;
use std::{i8, i16, i32, i64};
use syntax::abi::{X86, X86_64, Arm, Mips, Mipsel, Rust, RustCall};
use syntax::abi::{RustIntrinsic, Abi, OsWindows};
use syntax::ast_util::{local_def, is_local};
use syntax::attr::AttrMetaMethods;
use syntax::attr;
use syntax::codemap::Span;
use syntax::parse::token::InternedString;
use syntax::visit::Visitor;
use syntax::visit;
use syntax::{ast, ast_util, ast_map};

use time;

local_data_key!(task_local_insn_key: RefCell<Vec<&'static str>>)

pub fn with_insn_ctxt(blk: |&[&'static str]|) {
    match task_local_insn_key.get() {
        Some(ctx) => blk(ctx.borrow().as_slice()),
        None => ()
    }
}

pub fn init_insn_ctxt() {
    task_local_insn_key.replace(Some(RefCell::new(Vec::new())));
}

pub struct _InsnCtxt {
    _cannot_construct_outside_of_this_module: ()
}

#[unsafe_destructor]
impl Drop for _InsnCtxt {
    fn drop(&mut self) {
        match task_local_insn_key.get() {
            Some(ctx) => { ctx.borrow_mut().pop(); }
            None => {}
        }
    }
}

pub fn push_ctxt(s: &'static str) -> _InsnCtxt {
    debug!("new InsnCtxt: {}", s);
    match task_local_insn_key.get() {
        Some(ctx) => ctx.borrow_mut().push(s),
        None => {}
    }
    _InsnCtxt { _cannot_construct_outside_of_this_module: () }
}

pub struct StatRecorder<'a> {
    ccx: &'a CrateContext<'a>,
    name: Option<String>,
    start: u64,
    istart: uint,
}

impl<'a> StatRecorder<'a> {
    pub fn new(ccx: &'a CrateContext, name: String) -> StatRecorder<'a> {
        let start = if ccx.sess().trans_stats() {
            time::precise_time_ns()
        } else {
            0
        };
        let istart = ccx.stats().n_llvm_insns.get();
        StatRecorder {
            ccx: ccx,
            name: Some(name),
            start: start,
            istart: istart,
        }
    }
}

#[unsafe_destructor]
impl<'a> Drop for StatRecorder<'a> {
    fn drop(&mut self) {
        if self.ccx.sess().trans_stats() {
            let end = time::precise_time_ns();
            let elapsed = ((end - self.start) / 1_000_000) as uint;
            let iend = self.ccx.stats().n_llvm_insns.get();
            self.ccx.stats().fn_stats.borrow_mut().push((self.name.take().unwrap(),
                                                       elapsed,
                                                       iend - self.istart));
            self.ccx.stats().n_fns.set(self.ccx.stats().n_fns.get() + 1);
            // Reset LLVM insn count to avoid compound costs.
            self.ccx.stats().n_llvm_insns.set(self.istart);
        }
    }
}

// only use this for foreign function ABIs and glue, use `decl_rust_fn` for Rust functions
pub fn decl_fn(ccx: &CrateContext, name: &str, cc: llvm::CallConv,
           ty: Type, output: ty::t) -> ValueRef {

    let llfn: ValueRef = name.with_c_str(|buf| {
        unsafe {
            llvm::LLVMGetOrInsertFunction(ccx.llmod(), buf, ty.to_ref())
        }
    });

    match ty::get(output).sty {
        // functions returning bottom may unwind, but can never return normally
        ty::ty_bot => {
            unsafe {
                llvm::LLVMAddFunctionAttribute(llfn,
                                               llvm::FunctionIndex as c_uint,
                                               llvm::NoReturnAttribute as uint64_t)
            }
        }
        _ => {}
    }

    if ccx.tcx().sess.opts.cg.no_redzone {
        unsafe {
            llvm::LLVMAddFunctionAttribute(llfn,
                                           llvm::FunctionIndex as c_uint,
                                           llvm::NoRedZoneAttribute as uint64_t)
        }
    }

    llvm::SetFunctionCallConv(llfn, cc);
    // Function addresses in Rust are never significant, allowing functions to be merged.
    llvm::SetUnnamedAddr(llfn, true);

    if ccx.is_split_stack_supported() {
        set_split_stack(llfn);
    }

    llfn
}

// only use this for foreign function ABIs and glue, use `decl_rust_fn` for Rust functions
pub fn decl_cdecl_fn(ccx: &CrateContext,
                     name: &str,
                     ty: Type,
                     output: ty::t) -> ValueRef {
    decl_fn(ccx, name, llvm::CCallConv, ty, output)
}

// only use this for foreign function ABIs and glue, use `get_extern_rust_fn` for Rust functions
pub fn get_extern_fn(ccx: &CrateContext,
                     externs: &mut ExternMap,
                     name: &str,
                     cc: llvm::CallConv,
                     ty: Type,
                     output: ty::t)
                     -> ValueRef {
    match externs.find_equiv(&name) {
        Some(n) => return *n,
        None => {}
    }
    let f = decl_fn(ccx, name, cc, ty, output);
    externs.insert(name.to_string(), f);
    f
}

fn get_extern_rust_fn(ccx: &CrateContext, fn_ty: ty::t, name: &str, did: ast::DefId) -> ValueRef {
    match ccx.externs().borrow().find_equiv(&name) {
        Some(n) => return *n,
        None => ()
    }

    let f = decl_rust_fn(ccx, fn_ty, name);

    csearch::get_item_attrs(&ccx.sess().cstore, did, |attrs| {
        set_llvm_fn_attrs(attrs.as_slice(), f)
    });

    ccx.externs().borrow_mut().insert(name.to_string(), f);
    f
}

pub fn self_type_for_unboxed_closure(ccx: &CrateContext,
                                     closure_id: ast::DefId)
                                     -> ty::t {
    let unboxed_closure_type = ty::mk_unboxed_closure(ccx.tcx(),
                                                      closure_id,
                                                      ty::ReStatic);
    let unboxed_closures = ccx.tcx().unboxed_closures.borrow();
    let unboxed_closure = unboxed_closures.get(&closure_id);
    match unboxed_closure.kind {
        ty::FnUnboxedClosureKind => {
            ty::mk_imm_rptr(ccx.tcx(), ty::ReStatic, unboxed_closure_type)
        }
        ty::FnMutUnboxedClosureKind => {
            ty::mk_mut_rptr(ccx.tcx(), ty::ReStatic, unboxed_closure_type)
        }
        ty::FnOnceUnboxedClosureKind => unboxed_closure_type,
    }
}

pub fn kind_for_unboxed_closure(ccx: &CrateContext, closure_id: ast::DefId)
                                -> ty::UnboxedClosureKind {
    let unboxed_closures = ccx.tcx().unboxed_closures.borrow();
    unboxed_closures.get(&closure_id).kind
}

pub fn decl_rust_fn(ccx: &CrateContext, fn_ty: ty::t, name: &str) -> ValueRef {
    let (inputs, output, abi, env) = match ty::get(fn_ty).sty {
        ty::ty_bare_fn(ref f) => {
            (f.sig.inputs.clone(), f.sig.output, f.abi, None)
        }
        ty::ty_closure(ref f) => {
            (f.sig.inputs.clone(), f.sig.output, f.abi, Some(Type::i8p(ccx)))
        }
        ty::ty_unboxed_closure(closure_did, _) => {
            let unboxed_closures = ccx.tcx().unboxed_closures.borrow();
            let unboxed_closure = unboxed_closures.get(&closure_did);
            let function_type = unboxed_closure.closure_type.clone();
            let self_type = self_type_for_unboxed_closure(ccx, closure_did);
            let llenvironment_type = type_of_explicit_arg(ccx, self_type);
            (function_type.sig.inputs.clone(),
             function_type.sig.output,
             RustCall,
             Some(llenvironment_type))
        }
        _ => fail!("expected closure or fn")
    };

    let llfty = type_of_rust_fn(ccx, env, inputs.as_slice(), output, abi);
    debug!("decl_rust_fn(input count={},type={})",
           inputs.len(),
           ccx.tn().type_to_string(llfty));

    let llfn = decl_fn(ccx, name, llvm::CCallConv, llfty, output);
    let attrs = get_fn_llvm_attributes(ccx, fn_ty);
    attrs.apply_llfn(llfn);

    llfn
}

pub fn decl_internal_rust_fn(ccx: &CrateContext, fn_ty: ty::t, name: &str) -> ValueRef {
    let llfn = decl_rust_fn(ccx, fn_ty, name);
    llvm::SetLinkage(llfn, llvm::InternalLinkage);
    llfn
}

pub fn get_extern_const(externs: &mut ExternMap, llmod: ModuleRef,
                        name: &str, ty: Type) -> ValueRef {
    match externs.find_equiv(&name) {
        Some(n) => return *n,
        None => ()
    }
    unsafe {
        let c = name.with_c_str(|buf| {
            llvm::LLVMAddGlobal(llmod, ty.to_ref(), buf)
        });
        externs.insert(name.to_string(), c);
        return c;
    }
}

// Returns a pointer to the body for the box. The box may be an opaque
// box. The result will be casted to the type of body_t, if it is statically
// known.
pub fn at_box_body(bcx: &Block, body_t: ty::t, boxptr: ValueRef) -> ValueRef {
    let _icx = push_ctxt("at_box_body");
    let ccx = bcx.ccx();
    let ty = Type::at_box(ccx, type_of(ccx, body_t));
    let boxptr = PointerCast(bcx, boxptr, ty.ptr_to());
    GEPi(bcx, boxptr, [0u, abi::box_field_body])
}

fn require_alloc_fn(bcx: &Block, info_ty: ty::t, it: LangItem) -> ast::DefId {
    match bcx.tcx().lang_items.require(it) {
        Ok(id) => id,
        Err(s) => {
            bcx.sess().fatal(format!("allocation of `{}` {}",
                                     bcx.ty_to_string(info_ty),
                                     s).as_slice());
        }
    }
}

// The following malloc_raw_dyn* functions allocate a box to contain
// a given type, but with a potentially dynamic size.

pub fn malloc_raw_dyn<'a>(bcx: &'a Block<'a>,
                          llty_ptr: Type,
                          info_ty: ty::t,
                          size: ValueRef,
                          align: ValueRef)
                          -> Result<'a> {
    let _icx = push_ctxt("malloc_raw_exchange");

    // Allocate space:
    let r = callee::trans_lang_call(bcx,
        require_alloc_fn(bcx, info_ty, ExchangeMallocFnLangItem),
        [size, align],
        None);

    Result::new(r.bcx, PointerCast(r.bcx, r.val, llty_ptr))
}

pub fn malloc_raw_dyn_managed<'a>(
                      bcx: &'a Block<'a>,
                      t: ty::t,
                      alloc_fn: LangItem,
                      size: ValueRef)
                      -> Result<'a> {
    let _icx = push_ctxt("malloc_raw_managed");
    let ccx = bcx.ccx();

    let langcall = require_alloc_fn(bcx, t, alloc_fn);

    // Grab the TypeRef type of box_ptr_ty.
    let box_ptr_ty = ty::mk_box(bcx.tcx(), t);
    let llty = type_of(ccx, box_ptr_ty);
    let llalign = C_uint(ccx, type_of::align_of(ccx, box_ptr_ty) as uint);

    // Allocate space:
    let drop_glue = glue::get_drop_glue(ccx, t);
    let r = callee::trans_lang_call(
        bcx,
        langcall,
        [
            PointerCast(bcx, drop_glue, Type::glue_fn(ccx, Type::i8p(ccx)).ptr_to()),
            size,
            llalign
        ],
        None);
    Result::new(r.bcx, PointerCast(r.bcx, r.val, llty))
}

// Type descriptor and type glue stuff

pub fn get_tydesc(ccx: &CrateContext, t: ty::t) -> Rc<tydesc_info> {
    match ccx.tydescs().borrow().find(&t) {
        Some(inf) => return inf.clone(),
        _ => { }
    }

    ccx.stats().n_static_tydescs.set(ccx.stats().n_static_tydescs.get() + 1u);
    let inf = Rc::new(glue::declare_tydesc(ccx, t));

    ccx.tydescs().borrow_mut().insert(t, inf.clone());
    inf
}

#[allow(dead_code)] // useful
pub fn set_optimize_for_size(f: ValueRef) {
    llvm::SetFunctionAttribute(f, llvm::OptimizeForSizeAttribute)
}

pub fn set_no_inline(f: ValueRef) {
    llvm::SetFunctionAttribute(f, llvm::NoInlineAttribute)
}

#[allow(dead_code)] // useful
pub fn set_no_unwind(f: ValueRef) {
    llvm::SetFunctionAttribute(f, llvm::NoUnwindAttribute)
}

// Tell LLVM to emit the information necessary to unwind the stack for the
// function f.
pub fn set_uwtable(f: ValueRef) {
    llvm::SetFunctionAttribute(f, llvm::UWTableAttribute)
}

pub fn set_inline_hint(f: ValueRef) {
    llvm::SetFunctionAttribute(f, llvm::InlineHintAttribute)
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
        unset_split_stack(llfn);
    }

    if contains_name(attrs, "cold") {
        unsafe {
            llvm::LLVMAddFunctionAttribute(llfn,
                                           llvm::FunctionIndex as c_uint,
                                           llvm::ColdAttribute as uint64_t)
        }
    }
}

pub fn set_always_inline(f: ValueRef) {
    llvm::SetFunctionAttribute(f, llvm::AlwaysInlineAttribute)
}

pub fn set_split_stack(f: ValueRef) {
    "split-stack".with_c_str(|buf| {
        unsafe { llvm::LLVMAddFunctionAttrString(f, llvm::FunctionIndex as c_uint, buf); }
    })
}

pub fn unset_split_stack(f: ValueRef) {
    "split-stack".with_c_str(|buf| {
        unsafe { llvm::LLVMRemoveFunctionAttrString(f, llvm::FunctionIndex as c_uint, buf); }
    })
}

// Double-check that we never ask LLVM to declare the same symbol twice. It
// silently mangles such symbols, breaking our linkage model.
pub fn note_unique_llvm_symbol(ccx: &CrateContext, sym: String) {
    if ccx.all_llvm_symbols().borrow().contains(&sym) {
        ccx.sess().bug(format!("duplicate LLVM symbol: {}", sym).as_slice());
    }
    ccx.all_llvm_symbols().borrow_mut().insert(sym);
}


pub fn get_res_dtor(ccx: &CrateContext,
                    did: ast::DefId,
                    t: ty::t,
                    parent_id: ast::DefId,
                    substs: &subst::Substs)
                 -> ValueRef {
    let _icx = push_ctxt("trans_res_dtor");
    let did = if did.krate != ast::LOCAL_CRATE {
        inline::maybe_instantiate_inline(ccx, did)
    } else {
        did
    };

    if !substs.types.is_empty() {
        assert_eq!(did.krate, ast::LOCAL_CRATE);

        // Since we're in trans we don't care for any region parameters
        let ref substs = subst::Substs::erased(substs.types.clone());

        let vtables = typeck::check::vtable::trans_resolve_method(ccx.tcx(), did.node, substs);
        let (val, _) = monomorphize::monomorphic_fn(ccx, did, substs, vtables, None);

        val
    } else if did.krate == ast::LOCAL_CRATE {
        get_item_val(ccx, did.node)
    } else {
        let tcx = ccx.tcx();
        let name = csearch::get_symbol(&ccx.sess().cstore, did);
        let class_ty = ty::lookup_item_type(tcx, parent_id).ty.subst(tcx, substs);
        let llty = type_of_dtor(ccx, class_ty);
        let dtor_ty = ty::mk_ctor_fn(ccx.tcx(), ast::DUMMY_NODE_ID,
                                     [glue::get_drop_glue_type(ccx, t)], ty::mk_nil());
        get_extern_fn(ccx,
                      &mut *ccx.externs().borrow_mut(),
                      name.as_slice(),
                      llvm::CCallConv,
                      llty,
                      dtor_ty)
    }
}

// Structural comparison: a rather involved form of glue.
pub fn maybe_name_value(cx: &CrateContext, v: ValueRef, s: &str) {
    if cx.sess().opts.cg.save_temps {
        s.with_c_str(|buf| {
            unsafe {
                llvm::LLVMSetValueName(v, buf)
            }
        })
    }
}


// Used only for creating scalar comparison glue.
pub enum scalar_type { nil_type, signed_int, unsigned_int, floating_point, }

pub fn compare_scalar_types<'a>(
                            cx: &'a Block<'a>,
                            lhs: ValueRef,
                            rhs: ValueRef,
                            t: ty::t,
                            op: ast::BinOp)
                            -> Result<'a> {
    let f = |a| Result::new(cx, compare_scalar_values(cx, lhs, rhs, a, op));

    match ty::get(t).sty {
        ty::ty_nil => f(nil_type),
        ty::ty_bool | ty::ty_uint(_) | ty::ty_char => f(unsigned_int),
        ty::ty_ptr(mt) if ty::type_is_sized(cx.tcx(), mt.ty) => f(unsigned_int),
        ty::ty_int(_) => f(signed_int),
        ty::ty_float(_) => f(floating_point),
            // Should never get here, because t is scalar.
        _ => cx.sess().bug("non-scalar type passed to compare_scalar_types")
    }
}


// A helper function to do the actual comparison of scalar values.
pub fn compare_scalar_values<'a>(
                             cx: &'a Block<'a>,
                             lhs: ValueRef,
                             rhs: ValueRef,
                             nt: scalar_type,
                             op: ast::BinOp)
                             -> ValueRef {
    let _icx = push_ctxt("compare_scalar_values");
    fn die(cx: &Block) -> ! {
        cx.sess().bug("compare_scalar_values: must be a comparison operator");
    }
    match nt {
      nil_type => {
        // We don't need to do actual comparisons for nil.
        // () == () holds but () < () does not.
        match op {
          ast::BiEq | ast::BiLe | ast::BiGe => return C_bool(cx.ccx(), true),
          ast::BiNe | ast::BiLt | ast::BiGt => return C_bool(cx.ccx(), false),
          // refinements would be nice
          _ => die(cx)
        }
      }
      floating_point => {
        let cmp = match op {
          ast::BiEq => llvm::RealOEQ,
          ast::BiNe => llvm::RealUNE,
          ast::BiLt => llvm::RealOLT,
          ast::BiLe => llvm::RealOLE,
          ast::BiGt => llvm::RealOGT,
          ast::BiGe => llvm::RealOGE,
          _ => die(cx)
        };
        return FCmp(cx, cmp, lhs, rhs);
      }
      signed_int => {
        let cmp = match op {
          ast::BiEq => llvm::IntEQ,
          ast::BiNe => llvm::IntNE,
          ast::BiLt => llvm::IntSLT,
          ast::BiLe => llvm::IntSLE,
          ast::BiGt => llvm::IntSGT,
          ast::BiGe => llvm::IntSGE,
          _ => die(cx)
        };
        return ICmp(cx, cmp, lhs, rhs);
      }
      unsigned_int => {
        let cmp = match op {
          ast::BiEq => llvm::IntEQ,
          ast::BiNe => llvm::IntNE,
          ast::BiLt => llvm::IntULT,
          ast::BiLe => llvm::IntULE,
          ast::BiGt => llvm::IntUGT,
          ast::BiGe => llvm::IntUGE,
          _ => die(cx)
        };
        return ICmp(cx, cmp, lhs, rhs);
      }
    }
}

pub fn compare_simd_types(
                    cx: &Block,
                    lhs: ValueRef,
                    rhs: ValueRef,
                    t: ty::t,
                    size: uint,
                    op: ast::BinOp)
                    -> ValueRef {
    match ty::get(t).sty {
        ty::ty_float(_) => {
            // The comparison operators for floating point vectors are challenging.
            // LLVM outputs a `< size x i1 >`, but if we perform a sign extension
            // then bitcast to a floating point vector, the result will be `-NaN`
            // for each truth value. Because of this they are unsupported.
            cx.sess().bug("compare_simd_types: comparison operators \
                           not supported for floating point SIMD types")
        },
        ty::ty_uint(_) | ty::ty_int(_) => {
            let cmp = match op {
                ast::BiEq => llvm::IntEQ,
                ast::BiNe => llvm::IntNE,
                ast::BiLt => llvm::IntSLT,
                ast::BiLe => llvm::IntSLE,
                ast::BiGt => llvm::IntSGT,
                ast::BiGe => llvm::IntSGE,
                _ => cx.sess().bug("compare_simd_types: must be a comparison operator"),
            };
            let return_ty = Type::vector(&type_of(cx.ccx(), t), size as u64);
            // LLVM outputs an `< size x i1 >`, so we need to perform a sign extension
            // to get the correctly sized type. This will compile to a single instruction
            // once the IR is converted to assembly if the SIMD instruction is supported
            // by the target architecture.
            SExt(cx, ICmp(cx, cmp, lhs, rhs), return_ty)
        },
        _ => cx.sess().bug("compare_simd_types: invalid SIMD type"),
    }
}

pub type val_and_ty_fn<'r,'b> =
    |&'b Block<'b>, ValueRef, ty::t|: 'r -> &'b Block<'b>;

// Iterates through the elements of a structural type.
pub fn iter_structural_ty<'r,
                          'b>(
                          cx: &'b Block<'b>,
                          av: ValueRef,
                          t: ty::t,
                          f: val_and_ty_fn<'r,'b>)
                          -> &'b Block<'b> {
    let _icx = push_ctxt("iter_structural_ty");

    fn iter_variant<'r,
                    'b>(
                    cx: &'b Block<'b>,
                    repr: &adt::Repr,
                    av: ValueRef,
                    variant: &ty::VariantInfo,
                    substs: &subst::Substs,
                    f: val_and_ty_fn<'r,'b>)
                    -> &'b Block<'b> {
        let _icx = push_ctxt("iter_variant");
        let tcx = cx.tcx();
        let mut cx = cx;

        for (i, &arg) in variant.args.iter().enumerate() {
            cx = f(cx,
                   adt::trans_field_ptr(cx, repr, av, variant.disr_val, i),
                   arg.subst(tcx, substs));
        }
        return cx;
    }

    let (data_ptr, info) = if ty::type_is_sized(cx.tcx(), t) {
        (av, None)
    } else {
        let data = GEPi(cx, av, [0, abi::slice_elt_base]);
        let info = GEPi(cx, av, [0, abi::slice_elt_len]);
        (Load(cx, data), Some(Load(cx, info)))
    };

    let mut cx = cx;
    match ty::get(t).sty {
      ty::ty_struct(..) => {
          let repr = adt::represent_type(cx.ccx(), t);
          expr::with_field_tys(cx.tcx(), t, None, |discr, field_tys| {
              for (i, field_ty) in field_tys.iter().enumerate() {
                  let field_ty = field_ty.mt.ty;
                  let llfld_a = adt::trans_field_ptr(cx, &*repr, data_ptr, discr, i);

                  let val = if ty::type_is_sized(cx.tcx(), field_ty) {
                      llfld_a
                  } else {
                      let boxed_ty = ty::mk_open(cx.tcx(), field_ty);
                      let scratch = datum::rvalue_scratch_datum(cx, boxed_ty, "__fat_ptr_iter");
                      Store(cx, llfld_a, GEPi(cx, scratch.val, [0, abi::slice_elt_base]));
                      Store(cx, info.unwrap(), GEPi(cx, scratch.val, [0, abi::slice_elt_len]));
                      scratch.val
                  };
                  cx = f(cx, val, field_ty);
              }
          })
      }
      ty::ty_unboxed_closure(def_id, _) => {
          let repr = adt::represent_type(cx.ccx(), t);
          let upvars = ty::unboxed_closure_upvars(cx.tcx(), def_id);
          for (i, upvar) in upvars.iter().enumerate() {
              let llupvar = adt::trans_field_ptr(cx, &*repr, data_ptr, 0, i);
              cx = f(cx, llupvar, upvar.ty);
          }
      }
      ty::ty_vec(_, Some(n)) => {
        let (base, len) = tvec::get_fixed_base_and_len(cx, data_ptr, n);
        let unit_ty = ty::sequence_element_type(cx.tcx(), t);
        cx = tvec::iter_vec_raw(cx, base, unit_ty, len, f);
      }
      ty::ty_tup(ref args) => {
          let repr = adt::represent_type(cx.ccx(), t);
          for (i, arg) in args.iter().enumerate() {
              let llfld_a = adt::trans_field_ptr(cx, &*repr, data_ptr, 0, i);
              cx = f(cx, llfld_a, *arg);
          }
      }
      ty::ty_enum(tid, ref substs) => {
          let fcx = cx.fcx;
          let ccx = fcx.ccx;

          let repr = adt::represent_type(ccx, t);
          let variants = ty::enum_variants(ccx.tcx(), tid);
          let n_variants = (*variants).len();

          // NB: we must hit the discriminant first so that structural
          // comparison know not to proceed when the discriminants differ.

          match adt::trans_switch(cx, &*repr, av) {
              (_match::Single, None) => {
                  cx = iter_variant(cx, &*repr, av, &**variants.get(0),
                                    substs, f);
              }
              (_match::Switch, Some(lldiscrim_a)) => {
                  cx = f(cx, lldiscrim_a, ty::mk_int());
                  let unr_cx = fcx.new_temp_block("enum-iter-unr");
                  Unreachable(unr_cx);
                  let llswitch = Switch(cx, lldiscrim_a, unr_cx.llbb,
                                        n_variants);
                  let next_cx = fcx.new_temp_block("enum-iter-next");

                  for variant in (*variants).iter() {
                      let variant_cx =
                          fcx.new_temp_block(
                              format!("enum-iter-variant-{}",
                                      variant.disr_val.to_string().as_slice())
                                     .as_slice());
                      match adt::trans_case(cx, &*repr, variant.disr_val) {
                          _match::SingleResult(r) => {
                              AddCase(llswitch, r.val, variant_cx.llbb)
                          }
                          _ => ccx.sess().unimpl("value from adt::trans_case \
                                                  in iter_structural_ty")
                      }
                      let variant_cx =
                          iter_variant(variant_cx,
                                       &*repr,
                                       data_ptr,
                                       &**variant,
                                       substs,
                                       |x,y,z| f(x,y,z));
                      Br(variant_cx, next_cx.llbb);
                  }
                  cx = next_cx;
              }
              _ => ccx.sess().unimpl("value from adt::trans_switch \
                                      in iter_structural_ty")
          }
      }
      _ => cx.sess().unimpl("type in iter_structural_ty")
    }
    return cx;
}

pub fn cast_shift_expr_rhs<'a>(
                           cx: &'a Block<'a>,
                           op: ast::BinOp,
                           lhs: ValueRef,
                           rhs: ValueRef)
                           -> ValueRef {
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
                      lhs: ValueRef,
                      rhs: ValueRef,
                      trunc: |ValueRef, Type| -> ValueRef,
                      zext: |ValueRef, Type| -> ValueRef)
                      -> ValueRef {
    // Shifts may have any size int on the rhs
    unsafe {
        if ast_util::is_shift_binop(op) {
            let mut rhs_llty = val_ty(rhs);
            let mut lhs_llty = val_ty(lhs);
            if rhs_llty.kind() == Vector { rhs_llty = rhs_llty.element_type() }
            if lhs_llty.kind() == Vector { lhs_llty = lhs_llty.element_type() }
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

pub fn fail_if_zero_or_overflows<'a>(
                    cx: &'a Block<'a>,
                    span: Span,
                    divrem: ast::BinOp,
                    lhs: ValueRef,
                    rhs: ValueRef,
                    rhs_t: ty::t)
                    -> &'a Block<'a> {
    let (zero_text, overflow_text) = if divrem == ast::BiDiv {
        ("attempted to divide by zero",
         "attempted to divide with overflow")
    } else {
        ("attempted remainder with a divisor of zero",
         "attempted remainder with overflow")
    };
    let (is_zero, is_signed) = match ty::get(rhs_t).sty {
        ty::ty_int(t) => {
            let zero = C_integral(Type::int_from_ty(cx.ccx(), t), 0u64, false);
            (ICmp(cx, llvm::IntEQ, rhs, zero), true)
        }
        ty::ty_uint(t) => {
            let zero = C_integral(Type::uint_from_ty(cx.ccx(), t), 0u64, false);
            (ICmp(cx, llvm::IntEQ, rhs, zero), false)
        }
        _ => {
            cx.sess().bug(format!("fail-if-zero on unexpected type: {}",
                                  ty_to_string(cx.tcx(), rhs_t)).as_slice());
        }
    };
    let bcx = with_cond(cx, is_zero, |bcx| {
        controlflow::trans_fail(bcx, span, InternedString::new(zero_text))
    });

    // To quote LLVM's documentation for the sdiv instruction:
    //
    //      Division by zero leads to undefined behavior. Overflow also leads
    //      to undefined behavior; this is a rare case, but can occur, for
    //      example, by doing a 32-bit division of -2147483648 by -1.
    //
    // In order to avoid undefined behavior, we perform runtime checks for
    // signed division/remainder which would trigger overflow. For unsigned
    // integers, no action beyond checking for zero need be taken.
    if is_signed {
        let (llty, min) = match ty::get(rhs_t).sty {
            ty::ty_int(t) => {
                let llty = Type::int_from_ty(cx.ccx(), t);
                let min = match t {
                    ast::TyI if llty == Type::i32(cx.ccx()) => i32::MIN as u64,
                    ast::TyI => i64::MIN as u64,
                    ast::TyI8 => i8::MIN as u64,
                    ast::TyI16 => i16::MIN as u64,
                    ast::TyI32 => i32::MIN as u64,
                    ast::TyI64 => i64::MIN as u64,
                };
                (llty, min)
            }
            _ => unreachable!(),
        };
        let minus_one = ICmp(bcx, llvm::IntEQ, rhs,
                             C_integral(llty, -1, false));
        with_cond(bcx, minus_one, |bcx| {
            let is_min = ICmp(bcx, llvm::IntEQ, lhs,
                              C_integral(llty, min, true));
            with_cond(bcx, is_min, |bcx| {
                controlflow::trans_fail(bcx, span,
                                        InternedString::new(overflow_text))
            })
        })
    } else {
        bcx
    }
}

pub fn trans_external_path(ccx: &CrateContext, did: ast::DefId, t: ty::t) -> ValueRef {
    let name = csearch::get_symbol(&ccx.sess().cstore, did);
    match ty::get(t).sty {
        ty::ty_bare_fn(ref fn_ty) => {
            match fn_ty.abi.for_target(ccx.sess().targ_cfg.os,
                                       ccx.sess().targ_cfg.arch) {
                Some(Rust) | Some(RustCall) => {
                    get_extern_rust_fn(ccx, t, name.as_slice(), did)
                }
                Some(RustIntrinsic) => {
                    ccx.sess().bug("unexpected intrinsic in trans_external_path")
                }
                Some(..) | None => {
                    foreign::register_foreign_item_fn(ccx, fn_ty.abi, t,
                                                      name.as_slice(), None)
                }
            }
        }
        ty::ty_closure(_) => {
            get_extern_rust_fn(ccx, t, name.as_slice(), did)
        }
        _ => {
            let llty = type_of(ccx, t);
            get_extern_const(&mut *ccx.externs().borrow_mut(),
                             ccx.llmod(),
                             name.as_slice(),
                             llty)
        }
    }
}

pub fn invoke<'a>(
              bcx: &'a Block<'a>,
              llfn: ValueRef,
              llargs: Vec<ValueRef> ,
              fn_ty: ty::t,
              call_info: Option<NodeInfo>,
              // FIXME(15064) is_lang_item is a horrible hack, please remove it
              // at the soonest opportunity.
              is_lang_item: bool)
              -> (ValueRef, &'a Block<'a>) {
    let _icx = push_ctxt("invoke_");
    if bcx.unreachable.get() {
        return (C_null(Type::i8(bcx.ccx())), bcx);
    }

    // FIXME(15064) Lang item methods may (in the reflect case) not have proper
    // types, so doing an attribute lookup will fail.
    let attributes = if is_lang_item {
        llvm::AttrBuilder::new()
    } else {
        get_fn_llvm_attributes(bcx.ccx(), fn_ty)
    };

    match bcx.opt_node_id {
        None => {
            debug!("invoke at ???");
        }
        Some(id) => {
            debug!("invoke at {}", bcx.tcx().map.node_to_string(id));
        }
    }

    if need_invoke(bcx) {
        debug!("invoking {} at {}", llfn, bcx.llbb);
        for &llarg in llargs.iter() {
            debug!("arg: {}", llarg);
        }
        let normal_bcx = bcx.fcx.new_temp_block("normal-return");
        let landing_pad = bcx.fcx.get_landing_pad();

        match call_info {
            Some(info) => debuginfo::set_source_location(bcx.fcx, info.id, info.span),
            None => debuginfo::clear_source_location(bcx.fcx)
        };

        let llresult = Invoke(bcx,
                              llfn,
                              llargs.as_slice(),
                              normal_bcx.llbb,
                              landing_pad,
                              Some(attributes));
        return (llresult, normal_bcx);
    } else {
        debug!("calling {} at {}", llfn, bcx.llbb);
        for &llarg in llargs.iter() {
            debug!("arg: {}", llarg);
        }

        match call_info {
            Some(info) => debuginfo::set_source_location(bcx.fcx, info.id, info.span),
            None => debuginfo::clear_source_location(bcx.fcx)
        };

        let llresult = Call(bcx, llfn, llargs.as_slice(), Some(attributes));
        return (llresult, bcx);
    }
}

pub fn need_invoke(bcx: &Block) -> bool {
    if bcx.sess().no_landing_pads() {
        return false;
    }

    // Avoid using invoke if we are already inside a landing pad.
    if bcx.is_lpad {
        return false;
    }

    bcx.fcx.needs_invoke()
}

pub fn load_if_immediate(cx: &Block, v: ValueRef, t: ty::t) -> ValueRef {
    let _icx = push_ctxt("load_if_immediate");
    if type_is_immediate(cx.ccx(), t) { return load_ty(cx, v, t); }
    return v;
}

pub fn load_ty(cx: &Block, ptr: ValueRef, t: ty::t) -> ValueRef {
    /*!
     * Helper for loading values from memory. Does the necessary conversion if
     * the in-memory type differs from the type used for SSA values. Also
     * handles various special cases where the type gives us better information
     * about what we are loading.
     */
    if type_is_zero_size(cx.ccx(), t) {
        C_undef(type_of::type_of(cx.ccx(), t))
    } else if ty::type_is_bool(t) {
        Trunc(cx, LoadRangeAssert(cx, ptr, 0, 2, llvm::False), Type::i1(cx.ccx()))
    } else if ty::type_is_char(t) {
        // a char is a Unicode codepoint, and so takes values from 0
        // to 0x10FFFF inclusive only.
        LoadRangeAssert(cx, ptr, 0, 0x10FFFF + 1, llvm::False)
    } else {
        Load(cx, ptr)
    }
}

pub fn store_ty(cx: &Block, v: ValueRef, dst: ValueRef, t: ty::t) {
    /*!
     * Helper for storing values in memory. Does the necessary conversion if
     * the in-memory type differs from the type used for SSA values.
     */
    if ty::type_is_bool(t) {
        Store(cx, ZExt(cx, v, Type::i8(cx.ccx())), dst);
    } else {
        Store(cx, v, dst);
    };
}

pub fn ignore_lhs(_bcx: &Block, local: &ast::Local) -> bool {
    match local.pat.node {
        ast::PatWild(ast::PatWildSingle) => true, _ => false
    }
}

pub fn init_local<'a>(bcx: &'a Block<'a>, local: &ast::Local)
                  -> &'a Block<'a> {
    debug!("init_local(bcx={}, local.id={:?})", bcx.to_str(), local.id);
    let _indenter = indenter();
    let _icx = push_ctxt("init_local");
    _match::store_local(bcx, local)
}

pub fn raw_block<'a>(
                 fcx: &'a FunctionContext<'a>,
                 is_lpad: bool,
                 llbb: BasicBlockRef)
                 -> &'a Block<'a> {
    common::Block::new(llbb, is_lpad, None, fcx)
}

pub fn with_cond<'a>(
                 bcx: &'a Block<'a>,
                 val: ValueRef,
                 f: |&'a Block<'a>| -> &'a Block<'a>)
                 -> &'a Block<'a> {
    let _icx = push_ctxt("with_cond");
    let fcx = bcx.fcx;
    let next_cx = fcx.new_temp_block("next");
    let cond_cx = fcx.new_temp_block("cond");
    CondBr(bcx, val, cond_cx.llbb, next_cx.llbb);
    let after_cx = f(cond_cx);
    if !after_cx.terminated.get() {
        Br(after_cx, next_cx.llbb);
    }
    next_cx
}

pub fn call_lifetime_start(cx: &Block, ptr: ValueRef) {
    if cx.sess().opts.optimize == config::No {
        return;
    }

    let _icx = push_ctxt("lifetime_start");
    let ccx = cx.ccx();

    let llsize = C_u64(ccx, machine::llsize_of_alloc(ccx, val_ty(ptr).element_type()));
    let ptr = PointerCast(cx, ptr, Type::i8p(ccx));
    let lifetime_start = ccx.get_intrinsic(&"llvm.lifetime.start");
    Call(cx, lifetime_start, [llsize, ptr], None);
}

pub fn call_lifetime_end(cx: &Block, ptr: ValueRef) {
    if cx.sess().opts.optimize == config::No {
        return;
    }

    let _icx = push_ctxt("lifetime_end");
    let ccx = cx.ccx();

    let llsize = C_u64(ccx, machine::llsize_of_alloc(ccx, val_ty(ptr).element_type()));
    let ptr = PointerCast(cx, ptr, Type::i8p(ccx));
    let lifetime_end = ccx.get_intrinsic(&"llvm.lifetime.end");
    Call(cx, lifetime_end, [llsize, ptr], None);
}

pub fn call_memcpy(cx: &Block, dst: ValueRef, src: ValueRef, n_bytes: ValueRef, align: u32) {
    let _icx = push_ctxt("call_memcpy");
    let ccx = cx.ccx();
    let key = match ccx.sess().targ_cfg.arch {
        X86 | Arm | Mips | Mipsel => "llvm.memcpy.p0i8.p0i8.i32",
        X86_64 => "llvm.memcpy.p0i8.p0i8.i64"
    };
    let memcpy = ccx.get_intrinsic(&key);
    let src_ptr = PointerCast(cx, src, Type::i8p(ccx));
    let dst_ptr = PointerCast(cx, dst, Type::i8p(ccx));
    let size = IntCast(cx, n_bytes, ccx.int_type());
    let align = C_i32(ccx, align as i32);
    let volatile = C_bool(ccx, false);
    Call(cx, memcpy, [dst_ptr, src_ptr, size, align, volatile], None);
}

pub fn memcpy_ty(bcx: &Block, dst: ValueRef, src: ValueRef, t: ty::t) {
    let _icx = push_ctxt("memcpy_ty");
    let ccx = bcx.ccx();
    if ty::type_is_structural(t) {
        let llty = type_of::type_of(ccx, t);
        let llsz = llsize_of(ccx, llty);
        let llalign = type_of::align_of(ccx, t);
        call_memcpy(bcx, dst, src, llsz, llalign as u32);
    } else {
        store_ty(bcx, Load(bcx, src), dst, t);
    }
}

pub fn zero_mem(cx: &Block, llptr: ValueRef, t: ty::t) {
    if cx.unreachable.get() { return; }
    let _icx = push_ctxt("zero_mem");
    let bcx = cx;
    memzero(&B(bcx), llptr, t);
}

// Always use this function instead of storing a zero constant to the memory
// in question. If you store a zero constant, LLVM will drown in vreg
// allocation for large data structures, and the generated code will be
// awful. (A telltale sign of this is large quantities of
// `mov [byte ptr foo],0` in the generated code.)
fn memzero(b: &Builder, llptr: ValueRef, ty: ty::t) {
    let _icx = push_ctxt("memzero");
    let ccx = b.ccx;

    let llty = type_of::type_of(ccx, ty);

    let intrinsic_key = match ccx.sess().targ_cfg.arch {
        X86 | Arm | Mips | Mipsel => "llvm.memset.p0i8.i32",
        X86_64 => "llvm.memset.p0i8.i64"
    };

    let llintrinsicfn = ccx.get_intrinsic(&intrinsic_key);
    let llptr = b.pointercast(llptr, Type::i8(ccx).ptr_to());
    let llzeroval = C_u8(ccx, 0);
    let size = machine::llsize_of(ccx, llty);
    let align = C_i32(ccx, type_of::align_of(ccx, ty) as i32);
    let volatile = C_bool(ccx, false);
    b.call(llintrinsicfn, [llptr, llzeroval, size, align, volatile], None);
}

pub fn alloc_ty(bcx: &Block, t: ty::t, name: &str) -> ValueRef {
    let _icx = push_ctxt("alloc_ty");
    let ccx = bcx.ccx();
    let ty = type_of::type_of(ccx, t);
    assert!(!ty::type_has_params(t));
    let val = alloca(bcx, ty, name);
    return val;
}

pub fn alloca(cx: &Block, ty: Type, name: &str) -> ValueRef {
    let p = alloca_no_lifetime(cx, ty, name);
    call_lifetime_start(cx, p);
    p
}

pub fn alloca_no_lifetime(cx: &Block, ty: Type, name: &str) -> ValueRef {
    let _icx = push_ctxt("alloca");
    if cx.unreachable.get() {
        unsafe {
            return llvm::LLVMGetUndef(ty.ptr_to().to_ref());
        }
    }
    debuginfo::clear_source_location(cx.fcx);
    Alloca(cx, ty, name)
}

pub fn alloca_zeroed(cx: &Block, ty: ty::t, name: &str) -> ValueRef {
    let llty = type_of::type_of(cx.ccx(), ty);
    if cx.unreachable.get() {
        unsafe {
            return llvm::LLVMGetUndef(llty.ptr_to().to_ref());
        }
    }
    let p = alloca_no_lifetime(cx, llty, name);
    let b = cx.fcx.ccx.builder();
    b.position_before(cx.fcx.alloca_insert_pt.get().unwrap());
    memzero(&b, p, ty);
    p
}

pub fn arrayalloca(cx: &Block, ty: Type, v: ValueRef) -> ValueRef {
    let _icx = push_ctxt("arrayalloca");
    if cx.unreachable.get() {
        unsafe {
            return llvm::LLVMGetUndef(ty.to_ref());
        }
    }
    debuginfo::clear_source_location(cx.fcx);
    let p = ArrayAlloca(cx, ty, v);
    call_lifetime_start(cx, p);
    p
}

// Creates the alloca slot which holds the pointer to the slot for the final return value
pub fn make_return_slot_pointer(fcx: &FunctionContext, output_type: ty::t) -> ValueRef {
    let lloutputtype = type_of::type_of(fcx.ccx, output_type);

    // We create an alloca to hold a pointer of type `output_type`
    // which will hold the pointer to the right alloca which has the
    // final ret value
    if fcx.needs_ret_allocas {
        // Let's create the stack slot
        let slot = AllocaFcx(fcx, lloutputtype.ptr_to(), "llretslotptr");

        // and if we're using an out pointer, then store that in our newly made slot
        if type_of::return_uses_outptr(fcx.ccx, output_type) {
            let outptr = get_param(fcx.llfn, 0);

            let b = fcx.ccx.builder();
            b.position_before(fcx.alloca_insert_pt.get().unwrap());
            b.store(outptr, slot);
        }

        slot

    // But if there are no nested returns, we skip the indirection and have a single
    // retslot
    } else {
        if type_of::return_uses_outptr(fcx.ccx, output_type) {
            get_param(fcx.llfn, 0)
        } else {
            AllocaFcx(fcx, lloutputtype, "sret_slot")
        }
    }
}

struct CheckForNestedReturnsVisitor {
    found: bool
}

impl Visitor<bool> for CheckForNestedReturnsVisitor {
    fn visit_expr(&mut self, e: &ast::Expr, in_return: bool) {
        match e.node {
            ast::ExprRet(..) if in_return => {
                self.found = true;
                return;
            }
            ast::ExprRet(..) => visit::walk_expr(self, e, true),
            _ => visit::walk_expr(self, e, in_return)
        }
    }
}

fn has_nested_returns(tcx: &ty::ctxt, id: ast::NodeId) -> bool {
    match tcx.map.find(id) {
        Some(ast_map::NodeItem(i)) => {
            match i.node {
                ast::ItemFn(_, _, _, _, blk) => {
                    let mut explicit = CheckForNestedReturnsVisitor { found: false };
                    let mut implicit = CheckForNestedReturnsVisitor { found: false };
                    visit::walk_item(&mut explicit, &*i, false);
                    visit::walk_expr_opt(&mut implicit, blk.expr, true);
                    explicit.found || implicit.found
                }
                _ => tcx.sess.bug("unexpected item variant in has_nested_returns")
            }
        }
        Some(ast_map::NodeTraitItem(trait_method)) => {
            match *trait_method {
                ast::ProvidedMethod(m) => {
                    match m.node {
                        ast::MethDecl(_, _, _, _, _, _, blk, _) => {
                            let mut explicit = CheckForNestedReturnsVisitor { found: false };
                            let mut implicit = CheckForNestedReturnsVisitor { found: false };
                            visit::walk_method_helper(&mut explicit, &*m, false);
                            visit::walk_expr_opt(&mut implicit, blk.expr, true);
                            explicit.found || implicit.found
                        }
                        ast::MethMac(_) => tcx.sess.bug("unexpanded macro")
                    }
                }
                ast::RequiredMethod(_) => {
                    tcx.sess.bug("unexpected variant: required trait method \
                                  in has_nested_returns")
                }
            }
        }
        Some(ast_map::NodeImplItem(ref ii)) => {
            match **ii {
                ast::MethodImplItem(ref m) => {
                    match m.node {
                        ast::MethDecl(_, _, _, _, _, _, blk, _) => {
                            let mut explicit = CheckForNestedReturnsVisitor {
                                found: false,
                            };
                            let mut implicit = CheckForNestedReturnsVisitor {
                                found: false,
                            };
                            visit::walk_method_helper(&mut explicit,
                                                      &**m,
                                                      false);
                            visit::walk_expr_opt(&mut implicit,
                                                 blk.expr,
                                                 true);
                            explicit.found || implicit.found
                        }
                        ast::MethMac(_) => tcx.sess.bug("unexpanded macro")
                    }
                }
            }
        }
        Some(ast_map::NodeExpr(e)) => {
            match e.node {
                ast::ExprFnBlock(_, _, blk) |
                ast::ExprProc(_, blk) |
                ast::ExprUnboxedFn(_, _, _, blk) => {
                    let mut explicit = CheckForNestedReturnsVisitor { found: false };
                    let mut implicit = CheckForNestedReturnsVisitor { found: false };
                    visit::walk_expr(&mut explicit, &*e, false);
                    visit::walk_expr_opt(&mut implicit, blk.expr, true);
                    explicit.found || implicit.found
                }
                _ => tcx.sess.bug("unexpected expr variant in has_nested_returns")
            }
        }

        Some(ast_map::NodeVariant(..)) | Some(ast_map::NodeStructCtor(..)) => false,

        // glue, shims, etc
        None if id == ast::DUMMY_NODE_ID => false,

        _ => tcx.sess.bug(format!("unexpected variant in has_nested_returns: {}",
                                  tcx.map.path_to_string(id)).as_slice())
    }
}

// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_datums_for_fn_args.
//  - new_fn_ctxt
//  - trans_args
//
// Be warned! You must call `init_function` before doing anything with the
// returned function context.
pub fn new_fn_ctxt<'a>(ccx: &'a CrateContext,
                       llfndecl: ValueRef,
                       id: ast::NodeId,
                       has_env: bool,
                       output_type: ty::t,
                       param_substs: &'a param_substs,
                       sp: Option<Span>,
                       block_arena: &'a TypedArena<Block<'a>>)
                       -> FunctionContext<'a> {
    param_substs.validate();

    debug!("new_fn_ctxt(path={}, id={}, param_substs={})",
           if id == -1 {
               "".to_string()
           } else {
               ccx.tcx().map.path_to_string(id).to_string()
           },
           id, param_substs.repr(ccx.tcx()));

    let substd_output_type = output_type.substp(ccx.tcx(), param_substs);
    let uses_outptr = type_of::return_uses_outptr(ccx, substd_output_type);
    let debug_context = debuginfo::create_function_debug_context(ccx, id, param_substs, llfndecl);
    let nested_returns = has_nested_returns(ccx.tcx(), id);

    let mut fcx = FunctionContext {
          llfn: llfndecl,
          llenv: None,
          llretslotptr: Cell::new(None),
          alloca_insert_pt: Cell::new(None),
          llreturn: Cell::new(None),
          needs_ret_allocas: nested_returns,
          personality: Cell::new(None),
          caller_expects_out_pointer: uses_outptr,
          llargs: RefCell::new(NodeMap::new()),
          lllocals: RefCell::new(NodeMap::new()),
          llupvars: RefCell::new(NodeMap::new()),
          id: id,
          param_substs: param_substs,
          span: sp,
          block_arena: block_arena,
          ccx: ccx,
          debug_context: debug_context,
          scopes: RefCell::new(Vec::new())
    };

    if has_env {
        fcx.llenv = Some(get_param(fcx.llfn, fcx.env_arg_pos() as c_uint))
    }

    fcx
}

/// Performs setup on a newly created function, creating the entry scope block
/// and allocating space for the return pointer.
pub fn init_function<'a>(fcx: &'a FunctionContext<'a>,
                         skip_retptr: bool,
                         output_type: ty::t) -> &'a Block<'a> {
    let entry_bcx = fcx.new_temp_block("entry-block");

    // Use a dummy instruction as the insertion point for all allocas.
    // This is later removed in FunctionContext::cleanup.
    fcx.alloca_insert_pt.set(Some(unsafe {
        Load(entry_bcx, C_null(Type::i8p(fcx.ccx)));
        llvm::LLVMGetFirstInstruction(entry_bcx.llbb)
    }));

    // This shouldn't need to recompute the return type,
    // as new_fn_ctxt did it already.
    let substd_output_type = output_type.substp(fcx.ccx.tcx(), fcx.param_substs);

    if !return_type_is_void(fcx.ccx, substd_output_type) {
        // If the function returns nil/bot, there is no real return
        // value, so do not set `llretslotptr`.
        if !skip_retptr || fcx.caller_expects_out_pointer {
            // Otherwise, we normally allocate the llretslotptr, unless we
            // have been instructed to skip it for immediate return
            // values.
            fcx.llretslotptr.set(Some(make_return_slot_pointer(fcx, substd_output_type)));
        }
    }

    entry_bcx
}

// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_datums_for_fn_args.
//  - new_fn_ctxt
//  - trans_args

pub fn arg_kind(cx: &FunctionContext, t: ty::t) -> datum::Rvalue {
    use middle::trans::datum::{ByRef, ByValue};

    datum::Rvalue {
        mode: if arg_is_indirect(cx.ccx, t) { ByRef } else { ByValue }
    }
}

// work around bizarre resolve errors
pub type RvalueDatum = datum::Datum<datum::Rvalue>;
pub type LvalueDatum = datum::Datum<datum::Lvalue>;

// create_datums_for_fn_args: creates rvalue datums for each of the
// incoming function arguments. These will later be stored into
// appropriate lvalue datums.
pub fn create_datums_for_fn_args(fcx: &FunctionContext,
                                 arg_tys: &[ty::t])
                                 -> Vec<RvalueDatum> {
    let _icx = push_ctxt("create_datums_for_fn_args");

    // Return an array wrapping the ValueRefs that we get from `get_param` for
    // each argument into datums.
    arg_tys.iter().enumerate().map(|(i, &arg_ty)| {
        let llarg = get_param(fcx.llfn, fcx.arg_pos(i) as c_uint);
        datum::Datum::new(llarg, arg_ty, arg_kind(fcx, arg_ty))
    }).collect()
}

/// Creates rvalue datums for each of the incoming function arguments and
/// tuples the arguments. These will later be stored into appropriate lvalue
/// datums.
///
/// FIXME(pcwalton): Reduce the amount of code bloat this is responsible for.
fn create_datums_for_fn_args_under_call_abi<
        'a>(
        mut bcx: &'a Block<'a>,
        arg_scope: cleanup::CustomScopeIndex,
        arg_tys: &[ty::t])
        -> Vec<RvalueDatum> {
    let mut result = Vec::new();
    for (i, &arg_ty) in arg_tys.iter().enumerate() {
        if i < arg_tys.len() - 1 {
            // Regular argument.
            let llarg = get_param(bcx.fcx.llfn, bcx.fcx.arg_pos(i) as c_uint);
            result.push(datum::Datum::new(llarg, arg_ty, arg_kind(bcx.fcx,
                                                                  arg_ty)));
            continue
        }

        // This is the last argument. Tuple it.
        match ty::get(arg_ty).sty {
            ty::ty_tup(ref tupled_arg_tys) => {
                let tuple_args_scope_id = cleanup::CustomScope(arg_scope);
                let tuple =
                    unpack_datum!(bcx,
                                  datum::lvalue_scratch_datum(bcx,
                                                              arg_ty,
                                                              "tupled_args",
                                                              false,
                                                              tuple_args_scope_id,
                                                              (),
                                                              |(),
                                                               mut bcx,
                                                               llval| {
                        for (j, &tupled_arg_ty) in
                                    tupled_arg_tys.iter().enumerate() {
                            let llarg =
                                get_param(bcx.fcx.llfn,
                                          bcx.fcx.arg_pos(i + j) as c_uint);
                            let lldest = GEPi(bcx, llval, [0, j]);
                            let datum = datum::Datum::new(
                                llarg,
                                tupled_arg_ty,
                                arg_kind(bcx.fcx, tupled_arg_ty));
                            bcx = datum.store_to(bcx, lldest);
                        }
                        bcx
                    }));
                let tuple = unpack_datum!(bcx,
                                          tuple.to_expr_datum()
                                               .to_rvalue_datum(bcx,
                                                                "argtuple"));
                result.push(tuple);
            }
            ty::ty_nil => {
                let mode = datum::Rvalue::new(datum::ByValue);
                result.push(datum::Datum::new(C_nil(bcx.ccx()),
                                              ty::mk_nil(),
                                              mode))
            }
            _ => {
                bcx.tcx().sess.bug("last argument of a function with \
                                    `rust-call` ABI isn't a tuple?!")
            }
        };

    }

    result
}

fn copy_args_to_allocas<'a>(fcx: &FunctionContext<'a>,
                            arg_scope: cleanup::CustomScopeIndex,
                            bcx: &'a Block<'a>,
                            args: &[ast::Arg],
                            arg_datums: Vec<RvalueDatum> )
                            -> &'a Block<'a> {
    debug!("copy_args_to_allocas");

    let _icx = push_ctxt("copy_args_to_allocas");
    let mut bcx = bcx;

    let arg_scope_id = cleanup::CustomScope(arg_scope);

    for (i, arg_datum) in arg_datums.move_iter().enumerate() {
        // For certain mode/type combinations, the raw llarg values are passed
        // by value.  However, within the fn body itself, we want to always
        // have all locals and arguments be by-ref so that we can cancel the
        // cleanup and for better interaction with LLVM's debug info.  So, if
        // the argument would be passed by value, we store it into an alloca.
        // This alloca should be optimized away by LLVM's mem-to-reg pass in
        // the event it's not truly needed.

        bcx = _match::store_arg(bcx, args[i].pat, arg_datum, arg_scope_id);

        if fcx.ccx.sess().opts.debuginfo == FullDebugInfo {
            debuginfo::create_argument_metadata(bcx, &args[i]);
        }
    }

    bcx
}

fn copy_unboxed_closure_args_to_allocas<'a>(
                                        mut bcx: &'a Block<'a>,
                                        arg_scope: cleanup::CustomScopeIndex,
                                        args: &[ast::Arg],
                                        arg_datums: Vec<RvalueDatum>,
                                        monomorphized_arg_types: &[ty::t])
                                        -> &'a Block<'a> {
    let _icx = push_ctxt("copy_unboxed_closure_args_to_allocas");
    let arg_scope_id = cleanup::CustomScope(arg_scope);

    assert_eq!(arg_datums.len(), 1);

    let arg_datum = arg_datums.move_iter().next().unwrap();

    // Untuple the rest of the arguments.
    let tuple_datum =
        unpack_datum!(bcx,
                      arg_datum.to_lvalue_datum_in_scope(bcx,
                                                         "argtuple",
                                                         arg_scope_id));
    let empty = Vec::new();
    let untupled_arg_types = match ty::get(monomorphized_arg_types[0]).sty {
        ty::ty_tup(ref types) => types.as_slice(),
        ty::ty_nil => empty.as_slice(),
        _ => {
            bcx.tcx().sess.span_bug(args[0].pat.span,
                                    "first arg to `rust-call` ABI function \
                                     wasn't a tuple?!")
        }
    };
    for j in range(0, args.len()) {
        let tuple_element_type = untupled_arg_types[j];
        let tuple_element_datum =
            tuple_datum.get_element(bcx,
                                    tuple_element_type,
                                    |llval| GEPi(bcx, llval, [0, j]));
        let tuple_element_datum = tuple_element_datum.to_expr_datum();
        let tuple_element_datum =
            unpack_datum!(bcx,
                          tuple_element_datum.to_rvalue_datum(bcx,
                                                              "arg"));
        bcx = _match::store_arg(bcx,
                                args[j].pat,
                                tuple_element_datum,
                                arg_scope_id);

        if bcx.fcx.ccx.sess().opts.debuginfo == FullDebugInfo {
            debuginfo::create_argument_metadata(bcx, &args[j]);
        }
    }

    bcx
}

// Ties up the llstaticallocas -> llloadenv -> lltop edges,
// and builds the return block.
pub fn finish_fn<'a>(fcx: &'a FunctionContext<'a>,
                     last_bcx: &'a Block<'a>,
                     retty: ty::t) {
    let _icx = push_ctxt("finish_fn");

    // This shouldn't need to recompute the return type,
    // as new_fn_ctxt did it already.
    let substd_retty = retty.substp(fcx.ccx.tcx(), fcx.param_substs);

    let ret_cx = match fcx.llreturn.get() {
        Some(llreturn) => {
            if !last_bcx.terminated.get() {
                Br(last_bcx, llreturn);
            }
            raw_block(fcx, false, llreturn)
        }
        None => last_bcx
    };
    build_return_block(fcx, ret_cx, substd_retty);
    debuginfo::clear_source_location(fcx);
    fcx.cleanup();
}

// Builds the return block for a function.
pub fn build_return_block(fcx: &FunctionContext, ret_cx: &Block, retty: ty::t) {
    if fcx.llretslotptr.get().is_none() ||
       (!fcx.needs_ret_allocas && fcx.caller_expects_out_pointer) {
        return RetVoid(ret_cx);
    }

    let retslot = if fcx.needs_ret_allocas {
        Load(ret_cx, fcx.llretslotptr.get().unwrap())
    } else {
        fcx.llretslotptr.get().unwrap()
    };
    let retptr = Value(retslot);
    match retptr.get_dominating_store(ret_cx) {
        // If there's only a single store to the ret slot, we can directly return
        // the value that was stored and omit the store and the alloca
        Some(s) => {
            let retval = s.get_operand(0).unwrap().get();
            s.erase_from_parent();

            if retptr.has_no_uses() {
                retptr.erase_from_parent();
            }

            let retval = if ty::type_is_bool(retty) {
                Trunc(ret_cx, retval, Type::i1(fcx.ccx))
            } else {
                retval
            };

            if fcx.caller_expects_out_pointer {
                store_ty(ret_cx, retval, get_param(fcx.llfn, 0), retty);
                return RetVoid(ret_cx);
            } else {
                return Ret(ret_cx, retval);
            }
        }
        // Otherwise, copy the return value to the ret slot
        None => {
            if fcx.caller_expects_out_pointer {
                memcpy_ty(ret_cx, get_param(fcx.llfn, 0), retslot, retty);
                return RetVoid(ret_cx);
            } else {
                return Ret(ret_cx, load_ty(ret_cx, retslot, retty));
            }
        }
    }
}

#[deriving(Clone, Eq, PartialEq)]
pub enum IsUnboxedClosureFlag {
    NotUnboxedClosure,
    IsUnboxedClosure,
}

// trans_closure: Builds an LLVM function out of a source function.
// If the function closes over its environment a closure will be
// returned.
pub fn trans_closure(ccx: &CrateContext,
                     decl: &ast::FnDecl,
                     body: &ast::Block,
                     llfndecl: ValueRef,
                     param_substs: &param_substs,
                     id: ast::NodeId,
                     _attributes: &[ast::Attribute],
                     arg_types: Vec<ty::t>,
                     output_type: ty::t,
                     abi: Abi,
                     has_env: bool,
                     is_unboxed_closure: IsUnboxedClosureFlag,
                     maybe_load_env: <'a>|&'a Block<'a>, ScopeId|
                                         -> &'a Block<'a>) {
    ccx.stats().n_closures.set(ccx.stats().n_closures.get() + 1);

    let _icx = push_ctxt("trans_closure");
    set_uwtable(llfndecl);

    debug!("trans_closure(..., param_substs={})",
           param_substs.repr(ccx.tcx()));

    let arena = TypedArena::new();
    let fcx = new_fn_ctxt(ccx,
                          llfndecl,
                          id,
                          has_env,
                          output_type,
                          param_substs,
                          Some(body.span),
                          &arena);
    let mut bcx = init_function(&fcx, false, output_type);

    // cleanup scope for the incoming arguments
    let arg_scope = fcx.push_custom_cleanup_scope();

    let block_ty = node_id_type(bcx, body.id);

    // Set up arguments to the function.
    let monomorphized_arg_types =
        arg_types.iter()
                 .map(|at| monomorphize_type(bcx, *at))
                 .collect::<Vec<_>>();
    for monomorphized_arg_type in monomorphized_arg_types.iter() {
        debug!("trans_closure: monomorphized_arg_type: {}",
               ty_to_string(ccx.tcx(), *monomorphized_arg_type));
    }
    debug!("trans_closure: function lltype: {}",
           bcx.fcx.ccx.tn().val_to_string(bcx.fcx.llfn));

    let arg_datums = if abi != RustCall {
        create_datums_for_fn_args(&fcx,
                                  monomorphized_arg_types.as_slice())
    } else {
        create_datums_for_fn_args_under_call_abi(
            bcx,
            arg_scope,
            monomorphized_arg_types.as_slice())
    };

    bcx = match is_unboxed_closure {
        NotUnboxedClosure => {
            copy_args_to_allocas(&fcx,
                                 arg_scope,
                                 bcx,
                                 decl.inputs.as_slice(),
                                 arg_datums)
        }
        IsUnboxedClosure => {
            copy_unboxed_closure_args_to_allocas(
                bcx,
                arg_scope,
                decl.inputs.as_slice(),
                arg_datums,
                monomorphized_arg_types.as_slice())
        }
    };

    bcx = maybe_load_env(bcx, cleanup::CustomScope(arg_scope));

    // Up until here, IR instructions for this function have explicitly not been annotated with
    // source code location, so we don't step into call setup code. From here on, source location
    // emitting should be enabled.
    debuginfo::start_emitting_source_locations(&fcx);

    let dest = match fcx.llretslotptr.get() {
        Some(_) => expr::SaveIn(fcx.get_ret_slot(bcx, block_ty, "iret_slot")),
        None => {
            assert!(type_is_zero_size(bcx.ccx(), block_ty));
            expr::Ignore
        }
    };

    // This call to trans_block is the place where we bridge between
    // translation calls that don't have a return value (trans_crate,
    // trans_mod, trans_item, et cetera) and those that do
    // (trans_block, trans_expr, et cetera).
    bcx = controlflow::trans_block(bcx, body, dest);

    match dest {
        expr::SaveIn(slot) if fcx.needs_ret_allocas => {
            Store(bcx, slot, fcx.llretslotptr.get().unwrap());
        }
        _ => {}
    }

    match fcx.llreturn.get() {
        Some(_) => {
            Br(bcx, fcx.return_exit_block());
            fcx.pop_custom_cleanup_scope(arg_scope);
        }
        None => {
            // Microoptimization writ large: avoid creating a separate
            // llreturn basic block
            bcx = fcx.pop_and_trans_custom_cleanup_scope(bcx, arg_scope);
        }
    };

    // Put return block after all other blocks.
    // This somewhat improves single-stepping experience in debugger.
    unsafe {
        let llreturn = fcx.llreturn.get();
        for &llreturn in llreturn.iter() {
            llvm::LLVMMoveBasicBlockAfter(llreturn, bcx.llbb);
        }
    }

    // Insert the mandatory first few basic blocks before lltop.
    finish_fn(&fcx, bcx, output_type);
}

// trans_fn: creates an LLVM function corresponding to a source language
// function.
pub fn trans_fn(ccx: &CrateContext,
                decl: &ast::FnDecl,
                body: &ast::Block,
                llfndecl: ValueRef,
                param_substs: &param_substs,
                id: ast::NodeId,
                attrs: &[ast::Attribute]) {
    let _s = StatRecorder::new(ccx, ccx.tcx().map.path_to_string(id).to_string());
    debug!("trans_fn(param_substs={})", param_substs.repr(ccx.tcx()));
    let _icx = push_ctxt("trans_fn");
    let fn_ty = ty::node_id_to_type(ccx.tcx(), id);
    let arg_types = ty::ty_fn_args(fn_ty);
    let output_type = ty::ty_fn_ret(fn_ty);
    let abi = ty::ty_fn_abi(fn_ty);
    trans_closure(ccx,
                  decl,
                  body,
                  llfndecl,
                  param_substs,
                  id,
                  attrs,
                  arg_types,
                  output_type,
                  abi,
                  false,
                  NotUnboxedClosure,
                  |bcx, _| bcx);
}

pub fn trans_enum_variant(ccx: &CrateContext,
                          _enum_id: ast::NodeId,
                          variant: &ast::Variant,
                          _args: &[ast::VariantArg],
                          disr: ty::Disr,
                          param_substs: &param_substs,
                          llfndecl: ValueRef) {
    let _icx = push_ctxt("trans_enum_variant");

    trans_enum_variant_or_tuple_like_struct(
        ccx,
        variant.node.id,
        disr,
        param_substs,
        llfndecl);
}

pub fn trans_named_tuple_constructor<'a>(mut bcx: &'a Block<'a>,
                                         ctor_ty: ty::t,
                                         disr: ty::Disr,
                                         args: callee::CallArgs,
                                         dest: expr::Dest) -> Result<'a> {

    let ccx = bcx.fcx.ccx;
    let tcx = ccx.tcx();

    let result_ty = match ty::get(ctor_ty).sty {
        ty::ty_bare_fn(ref bft) => bft.sig.output,
        _ => ccx.sess().bug(
            format!("trans_enum_variant_constructor: \
                     unexpected ctor return type {}",
                     ctor_ty.repr(tcx)).as_slice())
    };

    // Get location to store the result. If the user does not care about
    // the result, just make a stack slot
    let llresult = match dest {
        expr::SaveIn(d) => d,
        expr::Ignore => {
            if !type_is_zero_size(ccx, result_ty) {
                alloc_ty(bcx, result_ty, "constructor_result")
            } else {
                C_undef(type_of::type_of(ccx, result_ty))
            }
        }
    };

    if !type_is_zero_size(ccx, result_ty) {
        match args {
            callee::ArgExprs(exprs) => {
                let fields = exprs.iter().map(|x| *x).enumerate().collect::<Vec<_>>();
                bcx = expr::trans_adt(bcx, result_ty, disr, fields.as_slice(),
                                      None, expr::SaveIn(llresult));
            }
            _ => ccx.sess().bug("expected expr as arguments for variant/struct tuple constructor")
        }
    }

    // If the caller doesn't care about the result
    // drop the temporary we made
    let bcx = match dest {
        expr::SaveIn(_) => bcx,
        expr::Ignore => glue::drop_ty(bcx, llresult, result_ty)
    };

    Result::new(bcx, llresult)
}

pub fn trans_tuple_struct(ccx: &CrateContext,
                          _fields: &[ast::StructField],
                          ctor_id: ast::NodeId,
                          param_substs: &param_substs,
                          llfndecl: ValueRef) {
    let _icx = push_ctxt("trans_tuple_struct");

    trans_enum_variant_or_tuple_like_struct(
        ccx,
        ctor_id,
        0,
        param_substs,
        llfndecl);
}

fn trans_enum_variant_or_tuple_like_struct(ccx: &CrateContext,
                                           ctor_id: ast::NodeId,
                                           disr: ty::Disr,
                                           param_substs: &param_substs,
                                           llfndecl: ValueRef) {
    let ctor_ty = ty::node_id_to_type(ccx.tcx(), ctor_id);
    let ctor_ty = ctor_ty.substp(ccx.tcx(), param_substs);

    let result_ty = match ty::get(ctor_ty).sty {
        ty::ty_bare_fn(ref bft) => bft.sig.output,
        _ => ccx.sess().bug(
            format!("trans_enum_variant_or_tuple_like_struct: \
                     unexpected ctor return type {}",
                    ty_to_string(ccx.tcx(), ctor_ty)).as_slice())
    };

    let arena = TypedArena::new();
    let fcx = new_fn_ctxt(ccx, llfndecl, ctor_id, false, result_ty,
                          param_substs, None, &arena);
    let bcx = init_function(&fcx, false, result_ty);

    assert!(!fcx.needs_ret_allocas);

    let arg_tys = ty::ty_fn_args(ctor_ty);

    let arg_datums = create_datums_for_fn_args(&fcx, arg_tys.as_slice());

    if !type_is_zero_size(fcx.ccx, result_ty) {
        let dest = fcx.get_ret_slot(bcx, result_ty, "eret_slot");
        let repr = adt::represent_type(ccx, result_ty);
        for (i, arg_datum) in arg_datums.move_iter().enumerate() {
            let lldestptr = adt::trans_field_ptr(bcx,
                                                 &*repr,
                                                 dest,
                                                 disr,
                                                 i);
            arg_datum.store_to(bcx, lldestptr);
        }
        adt::trans_set_discr(bcx, &*repr, dest, disr);
    }

    finish_fn(&fcx, bcx, result_ty);
}

fn enum_variant_size_lint(ccx: &CrateContext, enum_def: &ast::EnumDef, sp: Span, id: ast::NodeId) {
    let mut sizes = Vec::new(); // does no allocation if no pushes, thankfully

    let levels = ccx.tcx().node_lint_levels.borrow();
    let lint_id = lint::LintId::of(lint::builtin::VARIANT_SIZE_DIFFERENCE);
    let lvlsrc = match levels.find(&(id, lint_id)) {
        None | Some(&(lint::Allow, _)) => return,
        Some(&lvlsrc) => lvlsrc,
    };

    let avar = adt::represent_type(ccx, ty::node_id_to_type(ccx.tcx(), id));
    match *avar {
        adt::General(_, ref variants, _) => {
            for var in variants.iter() {
                let mut size = 0;
                for field in var.fields.iter().skip(1) {
                    // skip the discriminant
                    size += llsize_of_real(ccx, sizing_type_of(ccx, *field));
                }
                sizes.push(size);
            }
        },
        _ => { /* its size is either constant or unimportant */ }
    }

    let (largest, slargest, largest_index) = sizes.iter().enumerate().fold((0, 0, 0),
        |(l, s, li), (idx, &size)|
            if size > l {
                (size, l, idx)
            } else if size > s {
                (l, size, li)
            } else {
                (l, s, li)
            }
    );

    // we only warn if the largest variant is at least thrice as large as
    // the second-largest.
    if largest > slargest * 3 && slargest > 0 {
        // Use lint::raw_emit_lint rather than sess.add_lint because the lint-printing
        // pass for the latter already ran.
        lint::raw_emit_lint(&ccx.tcx().sess, lint::builtin::VARIANT_SIZE_DIFFERENCE,
                            lvlsrc, Some(sp),
                            format!("enum variant is more than three times larger \
                                     ({} bytes) than the next largest (ignoring padding)",
                                    largest).as_slice());

        ccx.sess().span_note(enum_def.variants.get(largest_index).span,
                             "this variant is the largest");
    }
}

pub struct TransItemVisitor<'a> {
    pub ccx: &'a CrateContext<'a>,
}

impl<'a> Visitor<()> for TransItemVisitor<'a> {
    fn visit_item(&mut self, i: &ast::Item, _:()) {
        trans_item(self.ccx, i);
    }
}

/// Enum describing the origin of an LLVM `Value`, for linkage purposes.
pub enum ValueOrigin {
    /// The LLVM `Value` is in this context because the corresponding item was
    /// assigned to the current compilation unit.
    OriginalTranslation,
    /// The `Value`'s corresponding item was assigned to some other compilation
    /// unit, but the `Value` was translated in this context anyway because the
    /// item is marked `#[inline]`.
    InlinedCopy,
}

/// Set the appropriate linkage for an LLVM `ValueRef` (function or global).
/// If the `llval` is the direct translation of a specific Rust item, `id`
/// should be set to the `NodeId` of that item.  (This mapping should be
/// 1-to-1, so monomorphizations and drop/visit glue should have `id` set to
/// `None`.)  `llval_origin` indicates whether `llval` is the translation of an
/// item assigned to `ccx`'s compilation unit or an inlined copy of an item
/// assigned to a different compilation unit.
pub fn update_linkage(ccx: &CrateContext,
                      llval: ValueRef,
                      id: Option<ast::NodeId>,
                      llval_origin: ValueOrigin) {
    match llval_origin {
        InlinedCopy => {
            // `llval` is a translation of an item defined in a separate
            // compilation unit.  This only makes sense if there are at least
            // two compilation units.
            assert!(ccx.sess().opts.cg.codegen_units > 1);
            // `llval` is a copy of something defined elsewhere, so use
            // `AvailableExternallyLinkage` to avoid duplicating code in the
            // output.
            llvm::SetLinkage(llval, llvm::AvailableExternallyLinkage);
            return;
        },
        OriginalTranslation => {},
    }

    match id {
        Some(id) if ccx.reachable().contains(&id) => {
            llvm::SetLinkage(llval, llvm::ExternalLinkage);
        },
        _ => {
            // `id` does not refer to an item in `ccx.reachable`.
            if ccx.sess().opts.cg.codegen_units > 1 {
                llvm::SetLinkage(llval, llvm::ExternalLinkage);
            } else {
                llvm::SetLinkage(llval, llvm::InternalLinkage);
            }
        },
    }
}

pub fn trans_item(ccx: &CrateContext, item: &ast::Item) {
    let _icx = push_ctxt("trans_item");

    let from_external = ccx.external_srcs().borrow().contains_key(&item.id);

    match item.node {
      ast::ItemFn(ref decl, _fn_style, abi, ref generics, ref body) => {
        if !generics.is_type_parameterized() {
            let trans_everywhere = attr::requests_inline(item.attrs.as_slice());
            // Ignore `trans_everywhere` for cross-crate inlined items
            // (`from_external`).  `trans_item` will be called once for each
            // compilation unit that references the item, so it will still get
            // translated everywhere it's needed.
            for (ref ccx, is_origin) in ccx.maybe_iter(!from_external && trans_everywhere) {
                let llfn = get_item_val(ccx, item.id);
                if abi != Rust {
                    foreign::trans_rust_fn_with_foreign_abi(ccx,
                                                            &**decl,
                                                            &**body,
                                                            item.attrs.as_slice(),
                                                            llfn,
                                                            &param_substs::empty(),
                                                            item.id,
                                                            None);
                } else {
                    trans_fn(ccx,
                             &**decl,
                             &**body,
                             llfn,
                             &param_substs::empty(),
                             item.id,
                             item.attrs.as_slice());
                }
                update_linkage(ccx,
                               llfn,
                               Some(item.id),
                               if is_origin { OriginalTranslation } else { InlinedCopy });
            }
        }

        // Be sure to travel more than just one layer deep to catch nested
        // items in blocks and such.
        let mut v = TransItemVisitor{ ccx: ccx };
        v.visit_block(&**body, ());
      }
      ast::ItemImpl(ref generics, _, _, ref impl_items) => {
        meth::trans_impl(ccx,
                         item.ident,
                         impl_items.as_slice(),
                         generics,
                         item.id);
      }
      ast::ItemMod(ref m) => {
        trans_mod(&ccx.rotate(), m);
      }
      ast::ItemEnum(ref enum_definition, _) => {
        enum_variant_size_lint(ccx, enum_definition, item.span, item.id);
      }
      ast::ItemStatic(_, m, ref expr) => {
          // Recurse on the expression to catch items in blocks
          let mut v = TransItemVisitor{ ccx: ccx };
          v.visit_expr(&**expr, ());

          let trans_everywhere = attr::requests_inline(item.attrs.as_slice());
          for (ref ccx, is_origin) in ccx.maybe_iter(!from_external && trans_everywhere) {
              consts::trans_const(ccx, m, item.id);

              let g = get_item_val(ccx, item.id);
              update_linkage(ccx,
                             g,
                             Some(item.id),
                             if is_origin { OriginalTranslation } else { InlinedCopy });
          }

          // Do static_assert checking. It can't really be done much earlier
          // because we need to get the value of the bool out of LLVM
          if attr::contains_name(item.attrs.as_slice(), "static_assert") {
              if m == ast::MutMutable {
                  ccx.sess().span_fatal(expr.span,
                                        "cannot have static_assert on a mutable \
                                         static");
              }

              let v = ccx.const_values().borrow().get_copy(&item.id);
              unsafe {
                  if !(llvm::LLVMConstIntGetZExtValue(v) != 0) {
                      ccx.sess().span_fatal(expr.span, "static assertion failed");
                  }
              }
          }
      },
      ast::ItemForeignMod(ref foreign_mod) => {
        foreign::trans_foreign_mod(ccx, foreign_mod);
      }
      ast::ItemTrait(..) => {
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

// Translate a module. Doing this amounts to translating the items in the
// module; there ends up being no artifact (aside from linkage names) of
// separate modules in the compiled program.  That's because modules exist
// only as a convenience for humans working with the code, to organize names
// and control visibility.
pub fn trans_mod(ccx: &CrateContext, m: &ast::Mod) {
    let _icx = push_ctxt("trans_mod");
    for item in m.items.iter() {
        trans_item(ccx, &**item);
    }
}

fn finish_register_fn(ccx: &CrateContext, sp: Span, sym: String, node_id: ast::NodeId,
                      llfn: ValueRef) {
    ccx.item_symbols().borrow_mut().insert(node_id, sym);

    // The stack exhaustion lang item shouldn't have a split stack because
    // otherwise it would continue to be exhausted (bad), and both it and the
    // eh_personality functions need to be externally linkable.
    let def = ast_util::local_def(node_id);
    if ccx.tcx().lang_items.stack_exhausted() == Some(def) {
        unset_split_stack(llfn);
        llvm::SetLinkage(llfn, llvm::ExternalLinkage);
    }
    if ccx.tcx().lang_items.eh_personality() == Some(def) {
        llvm::SetLinkage(llfn, llvm::ExternalLinkage);
    }


    if is_entry_fn(ccx.sess(), node_id) {
        create_entry_wrapper(ccx, sp, llfn);
    }
}

fn register_fn(ccx: &CrateContext,
               sp: Span,
               sym: String,
               node_id: ast::NodeId,
               node_type: ty::t)
               -> ValueRef {
    match ty::get(node_type).sty {
        ty::ty_bare_fn(ref f) => {
            assert!(f.abi == Rust || f.abi == RustCall);
        }
        _ => fail!("expected bare rust fn")
    };

    let llfn = decl_rust_fn(ccx, node_type, sym.as_slice());
    finish_register_fn(ccx, sp, sym, node_id, llfn);
    llfn
}

pub fn get_fn_llvm_attributes(ccx: &CrateContext, fn_ty: ty::t)
                              -> llvm::AttrBuilder {
    use middle::ty::{BrAnon, ReLateBound};

    let (fn_sig, abi, has_env) = match ty::get(fn_ty).sty {
        ty::ty_closure(ref f) => (f.sig.clone(), f.abi, true),
        ty::ty_bare_fn(ref f) => (f.sig.clone(), f.abi, false),
        ty::ty_unboxed_closure(closure_did, _) => {
            let unboxed_closures = ccx.tcx().unboxed_closures.borrow();
            let ref function_type = unboxed_closures.get(&closure_did)
                                                    .closure_type;

            (function_type.sig.clone(), RustCall, true)
        }
        _ => ccx.sess().bug("expected closure or function.")
    };


    // Since index 0 is the return value of the llvm func, we start
    // at either 1 or 2 depending on whether there's an env slot or not
    let mut first_arg_offset = if has_env { 2 } else { 1 };
    let mut attrs = llvm::AttrBuilder::new();
    let ret_ty = fn_sig.output;

    // These have an odd calling convention, so we need to manually
    // unpack the input ty's
    let input_tys = match ty::get(fn_ty).sty {
        ty::ty_unboxed_closure(_, _) => {
            assert!(abi == RustCall);

            match ty::get(fn_sig.inputs[0]).sty {
                ty::ty_nil => Vec::new(),
                ty::ty_tup(ref inputs) => inputs.clone(),
                _ => ccx.sess().bug("expected tuple'd inputs")
            }
        },
        ty::ty_bare_fn(_) if abi == RustCall => {
            let inputs = vec![fn_sig.inputs[0]];

            match ty::get(fn_sig.inputs[1]).sty {
                ty::ty_nil => inputs,
                ty::ty_tup(ref t_in) => inputs.append(t_in.as_slice()),
                _ => ccx.sess().bug("expected tuple'd inputs")
            }
        }
        _ => fn_sig.inputs.clone()
    };

    // A function pointer is called without the declaration
    // available, so we have to apply any attributes with ABI
    // implications directly to the call instruction. Right now,
    // the only attribute we need to worry about is `sret`.
    if type_of::return_uses_outptr(ccx, ret_ty) {
        let llret_sz = llsize_of_real(ccx, type_of::type_of(ccx, ret_ty));

        // The outptr can be noalias and nocapture because it's entirely
        // invisible to the program. We also know it's nonnull as well
        // as how many bytes we can dereference
        attrs.arg(1, llvm::StructRetAttribute)
             .arg(1, llvm::NoAliasAttribute)
             .arg(1, llvm::NoCaptureAttribute)
             .arg(1, llvm::DereferenceableAttribute(llret_sz));

        // Add one more since there's an outptr
        first_arg_offset += 1;
    } else {
        // The `noalias` attribute on the return value is useful to a
        // function ptr caller.
        match ty::get(ret_ty).sty {
            // `~` pointer return values never alias because ownership
            // is transferred
            ty::ty_uniq(it) if !ty::type_is_sized(ccx.tcx(), it) => {}
            ty::ty_uniq(_) => {
                attrs.ret(llvm::NoAliasAttribute);
            }
            _ => {}
        }

        // We can also mark the return value as `dereferenceable` in certain cases
        match ty::get(ret_ty).sty {
            // These are not really pointers but pairs, (pointer, len)
            ty::ty_uniq(it) |
            ty::ty_rptr(_, ty::mt { ty: it, .. }) if !ty::type_is_sized(ccx.tcx(), it) => {}
            ty::ty_uniq(inner) | ty::ty_rptr(_, ty::mt { ty: inner, .. }) => {
                let llret_sz = llsize_of_real(ccx, type_of::type_of(ccx, inner));
                attrs.ret(llvm::DereferenceableAttribute(llret_sz));
            }
            _ => {}
        }

        match ty::get(ret_ty).sty {
            ty::ty_bool => {
                attrs.ret(llvm::ZExtAttribute);
            }
            _ => {}
        }
    }

    for (idx, &t) in input_tys.iter().enumerate().map(|(i, v)| (i + first_arg_offset, v)) {
        match ty::get(t).sty {
            // this needs to be first to prevent fat pointers from falling through
            _ if !type_is_immediate(ccx, t) => {
                let llarg_sz = llsize_of_real(ccx, type_of::type_of(ccx, t));

                // For non-immediate arguments the callee gets its own copy of
                // the value on the stack, so there are no aliases. It's also
                // program-invisible so can't possibly capture
                attrs.arg(idx, llvm::NoAliasAttribute)
                     .arg(idx, llvm::NoCaptureAttribute)
                     .arg(idx, llvm::DereferenceableAttribute(llarg_sz));
            }

            ty::ty_bool => {
                attrs.arg(idx, llvm::ZExtAttribute);
            }

            // `~` pointer parameters never alias because ownership is transferred
            ty::ty_uniq(inner) => {
                let llsz = llsize_of_real(ccx, type_of::type_of(ccx, inner));

                attrs.arg(idx, llvm::NoAliasAttribute)
                     .arg(idx, llvm::DereferenceableAttribute(llsz));
            }

            // The visit glue deals only with opaque pointers so we don't
            // actually know the concrete type of Self thus we don't know how
            // many bytes to mark as dereferenceable so instead we just mark
            // it as nonnull which still holds true
            ty::ty_rptr(b, ty::mt { ty: it, mutbl }) if match ty::get(it).sty {
                ty::ty_param(_) => true, _ => false
            } && mutbl == ast::MutMutable => {
                attrs.arg(idx, llvm::NoAliasAttribute)
                     .arg(idx, llvm::NonNullAttribute);

                match b {
                    ReLateBound(_, BrAnon(_)) => {
                        attrs.arg(idx, llvm::NoCaptureAttribute);
                    }
                    _ => {}
                }
            }

            // `&mut` pointer parameters never alias other parameters, or mutable global data
            //
            // `&T` where `T` contains no `UnsafeCell<U>` is immutable, and can be marked as both
            // `readonly` and `noalias`, as LLVM's definition of `noalias` is based solely on
            // memory dependencies rather than pointer equality
            ty::ty_rptr(b, mt) if mt.mutbl == ast::MutMutable ||
                                  !ty::type_contents(ccx.tcx(), mt.ty).interior_unsafe() => {

                let llsz = llsize_of_real(ccx, type_of::type_of(ccx, mt.ty));
                attrs.arg(idx, llvm::NoAliasAttribute)
                     .arg(idx, llvm::DereferenceableAttribute(llsz));

                if mt.mutbl == ast::MutImmutable {
                    attrs.arg(idx, llvm::ReadOnlyAttribute);
                }

                match b {
                    ReLateBound(_, BrAnon(_)) => {
                        attrs.arg(idx, llvm::NoCaptureAttribute);
                    }
                    _ => {}
                }
            }

            // When a reference in an argument has no named lifetime, it's impossible for that
            // reference to escape this function (returned or stored beyond the call by a closure).
            ty::ty_rptr(ReLateBound(_, BrAnon(_)), mt) => {
                let llsz = llsize_of_real(ccx, type_of::type_of(ccx, mt.ty));
                attrs.arg(idx, llvm::NoCaptureAttribute)
                     .arg(idx, llvm::DereferenceableAttribute(llsz));
            }

            // & pointer parameters are also never null and we know exactly how
            // many bytes we can dereference
            ty::ty_rptr(_, mt) => {
                let llsz = llsize_of_real(ccx, type_of::type_of(ccx, mt.ty));
                attrs.arg(idx, llvm::DereferenceableAttribute(llsz));
            }
            _ => ()
        }
    }

    attrs
}

// only use this for foreign function ABIs and glue, use `register_fn` for Rust functions
pub fn register_fn_llvmty(ccx: &CrateContext,
                          sp: Span,
                          sym: String,
                          node_id: ast::NodeId,
                          cc: llvm::CallConv,
                          llfty: Type) -> ValueRef {
    debug!("register_fn_llvmty id={} sym={}", node_id, sym);

    let llfn = decl_fn(ccx, sym.as_slice(), cc, llfty, ty::mk_nil());
    finish_register_fn(ccx, sp, sym, node_id, llfn);
    llfn
}

pub fn is_entry_fn(sess: &Session, node_id: ast::NodeId) -> bool {
    match *sess.entry_fn.borrow() {
        Some((entry_id, _)) => node_id == entry_id,
        None => false
    }
}

// Create a _rust_main(args: ~[str]) function which will be called from the
// runtime rust_start function
pub fn create_entry_wrapper(ccx: &CrateContext,
                           _sp: Span,
                           main_llfn: ValueRef) {
    let et = ccx.sess().entry_type.get().unwrap();
    match et {
        config::EntryMain => {
            create_entry_fn(ccx, main_llfn, true);
        }
        config::EntryStart => create_entry_fn(ccx, main_llfn, false),
        config::EntryNone => {}    // Do nothing.
    }

    fn create_entry_fn(ccx: &CrateContext,
                       rust_main: ValueRef,
                       use_start_lang_item: bool) {
        let llfty = Type::func([ccx.int_type(), Type::i8p(ccx).ptr_to()],
                               &ccx.int_type());

        let llfn = decl_cdecl_fn(ccx, "main", llfty, ty::mk_nil());

        // FIXME: #16581: Marking a symbol in the executable with `dllexport`
        // linkage forces MinGW's linker to output a `.reloc` section for ASLR
        if ccx.sess().targ_cfg.os == OsWindows {
            unsafe { llvm::LLVMRustSetDLLExportStorageClass(llfn) }
        }

        let llbb = "top".with_c_str(|buf| {
            unsafe {
                llvm::LLVMAppendBasicBlockInContext(ccx.llcx(), llfn, buf)
            }
        });
        let bld = ccx.raw_builder();
        unsafe {
            llvm::LLVMPositionBuilderAtEnd(bld, llbb);

            let (start_fn, args) = if use_start_lang_item {
                let start_def_id = match ccx.tcx().lang_items.require(StartFnLangItem) {
                    Ok(id) => id,
                    Err(s) => { ccx.sess().fatal(s.as_slice()); }
                };
                let start_fn = if start_def_id.krate == ast::LOCAL_CRATE {
                    get_item_val(ccx, start_def_id.node)
                } else {
                    let start_fn_type = csearch::get_type(ccx.tcx(),
                                                          start_def_id).ty;
                    trans_external_path(ccx, start_def_id, start_fn_type)
                };

                let args = {
                    let opaque_rust_main = "rust_main".with_c_str(|buf| {
                        llvm::LLVMBuildPointerCast(bld, rust_main, Type::i8p(ccx).to_ref(), buf)
                    });

                    vec!(
                        opaque_rust_main,
                        get_param(llfn, 0),
                        get_param(llfn, 1)
                     )
                };
                (start_fn, args)
            } else {
                debug!("using user-defined start fn");
                let args = vec!(
                    get_param(llfn, 0 as c_uint),
                    get_param(llfn, 1 as c_uint)
                );

                (rust_main, args)
            };

            let result = llvm::LLVMBuildCall(bld,
                                             start_fn,
                                             args.as_ptr(),
                                             args.len() as c_uint,
                                             noname());

            llvm::LLVMBuildRet(bld, result);
        }
    }
}

fn exported_name(ccx: &CrateContext, id: ast::NodeId,
                 ty: ty::t, attrs: &[ast::Attribute]) -> String {
    match ccx.external_srcs().borrow().find(&id) {
        Some(&did) => {
            let sym = csearch::get_symbol(&ccx.sess().cstore, did);
            debug!("found item {} in other crate...", sym);
            return sym;
        }
        None => {}
    }

    match attr::first_attr_value_str_by_name(attrs, "export_name") {
        // Use provided name
        Some(name) => name.get().to_string(),

        _ => ccx.tcx().map.with_path(id, |mut path| {
            if attr::contains_name(attrs, "no_mangle") {
                // Don't mangle
                path.last().unwrap().to_string()
            } else {
                match weak_lang_items::link_name(attrs) {
                    Some(name) => name.get().to_string(),
                    None => {
                        // Usual name mangling
                        mangle_exported_name(ccx, path, ty, id)
                    }
                }
            }
        })
    }
}

pub fn get_item_val(ccx: &CrateContext, id: ast::NodeId) -> ValueRef {
    debug!("get_item_val(id=`{:?}`)", id);

    match ccx.item_vals().borrow().find_copy(&id) {
        Some(v) => return v,
        None => {}
    }

    let item = ccx.tcx().map.get(id);
    let val = match item {
        ast_map::NodeItem(i) => {
            let ty = ty::node_id_to_type(ccx.tcx(), i.id);
            let sym = exported_name(ccx, id, ty, i.attrs.as_slice());

            let v = match i.node {
                ast::ItemStatic(_, mutbl, ref expr) => {
                    // If this static came from an external crate, then
                    // we need to get the symbol from csearch instead of
                    // using the current crate's name/version
                    // information in the hash of the symbol
                    debug!("making {}", sym);
                    let is_local = !ccx.external_srcs().borrow().contains_key(&id);

                    // We need the translated value here, because for enums the
                    // LLVM type is not fully determined by the Rust type.
                    let (v, inlineable, _) = consts::const_expr(ccx, &**expr, is_local);
                    ccx.const_values().borrow_mut().insert(id, v);
                    let mut inlineable = inlineable;

                    unsafe {
                        let llty = llvm::LLVMTypeOf(v);
                        let g = sym.as_slice().with_c_str(|buf| {
                            llvm::LLVMAddGlobal(ccx.llmod(), llty, buf)
                        });

                        // Apply the `unnamed_addr` attribute if
                        // requested
                        if !ast_util::static_has_significant_address(
                                mutbl,
                                i.attrs.as_slice()) {
                            llvm::SetUnnamedAddr(g, true);

                            // This is a curious case where we must make
                            // all of these statics inlineable. If a
                            // global is not tagged as `#[inline(never)]`,
                            // then LLVM won't coalesce globals unless they
                            // have an internal linkage type. This means that
                            // external crates cannot use this global.
                            // This is a problem for things like inner
                            // statics in generic functions, because the
                            // function will be inlined into another
                            // crate and then attempt to link to the
                            // static in the original crate, only to
                            // find that it's not there. On the other
                            // side of inlining, the crates knows to
                            // not declare this static as
                            // available_externally (because it isn't)
                            inlineable = true;
                        }

                        if attr::contains_name(i.attrs.as_slice(),
                                               "thread_local") {
                            llvm::set_thread_local(g, true);
                        }

                        if !inlineable {
                            debug!("{} not inlined", sym);
                            ccx.non_inlineable_statics().borrow_mut()
                                                      .insert(id);
                        }

                        ccx.item_symbols().borrow_mut().insert(i.id, sym);
                        g
                    }
                }

                ast::ItemFn(_, _, abi, _, _) => {
                    let llfn = if abi == Rust {
                        register_fn(ccx, i.span, sym, i.id, ty)
                    } else {
                        foreign::register_rust_fn_with_foreign_abi(ccx,
                                                                   i.span,
                                                                   sym,
                                                                   i.id)
                    };
                    set_llvm_fn_attrs(i.attrs.as_slice(), llfn);
                    llfn
                }

                _ => fail!("get_item_val: weird result in table")
            };

            match attr::first_attr_value_str_by_name(i.attrs.as_slice(),
                                                     "link_section") {
                Some(sect) => unsafe {
                    sect.get().with_c_str(|buf| {
                        llvm::LLVMSetSection(v, buf);
                    })
                },
                None => ()
            }

            v
        }

        ast_map::NodeTraitItem(trait_method) => {
            debug!("get_item_val(): processing a NodeTraitItem");
            match *trait_method {
                ast::RequiredMethod(_) => {
                    ccx.sess().bug("unexpected variant: required trait method in \
                                   get_item_val()");
                }
                ast::ProvidedMethod(m) => {
                    register_method(ccx, id, &*m)
                }
            }
        }

        ast_map::NodeImplItem(ii) => {
            match *ii {
                ast::MethodImplItem(m) => register_method(ccx, id, &*m),
            }
        }

        ast_map::NodeForeignItem(ni) => {
            match ni.node {
                ast::ForeignItemFn(..) => {
                    let abi = ccx.tcx().map.get_foreign_abi(id);
                    let ty = ty::node_id_to_type(ccx.tcx(), ni.id);
                    let name = foreign::link_name(&*ni);
                    foreign::register_foreign_item_fn(ccx, abi, ty,
                                                      name.get().as_slice(),
                                                      Some(ni.span))
                }
                ast::ForeignItemStatic(..) => {
                    foreign::register_static(ccx, &*ni)
                }
            }
        }

        ast_map::NodeVariant(ref v) => {
            let llfn;
            let args = match v.node.kind {
                ast::TupleVariantKind(ref args) => args,
                ast::StructVariantKind(_) => {
                    fail!("struct variant kind unexpected in get_item_val")
                }
            };
            assert!(args.len() != 0u);
            let ty = ty::node_id_to_type(ccx.tcx(), id);
            let parent = ccx.tcx().map.get_parent(id);
            let enm = ccx.tcx().map.expect_item(parent);
            let sym = exported_name(ccx,
                                    id,
                                    ty,
                                    enm.attrs.as_slice());

            llfn = match enm.node {
                ast::ItemEnum(_, _) => {
                    register_fn(ccx, (*v).span, sym, id, ty)
                }
                _ => fail!("NodeVariant, shouldn't happen")
            };
            set_inline_hint(llfn);
            llfn
        }

        ast_map::NodeStructCtor(struct_def) => {
            // Only register the constructor if this is a tuple-like struct.
            let ctor_id = match struct_def.ctor_id {
                None => {
                    ccx.sess().bug("attempt to register a constructor of \
                                    a non-tuple-like struct")
                }
                Some(ctor_id) => ctor_id,
            };
            let parent = ccx.tcx().map.get_parent(id);
            let struct_item = ccx.tcx().map.expect_item(parent);
            let ty = ty::node_id_to_type(ccx.tcx(), ctor_id);
            let sym = exported_name(ccx,
                                    id,
                                    ty,
                                    struct_item.attrs
                                               .as_slice());
            let llfn = register_fn(ccx, struct_item.span,
                                   sym, ctor_id, ty);
            set_inline_hint(llfn);
            llfn
        }

        ref variant => {
            ccx.sess().bug(format!("get_item_val(): unexpected variant: {:?}",
                                   variant).as_slice())
        }
    };

    // All LLVM globals and functions are initially created as external-linkage
    // declarations.  If `trans_item`/`trans_fn` later turns the declaration
    // into a definition, it adjusts the linkage then (using `update_linkage`).
    //
    // The exception is foreign items, which have their linkage set inside the
    // call to `foreign::register_*` above.  We don't touch the linkage after
    // that (`foreign::trans_foreign_mod` doesn't adjust the linkage like the
    // other item translation functions do).

    ccx.item_vals().borrow_mut().insert(id, val);
    val
}

fn register_method(ccx: &CrateContext, id: ast::NodeId,
                   m: &ast::Method) -> ValueRef {
    let mty = ty::node_id_to_type(ccx.tcx(), id);

    let sym = exported_name(ccx, id, mty, m.attrs.as_slice());

    let llfn = register_fn(ccx, m.span, sym, id, mty);
    set_llvm_fn_attrs(m.attrs.as_slice(), llfn);
    llfn
}

pub fn p2i(ccx: &CrateContext, v: ValueRef) -> ValueRef {
    unsafe {
        return llvm::LLVMConstPtrToInt(v, ccx.int_type().to_ref());
    }
}

pub fn crate_ctxt_to_encode_parms<'r>(cx: &'r SharedCrateContext,
                                      ie: encoder::EncodeInlinedItem<'r>)
    -> encoder::EncodeParams<'r> {
        encoder::EncodeParams {
            diag: cx.sess().diagnostic(),
            tcx: cx.tcx(),
            reexports2: cx.exp_map2(),
            item_symbols: cx.item_symbols(),
            non_inlineable_statics: cx.non_inlineable_statics(),
            link_meta: cx.link_meta(),
            cstore: &cx.sess().cstore,
            encode_inlined_item: ie,
            reachable: cx.reachable(),
        }
}

pub fn write_metadata(cx: &SharedCrateContext, krate: &ast::Crate) -> Vec<u8> {
    use flate;

    let any_library = cx.sess().crate_types.borrow().iter().any(|ty| {
        *ty != config::CrateTypeExecutable
    });
    if !any_library {
        return Vec::new()
    }

    let encode_inlined_item: encoder::EncodeInlinedItem =
        |ecx, rbml_w, ii| astencode::encode_inlined_item(ecx, rbml_w, ii);

    let encode_parms = crate_ctxt_to_encode_parms(cx, encode_inlined_item);
    let metadata = encoder::encode_metadata(encode_parms, krate);
    let compressed = Vec::from_slice(encoder::metadata_encoding_version)
                     .append(match flate::deflate_bytes(metadata.as_slice()) {
                         Some(compressed) => compressed,
                         None => {
                             cx.sess().fatal("failed to compress metadata")
                         }
                     }.as_slice());
    let llmeta = C_bytes_in_context(cx.metadata_llcx(), compressed.as_slice());
    let llconst = C_struct_in_context(cx.metadata_llcx(), [llmeta], false);
    let name = format!("rust_metadata_{}_{}",
                       cx.link_meta().crate_name,
                       cx.link_meta().crate_hash);
    let llglobal = name.with_c_str(|buf| {
        unsafe {
            llvm::LLVMAddGlobal(cx.metadata_llmod(), val_ty(llconst).to_ref(), buf)
        }
    });
    unsafe {
        llvm::LLVMSetInitializer(llglobal, llconst);
        let name = loader::meta_section_name(cx.sess().targ_cfg.os);
        name.unwrap_or("rust_metadata").with_c_str(|buf| {
            llvm::LLVMSetSection(llglobal, buf)
        });
    }
    return metadata;
}

/// Find any symbols that are defined in one compilation unit, but not declared
/// in any other compilation unit.  Give these symbols internal linkage.
fn internalize_symbols(cx: &SharedCrateContext, reachable: &HashSet<String>) {
    use std::c_str::CString;

    unsafe {
        let mut declared = HashSet::new();

        let iter_globals = |llmod| {
            ValueIter {
                cur: llvm::LLVMGetFirstGlobal(llmod),
                step: llvm::LLVMGetNextGlobal,
            }
        };

        let iter_functions = |llmod| {
            ValueIter {
                cur: llvm::LLVMGetFirstFunction(llmod),
                step: llvm::LLVMGetNextFunction,
            }
        };

        // Collect all external declarations in all compilation units.
        for ccx in cx.iter() {
            for val in iter_globals(ccx.llmod()).chain(iter_functions(ccx.llmod())) {
                let linkage = llvm::LLVMGetLinkage(val);
                // We only care about external declarations (not definitions)
                // and available_externally definitions.
                if !(linkage == llvm::ExternalLinkage as c_uint &&
                     llvm::LLVMIsDeclaration(val) != 0) &&
                   !(linkage == llvm::AvailableExternallyLinkage as c_uint) {
                    continue
                }

                let name = CString::new(llvm::LLVMGetValueName(val), false);
                declared.insert(name);
            }
        }

        // Examine each external definition.  If the definition is not used in
        // any other compilation unit, and is not reachable from other crates,
        // then give it internal linkage.
        for ccx in cx.iter() {
            for val in iter_globals(ccx.llmod()).chain(iter_functions(ccx.llmod())) {
                // We only care about external definitions.
                if !(llvm::LLVMGetLinkage(val) == llvm::ExternalLinkage as c_uint &&
                     llvm::LLVMIsDeclaration(val) == 0) {
                    continue
                }

                let name = CString::new(llvm::LLVMGetValueName(val), false);
                if !declared.contains(&name) &&
                   !reachable.contains_equiv(&name.as_str().unwrap()) {
                    llvm::SetLinkage(val, llvm::InternalLinkage);
                }
            }
        }
    }


    struct ValueIter {
        cur: ValueRef,
        step: unsafe extern "C" fn(ValueRef) -> ValueRef,
    }

    impl Iterator<ValueRef> for ValueIter {
        fn next(&mut self) -> Option<ValueRef> {
            let old = self.cur;
            if !old.is_null() {
                self.cur = unsafe { (self.step)(old) };
                Some(old)
            } else {
                None
            }
        }
    }
}

pub fn trans_crate(krate: ast::Crate,
                   analysis: CrateAnalysis) -> (ty::ctxt, CrateTranslation) {
    let CrateAnalysis { ty_cx: tcx, exp_map2, reachable, name, .. } = analysis;

    // Before we touch LLVM, make sure that multithreading is enabled.
    unsafe {
        use std::sync::{Once, ONCE_INIT};
        static mut INIT: Once = ONCE_INIT;
        static mut POISONED: bool = false;
        INIT.doit(|| {
            if llvm::LLVMStartMultithreaded() != 1 {
                // use an extra bool to make sure that all future usage of LLVM
                // cannot proceed despite the Once not running more than once.
                POISONED = true;
            }
        });

        if POISONED {
            tcx.sess.bug("couldn't enable multi-threaded LLVM");
        }
    }

    let link_meta = link::build_link_meta(&tcx.sess, &krate, name);

    let codegen_units = tcx.sess.opts.cg.codegen_units;
    let shared_ccx = SharedCrateContext::new(link_meta.crate_name.as_slice(),
                                             codegen_units,
                                             tcx,
                                             exp_map2,
                                             Sha256::new(),
                                             link_meta.clone(),
                                             reachable);

    {
        let ccx = shared_ccx.get_ccx(0);

        // First, verify intrinsics.
        intrinsic::check_intrinsics(&ccx);

        // Next, translate the module.
        {
            let _icx = push_ctxt("text");
            trans_mod(&ccx, &krate.module);
        }
    }

    for ccx in shared_ccx.iter() {
        glue::emit_tydescs(&ccx);
        if ccx.sess().opts.debuginfo != NoDebugInfo {
            debuginfo::finalize(&ccx);
        }
    }

    // Translate the metadata.
    let metadata = write_metadata(&shared_ccx, &krate);

    if shared_ccx.sess().trans_stats() {
        let stats = shared_ccx.stats();
        println!("--- trans stats ---");
        println!("n_static_tydescs: {}", stats.n_static_tydescs.get());
        println!("n_glues_created: {}", stats.n_glues_created.get());
        println!("n_null_glues: {}", stats.n_null_glues.get());
        println!("n_real_glues: {}", stats.n_real_glues.get());

        println!("n_fns: {}", stats.n_fns.get());
        println!("n_monos: {}", stats.n_monos.get());
        println!("n_inlines: {}", stats.n_inlines.get());
        println!("n_closures: {}", stats.n_closures.get());
        println!("fn stats:");
        stats.fn_stats.borrow_mut().sort_by(|&(_, _, insns_a), &(_, _, insns_b)| {
            insns_b.cmp(&insns_a)
        });
        for tuple in stats.fn_stats.borrow().iter() {
            match *tuple {
                (ref name, ms, insns) => {
                    println!("{} insns, {} ms, {}", insns, ms, *name);
                }
            }
        }
    }
    if shared_ccx.sess().count_llvm_insns() {
        for (k, v) in shared_ccx.stats().llvm_insns.borrow().iter() {
            println!("{:7u} {}", *v, *k);
        }
    }

    let modules = shared_ccx.iter()
        .map(|ccx| ModuleTranslation { llcx: ccx.llcx(), llmod: ccx.llmod() })
        .collect();

    let mut reachable: Vec<String> = shared_ccx.reachable().iter().filter_map(|id| {
        shared_ccx.item_symbols().borrow().find(id).map(|s| s.to_string())
    }).collect();

    // For the purposes of LTO, we add to the reachable set all of the upstream
    // reachable extern fns. These functions are all part of the public ABI of
    // the final product, so LTO needs to preserve them.
    shared_ccx.sess().cstore.iter_crate_data(|cnum, _| {
        let syms = csearch::get_reachable_extern_fns(&shared_ccx.sess().cstore, cnum);
        reachable.extend(syms.move_iter().map(|did| {
            csearch::get_symbol(&shared_ccx.sess().cstore, did)
        }));
    });

    // Make sure that some other crucial symbols are not eliminated from the
    // module. This includes the main function, the crate map (used for debug
    // log settings and I/O), and finally the curious rust_stack_exhausted
    // symbol. This symbol is required for use by the libmorestack library that
    // we link in, so we must ensure that this symbol is not internalized (if
    // defined in the crate).
    reachable.push("main".to_string());
    reachable.push("rust_stack_exhausted".to_string());

    // referenced from .eh_frame section on some platforms
    reachable.push("rust_eh_personality".to_string());
    // referenced from rt/rust_try.ll
    reachable.push("rust_eh_personality_catch".to_string());

    if codegen_units > 1 {
        internalize_symbols(&shared_ccx, &reachable.iter().map(|x| x.clone()).collect());
    }

    let metadata_module = ModuleTranslation {
        llcx: shared_ccx.metadata_llcx(),
        llmod: shared_ccx.metadata_llmod(),
    };
    let formats = shared_ccx.tcx().dependency_formats.borrow().clone();
    let no_builtins = attr::contains_name(krate.attrs.as_slice(), "no_builtins");

    let translation = CrateTranslation {
        modules: modules,
        metadata_module: metadata_module,
        link: link_meta,
        metadata: metadata,
        reachable: reachable,
        crate_formats: formats,
        no_builtins: no_builtins,
    };

    (shared_ccx.take_tcx(), translation)
}
