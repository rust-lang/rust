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
//   * There's no way to find out the Ty type of a ValueRef.  Doing so
//     would be "trying to get the eggs out of an omelette" (credit:
//     pcwalton).  You can, instead, find out its TypeRef by calling val_ty,
//     but one TypeRef corresponds to many `Ty`s; for instance, tup(int, int,
//     int) and rec(x=int, y=int, z=int) will have the same TypeRef.

#![allow(non_camel_case_types)]

pub use self::ValueOrigin::*;

use super::CrateTranslation;
use super::ModuleTranslation;

use back::link::{mangle_exported_name};
use back::{link, abi};
use lint;
use llvm::{BasicBlockRef, Linkage, ValueRef, Vector, get_param};
use llvm;
use metadata::{csearch, encoder, loader};
use middle::astencode;
use middle::cfg;
use middle::lang_items::{LangItem, ExchangeMallocFnLangItem, StartFnLangItem};
use middle::weak_lang_items;
use middle::subst::{Subst, Substs};
use middle::ty::{self, Ty, ClosureTyper};
use session::config::{self, NoDebugInfo};
use session::Session;
use trans::_match;
use trans::adt;
use trans::build::*;
use trans::builder::{Builder, noname};
use trans::callee;
use trans::cleanup::CleanupMethods;
use trans::cleanup;
use trans::closure;
use trans::common::{Block, C_bool, C_bytes_in_context, C_i32, C_integral};
use trans::common::{C_null, C_struct_in_context, C_u64, C_u8, C_undef};
use trans::common::{CrateContext, ExternMap, FunctionContext};
use trans::common::{Result, NodeIdAndSpan};
use trans::common::{node_id_type, return_type_is_void};
use trans::common::{tydesc_info, type_is_immediate};
use trans::common::{type_is_zero_size, val_ty};
use trans::common;
use trans::consts;
use trans::context::SharedCrateContext;
use trans::controlflow;
use trans::datum;
use trans::debuginfo::{self, DebugLoc, ToDebugLoc};
use trans::expr;
use trans::foreign;
use trans::glue;
use trans::inline;
use trans::intrinsic;
use trans::machine;
use trans::machine::{llsize_of, llsize_of_real};
use trans::meth;
use trans::monomorphize;
use trans::tvec;
use trans::type_::Type;
use trans::type_of;
use trans::type_of::*;
use trans::value::Value;
use util::common::indenter;
use util::ppaux::{Repr, ty_to_string};
use util::sha2::Sha256;
use util::nodemap::NodeMap;

use arena::TypedArena;
use libc::{c_uint, uint64_t};
use std::ffi::{CStr, CString};
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::mem;
use std::rc::Rc;
use std::str;
use std::{i8, i16, i32, i64};
use syntax::abi::{Rust, RustCall, RustIntrinsic, Abi};
use syntax::ast_util::local_def;
use syntax::attr::AttrMetaMethods;
use syntax::attr;
use syntax::codemap::Span;
use syntax::parse::token::InternedString;
use syntax::visit::Visitor;
use syntax::visit;
use syntax::{ast, ast_util, ast_map};

thread_local! {
    static TASK_LOCAL_INSN_KEY: RefCell<Option<Vec<&'static str>>> = {
        RefCell::new(None)
    }
}

pub fn with_insn_ctxt<F>(blk: F) where
    F: FnOnce(&[&'static str]),
{
    TASK_LOCAL_INSN_KEY.with(move |slot| {
        slot.borrow().as_ref().map(move |s| blk(s));
    })
}

pub fn init_insn_ctxt() {
    TASK_LOCAL_INSN_KEY.with(|slot| {
        *slot.borrow_mut() = Some(Vec::new());
    });
}

pub struct _InsnCtxt {
    _cannot_construct_outside_of_this_module: ()
}

#[unsafe_destructor]
impl Drop for _InsnCtxt {
    fn drop(&mut self) {
        TASK_LOCAL_INSN_KEY.with(|slot| {
            match slot.borrow_mut().as_mut() {
                Some(ctx) => { ctx.pop(); }
                None => {}
            }
        })
    }
}

pub fn push_ctxt(s: &'static str) -> _InsnCtxt {
    debug!("new InsnCtxt: {}", s);
    TASK_LOCAL_INSN_KEY.with(|slot| {
        match slot.borrow_mut().as_mut() {
            Some(ctx) => ctx.push(s),
            None => {}
        }
    });
    _InsnCtxt { _cannot_construct_outside_of_this_module: () }
}

pub struct StatRecorder<'a, 'tcx: 'a> {
    ccx: &'a CrateContext<'a, 'tcx>,
    name: Option<String>,
    istart: uint,
}

impl<'a, 'tcx> StatRecorder<'a, 'tcx> {
    pub fn new(ccx: &'a CrateContext<'a, 'tcx>, name: String)
               -> StatRecorder<'a, 'tcx> {
        let istart = ccx.stats().n_llvm_insns.get();
        StatRecorder {
            ccx: ccx,
            name: Some(name),
            istart: istart,
        }
    }
}

#[unsafe_destructor]
impl<'a, 'tcx> Drop for StatRecorder<'a, 'tcx> {
    fn drop(&mut self) {
        if self.ccx.sess().trans_stats() {
            let iend = self.ccx.stats().n_llvm_insns.get();
            self.ccx.stats().fn_stats.borrow_mut().push((self.name.take().unwrap(),
                                                       iend - self.istart));
            self.ccx.stats().n_fns.set(self.ccx.stats().n_fns.get() + 1);
            // Reset LLVM insn count to avoid compound costs.
            self.ccx.stats().n_llvm_insns.set(self.istart);
        }
    }
}

// only use this for foreign function ABIs and glue, use `decl_rust_fn` for Rust functions
pub fn decl_fn(ccx: &CrateContext, name: &str, cc: llvm::CallConv,
               ty: Type, output: ty::FnOutput) -> ValueRef {

    let buf = CString::new(name).unwrap();
    let llfn: ValueRef = unsafe {
        llvm::LLVMGetOrInsertFunction(ccx.llmod(), buf.as_ptr(), ty.to_ref())
    };

    // diverging functions may unwind, but can never return normally
    if output == ty::FnDiverging {
        llvm::SetFunctionAttribute(llfn, llvm::NoReturnAttribute);
    }

    if ccx.tcx().sess.opts.cg.no_redzone
        .unwrap_or(ccx.tcx().sess.target.target.options.disable_redzone) {
        llvm::SetFunctionAttribute(llfn, llvm::NoRedZoneAttribute)
    }

    llvm::SetFunctionCallConv(llfn, cc);
    // Function addresses in Rust are never significant, allowing functions to be merged.
    llvm::SetUnnamedAddr(llfn, true);

    if ccx.is_split_stack_supported() && !ccx.sess().opts.cg.no_stack_check {
        set_split_stack(llfn);
    }

    llfn
}

// only use this for foreign function ABIs and glue, use `decl_rust_fn` for Rust functions
pub fn decl_cdecl_fn(ccx: &CrateContext,
                     name: &str,
                     ty: Type,
                     output: Ty) -> ValueRef {
    decl_fn(ccx, name, llvm::CCallConv, ty, ty::FnConverging(output))
}

// only use this for foreign function ABIs and glue, use `get_extern_rust_fn` for Rust functions
pub fn get_extern_fn(ccx: &CrateContext,
                     externs: &mut ExternMap,
                     name: &str,
                     cc: llvm::CallConv,
                     ty: Type,
                     output: Ty)
                     -> ValueRef {
    match externs.get(name) {
        Some(n) => return *n,
        None => {}
    }
    let f = decl_fn(ccx, name, cc, ty, ty::FnConverging(output));
    externs.insert(name.to_string(), f);
    f
}

fn get_extern_rust_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, fn_ty: Ty<'tcx>,
                                name: &str, did: ast::DefId) -> ValueRef {
    match ccx.externs().borrow().get(name) {
        Some(n) => return *n,
        None => ()
    }

    let f = decl_rust_fn(ccx, fn_ty, name);

    let attrs = csearch::get_item_attrs(&ccx.sess().cstore, did);
    set_llvm_fn_attrs(ccx, &attrs[..], f);

    ccx.externs().borrow_mut().insert(name.to_string(), f);
    f
}

pub fn self_type_for_closure<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                       closure_id: ast::DefId,
                                       fn_ty: Ty<'tcx>)
                                       -> Ty<'tcx>
{
    let closure_kind = ccx.tcx().closure_kind(closure_id);
    match closure_kind {
        ty::FnClosureKind => {
            ty::mk_imm_rptr(ccx.tcx(), ccx.tcx().mk_region(ty::ReStatic), fn_ty)
        }
        ty::FnMutClosureKind => {
            ty::mk_mut_rptr(ccx.tcx(), ccx.tcx().mk_region(ty::ReStatic), fn_ty)
        }
        ty::FnOnceClosureKind => fn_ty
    }
}

pub fn kind_for_closure(ccx: &CrateContext, closure_id: ast::DefId) -> ty::ClosureKind {
    ccx.tcx().closure_kinds.borrow()[closure_id]
}

pub fn decl_rust_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                              fn_ty: Ty<'tcx>, name: &str) -> ValueRef {
    debug!("decl_rust_fn(fn_ty={}, name={:?})",
           fn_ty.repr(ccx.tcx()),
           name);

    let fn_ty = monomorphize::normalize_associated_type(ccx.tcx(), &fn_ty);

    debug!("decl_rust_fn: fn_ty={} (after normalized associated types)",
           fn_ty.repr(ccx.tcx()));

    let function_type; // placeholder so that the memory ownership works out ok

    let (sig, abi, env) = match fn_ty.sty {
        ty::ty_bare_fn(_, ref f) => {
            (&f.sig, f.abi, None)
        }
        ty::ty_closure(closure_did, _, substs) => {
            let typer = common::NormalizingClosureTyper::new(ccx.tcx());
            function_type = typer.closure_type(closure_did, substs);
            let self_type = self_type_for_closure(ccx, closure_did, fn_ty);
            let llenvironment_type = type_of_explicit_arg(ccx, self_type);
            debug!("decl_rust_fn: function_type={} self_type={}",
                   function_type.repr(ccx.tcx()),
                   self_type.repr(ccx.tcx()));
            (&function_type.sig, RustCall, Some(llenvironment_type))
        }
        _ => panic!("expected closure or fn")
    };

    let sig = ty::erase_late_bound_regions(ccx.tcx(), sig);
    let sig = ty::Binder(sig);

    debug!("decl_rust_fn: sig={} (after erasing regions)",
           sig.repr(ccx.tcx()));

    let llfty = type_of_rust_fn(ccx, env, &sig, abi);

    debug!("decl_rust_fn: llfty={}",
           ccx.tn().type_to_string(llfty));

    let llfn = decl_fn(ccx, name, llvm::CCallConv, llfty, sig.0.output /* (1) */);
    let attrs = get_fn_llvm_attributes(ccx, fn_ty);
    attrs.apply_llfn(llfn);

    // (1) it's ok to directly access sig.0.output because we erased all late-bound-regions above

    llfn
}

pub fn decl_internal_rust_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                       fn_ty: Ty<'tcx>, name: &str) -> ValueRef {
    let llfn = decl_rust_fn(ccx, fn_ty, name);
    llvm::SetLinkage(llfn, llvm::InternalLinkage);
    llfn
}

pub fn get_extern_const<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, did: ast::DefId,
                                  t: Ty<'tcx>) -> ValueRef {
    let name = csearch::get_symbol(&ccx.sess().cstore, did);
    let ty = type_of(ccx, t);
    match ccx.externs().borrow_mut().get(&name) {
        Some(n) => return *n,
        None => ()
    }
    unsafe {
        let buf = CString::new(name.clone()).unwrap();
        let c = llvm::LLVMAddGlobal(ccx.llmod(), ty.to_ref(), buf.as_ptr());
        // Thread-local statics in some other crate need to *always* be linked
        // against in a thread-local fashion, so we need to be sure to apply the
        // thread-local attribute locally if it was present remotely. If we
        // don't do this then linker errors can be generated where the linker
        // complains that one object files has a thread local version of the
        // symbol and another one doesn't.
        for attr in &*ty::get_attrs(ccx.tcx(), did) {
            if attr.check_name("thread_local") {
                llvm::set_thread_local(c, true);
            }
        }
        ccx.externs().borrow_mut().insert(name.to_string(), c);
        return c;
    }
}

fn require_alloc_fn<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                info_ty: Ty<'tcx>, it: LangItem) -> ast::DefId {
    match bcx.tcx().lang_items.require(it) {
        Ok(id) => id,
        Err(s) => {
            bcx.sess().fatal(&format!("allocation of `{}` {}",
                                     bcx.ty_to_string(info_ty),
                                     s)[]);
        }
    }
}

// The following malloc_raw_dyn* functions allocate a box to contain
// a given type, but with a potentially dynamic size.

pub fn malloc_raw_dyn<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                  llty_ptr: Type,
                                  info_ty: Ty<'tcx>,
                                  size: ValueRef,
                                  align: ValueRef,
                                  debug_loc: DebugLoc)
                                  -> Result<'blk, 'tcx> {
    let _icx = push_ctxt("malloc_raw_exchange");

    // Allocate space:
    let r = callee::trans_lang_call(bcx,
        require_alloc_fn(bcx, info_ty, ExchangeMallocFnLangItem),
        &[size, align],
        None,
        debug_loc);

    Result::new(r.bcx, PointerCast(r.bcx, r.val, llty_ptr))
}

// Type descriptor and type glue stuff

pub fn get_tydesc<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                            t: Ty<'tcx>) -> Rc<tydesc_info<'tcx>> {
    match ccx.tydescs().borrow().get(&t) {
        Some(inf) => return inf.clone(),
        _ => { }
    }

    ccx.stats().n_static_tydescs.set(ccx.stats().n_static_tydescs.get() + 1);
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

pub fn set_llvm_fn_attrs(ccx: &CrateContext, attrs: &[ast::Attribute], llfn: ValueRef) {
    use syntax::attr::*;
    // Set the inline hint if there is one
    match find_inline_attr(attrs) {
        InlineHint   => set_inline_hint(llfn),
        InlineAlways => set_always_inline(llfn),
        InlineNever  => set_no_inline(llfn),
        InlineNone   => { /* fallthrough */ }
    }

    for attr in attrs {
        let mut used = true;
        match &attr.name()[] {
            "no_stack_check" => unset_split_stack(llfn),
            "no_split_stack" => {
                unset_split_stack(llfn);
                ccx.sess().span_warn(attr.span,
                                     "no_split_stack is a deprecated synonym for no_stack_check");
            }
            "cold" => unsafe {
                llvm::LLVMAddFunctionAttribute(llfn,
                                               llvm::FunctionIndex as c_uint,
                                               llvm::ColdAttribute as uint64_t)
            },
            _ => used = false,
        }
        if used {
            attr::mark_used(attr);
        }
    }
}

pub fn set_always_inline(f: ValueRef) {
    llvm::SetFunctionAttribute(f, llvm::AlwaysInlineAttribute)
}

pub fn set_split_stack(f: ValueRef) {
    unsafe {
        llvm::LLVMAddFunctionAttrString(f, llvm::FunctionIndex as c_uint,
                                        "split-stack\0".as_ptr() as *const _);
    }
}

pub fn unset_split_stack(f: ValueRef) {
    unsafe {
        llvm::LLVMRemoveFunctionAttrString(f, llvm::FunctionIndex as c_uint,
                                           "split-stack\0".as_ptr() as *const _);
    }
}

// Double-check that we never ask LLVM to declare the same symbol twice. It
// silently mangles such symbols, breaking our linkage model.
pub fn note_unique_llvm_symbol(ccx: &CrateContext, sym: String) {
    if ccx.all_llvm_symbols().borrow().contains(&sym) {
        ccx.sess().bug(&format!("duplicate LLVM symbol: {}", sym)[]);
    }
    ccx.all_llvm_symbols().borrow_mut().insert(sym);
}


pub fn get_res_dtor<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                              did: ast::DefId,
                              t: Ty<'tcx>,
                              parent_id: ast::DefId,
                              substs: &Substs<'tcx>)
                              -> ValueRef {
    let _icx = push_ctxt("trans_res_dtor");
    let did = inline::maybe_instantiate_inline(ccx, did);

    if !substs.types.is_empty() {
        assert_eq!(did.krate, ast::LOCAL_CRATE);

        // Since we're in trans we don't care for any region parameters
        let substs = ccx.tcx().mk_substs(Substs::erased(substs.types.clone()));

        let (val, _, _) = monomorphize::monomorphic_fn(ccx, did, substs, None);

        val
    } else if did.krate == ast::LOCAL_CRATE {
        get_item_val(ccx, did.node)
    } else {
        let tcx = ccx.tcx();
        let name = csearch::get_symbol(&ccx.sess().cstore, did);
        let class_ty = ty::lookup_item_type(tcx, parent_id).ty.subst(tcx, substs);
        let llty = type_of_dtor(ccx, class_ty);
        let dtor_ty = ty::mk_ctor_fn(ccx.tcx(),
                                     did,
                                     &[glue::get_drop_glue_type(ccx, t)],
                                     ty::mk_nil(ccx.tcx()));
        get_extern_fn(ccx,
                      &mut *ccx.externs().borrow_mut(),
                      &name[..],
                      llvm::CCallConv,
                      llty,
                      dtor_ty)
    }
}

pub fn bin_op_to_icmp_predicate(ccx: &CrateContext, op: ast::BinOp_, signed: bool)
                                -> llvm::IntPredicate {
    match op {
        ast::BiEq => llvm::IntEQ,
        ast::BiNe => llvm::IntNE,
        ast::BiLt => if signed { llvm::IntSLT } else { llvm::IntULT },
        ast::BiLe => if signed { llvm::IntSLE } else { llvm::IntULE },
        ast::BiGt => if signed { llvm::IntSGT } else { llvm::IntUGT },
        ast::BiGe => if signed { llvm::IntSGE } else { llvm::IntUGE },
        op => {
            ccx.sess().bug(&format!("comparison_op_to_icmp_predicate: expected \
                                     comparison operator, found {:?}", op)[]);
        }
    }
}

pub fn bin_op_to_fcmp_predicate(ccx: &CrateContext, op: ast::BinOp_)
                                -> llvm::RealPredicate {
    match op {
        ast::BiEq => llvm::RealOEQ,
        ast::BiNe => llvm::RealUNE,
        ast::BiLt => llvm::RealOLT,
        ast::BiLe => llvm::RealOLE,
        ast::BiGt => llvm::RealOGT,
        ast::BiGe => llvm::RealOGE,
        op => {
            ccx.sess().bug(&format!("comparison_op_to_fcmp_predicate: expected \
                                     comparison operator, found {:?}", op)[]);
        }
    }
}

pub fn compare_scalar_types<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                        lhs: ValueRef,
                                        rhs: ValueRef,
                                        t: Ty<'tcx>,
                                        op: ast::BinOp_,
                                        debug_loc: DebugLoc)
                                        -> ValueRef {
    match t.sty {
        ty::ty_tup(ref tys) if tys.is_empty() => {
            // We don't need to do actual comparisons for nil.
            // () == () holds but () < () does not.
            match op {
                ast::BiEq | ast::BiLe | ast::BiGe => return C_bool(bcx.ccx(), true),
                ast::BiNe | ast::BiLt | ast::BiGt => return C_bool(bcx.ccx(), false),
                // refinements would be nice
                _ => bcx.sess().bug("compare_scalar_types: must be a comparison operator")
            }
        }
        ty::ty_bool | ty::ty_uint(_) | ty::ty_char => {
            ICmp(bcx, bin_op_to_icmp_predicate(bcx.ccx(), op, false), lhs, rhs, debug_loc)
        }
        ty::ty_ptr(mt) if common::type_is_sized(bcx.tcx(), mt.ty) => {
            ICmp(bcx, bin_op_to_icmp_predicate(bcx.ccx(), op, false), lhs, rhs, debug_loc)
        }
        ty::ty_int(_) => {
            ICmp(bcx, bin_op_to_icmp_predicate(bcx.ccx(), op, true), lhs, rhs, debug_loc)
        }
        ty::ty_float(_) => {
            FCmp(bcx, bin_op_to_fcmp_predicate(bcx.ccx(), op), lhs, rhs, debug_loc)
        }
        // Should never get here, because t is scalar.
        _ => bcx.sess().bug("non-scalar type passed to compare_scalar_types")
    }
}

pub fn compare_simd_types<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                      lhs: ValueRef,
                                      rhs: ValueRef,
                                      t: Ty<'tcx>,
                                      op: ast::BinOp_,
                                      debug_loc: DebugLoc)
                                      -> ValueRef {
    let signed = match t.sty {
        ty::ty_float(_) => {
            // The comparison operators for floating point vectors are challenging.
            // LLVM outputs a `< size x i1 >`, but if we perform a sign extension
            // then bitcast to a floating point vector, the result will be `-NaN`
            // for each truth value. Because of this they are unsupported.
            bcx.sess().bug("compare_simd_types: comparison operators \
                            not supported for floating point SIMD types")
        },
        ty::ty_uint(_) => false,
        ty::ty_int(_) => true,
        _ => bcx.sess().bug("compare_simd_types: invalid SIMD type"),
    };

    let cmp = bin_op_to_icmp_predicate(bcx.ccx(), op, signed);
    // LLVM outputs an `< size x i1 >`, so we need to perform a sign extension
    // to get the correctly sized type. This will compile to a single instruction
    // once the IR is converted to assembly if the SIMD instruction is supported
    // by the target architecture.
    SExt(bcx, ICmp(bcx, cmp, lhs, rhs, debug_loc), val_ty(lhs))
}

// Iterates through the elements of a structural type.
pub fn iter_structural_ty<'blk, 'tcx, F>(cx: Block<'blk, 'tcx>,
                                         av: ValueRef,
                                         t: Ty<'tcx>,
                                         mut f: F)
                                         -> Block<'blk, 'tcx> where
    F: FnMut(Block<'blk, 'tcx>, ValueRef, Ty<'tcx>) -> Block<'blk, 'tcx>,
{
    let _icx = push_ctxt("iter_structural_ty");

    fn iter_variant<'blk, 'tcx, F>(cx: Block<'blk, 'tcx>,
                                   repr: &adt::Repr<'tcx>,
                                   av: ValueRef,
                                   variant: &ty::VariantInfo<'tcx>,
                                   substs: &Substs<'tcx>,
                                   f: &mut F)
                                   -> Block<'blk, 'tcx> where
        F: FnMut(Block<'blk, 'tcx>, ValueRef, Ty<'tcx>) -> Block<'blk, 'tcx>,
    {
        let _icx = push_ctxt("iter_variant");
        let tcx = cx.tcx();
        let mut cx = cx;

        for (i, &arg) in variant.args.iter().enumerate() {
            let arg = monomorphize::apply_param_substs(tcx, substs, &arg);
            cx = f(cx, adt::trans_field_ptr(cx, repr, av, variant.disr_val, i), arg);
        }
        return cx;
    }

    let (data_ptr, info) = if common::type_is_sized(cx.tcx(), t) {
        (av, None)
    } else {
        let data = GEPi(cx, av, &[0, abi::FAT_PTR_ADDR]);
        let info = GEPi(cx, av, &[0, abi::FAT_PTR_EXTRA]);
        (Load(cx, data), Some(Load(cx, info)))
    };

    let mut cx = cx;
    match t.sty {
      ty::ty_struct(..) => {
          let repr = adt::represent_type(cx.ccx(), t);
          expr::with_field_tys(cx.tcx(), t, None, |discr, field_tys| {
              for (i, field_ty) in field_tys.iter().enumerate() {
                  let field_ty = field_ty.mt.ty;
                  let llfld_a = adt::trans_field_ptr(cx, &*repr, data_ptr, discr, i);

                  let val = if common::type_is_sized(cx.tcx(), field_ty) {
                      llfld_a
                  } else {
                      let boxed_ty = ty::mk_open(cx.tcx(), field_ty);
                      let scratch = datum::rvalue_scratch_datum(cx, boxed_ty, "__fat_ptr_iter");
                      Store(cx, llfld_a, GEPi(cx, scratch.val, &[0, abi::FAT_PTR_ADDR]));
                      Store(cx, info.unwrap(), GEPi(cx, scratch.val, &[0, abi::FAT_PTR_EXTRA]));
                      scratch.val
                  };
                  cx = f(cx, val, field_ty);
              }
          })
      }
      ty::ty_closure(def_id, _, substs) => {
          let repr = adt::represent_type(cx.ccx(), t);
          let typer = common::NormalizingClosureTyper::new(cx.tcx());
          let upvars = typer.closure_upvars(def_id, substs).unwrap();
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
      ty::ty_enum(tid, substs) => {
          let fcx = cx.fcx;
          let ccx = fcx.ccx;

          let repr = adt::represent_type(ccx, t);
          let variants = ty::enum_variants(ccx.tcx(), tid);
          let n_variants = (*variants).len();

          // NB: we must hit the discriminant first so that structural
          // comparison know not to proceed when the discriminants differ.

          match adt::trans_switch(cx, &*repr, av) {
              (_match::Single, None) => {
                  cx = iter_variant(cx, &*repr, av, &*(*variants)[0],
                                    substs, &mut f);
              }
              (_match::Switch, Some(lldiscrim_a)) => {
                  cx = f(cx, lldiscrim_a, cx.tcx().types.int);
                  let unr_cx = fcx.new_temp_block("enum-iter-unr");
                  Unreachable(unr_cx);
                  let llswitch = Switch(cx, lldiscrim_a, unr_cx.llbb,
                                        n_variants);
                  let next_cx = fcx.new_temp_block("enum-iter-next");

                  for variant in &(*variants) {
                      let variant_cx =
                          fcx.new_temp_block(
                              &format!("enum-iter-variant-{}",
                                      &variant.disr_val.to_string()[])
                              []);
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
                                       &mut f);
                      Br(variant_cx, next_cx.llbb, DebugLoc::None);
                  }
                  cx = next_cx;
              }
              _ => ccx.sess().unimpl("value from adt::trans_switch \
                                      in iter_structural_ty")
          }
      }
      _ => {
          cx.sess().unimpl(&format!("type in iter_structural_ty: {}",
                                   ty_to_string(cx.tcx(), t))[])
      }
    }
    return cx;
}

pub fn cast_shift_expr_rhs(cx: Block,
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

pub fn cast_shift_rhs<F, G>(op: ast::BinOp,
                            lhs: ValueRef,
                            rhs: ValueRef,
                            trunc: F,
                            zext: G)
                            -> ValueRef where
    F: FnOnce(ValueRef, Type) -> ValueRef,
    G: FnOnce(ValueRef, Type) -> ValueRef,
{
    // Shifts may have any size int on the rhs
    if ast_util::is_shift_binop(op.node) {
        let mut rhs_llty = val_ty(rhs);
        let mut lhs_llty = val_ty(lhs);
        if rhs_llty.kind() == Vector { rhs_llty = rhs_llty.element_type() }
        if lhs_llty.kind() == Vector { lhs_llty = lhs_llty.element_type() }
        let rhs_sz = rhs_llty.int_width();
        let lhs_sz = lhs_llty.int_width();
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

pub fn fail_if_zero_or_overflows<'blk, 'tcx>(
                                cx: Block<'blk, 'tcx>,
                                call_info: NodeIdAndSpan,
                                divrem: ast::BinOp,
                                lhs: ValueRef,
                                rhs: ValueRef,
                                rhs_t: Ty<'tcx>)
                                -> Block<'blk, 'tcx> {
    let (zero_text, overflow_text) = if divrem.node == ast::BiDiv {
        ("attempted to divide by zero",
         "attempted to divide with overflow")
    } else {
        ("attempted remainder with a divisor of zero",
         "attempted remainder with overflow")
    };
    let debug_loc = call_info.debug_loc();

    let (is_zero, is_signed) = match rhs_t.sty {
        ty::ty_int(t) => {
            let zero = C_integral(Type::int_from_ty(cx.ccx(), t), 0u64, false);
            (ICmp(cx, llvm::IntEQ, rhs, zero, debug_loc), true)
        }
        ty::ty_uint(t) => {
            let zero = C_integral(Type::uint_from_ty(cx.ccx(), t), 0u64, false);
            (ICmp(cx, llvm::IntEQ, rhs, zero, debug_loc), false)
        }
        _ => {
            cx.sess().bug(&format!("fail-if-zero on unexpected type: {}",
                                  ty_to_string(cx.tcx(), rhs_t))[]);
        }
    };
    let bcx = with_cond(cx, is_zero, |bcx| {
        controlflow::trans_fail(bcx, call_info, InternedString::new(zero_text))
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
        let (llty, min) = match rhs_t.sty {
            ty::ty_int(t) => {
                let llty = Type::int_from_ty(cx.ccx(), t);
                let min = match t {
                    ast::TyIs(_) if llty == Type::i32(cx.ccx()) => i32::MIN as u64,
                    ast::TyIs(_) => i64::MIN as u64,
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
                             C_integral(llty, -1, false), debug_loc);
        with_cond(bcx, minus_one, |bcx| {
            let is_min = ICmp(bcx, llvm::IntEQ, lhs,
                              C_integral(llty, min, true), debug_loc);
            with_cond(bcx, is_min, |bcx| {
                controlflow::trans_fail(bcx,
                                        call_info,
                                        InternedString::new(overflow_text))
            })
        })
    } else {
        bcx
    }
}

pub fn trans_external_path<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                     did: ast::DefId, t: Ty<'tcx>) -> ValueRef {
    let name = csearch::get_symbol(&ccx.sess().cstore, did);
    match t.sty {
        ty::ty_bare_fn(_, ref fn_ty) => {
            match ccx.sess().target.target.adjust_abi(fn_ty.abi) {
                Rust | RustCall => {
                    get_extern_rust_fn(ccx, t, &name[..], did)
                }
                RustIntrinsic => {
                    ccx.sess().bug("unexpected intrinsic in trans_external_path")
                }
                _ => {
                    foreign::register_foreign_item_fn(ccx, fn_ty.abi, t,
                                                      &name[..])
                }
            }
        }
        _ => {
            get_extern_const(ccx, did, t)
        }
    }
}

pub fn invoke<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                          llfn: ValueRef,
                          llargs: &[ValueRef],
                          fn_ty: Ty<'tcx>,
                          debug_loc: DebugLoc)
                          -> (ValueRef, Block<'blk, 'tcx>) {
    let _icx = push_ctxt("invoke_");
    if bcx.unreachable.get() {
        return (C_null(Type::i8(bcx.ccx())), bcx);
    }

    let attributes = get_fn_llvm_attributes(bcx.ccx(), fn_ty);

    match bcx.opt_node_id {
        None => {
            debug!("invoke at ???");
        }
        Some(id) => {
            debug!("invoke at {}", bcx.tcx().map.node_to_string(id));
        }
    }

    if need_invoke(bcx) {
        debug!("invoking {} at {:?}", bcx.val_to_string(llfn), bcx.llbb);
        for &llarg in llargs {
            debug!("arg: {}", bcx.val_to_string(llarg));
        }
        let normal_bcx = bcx.fcx.new_temp_block("normal-return");
        let landing_pad = bcx.fcx.get_landing_pad();

        let llresult = Invoke(bcx,
                              llfn,
                              &llargs[..],
                              normal_bcx.llbb,
                              landing_pad,
                              Some(attributes),
                              debug_loc);
        return (llresult, normal_bcx);
    } else {
        debug!("calling {} at {:?}", bcx.val_to_string(llfn), bcx.llbb);
        for &llarg in llargs {
            debug!("arg: {}", bcx.val_to_string(llarg));
        }

        let llresult = Call(bcx,
                            llfn,
                            &llargs[..],
                            Some(attributes),
                            debug_loc);
        return (llresult, bcx);
    }
}

pub fn need_invoke(bcx: Block) -> bool {
    if bcx.sess().no_landing_pads() {
        return false;
    }

    // Avoid using invoke if we are already inside a landing pad.
    if bcx.is_lpad {
        return false;
    }

    bcx.fcx.needs_invoke()
}

pub fn load_if_immediate<'blk, 'tcx>(cx: Block<'blk, 'tcx>,
                                     v: ValueRef, t: Ty<'tcx>) -> ValueRef {
    let _icx = push_ctxt("load_if_immediate");
    if type_is_immediate(cx.ccx(), t) { return load_ty(cx, v, t); }
    return v;
}

/// Helper for loading values from memory. Does the necessary conversion if the in-memory type
/// differs from the type used for SSA values. Also handles various special cases where the type
/// gives us better information about what we are loading.
pub fn load_ty<'blk, 'tcx>(cx: Block<'blk, 'tcx>,
                           ptr: ValueRef, t: Ty<'tcx>) -> ValueRef {
    if type_is_zero_size(cx.ccx(), t) {
        C_undef(type_of::type_of(cx.ccx(), t))
    } else if type_is_immediate(cx.ccx(), t) && type_of::type_of(cx.ccx(), t).is_aggregate() {
        // We want to pass small aggregates as immediate values, but using an aggregate LLVM type
        // for this leads to bad optimizations, so its arg type is an appropriately sized integer
        // and we have to convert it
        Load(cx, BitCast(cx, ptr, type_of::arg_type_of(cx.ccx(), t).ptr_to()))
    } else {
        unsafe {
            let global = llvm::LLVMIsAGlobalVariable(ptr);
            if !global.is_null() && llvm::LLVMIsGlobalConstant(global) == llvm::True {
                let val = llvm::LLVMGetInitializer(global);
                if !val.is_null() {
                    // This could go into its own function, for DRY.
                    // (something like "pre-store packing/post-load unpacking")
                    if ty::type_is_bool(t) {
                        return Trunc(cx, val, Type::i1(cx.ccx()));
                    } else {
                        return val;
                    }
                }
            }
        }
        if ty::type_is_bool(t) {
            Trunc(cx, LoadRangeAssert(cx, ptr, 0, 2, llvm::False), Type::i1(cx.ccx()))
        } else if ty::type_is_char(t) {
            // a char is a Unicode codepoint, and so takes values from 0
            // to 0x10FFFF inclusive only.
            LoadRangeAssert(cx, ptr, 0, 0x10FFFF + 1, llvm::False)
        } else if (ty::type_is_region_ptr(t) || ty::type_is_unique(t))
                  && !common::type_is_fat_ptr(cx.tcx(), t) {
            LoadNonNull(cx, ptr)
        } else {
            Load(cx, ptr)
        }
    }
}

/// Helper for storing values in memory. Does the necessary conversion if the in-memory type
/// differs from the type used for SSA values.
pub fn store_ty<'blk, 'tcx>(cx: Block<'blk, 'tcx>, v: ValueRef, dst: ValueRef, t: Ty<'tcx>) {
    if ty::type_is_bool(t) {
        Store(cx, ZExt(cx, v, Type::i8(cx.ccx())), dst);
    } else if type_is_immediate(cx.ccx(), t) && type_of::type_of(cx.ccx(), t).is_aggregate() {
        // We want to pass small aggregates as immediate values, but using an aggregate LLVM type
        // for this leads to bad optimizations, so its arg type is an appropriately sized integer
        // and we have to convert it
        Store(cx, v, BitCast(cx, dst, type_of::arg_type_of(cx.ccx(), t).ptr_to()));
    } else {
        Store(cx, v, dst);
    }
}

pub fn init_local<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, local: &ast::Local)
                              -> Block<'blk, 'tcx> {
    debug!("init_local(bcx={}, local.id={})", bcx.to_str(), local.id);
    let _indenter = indenter();
    let _icx = push_ctxt("init_local");
    _match::store_local(bcx, local)
}

pub fn raw_block<'blk, 'tcx>(fcx: &'blk FunctionContext<'blk, 'tcx>,
                             is_lpad: bool,
                             llbb: BasicBlockRef)
                             -> Block<'blk, 'tcx> {
    common::BlockS::new(llbb, is_lpad, None, fcx)
}

pub fn with_cond<'blk, 'tcx, F>(bcx: Block<'blk, 'tcx>,
                                val: ValueRef,
                                f: F)
                                -> Block<'blk, 'tcx> where
    F: FnOnce(Block<'blk, 'tcx>) -> Block<'blk, 'tcx>,
{
    let _icx = push_ctxt("with_cond");

    if bcx.unreachable.get() ||
            (common::is_const(val) && common::const_to_uint(val) == 0) {
        return bcx;
    }

    let fcx = bcx.fcx;
    let next_cx = fcx.new_temp_block("next");
    let cond_cx = fcx.new_temp_block("cond");
    CondBr(bcx, val, cond_cx.llbb, next_cx.llbb, DebugLoc::None);
    let after_cx = f(cond_cx);
    if !after_cx.terminated.get() {
        Br(after_cx, next_cx.llbb, DebugLoc::None);
    }
    next_cx
}

pub fn call_lifetime_start(cx: Block, ptr: ValueRef) {
    if cx.sess().opts.optimize == config::No {
        return;
    }

    let _icx = push_ctxt("lifetime_start");
    let ccx = cx.ccx();

    let llsize = C_u64(ccx, machine::llsize_of_alloc(ccx, val_ty(ptr).element_type()));
    let ptr = PointerCast(cx, ptr, Type::i8p(ccx));
    let lifetime_start = ccx.get_intrinsic(&"llvm.lifetime.start");
    Call(cx, lifetime_start, &[llsize, ptr], None, DebugLoc::None);
}

pub fn call_lifetime_end(cx: Block, ptr: ValueRef) {
    if cx.sess().opts.optimize == config::No {
        return;
    }

    let _icx = push_ctxt("lifetime_end");
    let ccx = cx.ccx();

    let llsize = C_u64(ccx, machine::llsize_of_alloc(ccx, val_ty(ptr).element_type()));
    let ptr = PointerCast(cx, ptr, Type::i8p(ccx));
    let lifetime_end = ccx.get_intrinsic(&"llvm.lifetime.end");
    Call(cx, lifetime_end, &[llsize, ptr], None, DebugLoc::None);
}

pub fn call_memcpy(cx: Block, dst: ValueRef, src: ValueRef, n_bytes: ValueRef, align: u32) {
    let _icx = push_ctxt("call_memcpy");
    let ccx = cx.ccx();
    let key = match &ccx.sess().target.target.target_pointer_width[] {
        "32" => "llvm.memcpy.p0i8.p0i8.i32",
        "64" => "llvm.memcpy.p0i8.p0i8.i64",
        tws => panic!("Unsupported target word size for memcpy: {}", tws),
    };
    let memcpy = ccx.get_intrinsic(&key);
    let src_ptr = PointerCast(cx, src, Type::i8p(ccx));
    let dst_ptr = PointerCast(cx, dst, Type::i8p(ccx));
    let size = IntCast(cx, n_bytes, ccx.int_type());
    let align = C_i32(ccx, align as i32);
    let volatile = C_bool(ccx, false);
    Call(cx, memcpy, &[dst_ptr, src_ptr, size, align, volatile], None, DebugLoc::None);
}

pub fn memcpy_ty<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                             dst: ValueRef, src: ValueRef,
                             t: Ty<'tcx>) {
    let _icx = push_ctxt("memcpy_ty");
    let ccx = bcx.ccx();
    if ty::type_is_structural(t) {
        let llty = type_of::type_of(ccx, t);
        let llsz = llsize_of(ccx, llty);
        let llalign = type_of::align_of(ccx, t);
        call_memcpy(bcx, dst, src, llsz, llalign as u32);
    } else {
        store_ty(bcx, load_ty(bcx, src, t), dst, t);
    }
}

pub fn zero_mem<'blk, 'tcx>(cx: Block<'blk, 'tcx>, llptr: ValueRef, t: Ty<'tcx>) {
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
fn memzero<'a, 'tcx>(b: &Builder<'a, 'tcx>, llptr: ValueRef, ty: Ty<'tcx>) {
    let _icx = push_ctxt("memzero");
    let ccx = b.ccx;

    let llty = type_of::type_of(ccx, ty);

    let intrinsic_key = match &ccx.sess().target.target.target_pointer_width[] {
        "32" => "llvm.memset.p0i8.i32",
        "64" => "llvm.memset.p0i8.i64",
        tws => panic!("Unsupported target word size for memset: {}", tws),
    };

    let llintrinsicfn = ccx.get_intrinsic(&intrinsic_key);
    let llptr = b.pointercast(llptr, Type::i8(ccx).ptr_to());
    let llzeroval = C_u8(ccx, 0);
    let size = machine::llsize_of(ccx, llty);
    let align = C_i32(ccx, type_of::align_of(ccx, ty) as i32);
    let volatile = C_bool(ccx, false);
    b.call(llintrinsicfn, &[llptr, llzeroval, size, align, volatile], None);
}

pub fn alloc_ty<'blk, 'tcx>(bcx: Block<'blk, 'tcx>, t: Ty<'tcx>, name: &str) -> ValueRef {
    let _icx = push_ctxt("alloc_ty");
    let ccx = bcx.ccx();
    let ty = type_of::type_of(ccx, t);
    assert!(!ty::type_has_params(t));
    let val = alloca(bcx, ty, name);
    return val;
}

pub fn alloca(cx: Block, ty: Type, name: &str) -> ValueRef {
    let p = alloca_no_lifetime(cx, ty, name);
    call_lifetime_start(cx, p);
    p
}

pub fn alloca_no_lifetime(cx: Block, ty: Type, name: &str) -> ValueRef {
    let _icx = push_ctxt("alloca");
    if cx.unreachable.get() {
        unsafe {
            return llvm::LLVMGetUndef(ty.ptr_to().to_ref());
        }
    }
    debuginfo::clear_source_location(cx.fcx);
    Alloca(cx, ty, name)
}

pub fn alloca_zeroed<'blk, 'tcx>(cx: Block<'blk, 'tcx>, ty: Ty<'tcx>,
                                 name: &str) -> ValueRef {
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

// Creates the alloca slot which holds the pointer to the slot for the final return value
pub fn make_return_slot_pointer<'a, 'tcx>(fcx: &FunctionContext<'a, 'tcx>,
                                          output_type: Ty<'tcx>) -> ValueRef {
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

struct FindNestedReturn {
    found: bool,
}

impl FindNestedReturn {
    fn new() -> FindNestedReturn {
        FindNestedReturn { found: false }
    }
}

impl<'v> Visitor<'v> for FindNestedReturn {
    fn visit_expr(&mut self, e: &ast::Expr) {
        match e.node {
            ast::ExprRet(..) => {
                self.found = true;
            }
            _ => visit::walk_expr(self, e)
        }
    }
}

fn build_cfg(tcx: &ty::ctxt, id: ast::NodeId) -> (ast::NodeId, Option<cfg::CFG>) {
    let blk = match tcx.map.find(id) {
        Some(ast_map::NodeItem(i)) => {
            match i.node {
                ast::ItemFn(_, _, _, _, ref blk) => {
                    blk
                }
                _ => tcx.sess.bug("unexpected item variant in has_nested_returns")
            }
        }
        Some(ast_map::NodeTraitItem(trait_method)) => {
            match *trait_method {
                ast::ProvidedMethod(ref m) => {
                    match m.node {
                        ast::MethDecl(_, _, _, _, _, _, ref blk, _) => {
                            blk
                        }
                        ast::MethMac(_) => tcx.sess.bug("unexpanded macro")
                    }
                }
                ast::RequiredMethod(_) => {
                    tcx.sess.bug("unexpected variant: required trait method \
                                  in has_nested_returns")
                }
                ast::TypeTraitItem(_) => {
                    tcx.sess.bug("unexpected variant: type trait item in \
                                  has_nested_returns")
                }
            }
        }
        Some(ast_map::NodeImplItem(ii)) => {
            match *ii {
                ast::MethodImplItem(ref m) => {
                    match m.node {
                        ast::MethDecl(_, _, _, _, _, _, ref blk, _) => {
                            blk
                        }
                        ast::MethMac(_) => tcx.sess.bug("unexpanded macro")
                    }
                }
                ast::TypeImplItem(_) => {
                    tcx.sess.bug("unexpected variant: type impl item in \
                                  has_nested_returns")
                }
            }
        }
        Some(ast_map::NodeExpr(e)) => {
            match e.node {
                ast::ExprClosure(_, _, ref blk) => {
                    blk
                }
                _ => tcx.sess.bug("unexpected expr variant in has_nested_returns")
            }
        }
        Some(ast_map::NodeVariant(..)) |
        Some(ast_map::NodeStructCtor(..)) => return (ast::DUMMY_NODE_ID, None),

        // glue, shims, etc
        None if id == ast::DUMMY_NODE_ID => return (ast::DUMMY_NODE_ID, None),

        _ => tcx.sess.bug(&format!("unexpected variant in has_nested_returns: {}",
                                   tcx.map.path_to_string(id)))
    };

    (blk.id, Some(cfg::CFG::new(tcx, &**blk)))
}

// Checks for the presence of "nested returns" in a function.
// Nested returns are when the inner expression of a return expression
// (the 'expr' in 'return expr') contains a return expression. Only cases
// where the outer return is actually reachable are considered. Implicit
// returns from the end of blocks are considered as well.
//
// This check is needed to handle the case where the inner expression is
// part of a larger expression that may have already partially-filled the
// return slot alloca. This can cause errors related to clean-up due to
// the clobbering of the existing value in the return slot.
fn has_nested_returns(tcx: &ty::ctxt, cfg: &cfg::CFG, blk_id: ast::NodeId) -> bool {
    for n in cfg.graph.depth_traverse(cfg.entry) {
        match tcx.map.find(n.id) {
            Some(ast_map::NodeExpr(ex)) => {
                if let ast::ExprRet(Some(ref ret_expr)) = ex.node {
                    let mut visitor = FindNestedReturn::new();
                    visit::walk_expr(&mut visitor, &**ret_expr);
                    if visitor.found {
                        return true;
                    }
                }
            }
            Some(ast_map::NodeBlock(blk)) if blk.id == blk_id => {
                let mut visitor = FindNestedReturn::new();
                visit::walk_expr_opt(&mut visitor, &blk.expr);
                if visitor.found {
                    return true;
                }
            }
            _ => {}
        }
    }

    return false;
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
pub fn new_fn_ctxt<'a, 'tcx>(ccx: &'a CrateContext<'a, 'tcx>,
                             llfndecl: ValueRef,
                             id: ast::NodeId,
                             has_env: bool,
                             output_type: ty::FnOutput<'tcx>,
                             param_substs: &'tcx Substs<'tcx>,
                             sp: Option<Span>,
                             block_arena: &'a TypedArena<common::BlockS<'a, 'tcx>>)
                             -> FunctionContext<'a, 'tcx> {
    common::validate_substs(param_substs);

    debug!("new_fn_ctxt(path={}, id={}, param_substs={})",
           if id == -1 {
               "".to_string()
           } else {
               ccx.tcx().map.path_to_string(id).to_string()
           },
           id, param_substs.repr(ccx.tcx()));

    let uses_outptr = match output_type {
        ty::FnConverging(output_type) => {
            let substd_output_type =
                monomorphize::apply_param_substs(ccx.tcx(), param_substs, &output_type);
            type_of::return_uses_outptr(ccx, substd_output_type)
        }
        ty::FnDiverging => false
    };
    let debug_context = debuginfo::create_function_debug_context(ccx, id, param_substs, llfndecl);
    let (blk_id, cfg) = build_cfg(ccx.tcx(), id);
    let nested_returns = if let Some(ref cfg) = cfg {
        has_nested_returns(ccx.tcx(), cfg, blk_id)
    } else {
        false
    };

    let mut fcx = FunctionContext {
          llfn: llfndecl,
          llenv: None,
          llretslotptr: Cell::new(None),
          param_env: ty::empty_parameter_environment(ccx.tcx()),
          alloca_insert_pt: Cell::new(None),
          llreturn: Cell::new(None),
          needs_ret_allocas: nested_returns,
          personality: Cell::new(None),
          caller_expects_out_pointer: uses_outptr,
          lllocals: RefCell::new(NodeMap()),
          llupvars: RefCell::new(NodeMap()),
          id: id,
          param_substs: param_substs,
          span: sp,
          block_arena: block_arena,
          ccx: ccx,
          debug_context: debug_context,
          scopes: RefCell::new(Vec::new()),
          cfg: cfg
    };

    if has_env {
        fcx.llenv = Some(get_param(fcx.llfn, fcx.env_arg_pos() as c_uint))
    }

    fcx
}

/// Performs setup on a newly created function, creating the entry scope block
/// and allocating space for the return pointer.
pub fn init_function<'a, 'tcx>(fcx: &'a FunctionContext<'a, 'tcx>,
                               skip_retptr: bool,
                               output: ty::FnOutput<'tcx>)
                               -> Block<'a, 'tcx> {
    let entry_bcx = fcx.new_temp_block("entry-block");

    // Use a dummy instruction as the insertion point for all allocas.
    // This is later removed in FunctionContext::cleanup.
    fcx.alloca_insert_pt.set(Some(unsafe {
        Load(entry_bcx, C_null(Type::i8p(fcx.ccx)));
        llvm::LLVMGetFirstInstruction(entry_bcx.llbb)
    }));

    if let ty::FnConverging(output_type) = output {
        // This shouldn't need to recompute the return type,
        // as new_fn_ctxt did it already.
        let substd_output_type = fcx.monomorphize(&output_type);
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
    }

    entry_bcx
}

// NB: must keep 4 fns in sync:
//
//  - type_of_fn
//  - create_datums_for_fn_args.
//  - new_fn_ctxt
//  - trans_args

pub fn arg_kind<'a, 'tcx>(cx: &FunctionContext<'a, 'tcx>, t: Ty<'tcx>)
                          -> datum::Rvalue {
    use trans::datum::{ByRef, ByValue};

    datum::Rvalue {
        mode: if arg_is_indirect(cx.ccx, t) { ByRef } else { ByValue }
    }
}

// work around bizarre resolve errors
type RvalueDatum<'tcx> = datum::Datum<'tcx, datum::Rvalue>;

// create_datums_for_fn_args: creates rvalue datums for each of the
// incoming function arguments. These will later be stored into
// appropriate lvalue datums.
pub fn create_datums_for_fn_args<'a, 'tcx>(fcx: &FunctionContext<'a, 'tcx>,
                                           arg_tys: &[Ty<'tcx>])
                                           -> Vec<RvalueDatum<'tcx>> {
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
fn create_datums_for_fn_args_under_call_abi<'blk, 'tcx>(
        mut bcx: Block<'blk, 'tcx>,
        arg_scope: cleanup::CustomScopeIndex,
        arg_tys: &[Ty<'tcx>])
        -> Vec<RvalueDatum<'tcx>> {
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
        match arg_ty.sty {
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
                            let lldest = GEPi(bcx, llval, &[0, j]);
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
            _ => {
                bcx.tcx().sess.bug("last argument of a function with \
                                    `rust-call` ABI isn't a tuple?!")
            }
        };

    }

    result
}

fn copy_args_to_allocas<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                    arg_scope: cleanup::CustomScopeIndex,
                                    args: &[ast::Arg],
                                    arg_datums: Vec<RvalueDatum<'tcx>>)
                                    -> Block<'blk, 'tcx> {
    debug!("copy_args_to_allocas");

    let _icx = push_ctxt("copy_args_to_allocas");
    let mut bcx = bcx;

    let arg_scope_id = cleanup::CustomScope(arg_scope);

    for (i, arg_datum) in arg_datums.into_iter().enumerate() {
        // For certain mode/type combinations, the raw llarg values are passed
        // by value.  However, within the fn body itself, we want to always
        // have all locals and arguments be by-ref so that we can cancel the
        // cleanup and for better interaction with LLVM's debug info.  So, if
        // the argument would be passed by value, we store it into an alloca.
        // This alloca should be optimized away by LLVM's mem-to-reg pass in
        // the event it's not truly needed.

        bcx = _match::store_arg(bcx, &*args[i].pat, arg_datum, arg_scope_id);
        debuginfo::create_argument_metadata(bcx, &args[i]);
    }

    bcx
}

fn copy_closure_args_to_allocas<'blk, 'tcx>(mut bcx: Block<'blk, 'tcx>,
                                            arg_scope: cleanup::CustomScopeIndex,
                                            args: &[ast::Arg],
                                            arg_datums: Vec<RvalueDatum<'tcx>>,
                                            monomorphized_arg_types: &[Ty<'tcx>])
                                            -> Block<'blk, 'tcx> {
    let _icx = push_ctxt("copy_closure_args_to_allocas");
    let arg_scope_id = cleanup::CustomScope(arg_scope);

    assert_eq!(arg_datums.len(), 1);

    let arg_datum = arg_datums.into_iter().next().unwrap();

    // Untuple the rest of the arguments.
    let tuple_datum =
        unpack_datum!(bcx,
                      arg_datum.to_lvalue_datum_in_scope(bcx,
                                                         "argtuple",
                                                         arg_scope_id));
    let untupled_arg_types = match monomorphized_arg_types[0].sty {
        ty::ty_tup(ref types) => &types[..],
        _ => {
            bcx.tcx().sess.span_bug(args[0].pat.span,
                                    "first arg to `rust-call` ABI function \
                                     wasn't a tuple?!")
        }
    };
    for j in 0..args.len() {
        let tuple_element_type = untupled_arg_types[j];
        let tuple_element_datum =
            tuple_datum.get_element(bcx,
                                    tuple_element_type,
                                    |llval| GEPi(bcx, llval, &[0, j]));
        let tuple_element_datum = tuple_element_datum.to_expr_datum();
        let tuple_element_datum =
            unpack_datum!(bcx,
                          tuple_element_datum.to_rvalue_datum(bcx,
                                                              "arg"));
        bcx = _match::store_arg(bcx,
                                &*args[j].pat,
                                tuple_element_datum,
                                arg_scope_id);

        debuginfo::create_argument_metadata(bcx, &args[j]);
    }

    bcx
}

// Ties up the llstaticallocas -> llloadenv -> lltop edges,
// and builds the return block.
pub fn finish_fn<'blk, 'tcx>(fcx: &'blk FunctionContext<'blk, 'tcx>,
                             last_bcx: Block<'blk, 'tcx>,
                             retty: ty::FnOutput<'tcx>,
                             ret_debug_loc: DebugLoc) {
    let _icx = push_ctxt("finish_fn");

    let ret_cx = match fcx.llreturn.get() {
        Some(llreturn) => {
            if !last_bcx.terminated.get() {
                Br(last_bcx, llreturn, DebugLoc::None);
            }
            raw_block(fcx, false, llreturn)
        }
        None => last_bcx
    };

    // This shouldn't need to recompute the return type,
    // as new_fn_ctxt did it already.
    let substd_retty = fcx.monomorphize(&retty);
    build_return_block(fcx, ret_cx, substd_retty, ret_debug_loc);

    debuginfo::clear_source_location(fcx);
    fcx.cleanup();
}

// Builds the return block for a function.
pub fn build_return_block<'blk, 'tcx>(fcx: &FunctionContext<'blk, 'tcx>,
                                      ret_cx: Block<'blk, 'tcx>,
                                      retty: ty::FnOutput<'tcx>,
                                      ret_debug_location: DebugLoc) {
    if fcx.llretslotptr.get().is_none() ||
       (!fcx.needs_ret_allocas && fcx.caller_expects_out_pointer) {
        return RetVoid(ret_cx, ret_debug_location);
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

            let retval = if retty == ty::FnConverging(fcx.ccx.tcx().types.bool) {
                Trunc(ret_cx, retval, Type::i1(fcx.ccx))
            } else {
                retval
            };

            if fcx.caller_expects_out_pointer {
                if let ty::FnConverging(retty) = retty {
                    store_ty(ret_cx, retval, get_param(fcx.llfn, 0), retty);
                }
                RetVoid(ret_cx, ret_debug_location)
            } else {
                Ret(ret_cx, retval, ret_debug_location)
            }
        }
        // Otherwise, copy the return value to the ret slot
        None => match retty {
            ty::FnConverging(retty) => {
                if fcx.caller_expects_out_pointer {
                    memcpy_ty(ret_cx, get_param(fcx.llfn, 0), retslot, retty);
                    RetVoid(ret_cx, ret_debug_location)
                } else {
                    Ret(ret_cx, load_ty(ret_cx, retslot, retty), ret_debug_location)
                }
            }
            ty::FnDiverging => {
                if fcx.caller_expects_out_pointer {
                    RetVoid(ret_cx, ret_debug_location)
                } else {
                    Ret(ret_cx, C_undef(Type::nil(fcx.ccx)), ret_debug_location)
                }
            }
        }
    }
}

// trans_closure: Builds an LLVM function out of a source function.
// If the function closes over its environment a closure will be
// returned.
pub fn trans_closure<'a, 'b, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                   decl: &ast::FnDecl,
                                   body: &ast::Block,
                                   llfndecl: ValueRef,
                                   param_substs: &'tcx Substs<'tcx>,
                                   fn_ast_id: ast::NodeId,
                                   _attributes: &[ast::Attribute],
                                   output_type: ty::FnOutput<'tcx>,
                                   abi: Abi,
                                   closure_env: closure::ClosureEnv<'b>) {
    ccx.stats().n_closures.set(ccx.stats().n_closures.get() + 1);

    let _icx = push_ctxt("trans_closure");
    set_uwtable(llfndecl);

    debug!("trans_closure(..., param_substs={})",
           param_substs.repr(ccx.tcx()));

    let has_env = match closure_env {
        closure::ClosureEnv::Closure(_) => true,
        closure::ClosureEnv::NotClosure => false,
    };

    let (arena, fcx): (TypedArena<_>, FunctionContext);
    arena = TypedArena::new();
    fcx = new_fn_ctxt(ccx,
                      llfndecl,
                      fn_ast_id,
                      has_env,
                      output_type,
                      param_substs,
                      Some(body.span),
                      &arena);
    let mut bcx = init_function(&fcx, false, output_type);

    // cleanup scope for the incoming arguments
    let fn_cleanup_debug_loc =
        debuginfo::get_cleanup_debug_loc_for_ast_node(ccx, fn_ast_id, body.span, true);
    let arg_scope = fcx.push_custom_cleanup_scope_with_debug_loc(fn_cleanup_debug_loc);

    let block_ty = node_id_type(bcx, body.id);

    // Set up arguments to the function.
    let monomorphized_arg_types =
        decl.inputs.iter()
                   .map(|arg| node_id_type(bcx, arg.id))
                   .collect::<Vec<_>>();
    let monomorphized_arg_types = match closure_env {
        closure::ClosureEnv::NotClosure => {
            monomorphized_arg_types
        }

        // Tuple up closure argument types for the "rust-call" ABI.
        closure::ClosureEnv::Closure(_) => {
            vec![ty::mk_tup(ccx.tcx(), monomorphized_arg_types)]
        }
    };
    for monomorphized_arg_type in &monomorphized_arg_types {
        debug!("trans_closure: monomorphized_arg_type: {}",
               ty_to_string(ccx.tcx(), *monomorphized_arg_type));
    }
    debug!("trans_closure: function lltype: {}",
           bcx.fcx.ccx.tn().val_to_string(bcx.fcx.llfn));

    let arg_datums = if abi != RustCall {
        create_datums_for_fn_args(&fcx,
                                  &monomorphized_arg_types[..])
    } else {
        create_datums_for_fn_args_under_call_abi(
            bcx,
            arg_scope,
            &monomorphized_arg_types[..])
    };

    bcx = match closure_env {
        closure::ClosureEnv::NotClosure => {
            copy_args_to_allocas(bcx,
                                 arg_scope,
                                 &decl.inputs[],
                                 arg_datums)
        }
        closure::ClosureEnv::Closure(_) => {
            copy_closure_args_to_allocas(
                bcx,
                arg_scope,
                &decl.inputs[],
                arg_datums,
                &monomorphized_arg_types[..])
        }
    };

    bcx = closure_env.load(bcx, cleanup::CustomScope(arg_scope));

    // Up until here, IR instructions for this function have explicitly not been annotated with
    // source code location, so we don't step into call setup code. From here on, source location
    // emitting should be enabled.
    debuginfo::start_emitting_source_locations(&fcx);

    let dest = match fcx.llretslotptr.get() {
        Some(_) => expr::SaveIn(fcx.get_ret_slot(bcx, ty::FnConverging(block_ty), "iret_slot")),
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
            Br(bcx, fcx.return_exit_block(), DebugLoc::None);
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
        if let Some(llreturn) = llreturn {
            llvm::LLVMMoveBasicBlockAfter(llreturn, bcx.llbb);
        }
    }

    let ret_debug_loc = DebugLoc::At(fn_cleanup_debug_loc.id,
                                     fn_cleanup_debug_loc.span);

    // Insert the mandatory first few basic blocks before lltop.
    finish_fn(&fcx, bcx, output_type, ret_debug_loc);
}

// trans_fn: creates an LLVM function corresponding to a source language
// function.
pub fn trans_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                          decl: &ast::FnDecl,
                          body: &ast::Block,
                          llfndecl: ValueRef,
                          param_substs: &'tcx Substs<'tcx>,
                          id: ast::NodeId,
                          attrs: &[ast::Attribute]) {
    let _s = StatRecorder::new(ccx, ccx.tcx().map.path_to_string(id).to_string());
    debug!("trans_fn(param_substs={})", param_substs.repr(ccx.tcx()));
    let _icx = push_ctxt("trans_fn");
    let fn_ty = ty::node_id_to_type(ccx.tcx(), id);
    let output_type = ty::erase_late_bound_regions(ccx.tcx(), &ty::ty_fn_ret(fn_ty));
    let abi = ty::ty_fn_abi(fn_ty);
    trans_closure(ccx,
                  decl,
                  body,
                  llfndecl,
                  param_substs,
                  id,
                  attrs,
                  output_type,
                  abi,
                  closure::ClosureEnv::NotClosure);
}

pub fn trans_enum_variant<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                    _enum_id: ast::NodeId,
                                    variant: &ast::Variant,
                                    _args: &[ast::VariantArg],
                                    disr: ty::Disr,
                                    param_substs: &'tcx Substs<'tcx>,
                                    llfndecl: ValueRef) {
    let _icx = push_ctxt("trans_enum_variant");

    trans_enum_variant_or_tuple_like_struct(
        ccx,
        variant.node.id,
        disr,
        param_substs,
        llfndecl);
}

pub fn trans_named_tuple_constructor<'blk, 'tcx>(mut bcx: Block<'blk, 'tcx>,
                                                 ctor_ty: Ty<'tcx>,
                                                 disr: ty::Disr,
                                                 args: callee::CallArgs,
                                                 dest: expr::Dest,
                                                 debug_loc: DebugLoc)
                                                 -> Result<'blk, 'tcx> {

    let ccx = bcx.fcx.ccx;
    let tcx = ccx.tcx();

    let result_ty = match ctor_ty.sty {
        ty::ty_bare_fn(_, ref bft) => {
            ty::erase_late_bound_regions(bcx.tcx(), &bft.sig.output()).unwrap()
        }
        _ => ccx.sess().bug(
            &format!("trans_enum_variant_constructor: \
                     unexpected ctor return type {}",
                     ctor_ty.repr(tcx))[])
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
                let fields = exprs.iter().map(|x| &**x).enumerate().collect::<Vec<_>>();
                bcx = expr::trans_adt(bcx,
                                      result_ty,
                                      disr,
                                      &fields[..],
                                      None,
                                      expr::SaveIn(llresult),
                                      debug_loc);
            }
            _ => ccx.sess().bug("expected expr as arguments for variant/struct tuple constructor")
        }
    }

    // If the caller doesn't care about the result
    // drop the temporary we made
    let bcx = match dest {
        expr::SaveIn(_) => bcx,
        expr::Ignore => {
            let bcx = glue::drop_ty(bcx, llresult, result_ty, debug_loc);
            if !type_is_zero_size(ccx, result_ty) {
                call_lifetime_end(bcx, llresult);
            }
            bcx
        }
    };

    Result::new(bcx, llresult)
}

pub fn trans_tuple_struct<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                    _fields: &[ast::StructField],
                                    ctor_id: ast::NodeId,
                                    param_substs: &'tcx Substs<'tcx>,
                                    llfndecl: ValueRef) {
    let _icx = push_ctxt("trans_tuple_struct");

    trans_enum_variant_or_tuple_like_struct(
        ccx,
        ctor_id,
        0,
        param_substs,
        llfndecl);
}

fn trans_enum_variant_or_tuple_like_struct<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                                     ctor_id: ast::NodeId,
                                                     disr: ty::Disr,
                                                     param_substs: &'tcx Substs<'tcx>,
                                                     llfndecl: ValueRef) {
    let ctor_ty = ty::node_id_to_type(ccx.tcx(), ctor_id);
    let ctor_ty = monomorphize::apply_param_substs(ccx.tcx(), param_substs, &ctor_ty);

    let result_ty = match ctor_ty.sty {
        ty::ty_bare_fn(_, ref bft) => {
            ty::erase_late_bound_regions(ccx.tcx(), &bft.sig.output())
        }
        _ => ccx.sess().bug(
            &format!("trans_enum_variant_or_tuple_like_struct: \
                     unexpected ctor return type {}",
                    ty_to_string(ccx.tcx(), ctor_ty))[])
    };

    let (arena, fcx): (TypedArena<_>, FunctionContext);
    arena = TypedArena::new();
    fcx = new_fn_ctxt(ccx, llfndecl, ctor_id, false, result_ty,
                      param_substs, None, &arena);
    let bcx = init_function(&fcx, false, result_ty);

    assert!(!fcx.needs_ret_allocas);

    let arg_tys =
        ty::erase_late_bound_regions(
            ccx.tcx(), &ty::ty_fn_args(ctor_ty));

    let arg_datums = create_datums_for_fn_args(&fcx, &arg_tys[..]);

    if !type_is_zero_size(fcx.ccx, result_ty.unwrap()) {
        let dest = fcx.get_ret_slot(bcx, result_ty, "eret_slot");
        let repr = adt::represent_type(ccx, result_ty.unwrap());
        for (i, arg_datum) in arg_datums.into_iter().enumerate() {
            let lldestptr = adt::trans_field_ptr(bcx,
                                                 &*repr,
                                                 dest,
                                                 disr,
                                                 i);
            arg_datum.store_to(bcx, lldestptr);
        }
        adt::trans_set_discr(bcx, &*repr, dest, disr);
    }

    finish_fn(&fcx, bcx, result_ty, DebugLoc::None);
}

fn enum_variant_size_lint(ccx: &CrateContext, enum_def: &ast::EnumDef, sp: Span, id: ast::NodeId) {
    let mut sizes = Vec::new(); // does no allocation if no pushes, thankfully

    let print_info = ccx.sess().print_enum_sizes();

    let levels = ccx.tcx().node_lint_levels.borrow();
    let lint_id = lint::LintId::of(lint::builtin::VARIANT_SIZE_DIFFERENCES);
    let lvlsrc = levels.get(&(id, lint_id));
    let is_allow = lvlsrc.map_or(true, |&(lvl, _)| lvl == lint::Allow);

    if is_allow && !print_info {
        // we're not interested in anything here
        return
    }

    let ty = ty::node_id_to_type(ccx.tcx(), id);
    let avar = adt::represent_type(ccx, ty);
    match *avar {
        adt::General(_, ref variants, _) => {
            for var in variants {
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

    if print_info {
        let llty = type_of::sizing_type_of(ccx, ty);

        let sess = &ccx.tcx().sess;
        sess.span_note(sp, &*format!("total size: {} bytes", llsize_of_real(ccx, llty)));
        match *avar {
            adt::General(..) => {
                for (i, var) in enum_def.variants.iter().enumerate() {
                    ccx.tcx().sess.span_note(var.span,
                                             &*format!("variant data: {} bytes", sizes[i]));
                }
            }
            _ => {}
        }
    }

    // we only warn if the largest variant is at least thrice as large as
    // the second-largest.
    if !is_allow && largest > slargest * 3 && slargest > 0 {
        // Use lint::raw_emit_lint rather than sess.add_lint because the lint-printing
        // pass for the latter already ran.
        lint::raw_emit_lint(&ccx.tcx().sess, lint::builtin::VARIANT_SIZE_DIFFERENCES,
                            *lvlsrc.unwrap(), Some(sp),
                            &format!("enum variant is more than three times larger \
                                     ({} bytes) than the next largest (ignoring padding)",
                                    largest)[]);

        ccx.sess().span_note(enum_def.variants[largest_index].span,
                             "this variant is the largest");
    }
}

pub struct TransItemVisitor<'a, 'tcx: 'a> {
    pub ccx: &'a CrateContext<'a, 'tcx>,
}

impl<'a, 'tcx, 'v> Visitor<'v> for TransItemVisitor<'a, 'tcx> {
    fn visit_item(&mut self, i: &ast::Item) {
        trans_item(self.ccx, i);
    }
}

pub fn llvm_linkage_by_name(name: &str) -> Option<Linkage> {
    // Use the names from src/llvm/docs/LangRef.rst here. Most types are only
    // applicable to variable declarations and may not really make sense for
    // Rust code in the first place but whitelist them anyway and trust that
    // the user knows what s/he's doing. Who knows, unanticipated use cases
    // may pop up in the future.
    //
    // ghost, dllimport, dllexport and linkonce_odr_autohide are not supported
    // and don't have to be, LLVM treats them as no-ops.
    match name {
        "appending" => Some(llvm::AppendingLinkage),
        "available_externally" => Some(llvm::AvailableExternallyLinkage),
        "common" => Some(llvm::CommonLinkage),
        "extern_weak" => Some(llvm::ExternalWeakLinkage),
        "external" => Some(llvm::ExternalLinkage),
        "internal" => Some(llvm::InternalLinkage),
        "linkonce" => Some(llvm::LinkOnceAnyLinkage),
        "linkonce_odr" => Some(llvm::LinkOnceODRLinkage),
        "private" => Some(llvm::PrivateLinkage),
        "weak" => Some(llvm::WeakAnyLinkage),
        "weak_odr" => Some(llvm::WeakODRLinkage),
        _ => None,
    }
}


/// Enum describing the origin of an LLVM `Value`, for linkage purposes.
#[derive(Copy)]
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

    if let Some(id) = id {
        let item = ccx.tcx().map.get(id);
        if let ast_map::NodeItem(i) = item {
            if let Some(name) = attr::first_attr_value_str_by_name(&i.attrs, "linkage") {
                if let Some(linkage) = llvm_linkage_by_name(&name) {
                    llvm::SetLinkage(llval, linkage);
                } else {
                    ccx.sess().span_fatal(i.span, "invalid linkage specified");
                }
                return;
            }
        }
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
            let trans_everywhere = attr::requests_inline(&item.attrs[]);
            // Ignore `trans_everywhere` for cross-crate inlined items
            // (`from_external`).  `trans_item` will be called once for each
            // compilation unit that references the item, so it will still get
            // translated everywhere it's needed.
            for (ref ccx, is_origin) in ccx.maybe_iter(!from_external && trans_everywhere) {
                let llfn = get_item_val(ccx, item.id);
                let empty_substs = ccx.tcx().mk_substs(Substs::trans_empty());
                if abi != Rust {
                    foreign::trans_rust_fn_with_foreign_abi(ccx,
                                                            &**decl,
                                                            &**body,
                                                            &item.attrs[],
                                                            llfn,
                                                            empty_substs,
                                                            item.id,
                                                            None);
                } else {
                    trans_fn(ccx,
                             &**decl,
                             &**body,
                             llfn,
                             empty_substs,
                             item.id,
                             &item.attrs[]);
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
        v.visit_block(&**body);
      }
      ast::ItemImpl(_, _, ref generics, _, _, ref impl_items) => {
        meth::trans_impl(ccx,
                         item.ident,
                         &impl_items[..],
                         generics,
                         item.id);
      }
      ast::ItemMod(ref m) => {
        trans_mod(&ccx.rotate(), m);
      }
      ast::ItemEnum(ref enum_definition, ref gens) => {
        if gens.ty_params.is_empty() {
            // sizes only make sense for non-generic types

            enum_variant_size_lint(ccx, enum_definition, item.span, item.id);
        }
      }
      ast::ItemConst(_, ref expr) => {
          // Recurse on the expression to catch items in blocks
          let mut v = TransItemVisitor{ ccx: ccx };
          v.visit_expr(&**expr);
      }
      ast::ItemStatic(_, m, ref expr) => {
          // Recurse on the expression to catch items in blocks
          let mut v = TransItemVisitor{ ccx: ccx };
          v.visit_expr(&**expr);

          consts::trans_static(ccx, m, item.id);
          let g = get_item_val(ccx, item.id);
          update_linkage(ccx, g, Some(item.id), OriginalTranslation);

          // Do static_assert checking. It can't really be done much earlier
          // because we need to get the value of the bool out of LLVM
          if attr::contains_name(&item.attrs[], "static_assert") {
              if m == ast::MutMutable {
                  ccx.sess().span_fatal(expr.span,
                                        "cannot have static_assert on a mutable \
                                         static");
              }

              let v = ccx.static_values().borrow()[item.id].clone();
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
        visit::walk_item(&mut v, item);
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
    for item in &m.items {
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
        // check for the #[rustc_error] annotation, which forces an
        // error in trans. This is used to write compile-fail tests
        // that actually test that compilation succeeds without
        // reporting an error.
        if ty::has_attr(ccx.tcx(), local_def(node_id), "rustc_error") {
            ccx.tcx().sess.span_fatal(sp, "compilation successful");
        }

        create_entry_wrapper(ccx, sp, llfn);
    }
}

fn register_fn<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                         sp: Span,
                         sym: String,
                         node_id: ast::NodeId,
                         node_type: Ty<'tcx>)
                         -> ValueRef {
    match node_type.sty {
        ty::ty_bare_fn(_, ref f) => {
            assert!(f.abi == Rust || f.abi == RustCall);
        }
        _ => panic!("expected bare rust fn")
    };

    let llfn = decl_rust_fn(ccx, node_type, &sym[..]);
    finish_register_fn(ccx, sp, sym, node_id, llfn);
    llfn
}

pub fn get_fn_llvm_attributes<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, fn_ty: Ty<'tcx>)
                                        -> llvm::AttrBuilder
{
    use middle::ty::{BrAnon, ReLateBound};

    let function_type;
    let (fn_sig, abi, has_env) = match fn_ty.sty {
        ty::ty_bare_fn(_, ref f) => (&f.sig, f.abi, false),
        ty::ty_closure(closure_did, _, substs) => {
            let typer = common::NormalizingClosureTyper::new(ccx.tcx());
            function_type = typer.closure_type(closure_did, substs);
            (&function_type.sig, RustCall, true)
        }
        _ => ccx.sess().bug("expected closure or function.")
    };

    let fn_sig = ty::erase_late_bound_regions(ccx.tcx(), fn_sig);

    // Since index 0 is the return value of the llvm func, we start
    // at either 1 or 2 depending on whether there's an env slot or not
    let mut first_arg_offset = if has_env { 2 } else { 1 };
    let mut attrs = llvm::AttrBuilder::new();
    let ret_ty = fn_sig.output;

    // These have an odd calling convention, so we need to manually
    // unpack the input ty's
    let input_tys = match fn_ty.sty {
        ty::ty_closure(_, _, _) => {
            assert!(abi == RustCall);

            match fn_sig.inputs[0].sty {
                ty::ty_tup(ref inputs) => inputs.clone(),
                _ => ccx.sess().bug("expected tuple'd inputs")
            }
        },
        ty::ty_bare_fn(..) if abi == RustCall => {
            let mut inputs = vec![fn_sig.inputs[0]];

            match fn_sig.inputs[1].sty {
                ty::ty_tup(ref t_in) => {
                    inputs.push_all(&t_in[..]);
                    inputs
                }
                _ => ccx.sess().bug("expected tuple'd inputs")
            }
        }
        _ => fn_sig.inputs.clone()
    };

    if let ty::FnConverging(ret_ty) = ret_ty {
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
            match ret_ty.sty {
                // `~` pointer return values never alias because ownership
                // is transferred
                ty::ty_uniq(it) if !common::type_is_sized(ccx.tcx(), it) => {}
                ty::ty_uniq(_) => {
                    attrs.ret(llvm::NoAliasAttribute);
                }
                _ => {}
            }

            // We can also mark the return value as `dereferenceable` in certain cases
            match ret_ty.sty {
                // These are not really pointers but pairs, (pointer, len)
                ty::ty_uniq(it) |
                ty::ty_rptr(_, ty::mt { ty: it, .. }) if !common::type_is_sized(ccx.tcx(), it) => {}
                ty::ty_uniq(inner) | ty::ty_rptr(_, ty::mt { ty: inner, .. }) => {
                    let llret_sz = llsize_of_real(ccx, type_of::type_of(ccx, inner));
                    attrs.ret(llvm::DereferenceableAttribute(llret_sz));
                }
                _ => {}
            }

            if let ty::ty_bool = ret_ty.sty {
                attrs.ret(llvm::ZExtAttribute);
            }
        }
    }

    for (idx, &t) in input_tys.iter().enumerate().map(|(i, v)| (i + first_arg_offset, v)) {
        match t.sty {
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

                if let ReLateBound(_, BrAnon(_)) = *b {
                    attrs.arg(idx, llvm::NoCaptureAttribute);
                }
            }

            // When a reference in an argument has no named lifetime, it's impossible for that
            // reference to escape this function (returned or stored beyond the call by a closure).
            ty::ty_rptr(&ReLateBound(_, BrAnon(_)), mt) => {
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

    let llfn = decl_fn(ccx,
                       &sym[..],
                       cc,
                       llfty,
                       ty::FnConverging(ty::mk_nil(ccx.tcx())));
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
        let llfty = Type::func(&[ccx.int_type(), Type::i8p(ccx).ptr_to()],
                               &ccx.int_type());

        let llfn = decl_cdecl_fn(ccx, "main", llfty, ty::mk_nil(ccx.tcx()));

        // FIXME: #16581: Marking a symbol in the executable with `dllexport`
        // linkage forces MinGW's linker to output a `.reloc` section for ASLR
        if ccx.sess().target.target.options.is_like_windows {
            unsafe { llvm::LLVMRustSetDLLExportStorageClass(llfn) }
        }

        let llbb = unsafe {
            llvm::LLVMAppendBasicBlockInContext(ccx.llcx(), llfn,
                                                "top\0".as_ptr() as *const _)
        };
        let bld = ccx.raw_builder();
        unsafe {
            llvm::LLVMPositionBuilderAtEnd(bld, llbb);

            debuginfo::insert_reference_to_gdb_debug_scripts_section_global(ccx);

            let (start_fn, args) = if use_start_lang_item {
                let start_def_id = match ccx.tcx().lang_items.require(StartFnLangItem) {
                    Ok(id) => id,
                    Err(s) => { ccx.sess().fatal(&s[..]); }
                };
                let start_fn = if start_def_id.krate == ast::LOCAL_CRATE {
                    get_item_val(ccx, start_def_id.node)
                } else {
                    let start_fn_type = csearch::get_type(ccx.tcx(),
                                                          start_def_id).ty;
                    trans_external_path(ccx, start_def_id, start_fn_type)
                };

                let args = {
                    let opaque_rust_main = llvm::LLVMBuildPointerCast(bld,
                        rust_main, Type::i8p(ccx).to_ref(),
                        "rust_main\0".as_ptr() as *const _);

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

fn exported_name<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, id: ast::NodeId,
                           ty: Ty<'tcx>, attrs: &[ast::Attribute]) -> String {
    match ccx.external_srcs().borrow().get(&id) {
        Some(&did) => {
            let sym = csearch::get_symbol(&ccx.sess().cstore, did);
            debug!("found item {} in other crate...", sym);
            return sym;
        }
        None => {}
    }

    match attr::first_attr_value_str_by_name(attrs, "export_name") {
        // Use provided name
        Some(name) => name.to_string(),

        _ => ccx.tcx().map.with_path(id, |path| {
            if attr::contains_name(attrs, "no_mangle") {
                // Don't mangle
                path.last().unwrap().to_string()
            } else {
                match weak_lang_items::link_name(attrs) {
                    Some(name) => name.to_string(),
                    None => {
                        // Usual name mangling
                        mangle_exported_name(ccx, path, ty, id)
                    }
                }
            }
        })
    }
}

fn contains_null(s: &str) -> bool {
    s.bytes().any(|b| b == 0)
}

pub fn get_item_val(ccx: &CrateContext, id: ast::NodeId) -> ValueRef {
    debug!("get_item_val(id=`{}`)", id);

    match ccx.item_vals().borrow().get(&id).cloned() {
        Some(v) => return v,
        None => {}
    }

    let item = ccx.tcx().map.get(id);
    debug!("get_item_val: id={} item={:?}", id, item);
    let val = match item {
        ast_map::NodeItem(i) => {
            let ty = ty::node_id_to_type(ccx.tcx(), i.id);
            let sym = || exported_name(ccx, id, ty, &i.attrs[]);

            let v = match i.node {
                ast::ItemStatic(_, _, ref expr) => {
                    // If this static came from an external crate, then
                    // we need to get the symbol from csearch instead of
                    // using the current crate's name/version
                    // information in the hash of the symbol
                    let sym = sym();
                    debug!("making {}", sym);

                    // We need the translated value here, because for enums the
                    // LLVM type is not fully determined by the Rust type.
                    let empty_substs = ccx.tcx().mk_substs(Substs::trans_empty());
                    let (v, ty) = consts::const_expr(ccx, &**expr, empty_substs);
                    ccx.static_values().borrow_mut().insert(id, v);
                    unsafe {
                        // boolean SSA values are i1, but they have to be stored in i8 slots,
                        // otherwise some LLVM optimization passes don't work as expected
                        let llty = if ty::type_is_bool(ty) {
                            llvm::LLVMInt8TypeInContext(ccx.llcx())
                        } else {
                            llvm::LLVMTypeOf(v)
                        };
                        if contains_null(&sym[..]) {
                            ccx.sess().fatal(
                                &format!("Illegal null byte in export_name \
                                         value: `{}`", sym)[]);
                        }
                        let buf = CString::new(sym.clone()).unwrap();
                        let g = llvm::LLVMAddGlobal(ccx.llmod(), llty,
                                                    buf.as_ptr());

                        if attr::contains_name(&i.attrs[],
                                               "thread_local") {
                            llvm::set_thread_local(g, true);
                        }
                        ccx.item_symbols().borrow_mut().insert(i.id, sym);
                        g
                    }
                }

                ast::ItemFn(_, _, abi, _, _) => {
                    let sym = sym();
                    let llfn = if abi == Rust {
                        register_fn(ccx, i.span, sym, i.id, ty)
                    } else {
                        foreign::register_rust_fn_with_foreign_abi(ccx,
                                                                   i.span,
                                                                   sym,
                                                                   i.id)
                    };
                    set_llvm_fn_attrs(ccx, &i.attrs[], llfn);
                    llfn
                }

                _ => panic!("get_item_val: weird result in table")
            };

            match attr::first_attr_value_str_by_name(&i.attrs[],
                                                     "link_section") {
                Some(sect) => {
                    if contains_null(&sect) {
                        ccx.sess().fatal(&format!("Illegal null byte in link_section value: `{}`",
                                                 &sect)[]);
                    }
                    unsafe {
                        let buf = CString::new(sect.as_bytes()).unwrap();
                        llvm::LLVMSetSection(v, buf.as_ptr());
                    }
                },
                None => ()
            }

            v
        }

        ast_map::NodeTraitItem(trait_method) => {
            debug!("get_item_val(): processing a NodeTraitItem");
            match *trait_method {
                ast::RequiredMethod(_) | ast::TypeTraitItem(_) => {
                    ccx.sess().bug("unexpected variant: required trait \
                                    method in get_item_val()");
                }
                ast::ProvidedMethod(ref m) => {
                    register_method(ccx, id, &**m)
                }
            }
        }

        ast_map::NodeImplItem(ii) => {
            match *ii {
                ast::MethodImplItem(ref m) => register_method(ccx, id, &**m),
                ast::TypeImplItem(ref typedef) => {
                    ccx.sess().span_bug(typedef.span,
                                        "unexpected variant: required impl \
                                         method in get_item_val()")
                }
            }
        }

        ast_map::NodeForeignItem(ni) => {
            match ni.node {
                ast::ForeignItemFn(..) => {
                    let abi = ccx.tcx().map.get_foreign_abi(id);
                    let ty = ty::node_id_to_type(ccx.tcx(), ni.id);
                    let name = foreign::link_name(&*ni);
                    foreign::register_foreign_item_fn(ccx, abi, ty, &name)
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
                    panic!("struct variant kind unexpected in get_item_val")
                }
            };
            assert!(args.len() != 0);
            let ty = ty::node_id_to_type(ccx.tcx(), id);
            let parent = ccx.tcx().map.get_parent(id);
            let enm = ccx.tcx().map.expect_item(parent);
            let sym = exported_name(ccx,
                                    id,
                                    ty,
                                    &enm.attrs[]);

            llfn = match enm.node {
                ast::ItemEnum(_, _) => {
                    register_fn(ccx, (*v).span, sym, id, ty)
                }
                _ => panic!("NodeVariant, shouldn't happen")
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
                                    &struct_item.attrs[]);
            let llfn = register_fn(ccx, struct_item.span,
                                   sym, ctor_id, ty);
            set_inline_hint(llfn);
            llfn
        }

        ref variant => {
            ccx.sess().bug(&format!("get_item_val(): unexpected variant: {:?}",
                                   variant)[])
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

    let sym = exported_name(ccx, id, mty, &m.attrs[]);

    let llfn = register_fn(ccx, m.span, sym, id, mty);
    set_llvm_fn_attrs(ccx, &m.attrs[], llfn);
    llfn
}

pub fn crate_ctxt_to_encode_parms<'a, 'tcx>(cx: &'a SharedCrateContext<'tcx>,
                                            ie: encoder::EncodeInlinedItem<'a>)
                                            -> encoder::EncodeParams<'a, 'tcx> {
    encoder::EncodeParams {
        diag: cx.sess().diagnostic(),
        tcx: cx.tcx(),
        reexports: cx.export_map(),
        item_symbols: cx.item_symbols(),
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
        box |ecx, rbml_w, ii| astencode::encode_inlined_item(ecx, rbml_w, ii);

    let encode_parms = crate_ctxt_to_encode_parms(cx, encode_inlined_item);
    let metadata = encoder::encode_metadata(encode_parms, krate);
    let mut compressed = encoder::metadata_encoding_version.to_vec();
    compressed.push_all(&match flate::deflate_bytes(&metadata) {
        Some(compressed) => compressed,
        None => cx.sess().fatal("failed to compress metadata"),
    });
    let llmeta = C_bytes_in_context(cx.metadata_llcx(), &compressed[..]);
    let llconst = C_struct_in_context(cx.metadata_llcx(), &[llmeta], false);
    let name = format!("rust_metadata_{}_{}",
                       cx.link_meta().crate_name,
                       cx.link_meta().crate_hash);
    let buf = CString::new(name).unwrap();
    let llglobal = unsafe {
        llvm::LLVMAddGlobal(cx.metadata_llmod(), val_ty(llconst).to_ref(),
                            buf.as_ptr())
    };
    unsafe {
        llvm::LLVMSetInitializer(llglobal, llconst);
        let name = loader::meta_section_name(cx.sess().target.target.options.is_like_osx);
        let name = CString::new(name).unwrap();
        llvm::LLVMSetSection(llglobal, name.as_ptr())
    }
    return metadata;
}

/// Find any symbols that are defined in one compilation unit, but not declared
/// in any other compilation unit.  Give these symbols internal linkage.
fn internalize_symbols(cx: &SharedCrateContext, reachable: &HashSet<String>) {
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

                let name = CStr::from_ptr(llvm::LLVMGetValueName(val))
                                .to_bytes().to_vec();
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

                let name = CStr::from_ptr(llvm::LLVMGetValueName(val))
                                .to_bytes().to_vec();
                if !declared.contains(&name) &&
                   !reachable.contains(str::from_utf8(&name).unwrap()) {
                    llvm::SetLinkage(val, llvm::InternalLinkage);
                }
            }
        }
    }


    struct ValueIter {
        cur: ValueRef,
        step: unsafe extern "C" fn(ValueRef) -> ValueRef,
    }

    impl Iterator for ValueIter {
        type Item = ValueRef;

        fn next(&mut self) -> Option<ValueRef> {
            let old = self.cur;
            if !old.is_null() {
                self.cur = unsafe {
                    let step: unsafe extern "C" fn(ValueRef) -> ValueRef =
                        mem::transmute_copy(&self.step);
                    step(old)
                };
                Some(old)
            } else {
                None
            }
        }
    }
}

pub fn trans_crate<'tcx>(analysis: ty::CrateAnalysis<'tcx>)
                         -> (ty::ctxt<'tcx>, CrateTranslation) {
    let ty::CrateAnalysis { ty_cx: tcx, export_map, reachable, name, .. } = analysis;
    let krate = tcx.map.krate();

    // Before we touch LLVM, make sure that multithreading is enabled.
    unsafe {
        use std::sync::{Once, ONCE_INIT};
        static INIT: Once = ONCE_INIT;
        static mut POISONED: bool = false;
        INIT.call_once(|| {
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

    let link_meta = link::build_link_meta(&tcx.sess, krate, name);

    let codegen_units = tcx.sess.opts.cg.codegen_units;
    let shared_ccx = SharedCrateContext::new(&link_meta.crate_name[],
                                             codegen_units,
                                             tcx,
                                             export_map,
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
    let metadata = write_metadata(&shared_ccx, krate);

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
        stats.fn_stats.borrow_mut().sort_by(|&(_, insns_a), &(_, insns_b)| {
            insns_b.cmp(&insns_a)
        });
        for tuple in &*stats.fn_stats.borrow() {
            match *tuple {
                (ref name, insns) => {
                    println!("{} insns, {}", insns, *name);
                }
            }
        }
    }
    if shared_ccx.sess().count_llvm_insns() {
        for (k, v) in &*shared_ccx.stats().llvm_insns.borrow() {
            println!("{:7} {}", *v, *k);
        }
    }

    let modules = shared_ccx.iter()
        .map(|ccx| ModuleTranslation { llcx: ccx.llcx(), llmod: ccx.llmod() })
        .collect();

    let mut reachable: Vec<String> = shared_ccx.reachable().iter().filter_map(|id| {
        shared_ccx.item_symbols().borrow().get(id).map(|s| s.to_string())
    }).collect();

    // For the purposes of LTO, we add to the reachable set all of the upstream
    // reachable extern fns. These functions are all part of the public ABI of
    // the final product, so LTO needs to preserve them.
    shared_ccx.sess().cstore.iter_crate_data(|cnum, _| {
        let syms = csearch::get_reachable_extern_fns(&shared_ccx.sess().cstore, cnum);
        reachable.extend(syms.into_iter().map(|did| {
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
        internalize_symbols(&shared_ccx, &reachable.iter().cloned().collect());
    }

    let metadata_module = ModuleTranslation {
        llcx: shared_ccx.metadata_llcx(),
        llmod: shared_ccx.metadata_llmod(),
    };
    let formats = shared_ccx.tcx().dependency_formats.borrow().clone();
    let no_builtins = attr::contains_name(&krate.attrs[], "no_builtins");

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
