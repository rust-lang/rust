// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use back::abi;
use llvm;
use llvm::{ConstFCmp, ConstICmp, SetLinkage, SetUnnamedAddr};
use llvm::{InternalLinkage, ValueRef, Bool, True};
use middle::{check_const, def};
use middle::cstore::LOCAL_CRATE;
use middle::const_eval::{self, ConstVal, ConstEvalErr};
use middle::const_eval::{const_int_checked_neg, const_uint_checked_neg};
use middle::const_eval::{const_int_checked_add, const_uint_checked_add};
use middle::const_eval::{const_int_checked_sub, const_uint_checked_sub};
use middle::const_eval::{const_int_checked_mul, const_uint_checked_mul};
use middle::const_eval::{const_int_checked_div, const_uint_checked_div};
use middle::const_eval::{const_int_checked_rem, const_uint_checked_rem};
use middle::const_eval::{const_int_checked_shl, const_uint_checked_shl};
use middle::const_eval::{const_int_checked_shr, const_uint_checked_shr};
use middle::const_eval::EvalHint::ExprTypeChecked;
use middle::const_eval::eval_const_expr_partial;
use middle::def_id::DefId;
use trans::{adt, closure, debuginfo, expr, inline, machine};
use trans::base::{self, push_ctxt};
use trans::common::{self, type_is_sized, ExprOrMethodCall, node_id_substs, C_nil, const_get_elt};
use trans::common::{CrateContext, C_integral, C_floating, C_bool, C_str_slice, C_bytes, val_ty};
use trans::common::C_floating_f64;
use trans::common::{C_struct, C_undef, const_to_opt_int, const_to_opt_uint, VariantInfo, C_uint};
use trans::common::{type_is_fat_ptr, Field, C_vector, C_array, C_null, ExprId, MethodCallKey};
use trans::declare;
use trans::monomorphize;
use trans::type_::Type;
use trans::type_of;
use middle::subst::Substs;
use middle::ty::adjustment::{AdjustDerefRef, AdjustReifyFnPointer};
use middle::ty::adjustment::AdjustUnsafeFnPointer;
use middle::ty::{self, Ty};
use middle::ty::cast::{CastTy,IntTy};
use util::nodemap::NodeMap;

use rustc_front::hir;

use std::ffi::{CStr, CString};
use std::borrow::Cow;
use libc::c_uint;
use syntax::ast;
use syntax::attr;
use syntax::parse::token;
use syntax::ptr::P;

pub type FnArgMap<'a> = Option<&'a NodeMap<ValueRef>>;

pub fn const_lit(cx: &CrateContext, e: &hir::Expr, lit: &ast::Lit)
    -> ValueRef {
    let _icx = push_ctxt("trans_lit");
    debug!("const_lit: {:?}", lit);
    match lit.node {
        ast::LitByte(b) => C_integral(Type::uint_from_ty(cx, ast::TyU8), b as u64, false),
        ast::LitChar(i) => C_integral(Type::char(cx), i as u64, false),
        ast::LitInt(i, ast::SignedIntLit(t, _)) => {
            C_integral(Type::int_from_ty(cx, t), i, true)
        }
        ast::LitInt(u, ast::UnsignedIntLit(t)) => {
            C_integral(Type::uint_from_ty(cx, t), u, false)
        }
        ast::LitInt(i, ast::UnsuffixedIntLit(_)) => {
            let lit_int_ty = cx.tcx().node_id_to_type(e.id);
            match lit_int_ty.sty {
                ty::TyInt(t) => {
                    C_integral(Type::int_from_ty(cx, t), i as u64, true)
                }
                ty::TyUint(t) => {
                    C_integral(Type::uint_from_ty(cx, t), i as u64, false)
                }
                _ => cx.sess().span_bug(lit.span,
                        &format!("integer literal has type {:?} (expected int \
                                 or usize)",
                                lit_int_ty))
            }
        }
        ast::LitFloat(ref fs, t) => {
            C_floating(&fs, Type::float_from_ty(cx, t))
        }
        ast::LitFloatUnsuffixed(ref fs) => {
            let lit_float_ty = cx.tcx().node_id_to_type(e.id);
            match lit_float_ty.sty {
                ty::TyFloat(t) => {
                    C_floating(&fs, Type::float_from_ty(cx, t))
                }
                _ => {
                    cx.sess().span_bug(lit.span,
                        "floating point literal doesn't have the right type");
                }
            }
        }
        ast::LitBool(b) => C_bool(cx, b),
        ast::LitStr(ref s, _) => C_str_slice(cx, (*s).clone()),
        ast::LitByteStr(ref data) => {
            addr_of(cx, C_bytes(cx, &data[..]), 1, "byte_str")
        }
    }
}

pub fn trans_constval<'blk, 'tcx>(bcx: common::Block<'blk, 'tcx>,
                                cv: &ConstVal,
                                ty: Ty<'tcx>,
                                param_substs: &'tcx Substs<'tcx>)
                                -> ValueRef
{
    let ccx = bcx.ccx();
    let llty = type_of::type_of(ccx, ty);
    match *cv {
        ConstVal::Float(v) => C_floating_f64(v, llty),
        ConstVal::Bool(v) => C_bool(ccx, v),
        ConstVal::Int(v) => C_integral(llty, v as u64, true),
        ConstVal::Uint(v) => C_integral(llty, v, false),
        ConstVal::Str(ref v) => C_str_slice(ccx, v.clone()),
        ConstVal::ByteStr(ref v) => addr_of(ccx, C_bytes(ccx, v), 1, "byte_str"),
        ConstVal::Struct(id) | ConstVal::Tuple(id) => {
            let expr = bcx.tcx().map.expect_expr(id);
            match const_expr(ccx, expr, param_substs, None, TrueConst::Yes) {
                Ok((val, _)) => val,
                Err(e) => panic!("const eval failure: {}", e.description()),
            }
        },
        ConstVal::Array(id, _) | ConstVal::Repeat(id, _) => {
            let expr = bcx.tcx().map.expect_expr(id);
            expr::trans(bcx, expr).datum.val
        },
        ConstVal::Function(_) => {
            unimplemented!()
        },
    }
}

pub fn ptrcast(val: ValueRef, ty: Type) -> ValueRef {
    unsafe {
        llvm::LLVMConstPointerCast(val, ty.to_ref())
    }
}

fn addr_of_mut(ccx: &CrateContext,
               cv: ValueRef,
               align: machine::llalign,
               kind: &str)
               -> ValueRef {
    unsafe {
        // FIXME: this totally needs a better name generation scheme, perhaps a simple global
        // counter? Also most other uses of gensym in trans.
        let gsym = token::gensym("_");
        let name = format!("{}{}", kind, gsym.0);
        let gv = declare::define_global(ccx, &name[..], val_ty(cv)).unwrap_or_else(||{
            ccx.sess().bug(&format!("symbol `{}` is already defined", name));
        });
        llvm::LLVMSetInitializer(gv, cv);
        llvm::LLVMSetAlignment(gv, align);
        SetLinkage(gv, InternalLinkage);
        SetUnnamedAddr(gv, true);
        gv
    }
}

pub fn addr_of(ccx: &CrateContext,
               cv: ValueRef,
               align: machine::llalign,
               kind: &str)
               -> ValueRef {
    match ccx.const_globals().borrow().get(&cv) {
        Some(&gv) => {
            unsafe {
                // Upgrade the alignment in cases where the same constant is used with different
                // alignment requirements
                if align > llvm::LLVMGetAlignment(gv) {
                    llvm::LLVMSetAlignment(gv, align);
                }
            }
            return gv;
        }
        None => {}
    }
    let gv = addr_of_mut(ccx, cv, align, kind);
    unsafe {
        llvm::LLVMSetGlobalConstant(gv, True);
    }
    ccx.const_globals().borrow_mut().insert(cv, gv);
    gv
}

fn const_deref_ptr(cx: &CrateContext, v: ValueRef) -> ValueRef {
    let v = match cx.const_unsized().borrow().get(&v) {
        Some(&v) => v,
        None => v
    };
    unsafe {
        llvm::LLVMGetInitializer(v)
    }
}

fn const_deref<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                         v: ValueRef,
                         ty: Ty<'tcx>)
                         -> (ValueRef, Ty<'tcx>) {
    match ty.builtin_deref(true, ty::NoPreference) {
        Some(mt) => {
            if type_is_sized(cx.tcx(), mt.ty) {
                (const_deref_ptr(cx, v), mt.ty)
            } else {
                // Derefing a fat pointer does not change the representation,
                // just the type to the unsized contents.
                (v, mt.ty)
            }
        }
        None => {
            cx.sess().bug(&format!("unexpected dereferenceable type {:?}",
                                   ty))
        }
    }
}

fn const_fn_call<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                           node: ExprOrMethodCall,
                           def_id: DefId,
                           arg_vals: &[ValueRef],
                           param_substs: &'tcx Substs<'tcx>,
                           trueconst: TrueConst) -> Result<ValueRef, ConstEvalFailure> {
    let fn_like = const_eval::lookup_const_fn_by_id(ccx.tcx(), def_id);
    let fn_like = fn_like.expect("lookup_const_fn_by_id failed in const_fn_call");

    let args = &fn_like.decl().inputs;
    assert_eq!(args.len(), arg_vals.len());

    let arg_ids = args.iter().map(|arg| arg.pat.id);
    let fn_args = arg_ids.zip(arg_vals.iter().cloned()).collect();

    let substs = ccx.tcx().mk_substs(node_id_substs(ccx, node, param_substs));
    match fn_like.body().expr {
        Some(ref expr) => {
            const_expr(ccx, &**expr, substs, Some(&fn_args), trueconst).map(|(res, _)| res)
        },
        None => Ok(C_nil(ccx)),
    }
}

pub fn get_const_expr<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                def_id: DefId,
                                ref_expr: &hir::Expr)
                                -> &'tcx hir::Expr {
    let def_id = inline::maybe_instantiate_inline(ccx, def_id);

    if def_id.krate != LOCAL_CRATE {
        ccx.sess().span_bug(ref_expr.span,
                            "cross crate constant could not be inlined");
    }

    match const_eval::lookup_const_by_id(ccx.tcx(), def_id, Some(ref_expr.id)) {
        Some(ref expr) => expr,
        None => {
            ccx.sess().span_bug(ref_expr.span, "constant item not found")
        }
    }
}

pub enum ConstEvalFailure {
    /// in case the const evaluator failed on something that panic at runtime
    /// as defined in RFC 1229
    Runtime(ConstEvalErr),
    // in case we found a true constant
    Compiletime(ConstEvalErr),
}

impl ConstEvalFailure {
    fn into_inner(self) -> ConstEvalErr {
        match self {
            Runtime(e) => e,
            Compiletime(e) => e,
        }
    }
    pub fn description(&self) -> Cow<str> {
        match self {
            &Runtime(ref e) => e.description(),
            &Compiletime(ref e) => e.description(),
        }
    }
}

#[derive(Copy, Clone)]
pub enum TrueConst {
    Yes, No
}

use self::ConstEvalFailure::*;

fn get_const_val(ccx: &CrateContext,
                 def_id: DefId,
                 ref_expr: &hir::Expr) -> Result<ValueRef, ConstEvalFailure> {
    let expr = get_const_expr(ccx, def_id, ref_expr);
    let empty_substs = ccx.tcx().mk_substs(Substs::trans_empty());
    match get_const_expr_as_global(ccx, expr, check_const::ConstQualif::empty(),
                                   empty_substs, TrueConst::Yes) {
        Err(Runtime(err)) => {
            ccx.tcx().sess.span_err(expr.span, &err.description());
            Err(Compiletime(err))
        },
        other => other,
    }
}

pub fn get_const_expr_as_global<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                          expr: &hir::Expr,
                                          qualif: check_const::ConstQualif,
                                          param_substs: &'tcx Substs<'tcx>,
                                          trueconst: TrueConst)
                                          -> Result<ValueRef, ConstEvalFailure> {
    debug!("get_const_expr_as_global: {:?}", expr.id);
    // Special-case constants to cache a common global for all uses.
    if let hir::ExprPath(..) = expr.node {
        // `def` must be its own statement and cannot be in the `match`
        // otherwise the `def_map` will be borrowed for the entire match instead
        // of just to get the `def` value
        let def = ccx.tcx().def_map.borrow().get(&expr.id).unwrap().full_def();
        match def {
            def::DefConst(def_id) | def::DefAssociatedConst(def_id) => {
                if !ccx.tcx().tables.borrow().adjustments.contains_key(&expr.id) {
                    debug!("get_const_expr_as_global ({:?}): found const {:?}",
                           expr.id, def_id);
                    return get_const_val(ccx, def_id, expr);
                }
            },
            _ => {},
        }
    }

    let key = (expr.id, param_substs);
    if let Some(&val) = ccx.const_values().borrow().get(&key) {
        return Ok(val);
    }
    let ty = monomorphize::apply_param_substs(ccx.tcx(), param_substs,
                                              &ccx.tcx().expr_ty(expr));
    let val = if qualif.intersects(check_const::ConstQualif::NON_STATIC_BORROWS) {
        // Avoid autorefs as they would create global instead of stack
        // references, even when only the latter are correct.
        try!(const_expr_unadjusted(ccx, expr, ty, param_substs, None, trueconst))
    } else {
        try!(const_expr(ccx, expr, param_substs, None, trueconst)).0
    };

    // boolean SSA values are i1, but they have to be stored in i8 slots,
    // otherwise some LLVM optimization passes don't work as expected
    let val = unsafe {
        if llvm::LLVMTypeOf(val) == Type::i1(ccx).to_ref() {
            llvm::LLVMConstZExt(val, Type::i8(ccx).to_ref())
        } else {
            val
        }
    };

    let lvalue = addr_of(ccx, val, type_of::align_of(ccx, ty), "const");
    ccx.const_values().borrow_mut().insert(key, lvalue);
    Ok(lvalue)
}

pub fn const_expr<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                            e: &hir::Expr,
                            param_substs: &'tcx Substs<'tcx>,
                            fn_args: FnArgMap,
                            trueconst: TrueConst)
                            -> Result<(ValueRef, Ty<'tcx>), ConstEvalFailure> {
    let ety = monomorphize::apply_param_substs(cx.tcx(), param_substs,
                                               &cx.tcx().expr_ty(e));
    let llconst = try!(const_expr_unadjusted(cx, e, ety, param_substs, fn_args, trueconst));
    let mut llconst = llconst;
    let mut ety_adjusted = monomorphize::apply_param_substs(cx.tcx(), param_substs,
                                                            &cx.tcx().expr_ty_adjusted(e));
    let opt_adj = cx.tcx().tables.borrow().adjustments.get(&e.id).cloned();
    match opt_adj {
        Some(AdjustReifyFnPointer) => {
            // FIXME(#19925) once fn item types are
            // zero-sized, we'll need to do something here
        }
        Some(AdjustUnsafeFnPointer) => {
            // purely a type-level thing
        }
        Some(AdjustDerefRef(adj)) => {
            let mut ty = ety;
            // Save the last autoderef in case we can avoid it.
            if adj.autoderefs > 0 {
                for _ in 0..adj.autoderefs-1 {
                    let (dv, dt) = const_deref(cx, llconst, ty);
                    llconst = dv;
                    ty = dt;
                }
            }

            if adj.autoref.is_some() {
                if adj.autoderefs == 0 {
                    // Don't copy data to do a deref+ref
                    // (i.e., skip the last auto-deref).
                    llconst = addr_of(cx, llconst, type_of::align_of(cx, ty), "autoref");
                    ty = cx.tcx().mk_imm_ref(cx.tcx().mk_region(ty::ReStatic), ty);
                }
            } else {
                let (dv, dt) = const_deref(cx, llconst, ty);
                llconst = dv;

                // If we derefed a fat pointer then we will have an
                // open type here. So we need to update the type with
                // the one returned from const_deref.
                ety_adjusted = dt;
            }

            if let Some(target) = adj.unsize {
                let target = monomorphize::apply_param_substs(cx.tcx(),
                                                              param_substs,
                                                              &target);

                let pointee_ty = ty.builtin_deref(true, ty::NoPreference)
                    .expect("consts: unsizing got non-pointer type").ty;
                let (base, old_info) = if !type_is_sized(cx.tcx(), pointee_ty) {
                    // Normally, the source is a thin pointer and we are
                    // adding extra info to make a fat pointer. The exception
                    // is when we are upcasting an existing object fat pointer
                    // to use a different vtable. In that case, we want to
                    // load out the original data pointer so we can repackage
                    // it.
                    (const_get_elt(cx, llconst, &[abi::FAT_PTR_ADDR as u32]),
                     Some(const_get_elt(cx, llconst, &[abi::FAT_PTR_EXTRA as u32])))
                } else {
                    (llconst, None)
                };

                let unsized_ty = target.builtin_deref(true, ty::NoPreference)
                    .expect("consts: unsizing got non-pointer target type").ty;
                let ptr_ty = type_of::in_memory_type_of(cx, unsized_ty).ptr_to();
                let base = ptrcast(base, ptr_ty);
                let info = base::unsized_info(cx, pointee_ty, unsized_ty,
                                              old_info, param_substs);

                if old_info.is_none() {
                    let prev_const = cx.const_unsized().borrow_mut()
                                       .insert(base, llconst);
                    assert!(prev_const.is_none() || prev_const == Some(llconst));
                }
                assert_eq!(abi::FAT_PTR_ADDR, 0);
                assert_eq!(abi::FAT_PTR_EXTRA, 1);
                llconst = C_struct(cx, &[base, info], false);
            }
        }
        None => {}
    };

    let llty = type_of::sizing_type_of(cx, ety_adjusted);
    let csize = machine::llsize_of_alloc(cx, val_ty(llconst));
    let tsize = machine::llsize_of_alloc(cx, llty);
    if csize != tsize {
        cx.sess().abort_if_errors();
        unsafe {
            // FIXME these values could use some context
            llvm::LLVMDumpValue(llconst);
            llvm::LLVMDumpValue(C_undef(llty));
        }
        cx.sess().bug(&format!("const {:?} of type {:?} has size {} instead of {}",
                         e, ety_adjusted,
                         csize, tsize));
    }
    Ok((llconst, ety_adjusted))
}

fn check_unary_expr_validity(cx: &CrateContext, e: &hir::Expr, t: Ty,
                             te: ValueRef, trueconst: TrueConst) -> Result<(), ConstEvalFailure> {
    // The only kind of unary expression that we check for validity
    // here is `-expr`, to check if it "overflows" (e.g. `-i32::MIN`).
    if let hir::ExprUnary(hir::UnNeg, ref inner_e) = e.node {

        // An unfortunate special case: we parse e.g. -128 as a
        // negation of the literal 128, which means if we're expecting
        // a i8 (or if it was already suffixed, e.g. `-128_i8`), then
        // 128 will have already overflowed to -128, and so then the
        // constant evaluator thinks we're trying to negate -128.
        //
        // Catch this up front by looking for ExprLit directly,
        // and just accepting it.
        if let hir::ExprLit(_) = inner_e.node { return Ok(()); }

        let result = match t.sty {
            ty::TyInt(int_type) => {
                let input = match const_to_opt_int(te) {
                    Some(v) => v,
                    None => return Ok(()),
                };
                const_int_checked_neg(
                    input, e, Some(const_eval::IntTy::from(cx.tcx(), int_type)))
            }
            ty::TyUint(uint_type) => {
                let input = match const_to_opt_uint(te) {
                    Some(v) => v,
                    None => return Ok(()),
                };
                const_uint_checked_neg(
                    input, e, Some(const_eval::UintTy::from(cx.tcx(), uint_type)))
            }
            _ => return Ok(()),
        };
        const_err(cx, e, result, trueconst)
    } else {
        Ok(())
    }
}

fn const_err(cx: &CrateContext,
             e: &hir::Expr,
             result: Result<ConstVal, ConstEvalErr>,
             trueconst: TrueConst)
             -> Result<(), ConstEvalFailure> {
    match (result, trueconst) {
        (Ok(_), _) => {
            // We do not actually care about a successful result.
            Ok(())
        },
        (Err(err), TrueConst::Yes) => {
            cx.tcx().sess.span_err(e.span, &err.description());
            Err(Compiletime(err))
        },
        (Err(err), TrueConst::No) => {
            cx.tcx().sess.span_warn(e.span, &err.description());
            Err(Runtime(err))
        },
    }
}

fn check_binary_expr_validity(cx: &CrateContext, e: &hir::Expr, t: Ty,
                              te1: ValueRef, te2: ValueRef,
                              trueconst: TrueConst) -> Result<(), ConstEvalFailure> {
    let b = if let hir::ExprBinary(b, _, _) = e.node { b } else { unreachable!() };

    let result = match t.sty {
        ty::TyInt(int_type) => {
            let (lhs, rhs) = match (const_to_opt_int(te1),
                                    const_to_opt_int(te2)) {
                (Some(v1), Some(v2)) => (v1, v2),
                _ => return Ok(()),
            };

            let opt_ety = Some(const_eval::IntTy::from(cx.tcx(), int_type));
            match b.node {
                hir::BiAdd => const_int_checked_add(lhs, rhs, e, opt_ety),
                hir::BiSub => const_int_checked_sub(lhs, rhs, e, opt_ety),
                hir::BiMul => const_int_checked_mul(lhs, rhs, e, opt_ety),
                hir::BiDiv => const_int_checked_div(lhs, rhs, e, opt_ety),
                hir::BiRem => const_int_checked_rem(lhs, rhs, e, opt_ety),
                hir::BiShl => const_int_checked_shl(lhs, rhs, e, opt_ety),
                hir::BiShr => const_int_checked_shr(lhs, rhs, e, opt_ety),
                _ => return Ok(()),
            }
        }
        ty::TyUint(uint_type) => {
            let (lhs, rhs) = match (const_to_opt_uint(te1),
                                    const_to_opt_uint(te2)) {
                (Some(v1), Some(v2)) => (v1, v2),
                _ => return Ok(()),
            };

            let opt_ety = Some(const_eval::UintTy::from(cx.tcx(), uint_type));
            match b.node {
                hir::BiAdd => const_uint_checked_add(lhs, rhs, e, opt_ety),
                hir::BiSub => const_uint_checked_sub(lhs, rhs, e, opt_ety),
                hir::BiMul => const_uint_checked_mul(lhs, rhs, e, opt_ety),
                hir::BiDiv => const_uint_checked_div(lhs, rhs, e, opt_ety),
                hir::BiRem => const_uint_checked_rem(lhs, rhs, e, opt_ety),
                hir::BiShl => const_uint_checked_shl(lhs, rhs, e, opt_ety),
                hir::BiShr => const_uint_checked_shr(lhs, rhs, e, opt_ety),
                _ => return Ok(()),
            }
        }
        _ => return Ok(()),
    };
    const_err(cx, e, result, trueconst)
}

fn const_expr_unadjusted<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                   e: &hir::Expr,
                                   ety: Ty<'tcx>,
                                   param_substs: &'tcx Substs<'tcx>,
                                   fn_args: FnArgMap,
                                   trueconst: TrueConst)
                                   -> Result<ValueRef, ConstEvalFailure>
{
    debug!("const_expr_unadjusted(e={:?}, ety={:?}, param_substs={:?})",
           e,
           ety,
           param_substs);

    let map_list = |exprs: &[P<hir::Expr>]| -> Result<Vec<ValueRef>, ConstEvalFailure> {
        exprs.iter()
             .map(|e| const_expr(cx, &**e, param_substs, fn_args, trueconst).map(|(l, _)| l))
             .collect::<Vec<Result<ValueRef, ConstEvalFailure>>>()
             .into_iter()
             .collect()
         // this dance is necessary to eagerly run const_expr so all errors are reported
    };
    let _icx = push_ctxt("const_expr");
    Ok(match e.node {
        hir::ExprLit(ref lit) => const_lit(cx, e, &**lit),
        hir::ExprBinary(b, ref e1, ref e2) => {
            /* Neither type is bottom, and we expect them to be unified
             * already, so the following is safe. */
            let (te1, ty) = try!(const_expr(cx, &**e1, param_substs, fn_args, trueconst));
            debug!("const_expr_unadjusted: te1={}, ty={:?}",
                   cx.tn().val_to_string(te1),
                   ty);
            assert!(!ty.is_simd());
            let is_float = ty.is_fp();
            let signed = ty.is_signed();

            let (te2, _) = try!(const_expr(cx, &**e2, param_substs, fn_args, trueconst));

            try!(check_binary_expr_validity(cx, e, ty, te1, te2, trueconst));

            unsafe { match b.node {
                hir::BiAdd if is_float => llvm::LLVMConstFAdd(te1, te2),
                hir::BiAdd             => llvm::LLVMConstAdd(te1, te2),

                hir::BiSub if is_float => llvm::LLVMConstFSub(te1, te2),
                hir::BiSub             => llvm::LLVMConstSub(te1, te2),

                hir::BiMul if is_float => llvm::LLVMConstFMul(te1, te2),
                hir::BiMul             => llvm::LLVMConstMul(te1, te2),

                hir::BiDiv if is_float => llvm::LLVMConstFDiv(te1, te2),
                hir::BiDiv if signed   => llvm::LLVMConstSDiv(te1, te2),
                hir::BiDiv             => llvm::LLVMConstUDiv(te1, te2),

                hir::BiRem if is_float => llvm::LLVMConstFRem(te1, te2),
                hir::BiRem if signed   => llvm::LLVMConstSRem(te1, te2),
                hir::BiRem             => llvm::LLVMConstURem(te1, te2),

                hir::BiAnd    => llvm::LLVMConstAnd(te1, te2),
                hir::BiOr     => llvm::LLVMConstOr(te1, te2),
                hir::BiBitXor => llvm::LLVMConstXor(te1, te2),
                hir::BiBitAnd => llvm::LLVMConstAnd(te1, te2),
                hir::BiBitOr  => llvm::LLVMConstOr(te1, te2),
                hir::BiShl    => {
                    let te2 = base::cast_shift_const_rhs(b.node, te1, te2);
                    llvm::LLVMConstShl(te1, te2)
                },
                hir::BiShr    => {
                    let te2 = base::cast_shift_const_rhs(b.node, te1, te2);
                    if signed { llvm::LLVMConstAShr(te1, te2) }
                    else      { llvm::LLVMConstLShr(te1, te2) }
                },
                hir::BiEq | hir::BiNe | hir::BiLt | hir::BiLe | hir::BiGt | hir::BiGe => {
                    if is_float {
                        let cmp = base::bin_op_to_fcmp_predicate(cx, b.node);
                        ConstFCmp(cmp, te1, te2)
                    } else {
                        let cmp = base::bin_op_to_icmp_predicate(cx, b.node, signed);
                        ConstICmp(cmp, te1, te2)
                    }
                },
            } } // unsafe { match b.node {
        },
        hir::ExprUnary(u, ref inner_e) => {
            let (te, ty) = try!(const_expr(cx, &**inner_e, param_substs, fn_args, trueconst));

            try!(check_unary_expr_validity(cx, e, ty, te, trueconst));

            let is_float = ty.is_fp();
            unsafe { match u {
                hir::UnDeref           => const_deref(cx, te, ty).0,
                hir::UnNot             => llvm::LLVMConstNot(te),
                hir::UnNeg if is_float => llvm::LLVMConstFNeg(te),
                hir::UnNeg             => llvm::LLVMConstNeg(te),
            } }
        },
        hir::ExprField(ref base, field) => {
            let (bv, bt) = try!(const_expr(cx, &**base, param_substs, fn_args, trueconst));
            let brepr = adt::represent_type(cx, bt);
            let vinfo = VariantInfo::from_ty(cx.tcx(), bt, None);
            let ix = vinfo.field_index(field.node);
            adt::const_get_field(cx, &*brepr, bv, vinfo.discr, ix)
        },
        hir::ExprTupField(ref base, idx) => {
            let (bv, bt) = try!(const_expr(cx, &**base, param_substs, fn_args, trueconst));
            let brepr = adt::represent_type(cx, bt);
            let vinfo = VariantInfo::from_ty(cx.tcx(), bt, None);
            adt::const_get_field(cx, &*brepr, bv, vinfo.discr, idx.node)
        },
        hir::ExprIndex(ref base, ref index) => {
            let (bv, bt) = try!(const_expr(cx, &**base, param_substs, fn_args, trueconst));
            let iv = match eval_const_expr_partial(cx.tcx(), &index, ExprTypeChecked, None) {
                Ok(ConstVal::Int(i)) => i as u64,
                Ok(ConstVal::Uint(u)) => u,
                _ => cx.sess().span_bug(index.span,
                                        "index is not an integer-constant expression")
            };
            let (arr, len) = match bt.sty {
                ty::TyArray(_, u) => (bv, C_uint(cx, u)),
                ty::TySlice(_) | ty::TyStr => {
                    let e1 = const_get_elt(cx, bv, &[0]);
                    (const_deref_ptr(cx, e1), const_get_elt(cx, bv, &[1]))
                },
                ty::TyRef(_, mt) => match mt.ty.sty {
                    ty::TyArray(_, u) => {
                        (const_deref_ptr(cx, bv), C_uint(cx, u))
                    },
                    _ => cx.sess().span_bug(base.span,
                                            &format!("index-expr base must be a vector \
                                                      or string type, found {:?}",
                                                     bt)),
                },
                _ => cx.sess().span_bug(base.span,
                                        &format!("index-expr base must be a vector \
                                                  or string type, found {:?}",
                                                 bt)),
            };

            let len = unsafe { llvm::LLVMConstIntGetZExtValue(len) as u64 };
            let len = match bt.sty {
                ty::TyBox(ty) | ty::TyRef(_, ty::TypeAndMut{ty, ..}) => match ty.sty {
                    ty::TyStr => {
                        assert!(len > 0);
                        len - 1
                    },
                    _ => len,
                },
                _ => len,
            };
            if iv >= len {
                // FIXME #3170: report this earlier on in the const-eval
                // pass. Reporting here is a bit late.
                span_err!(cx.sess(), e.span, E0515,
                          "const index-expr is out of bounds");
                C_undef(val_ty(arr).element_type())
            } else {
                const_get_elt(cx, arr, &[iv as c_uint])
            }
        },
        hir::ExprCast(ref base, _) => {
            let t_cast = ety;
            let llty = type_of::type_of(cx, t_cast);
            let (v, t_expr) = try!(const_expr(cx, &**base, param_substs, fn_args, trueconst));
            debug!("trans_const_cast({:?} as {:?})", t_expr, t_cast);
            if expr::cast_is_noop(cx.tcx(), base, t_expr, t_cast) {
                return Ok(v);
            }
            if type_is_fat_ptr(cx.tcx(), t_expr) {
                // Fat pointer casts.
                let t_cast_inner =
                    t_cast.builtin_deref(true, ty::NoPreference).expect("cast to non-pointer").ty;
                let ptr_ty = type_of::in_memory_type_of(cx, t_cast_inner).ptr_to();
                let addr = ptrcast(const_get_elt(cx, v, &[abi::FAT_PTR_ADDR as u32]),
                                   ptr_ty);
                if type_is_fat_ptr(cx.tcx(), t_cast) {
                    let info = const_get_elt(cx, v, &[abi::FAT_PTR_EXTRA as u32]);
                    return Ok(C_struct(cx, &[addr, info], false))
                } else {
                    return Ok(addr);
                }
            }
            unsafe { match (
                CastTy::from_ty(t_expr).expect("bad input type for cast"),
                CastTy::from_ty(t_cast).expect("bad output type for cast"),
            ) {
                (CastTy::Int(IntTy::CEnum), CastTy::Int(_)) => {
                    let repr = adt::represent_type(cx, t_expr);
                    let discr = adt::const_get_discrim(cx, &*repr, v);
                    let iv = C_integral(cx.int_type(), discr, false);
                    let s = adt::is_discr_signed(&*repr) as Bool;
                    llvm::LLVMConstIntCast(iv, llty.to_ref(), s)
                },
                (CastTy::Int(_), CastTy::Int(_)) => {
                    let s = t_expr.is_signed() as Bool;
                    llvm::LLVMConstIntCast(v, llty.to_ref(), s)
                },
                (CastTy::Int(_), CastTy::Float) => {
                    if t_expr.is_signed() {
                        llvm::LLVMConstSIToFP(v, llty.to_ref())
                    } else {
                        llvm::LLVMConstUIToFP(v, llty.to_ref())
                    }
                },
                (CastTy::Float, CastTy::Float) => llvm::LLVMConstFPCast(v, llty.to_ref()),
                (CastTy::Float, CastTy::Int(IntTy::I)) => llvm::LLVMConstFPToSI(v, llty.to_ref()),
                (CastTy::Float, CastTy::Int(_)) => llvm::LLVMConstFPToUI(v, llty.to_ref()),
                (CastTy::Ptr(_), CastTy::Ptr(_)) | (CastTy::FnPtr, CastTy::Ptr(_))
                | (CastTy::RPtr(_), CastTy::Ptr(_)) => {
                    ptrcast(v, llty)
                },
                (CastTy::FnPtr, CastTy::FnPtr) => ptrcast(v, llty), // isn't this a coercion?
                (CastTy::Int(_), CastTy::Ptr(_)) => llvm::LLVMConstIntToPtr(v, llty.to_ref()),
                (CastTy::Ptr(_), CastTy::Int(_)) | (CastTy::FnPtr, CastTy::Int(_)) => {
                  llvm::LLVMConstPtrToInt(v, llty.to_ref())
                },
                _ => {
                  cx.sess().impossible_case(e.span,
                                            "bad combination of types for cast")
                },
            } } // unsafe { match ( ... ) {
        },
        hir::ExprAddrOf(hir::MutImmutable, ref sub) => {
            // If this is the address of some static, then we need to return
            // the actual address of the static itself (short circuit the rest
            // of const eval).
            let mut cur = sub;
            loop {
                match cur.node {
                    hir::ExprBlock(ref blk) => {
                        if let Some(ref sub) = blk.expr {
                            cur = sub;
                        } else {
                            break;
                        }
                    },
                    _ => break,
                }
            }
            let opt_def = cx.tcx().def_map.borrow().get(&cur.id).map(|d| d.full_def());
            if let Some(def::DefStatic(def_id, _)) = opt_def {
                common::get_static_val(cx, def_id, ety)
            } else {
                // If this isn't the address of a static, then keep going through
                // normal constant evaluation.
                let (v, ty) = try!(const_expr(cx, &**sub, param_substs, fn_args, trueconst));
                addr_of(cx, v, type_of::align_of(cx, ty), "ref")
            }
        },
        hir::ExprAddrOf(hir::MutMutable, ref sub) => {
            let (v, ty) = try!(const_expr(cx, &**sub, param_substs, fn_args, trueconst));
            addr_of_mut(cx, v, type_of::align_of(cx, ty), "ref_mut_slice")
        },
        hir::ExprTup(ref es) => {
            let repr = adt::represent_type(cx, ety);
            let vals = try!(map_list(&es[..]));
            adt::trans_const(cx, &*repr, 0, &vals[..])
        },
        hir::ExprStruct(_, ref fs, ref base_opt) => {
            let repr = adt::represent_type(cx, ety);

            let base_val = match *base_opt {
                Some(ref base) => Some(try!(const_expr(
                    cx,
                    &**base,
                    param_substs,
                    fn_args,
                    trueconst,
                ))),
                None => None
            };

            let VariantInfo { discr, fields } = VariantInfo::of_node(cx.tcx(), ety, e.id);
            let cs = fields.iter().enumerate().map(|(ix, &Field(f_name, _))| {
                match (fs.iter().find(|f| f_name == f.name.node), base_val) {
                    (Some(ref f), _) => {
                        const_expr(cx, &*f.expr, param_substs, fn_args, trueconst).map(|(l, _)| l)
                    },
                    (_, Some((bv, _))) => Ok(adt::const_get_field(cx, &*repr, bv, discr, ix)),
                    (_, None) => cx.sess().span_bug(e.span, "missing struct field"),
                }
            })
            .collect::<Vec<Result<_, ConstEvalFailure>>>()
            .into_iter()
            .collect::<Result<Vec<_>,ConstEvalFailure>>();
            let cs = try!(cs);
            if ety.is_simd() {
                C_vector(&cs[..])
            } else {
                adt::trans_const(cx, &*repr, discr, &cs[..])
            }
        },
        hir::ExprVec(ref es) => {
            let unit_ty = ety.sequence_element_type(cx.tcx());
            let llunitty = type_of::type_of(cx, unit_ty);
            let vs = es.iter()
                       .map(|e| const_expr(
                           cx,
                           &**e,
                           param_substs,
                           fn_args,
                           trueconst,
                       ).map(|(l, _)| l))
                       .collect::<Vec<Result<_, ConstEvalFailure>>>()
                       .into_iter()
                       .collect::<Result<Vec<_>, ConstEvalFailure>>();
            let vs = try!(vs);
            // If the vector contains enums, an LLVM array won't work.
            if vs.iter().any(|vi| val_ty(*vi) != llunitty) {
                C_struct(cx, &vs[..], false)
            } else {
                C_array(llunitty, &vs[..])
            }
        },
        hir::ExprRepeat(ref elem, ref count) => {
            let unit_ty = ety.sequence_element_type(cx.tcx());
            let llunitty = type_of::type_of(cx, unit_ty);
            let n = cx.tcx().eval_repeat_count(count);
            let unit_val = try!(const_expr(cx, &**elem, param_substs, fn_args, trueconst)).0;
            let vs = vec![unit_val; n];
            if val_ty(unit_val) != llunitty {
                C_struct(cx, &vs[..], false)
            } else {
                C_array(llunitty, &vs[..])
            }
        },
        hir::ExprPath(..) => {
            let def = cx.tcx().def_map.borrow().get(&e.id).unwrap().full_def();
            match def {
                def::DefLocal(_, id) => {
                    if let Some(val) = fn_args.and_then(|args| args.get(&id).cloned()) {
                        val
                    } else {
                        cx.sess().span_bug(e.span, "const fn argument not found")
                    }
                }
                def::DefFn(..) | def::DefMethod(..) => {
                    expr::trans_def_fn_unadjusted(cx, e, def, param_substs).val
                }
                def::DefConst(def_id) | def::DefAssociatedConst(def_id) => {
                    const_deref_ptr(cx, try!(get_const_val(cx, def_id, e)))
                }
                def::DefVariant(enum_did, variant_did, _) => {
                    let vinfo = cx.tcx().lookup_adt_def(enum_did).variant_with_id(variant_did);
                    match vinfo.kind() {
                        ty::VariantKind::Unit => {
                            let repr = adt::represent_type(cx, ety);
                            adt::trans_const(cx, &*repr, vinfo.disr_val, &[])
                        }
                        ty::VariantKind::Tuple => {
                            expr::trans_def_fn_unadjusted(cx, e, def, param_substs).val
                        }
                        ty::VariantKind::Struct => {
                            cx.sess().span_bug(e.span, "path-expr refers to a dict variant!")
                        }
                    }
                }
                def::DefStruct(_) => {
                    if let ty::TyBareFn(..) = ety.sty {
                        // Tuple struct.
                        expr::trans_def_fn_unadjusted(cx, e, def, param_substs).val
                    } else {
                        // Unit struct.
                        C_null(type_of::type_of(cx, ety))
                    }
                }
                _ => {
                    cx.sess().span_bug(e.span, "expected a const, fn, struct, \
                                                or variant def")
                }
            }
        },
        hir::ExprCall(ref callee, ref args) => {
            let mut callee = &**callee;
            loop {
                callee = match callee.node {
                    hir::ExprBlock(ref block) => match block.expr {
                        Some(ref tail) => &**tail,
                        None => break,
                    },
                    _ => break,
                };
            }
            let def = cx.tcx().def_map.borrow()[&callee.id].full_def();
            let arg_vals = try!(map_list(args));
            match def {
                def::DefFn(did, _) | def::DefMethod(did) => {
                    try!(const_fn_call(
                        cx,
                        ExprId(callee.id),
                        did,
                        &arg_vals,
                        param_substs,
                        trueconst,
                    ))
                }
                def::DefStruct(_) => {
                    if ety.is_simd() {
                        C_vector(&arg_vals[..])
                    } else {
                        let repr = adt::represent_type(cx, ety);
                        adt::trans_const(cx, &*repr, 0, &arg_vals[..])
                    }
                }
                def::DefVariant(enum_did, variant_did, _) => {
                    let repr = adt::represent_type(cx, ety);
                    let vinfo = cx.tcx().lookup_adt_def(enum_did).variant_with_id(variant_did);
                    adt::trans_const(cx,
                                     &*repr,
                                     vinfo.disr_val,
                                     &arg_vals[..])
                }
                _ => cx.sess().span_bug(e.span, "expected a struct, variant, or const fn def"),
            }
        },
        hir::ExprMethodCall(_, _, ref args) => {
            let arg_vals = try!(map_list(args));
            let method_call = ty::MethodCall::expr(e.id);
            let method_did = cx.tcx().tables.borrow().method_map[&method_call].def_id;
            try!(const_fn_call(cx, MethodCallKey(method_call),
                               method_did, &arg_vals, param_substs, trueconst))
        },
        hir::ExprBlock(ref block) => {
            match block.expr {
                Some(ref expr) => try!(const_expr(
                    cx,
                    &**expr,
                    param_substs,
                    fn_args,
                    trueconst,
                )).0,
                None => C_nil(cx),
            }
        },
        hir::ExprClosure(_, ref decl, ref body) => {
            match ety.sty {
                ty::TyClosure(def_id, ref substs) => {
                    closure::trans_closure_expr(closure::Dest::Ignore(cx), decl,
                                                body, e.id, def_id, substs);
                }
                _ =>
                    cx.sess().span_bug(
                        e.span,
                        &format!("bad type for closure expr: {:?}", ety))
            }
            C_null(type_of::type_of(cx, ety))
        },
        _ => cx.sess().span_bug(e.span,
                                "bad constant expression type in consts::const_expr"),
    })
}

pub fn trans_static(ccx: &CrateContext,
                    m: hir::Mutability,
                    expr: &hir::Expr,
                    id: ast::NodeId,
                    attrs: &Vec<ast::Attribute>)
                    -> Result<ValueRef, ConstEvalErr> {
    unsafe {
        let _icx = push_ctxt("trans_static");
        let g = base::get_item_val(ccx, id);

        let empty_substs = ccx.tcx().mk_substs(Substs::trans_empty());
        let (v, _) = try!(const_expr(
            ccx,
            expr,
            empty_substs,
            None,
            TrueConst::Yes,
        ).map_err(|e| e.into_inner()));

        // boolean SSA values are i1, but they have to be stored in i8 slots,
        // otherwise some LLVM optimization passes don't work as expected
        let mut val_llty = llvm::LLVMTypeOf(v);
        let v = if val_llty == Type::i1(ccx).to_ref() {
            val_llty = Type::i8(ccx).to_ref();
            llvm::LLVMConstZExt(v, val_llty)
        } else {
            v
        };

        let ty = ccx.tcx().node_id_to_type(id);
        let llty = type_of::type_of(ccx, ty);
        let g = if val_llty == llty.to_ref() {
            g
        } else {
            // If we created the global with the wrong type,
            // correct the type.
            let empty_string = CString::new("").unwrap();
            let name_str_ref = CStr::from_ptr(llvm::LLVMGetValueName(g));
            let name_string = CString::new(name_str_ref.to_bytes()).unwrap();
            llvm::LLVMSetValueName(g, empty_string.as_ptr());
            let new_g = llvm::LLVMGetOrInsertGlobal(
                ccx.llmod(), name_string.as_ptr(), val_llty);
            // To avoid breaking any invariants, we leave around the old
            // global for the moment; we'll replace all references to it
            // with the new global later. (See base::trans_crate.)
            ccx.statics_to_rauw().borrow_mut().push((g, new_g));
            new_g
        };
        llvm::LLVMSetAlignment(g, type_of::align_of(ccx, ty));
        llvm::LLVMSetInitializer(g, v);

        // As an optimization, all shared statics which do not have interior
        // mutability are placed into read-only memory.
        if m != hir::MutMutable {
            let tcontents = ty.type_contents(ccx.tcx());
            if !tcontents.interior_unsafe() {
                llvm::LLVMSetGlobalConstant(g, llvm::True);
            }
        }

        debuginfo::create_global_var_metadata(ccx, id, g);

        if attr::contains_name(attrs,
                               "thread_local") {
            llvm::set_thread_local(g, true);
        }
        Ok(g)
    }
}
