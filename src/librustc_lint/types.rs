// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::{infer};
use middle::def_id::DefId;
use middle::subst::Substs;
use middle::ty::{self, Ty};
use middle::const_eval::{eval_const_expr_partial, ConstVal};
use middle::const_eval::EvalHint::ExprTypeChecked;
use util::nodemap::{FnvHashSet};
use lint::{LateContext, LintContext, LintArray};
use lint::{LintPass, LateLintPass};

use std::cmp;
use std::{i8, i16, i32, i64, u8, u16, u32, u64, f32, f64};

use syntax::{abi, ast};
use syntax::attr::{self, AttrMetaMethods};
use syntax::codemap::{self, Span};
use syntax::feature_gate::{emit_feature_err, GateIssue};
use syntax::ast::{TyIs, TyUs, TyI8, TyU8, TyI16, TyU16, TyI32, TyU32, TyI64, TyU64};

use rustc_front::hir;
use rustc_front::intravisit::{self, Visitor};
use rustc_front::util::is_shift_binop;

declare_lint! {
    UNUSED_COMPARISONS,
    Warn,
    "comparisons made useless by limits of the types involved"
}

declare_lint! {
    OVERFLOWING_LITERALS,
    Warn,
    "literal out of range for its type"
}

declare_lint! {
    EXCEEDING_BITSHIFTS,
    Deny,
    "shift exceeds the type's number of bits"
}

#[derive(Copy, Clone)]
pub struct TypeLimits {
    /// Id of the last visited negated expression
    negated_expr_id: ast::NodeId,
}

impl TypeLimits {
    pub fn new() -> TypeLimits {
        TypeLimits {
            negated_expr_id: !0,
        }
    }
}

impl LintPass for TypeLimits {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_COMPARISONS, OVERFLOWING_LITERALS, EXCEEDING_BITSHIFTS)
    }
}

impl LateLintPass for TypeLimits {
    fn check_expr(&mut self, cx: &LateContext, e: &hir::Expr) {
        match e.node {
            hir::ExprUnary(hir::UnNeg, ref expr) => {
                match expr.node  {
                    hir::ExprLit(ref lit) => {
                        match lit.node {
                            ast::LitInt(_, ast::UnsignedIntLit(_)) => {
                                check_unsigned_negation_feature(cx, e.span);
                            },
                            ast::LitInt(_, ast::UnsuffixedIntLit(_)) => {
                                if let ty::TyUint(_) = cx.tcx.node_id_to_type(e.id).sty {
                                    check_unsigned_negation_feature(cx, e.span);
                                }
                            },
                            _ => ()
                        }
                    },
                    _ => {
                        let t = cx.tcx.node_id_to_type(expr.id);
                        match t.sty {
                            ty::TyUint(_) => {
                                check_unsigned_negation_feature(cx, e.span);
                            },
                            _ => ()
                        }
                    }
                };
                // propagate negation, if the negation itself isn't negated
                if self.negated_expr_id != e.id {
                    self.negated_expr_id = expr.id;
                }
            },
            hir::ExprBinary(binop, ref l, ref r) => {
                if is_comparison(binop) && !check_limits(cx.tcx, binop, &**l, &**r) {
                    cx.span_lint(UNUSED_COMPARISONS, e.span,
                                 "comparison is useless due to type limits");
                }

                if is_shift_binop(binop.node) {
                    let opt_ty_bits = match cx.tcx.node_id_to_type(l.id).sty {
                        ty::TyInt(t) => Some(int_ty_bits(t, cx.sess().target.int_type)),
                        ty::TyUint(t) => Some(uint_ty_bits(t, cx.sess().target.uint_type)),
                        _ => None
                    };

                    if let Some(bits) = opt_ty_bits {
                        let exceeding = if let hir::ExprLit(ref lit) = r.node {
                            if let ast::LitInt(shift, _) = lit.node { shift >= bits }
                            else { false }
                        } else {
                            match eval_const_expr_partial(cx.tcx, &r, ExprTypeChecked, None) {
                                Ok(ConstVal::Int(shift)) => { shift as u64 >= bits },
                                Ok(ConstVal::Uint(shift)) => { shift >= bits },
                                _ => { false }
                            }
                        };
                        if exceeding {
                            cx.span_lint(EXCEEDING_BITSHIFTS, e.span,
                                         "bitshift exceeds the type's number of bits");
                        }
                    };
                }
            },
            hir::ExprLit(ref lit) => {
                match cx.tcx.node_id_to_type(e.id).sty {
                    ty::TyInt(t) => {
                        match lit.node {
                            ast::LitInt(v, ast::SignedIntLit(_, ast::Plus)) |
                            ast::LitInt(v, ast::UnsuffixedIntLit(ast::Plus)) => {
                                let int_type = if let ast::TyIs = t {
                                    cx.sess().target.int_type
                                } else {
                                    t
                                };
                                let (_, max) = int_ty_range(int_type);
                                let negative = self.negated_expr_id == e.id;

                                // Detect literal value out of range [min, max] inclusive
                                // avoiding use of -min to prevent overflow/panic
                                if (negative && v > max as u64 + 1) ||
                                   (!negative && v > max as u64) {
                                    cx.span_lint(OVERFLOWING_LITERALS, e.span,
                                                 &*format!("literal out of range for {:?}", t));
                                    return;
                                }
                            }
                            _ => panic!()
                        };
                    },
                    ty::TyUint(t) => {
                        let uint_type = if let ast::TyUs = t {
                            cx.sess().target.uint_type
                        } else {
                            t
                        };
                        let (min, max) = uint_ty_range(uint_type);
                        let lit_val: u64 = match lit.node {
                            ast::LitByte(_v) => return,  // _v is u8, within range by definition
                            ast::LitInt(v, _) => v,
                            _ => panic!()
                        };
                        if lit_val < min || lit_val > max {
                            cx.span_lint(OVERFLOWING_LITERALS, e.span,
                                         &*format!("literal out of range for {:?}", t));
                        }
                    },
                    ty::TyFloat(t) => {
                        let (min, max) = float_ty_range(t);
                        let lit_val: f64 = match lit.node {
                            ast::LitFloat(ref v, _) |
                            ast::LitFloatUnsuffixed(ref v) => {
                                match v.parse() {
                                    Ok(f) => f,
                                    Err(_) => return
                                }
                            }
                            _ => panic!()
                        };
                        if lit_val < min || lit_val > max {
                            cx.span_lint(OVERFLOWING_LITERALS, e.span,
                                         &*format!("literal out of range for {:?}", t));
                        }
                    },
                    _ => ()
                };
            },
            _ => ()
        };

        fn is_valid<T:cmp::PartialOrd>(binop: hir::BinOp, v: T,
                                min: T, max: T) -> bool {
            match binop.node {
                hir::BiLt => v >  min && v <= max,
                hir::BiLe => v >= min && v <  max,
                hir::BiGt => v >= min && v <  max,
                hir::BiGe => v >  min && v <= max,
                hir::BiEq | hir::BiNe => v >= min && v <= max,
                _ => panic!()
            }
        }

        fn rev_binop(binop: hir::BinOp) -> hir::BinOp {
            codemap::respan(binop.span, match binop.node {
                hir::BiLt => hir::BiGt,
                hir::BiLe => hir::BiGe,
                hir::BiGt => hir::BiLt,
                hir::BiGe => hir::BiLe,
                _ => return binop
            })
        }

        // for isize & usize, be conservative with the warnings, so that the
        // warnings are consistent between 32- and 64-bit platforms
        fn int_ty_range(int_ty: ast::IntTy) -> (i64, i64) {
            match int_ty {
                ast::TyIs => (i64::MIN,        i64::MAX),
                ast::TyI8 =>    (i8::MIN  as i64, i8::MAX  as i64),
                ast::TyI16 =>   (i16::MIN as i64, i16::MAX as i64),
                ast::TyI32 =>   (i32::MIN as i64, i32::MAX as i64),
                ast::TyI64 =>   (i64::MIN,        i64::MAX)
            }
        }

        fn uint_ty_range(uint_ty: ast::UintTy) -> (u64, u64) {
            match uint_ty {
                ast::TyUs => (u64::MIN,         u64::MAX),
                ast::TyU8 =>    (u8::MIN   as u64, u8::MAX   as u64),
                ast::TyU16 =>   (u16::MIN  as u64, u16::MAX  as u64),
                ast::TyU32 =>   (u32::MIN  as u64, u32::MAX  as u64),
                ast::TyU64 =>   (u64::MIN,         u64::MAX)
            }
        }

        fn float_ty_range(float_ty: ast::FloatTy) -> (f64, f64) {
            match float_ty {
                ast::TyF32 => (f32::MIN as f64, f32::MAX as f64),
                ast::TyF64 => (f64::MIN,        f64::MAX)
            }
        }

        fn int_ty_bits(int_ty: ast::IntTy, target_int_ty: ast::IntTy) -> u64 {
            match int_ty {
                ast::TyIs => int_ty_bits(target_int_ty, target_int_ty),
                ast::TyI8 =>    i8::BITS  as u64,
                ast::TyI16 =>   i16::BITS as u64,
                ast::TyI32 =>   i32::BITS as u64,
                ast::TyI64 =>   i64::BITS as u64
            }
        }

        fn uint_ty_bits(uint_ty: ast::UintTy, target_uint_ty: ast::UintTy) -> u64 {
            match uint_ty {
                ast::TyUs => uint_ty_bits(target_uint_ty, target_uint_ty),
                ast::TyU8 =>    u8::BITS  as u64,
                ast::TyU16 =>   u16::BITS as u64,
                ast::TyU32 =>   u32::BITS as u64,
                ast::TyU64 =>   u64::BITS as u64
            }
        }

        fn check_limits(tcx: &ty::ctxt, binop: hir::BinOp,
                        l: &hir::Expr, r: &hir::Expr) -> bool {
            let (lit, expr, swap) = match (&l.node, &r.node) {
                (&hir::ExprLit(_), _) => (l, r, true),
                (_, &hir::ExprLit(_)) => (r, l, false),
                _ => return true
            };
            // Normalize the binop so that the literal is always on the RHS in
            // the comparison
            let norm_binop = if swap {
                rev_binop(binop)
            } else {
                binop
            };
            match tcx.node_id_to_type(expr.id).sty {
                ty::TyInt(int_ty) => {
                    let (min, max) = int_ty_range(int_ty);
                    let lit_val: i64 = match lit.node {
                        hir::ExprLit(ref li) => match li.node {
                            ast::LitInt(v, ast::SignedIntLit(_, ast::Plus)) |
                            ast::LitInt(v, ast::UnsuffixedIntLit(ast::Plus)) => v as i64,
                            ast::LitInt(v, ast::SignedIntLit(_, ast::Minus)) |
                            ast::LitInt(v, ast::UnsuffixedIntLit(ast::Minus)) => -(v as i64),
                            _ => return true
                        },
                        _ => panic!()
                    };
                    is_valid(norm_binop, lit_val, min, max)
                }
                ty::TyUint(uint_ty) => {
                    let (min, max): (u64, u64) = uint_ty_range(uint_ty);
                    let lit_val: u64 = match lit.node {
                        hir::ExprLit(ref li) => match li.node {
                            ast::LitInt(v, _) => v,
                            _ => return true
                        },
                        _ => panic!()
                    };
                    is_valid(norm_binop, lit_val, min, max)
                }
                _ => true
            }
        }

        fn is_comparison(binop: hir::BinOp) -> bool {
            match binop.node {
                hir::BiEq | hir::BiLt | hir::BiLe |
                hir::BiNe | hir::BiGe | hir::BiGt => true,
                _ => false
            }
        }

        fn check_unsigned_negation_feature(cx: &LateContext, span: Span) {
            if !cx.sess().features.borrow().negate_unsigned {
                emit_feature_err(
                    &cx.sess().parse_sess.span_diagnostic,
                    "negate_unsigned",
                    span,
                    GateIssue::Language,
                    "unary negation of unsigned integers may be removed in the future");
            }
        }
    }
}

declare_lint! {
    IMPROPER_CTYPES,
    Warn,
    "proper use of libc types in foreign modules"
}

struct ImproperCTypesVisitor<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>
}

enum FfiResult {
    FfiSafe,
    FfiUnsafe(&'static str),
    FfiBadStruct(DefId, &'static str),
    FfiBadEnum(DefId, &'static str)
}

/// Check if this enum can be safely exported based on the
/// "nullable pointer optimization". Currently restricted
/// to function pointers and references, but could be
/// expanded to cover NonZero raw pointers and newtypes.
/// FIXME: This duplicates code in trans.
fn is_repr_nullable_ptr<'tcx>(tcx: &ty::ctxt<'tcx>,
                              def: ty::AdtDef<'tcx>,
                              substs: &Substs<'tcx>)
                              -> bool {
    if def.variants.len() == 2 {
        let data_idx;

        if def.variants[0].fields.is_empty() {
            data_idx = 1;
        } else if def.variants[1].fields.is_empty() {
            data_idx = 0;
        } else {
            return false;
        }

        if def.variants[data_idx].fields.len() == 1 {
            match def.variants[data_idx].fields[0].ty(tcx, substs).sty {
                ty::TyBareFn(None, _) => { return true; }
                ty::TyRef(..) => { return true; }
                _ => { }
            }
        }
    }
    false
}

fn ast_ty_to_normalized<'tcx>(tcx: &ty::ctxt<'tcx>,
                              id: ast::NodeId)
                              -> Ty<'tcx> {
    let tty = match tcx.ast_ty_to_ty_cache.borrow().get(&id) {
        Some(&t) => t,
        None => panic!("ast_ty_to_ty_cache was incomplete after typeck!")
    };
    infer::normalize_associated_type(tcx, &tty)
}

impl<'a, 'tcx> ImproperCTypesVisitor<'a, 'tcx> {
    /// Check if the given type is "ffi-safe" (has a stable, well-defined
    /// representation which can be exported to C code).
    fn check_type_for_ffi(&self,
                          cache: &mut FnvHashSet<Ty<'tcx>>,
                          ty: Ty<'tcx>)
                          -> FfiResult {
        use self::FfiResult::*;
        let cx = &self.cx.tcx;

        // Protect against infinite recursion, for example
        // `struct S(*mut S);`.
        // FIXME: A recursion limit is necessary as well, for irregular
        // recusive types.
        if !cache.insert(ty) {
            return FfiSafe;
        }

        match ty.sty {
            ty::TyStruct(def, substs) => {
                if !cx.lookup_repr_hints(def.did).contains(&attr::ReprExtern) {
                    return FfiUnsafe(
                        "found struct without foreign-function-safe \
                         representation annotation in foreign module, \
                         consider adding a #[repr(C)] attribute to \
                         the type");
                }

                // We can't completely trust repr(C) markings; make sure the
                // fields are actually safe.
                if def.struct_variant().fields.is_empty() {
                    return FfiUnsafe(
                        "found zero-size struct in foreign module, consider \
                         adding a member to this struct");
                }

                for field in &def.struct_variant().fields {
                    let field_ty = infer::normalize_associated_type(cx, &field.ty(cx, substs));
                    let r = self.check_type_for_ffi(cache, field_ty);
                    match r {
                        FfiSafe => {}
                        FfiBadStruct(..) | FfiBadEnum(..) => { return r; }
                        FfiUnsafe(s) => { return FfiBadStruct(def.did, s); }
                    }
                }
                FfiSafe
            }
            ty::TyEnum(def, substs) => {
                if def.variants.is_empty() {
                    // Empty enums are okay... although sort of useless.
                    return FfiSafe
                }

                // Check for a repr() attribute to specify the size of the
                // discriminant.
                let repr_hints = cx.lookup_repr_hints(def.did);
                match &**repr_hints {
                    [] => {
                        // Special-case types like `Option<extern fn()>`.
                        if !is_repr_nullable_ptr(cx, def, substs) {
                            return FfiUnsafe(
                                "found enum without foreign-function-safe \
                                 representation annotation in foreign module, \
                                 consider adding a #[repr(...)] attribute to \
                                 the type")
                        }
                    }
                    [ref hint] => {
                        if !hint.is_ffi_safe() {
                            // FIXME: This shouldn't be reachable: we should check
                            // this earlier.
                            return FfiUnsafe(
                                "enum has unexpected #[repr(...)] attribute")
                        }

                        // Enum with an explicitly sized discriminant; either
                        // a C-style enum or a discriminated union.

                        // The layout of enum variants is implicitly repr(C).
                        // FIXME: Is that correct?
                    }
                    _ => {
                        // FIXME: This shouldn't be reachable: we should check
                        // this earlier.
                        return FfiUnsafe(
                            "enum has too many #[repr(...)] attributes");
                    }
                }

                // Check the contained variants.
                for variant in &def.variants {
                    for field in &variant.fields {
                        let arg = infer::normalize_associated_type(cx, &field.ty(cx, substs));
                        let r = self.check_type_for_ffi(cache, arg);
                        match r {
                            FfiSafe => {}
                            FfiBadStruct(..) | FfiBadEnum(..) => { return r; }
                            FfiUnsafe(s) => { return FfiBadEnum(def.did, s); }
                        }
                    }
                }
                FfiSafe
            }

            ty::TyChar => {
                FfiUnsafe("found Rust type `char` in foreign module, while \
                           `u32` or `libc::wchar_t` should be used")
            }

            // Primitive types with a stable representation.
            ty::TyBool | ty::TyInt(..) | ty::TyUint(..) |
            ty::TyFloat(..) => FfiSafe,

            ty::TyBox(..) => {
                FfiUnsafe("found Rust type Box<_> in foreign module, \
                           consider using a raw pointer instead")
            }

            ty::TySlice(_) => {
                FfiUnsafe("found Rust slice type in foreign module, \
                           consider using a raw pointer instead")
            }

            ty::TyTrait(..) => {
                FfiUnsafe("found Rust trait type in foreign module, \
                           consider using a raw pointer instead")
            }

            ty::TyStr => {
                FfiUnsafe("found Rust type `str` in foreign module; \
                           consider using a `*const libc::c_char`")
            }

            ty::TyTuple(_) => {
                FfiUnsafe("found Rust tuple type in foreign module; \
                           consider using a struct instead`")
            }

            ty::TyRawPtr(ref m) | ty::TyRef(_, ref m) => {
                self.check_type_for_ffi(cache, m.ty)
            }

            ty::TyArray(ty, _) => {
                self.check_type_for_ffi(cache, ty)
            }

            ty::TyBareFn(None, bare_fn) => {
                match bare_fn.abi {
                    abi::Rust |
                    abi::RustIntrinsic |
                    abi::PlatformIntrinsic |
                    abi::RustCall => {
                        return FfiUnsafe(
                            "found function pointer with Rust calling \
                             convention in foreign module; consider using an \
                             `extern` function pointer")
                    }
                    _ => {}
                }

                let sig = cx.erase_late_bound_regions(&bare_fn.sig);
                match sig.output {
                    ty::FnDiverging => {}
                    ty::FnConverging(output) => {
                        if !output.is_nil() {
                            let r = self.check_type_for_ffi(cache, output);
                            match r {
                                FfiSafe => {}
                                _ => { return r; }
                            }
                        }
                    }
                }
                for arg in sig.inputs {
                    let r = self.check_type_for_ffi(cache, arg);
                    match r {
                        FfiSafe => {}
                        _ => { return r; }
                    }
                }
                FfiSafe
            }

            ty::TyParam(..) | ty::TyInfer(..) | ty::TyError |
            ty::TyClosure(..) | ty::TyProjection(..) |
            ty::TyBareFn(Some(_), _) => {
                panic!("Unexpected type in foreign function")
            }
        }
    }

    fn check_def(&mut self, sp: Span, id: ast::NodeId) {
        let tty = ast_ty_to_normalized(self.cx.tcx, id);

        match ImproperCTypesVisitor::check_type_for_ffi(self, &mut FnvHashSet(), tty) {
            FfiResult::FfiSafe => {}
            FfiResult::FfiUnsafe(s) => {
                self.cx.span_lint(IMPROPER_CTYPES, sp, s);
            }
            FfiResult::FfiBadStruct(_, s) => {
                // FIXME: This diagnostic is difficult to read, and doesn't
                // point at the relevant field.
                self.cx.span_lint(IMPROPER_CTYPES, sp,
                    &format!("found non-foreign-function-safe member in \
                              struct marked #[repr(C)]: {}", s));
            }
            FfiResult::FfiBadEnum(_, s) => {
                // FIXME: This diagnostic is difficult to read, and doesn't
                // point at the relevant variant.
                self.cx.span_lint(IMPROPER_CTYPES, sp,
                    &format!("found non-foreign-function-safe member in \
                              enum: {}", s));
            }
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for ImproperCTypesVisitor<'a, 'tcx> {
    fn visit_ty(&mut self, ty: &hir::Ty) {
        match ty.node {
            hir::TyPath(..) |
            hir::TyBareFn(..) => self.check_def(ty.span, ty.id),
            hir::TyVec(..) => {
                self.cx.span_lint(IMPROPER_CTYPES, ty.span,
                    "found Rust slice type in foreign module, consider \
                     using a raw pointer instead");
            }
            hir::TyFixedLengthVec(ref ty, _) => self.visit_ty(ty),
            hir::TyTup(..) => {
                self.cx.span_lint(IMPROPER_CTYPES, ty.span,
                    "found Rust tuple type in foreign module; \
                     consider using a struct instead`")
            }
            _ => intravisit::walk_ty(self, ty)
        }
    }
}

#[derive(Copy, Clone)]
pub struct ImproperCTypes;

impl LintPass for ImproperCTypes {
    fn get_lints(&self) -> LintArray {
        lint_array!(IMPROPER_CTYPES)
    }
}

fn check_ty(cx: &LateContext, ty: &hir::Ty) {
    let mut vis = ImproperCTypesVisitor { cx: cx };
    vis.visit_ty(ty);
}

fn check_foreign_fn(cx: &LateContext, decl: &hir::FnDecl) {
    for input in &decl.inputs {
        check_ty(cx, &*input.ty);
    }
    if let hir::Return(ref ret_ty) = decl.output {
        let tty = ast_ty_to_normalized(cx.tcx, ret_ty.id);
        if !tty.is_nil() {
            check_ty(cx, &ret_ty);
        }
    }
}

fn should_check_abi(abi: abi::Abi) -> bool {
    ![abi::RustIntrinsic, abi::PlatformIntrinsic, abi::Rust, abi::RustCall].contains(&abi)
}

impl LateLintPass for ImproperCTypes {
    fn check_item(&mut self, cx: &LateContext, it: &hir::Item) {
        if let hir::ItemForeignMod(ref nmod) = it.node {
            if should_check_abi(nmod.abi) {
                for ni in &nmod.items {
                    match ni.node {
                        hir::ForeignItemFn(ref decl, _) => check_foreign_fn(cx, &**decl),
                        hir::ForeignItemStatic(ref t, _) => check_ty(cx, &**t)
                    }
                }
            }
        }
    }
}
