// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_snake_case)]

use rustc::hir::def_id::DefId;
use rustc::ty::subst::Substs;
use rustc::ty::{self, AdtKind, Ty, TyCtxt};
use rustc::ty::layout::{Layout, Primitive};
use rustc::traits::Reveal;
use middle::const_val::ConstVal;
use rustc_const_eval::ConstContext;
use rustc_const_eval::EvalHint::ExprTypeChecked;
use util::nodemap::FxHashSet;
use lint::{LateContext, LintContext, LintArray};
use lint::{LintPass, LateLintPass};

use std::cmp;
use std::{i8, i16, i32, i64, u8, u16, u32, u64, f32, f64};

use syntax::ast;
use syntax::abi::Abi;
use syntax::attr;
use syntax_pos::Span;
use syntax::codemap;

use rustc::hir;

use rustc_i128::{i128, u128};

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

declare_lint! {
    VARIANT_SIZE_DIFFERENCES,
    Allow,
    "detects enums with widely varying variant sizes"
}

#[derive(Copy, Clone)]
pub struct TypeLimits {
    /// Id of the last visited negated expression
    negated_expr_id: ast::NodeId,
}

impl TypeLimits {
    pub fn new() -> TypeLimits {
        TypeLimits { negated_expr_id: ast::DUMMY_NODE_ID }
    }
}

impl LintPass for TypeLimits {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_COMPARISONS,
                    OVERFLOWING_LITERALS,
                    EXCEEDING_BITSHIFTS)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for TypeLimits {
    fn check_expr(&mut self, cx: &LateContext, e: &hir::Expr) {
        match e.node {
            hir::ExprUnary(hir::UnNeg, ref expr) => {
                // propagate negation, if the negation itself isn't negated
                if self.negated_expr_id != e.id {
                    self.negated_expr_id = expr.id;
                }
            }
            hir::ExprBinary(binop, ref l, ref r) => {
                if is_comparison(binop) && !check_limits(cx, binop, &l, &r) {
                    cx.span_lint(UNUSED_COMPARISONS,
                                 e.span,
                                 "comparison is useless due to type limits");
                }

                if binop.node.is_shift() {
                    let opt_ty_bits = match cx.tables.node_id_to_type(l.id).sty {
                        ty::TyInt(t) => Some(int_ty_bits(t, cx.sess().target.int_type)),
                        ty::TyUint(t) => Some(uint_ty_bits(t, cx.sess().target.uint_type)),
                        _ => None,
                    };

                    if let Some(bits) = opt_ty_bits {
                        let exceeding = if let hir::ExprLit(ref lit) = r.node {
                            if let ast::LitKind::Int(shift, _) = lit.node {
                                shift as u64 >= bits
                            } else {
                                false
                            }
                        } else {
                            let const_cx = ConstContext::with_tables(cx.tcx, cx.tables);
                            match const_cx.eval(&r, ExprTypeChecked) {
                                Ok(ConstVal::Integral(i)) => {
                                    i.is_negative() ||
                                    i.to_u64()
                                        .map(|i| i >= bits)
                                        .unwrap_or(true)
                                }
                                _ => false,
                            }
                        };
                        if exceeding {
                            cx.span_lint(EXCEEDING_BITSHIFTS,
                                         e.span,
                                         "bitshift exceeds the type's number of bits");
                        }
                    };
                }
            }
            hir::ExprLit(ref lit) => {
                match cx.tables.node_id_to_type(e.id).sty {
                    ty::TyInt(t) => {
                        match lit.node {
                            ast::LitKind::Int(v, ast::LitIntType::Signed(_)) |
                            ast::LitKind::Int(v, ast::LitIntType::Unsuffixed) => {
                                let int_type = if let ast::IntTy::Is = t {
                                    cx.sess().target.int_type
                                } else {
                                    t
                                };
                                let (_, max) = int_ty_range(int_type);
                                let max = max as u128;
                                let negative = self.negated_expr_id == e.id;

                                // Detect literal value out of range [min, max] inclusive
                                // avoiding use of -min to prevent overflow/panic
                                if (negative && v > max + 1) ||
                                   (!negative && v > max) {
                                    cx.span_lint(OVERFLOWING_LITERALS,
                                                 e.span,
                                                 &format!("literal out of range for {:?}", t));
                                    return;
                                }
                            }
                            _ => bug!(),
                        };
                    }
                    ty::TyUint(t) => {
                        let uint_type = if let ast::UintTy::Us = t {
                            cx.sess().target.uint_type
                        } else {
                            t
                        };
                        let (min, max) = uint_ty_range(uint_type);
                        let lit_val: u128 = match lit.node {
                            // _v is u8, within range by definition
                            ast::LitKind::Byte(_v) => return,
                            ast::LitKind::Int(v, _) => v,
                            _ => bug!(),
                        };
                        if lit_val < min || lit_val > max {
                            cx.span_lint(OVERFLOWING_LITERALS,
                                         e.span,
                                         &format!("literal out of range for {:?}", t));
                        }
                    }
                    ty::TyFloat(t) => {
                        let (min, max) = float_ty_range(t);
                        let lit_val: f64 = match lit.node {
                            ast::LitKind::Float(v, _) |
                            ast::LitKind::FloatUnsuffixed(v) => {
                                match v.as_str().parse() {
                                    Ok(f) => f,
                                    Err(_) => return,
                                }
                            }
                            _ => bug!(),
                        };
                        if lit_val < min || lit_val > max {
                            cx.span_lint(OVERFLOWING_LITERALS,
                                         e.span,
                                         &format!("literal out of range for {:?}", t));
                        }
                    }
                    _ => (),
                };
            }
            _ => (),
        };

        fn is_valid<T: cmp::PartialOrd>(binop: hir::BinOp, v: T, min: T, max: T) -> bool {
            match binop.node {
                hir::BiLt => v > min && v <= max,
                hir::BiLe => v >= min && v < max,
                hir::BiGt => v >= min && v < max,
                hir::BiGe => v > min && v <= max,
                hir::BiEq | hir::BiNe => v >= min && v <= max,
                _ => bug!(),
            }
        }

        fn rev_binop(binop: hir::BinOp) -> hir::BinOp {
            codemap::respan(binop.span,
                            match binop.node {
                                hir::BiLt => hir::BiGt,
                                hir::BiLe => hir::BiGe,
                                hir::BiGt => hir::BiLt,
                                hir::BiGe => hir::BiLe,
                                _ => return binop,
                            })
        }

        // for isize & usize, be conservative with the warnings, so that the
        // warnings are consistent between 32- and 64-bit platforms
        fn int_ty_range(int_ty: ast::IntTy) -> (i128, i128) {
            match int_ty {
                ast::IntTy::Is => (i64::min_value() as i128, i64::max_value() as i128),
                ast::IntTy::I8 => (i8::min_value() as i64 as i128, i8::max_value() as i128),
                ast::IntTy::I16 => (i16::min_value() as i64 as i128, i16::max_value() as i128),
                ast::IntTy::I32 => (i32::min_value() as i64 as i128, i32::max_value() as i128),
                ast::IntTy::I64 => (i64::min_value() as i128, i64::max_value() as i128),
                ast::IntTy::I128 =>(i128::min_value() as i128, i128::max_value()),
            }
        }

        fn uint_ty_range(uint_ty: ast::UintTy) -> (u128, u128) {
            match uint_ty {
                ast::UintTy::Us => (u64::min_value() as u128, u64::max_value() as u128),
                ast::UintTy::U8 => (u8::min_value() as u128, u8::max_value() as u128),
                ast::UintTy::U16 => (u16::min_value() as u128, u16::max_value() as u128),
                ast::UintTy::U32 => (u32::min_value() as u128, u32::max_value() as u128),
                ast::UintTy::U64 => (u64::min_value() as u128, u64::max_value() as u128),
                ast::UintTy::U128 => (u128::min_value(), u128::max_value()),
            }
        }

        fn float_ty_range(float_ty: ast::FloatTy) -> (f64, f64) {
            match float_ty {
                ast::FloatTy::F32 => (f32::MIN as f64, f32::MAX as f64),
                ast::FloatTy::F64 => (f64::MIN, f64::MAX),
            }
        }

        fn int_ty_bits(int_ty: ast::IntTy, target_int_ty: ast::IntTy) -> u64 {
            match int_ty {
                ast::IntTy::Is => int_ty_bits(target_int_ty, target_int_ty),
                ast::IntTy::I8 => 8,
                ast::IntTy::I16 => 16 as u64,
                ast::IntTy::I32 => 32,
                ast::IntTy::I64 => 64,
                ast::IntTy::I128 => 128,
            }
        }

        fn uint_ty_bits(uint_ty: ast::UintTy, target_uint_ty: ast::UintTy) -> u64 {
            match uint_ty {
                ast::UintTy::Us => uint_ty_bits(target_uint_ty, target_uint_ty),
                ast::UintTy::U8 => 8,
                ast::UintTy::U16 => 16,
                ast::UintTy::U32 => 32,
                ast::UintTy::U64 => 64,
                ast::UintTy::U128 => 128,
            }
        }

        fn check_limits(cx: &LateContext,
                        binop: hir::BinOp,
                        l: &hir::Expr,
                        r: &hir::Expr)
                        -> bool {
            let (lit, expr, swap) = match (&l.node, &r.node) {
                (&hir::ExprLit(_), _) => (l, r, true),
                (_, &hir::ExprLit(_)) => (r, l, false),
                _ => return true,
            };
            // Normalize the binop so that the literal is always on the RHS in
            // the comparison
            let norm_binop = if swap { rev_binop(binop) } else { binop };
            match cx.tables.node_id_to_type(expr.id).sty {
                ty::TyInt(int_ty) => {
                    let (min, max) = int_ty_range(int_ty);
                    let lit_val: i128 = match lit.node {
                        hir::ExprLit(ref li) => {
                            match li.node {
                                ast::LitKind::Int(v, ast::LitIntType::Signed(_)) |
                                ast::LitKind::Int(v, ast::LitIntType::Unsuffixed) => v as i128,
                                _ => return true
                            }
                        },
                        _ => bug!()
                    };
                    is_valid(norm_binop, lit_val, min, max)
                }
                ty::TyUint(uint_ty) => {
                    let (min, max) :(u128, u128) = uint_ty_range(uint_ty);
                    let lit_val: u128 = match lit.node {
                        hir::ExprLit(ref li) => {
                            match li.node {
                                ast::LitKind::Int(v, _) => v,
                                _ => return true
                            }
                        },
                        _ => bug!()
                    };
                    is_valid(norm_binop, lit_val, min, max)
                }
                _ => true,
            }
        }

        fn is_comparison(binop: hir::BinOp) -> bool {
            match binop.node {
                hir::BiEq | hir::BiLt | hir::BiLe | hir::BiNe | hir::BiGe | hir::BiGt => true,
                _ => false,
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
    cx: &'a LateContext<'a, 'tcx>,
}

enum FfiResult {
    FfiSafe,
    FfiUnsafe(&'static str),
    FfiBadStruct(DefId, &'static str),
    FfiBadUnion(DefId, &'static str),
    FfiBadEnum(DefId, &'static str),
}

/// Check if this enum can be safely exported based on the
/// "nullable pointer optimization". Currently restricted
/// to function pointers and references, but could be
/// expanded to cover NonZero raw pointers and newtypes.
/// FIXME: This duplicates code in trans.
fn is_repr_nullable_ptr<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  def: &'tcx ty::AdtDef,
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
                ty::TyFnPtr(_) => {
                    return true;
                }
                ty::TyRef(..) => {
                    return true;
                }
                _ => {}
            }
        }
    }
    false
}

impl<'a, 'tcx> ImproperCTypesVisitor<'a, 'tcx> {
    /// Check if the given type is "ffi-safe" (has a stable, well-defined
    /// representation which can be exported to C code).
    fn check_type_for_ffi(&self, cache: &mut FxHashSet<Ty<'tcx>>, ty: Ty<'tcx>) -> FfiResult {
        use self::FfiResult::*;
        let cx = self.cx.tcx;

        // Protect against infinite recursion, for example
        // `struct S(*mut S);`.
        // FIXME: A recursion limit is necessary as well, for irregular
        // recusive types.
        if !cache.insert(ty) {
            return FfiSafe;
        }

        match ty.sty {
            ty::TyAdt(def, substs) => {
                match def.adt_kind() {
                    AdtKind::Struct => {
                        if !cx.lookup_repr_hints(def.did).contains(&attr::ReprExtern) {
                            return FfiUnsafe("found struct without foreign-function-safe \
                                              representation annotation in foreign module, \
                                              consider adding a #[repr(C)] attribute to the type");
                        }

                        // We can't completely trust repr(C) markings; make sure the
                        // fields are actually safe.
                        if def.struct_variant().fields.is_empty() {
                            return FfiUnsafe("found zero-size struct in foreign module, consider \
                                              adding a member to this struct");
                        }

                        for field in &def.struct_variant().fields {
                            let field_ty = cx.normalize_associated_type(&field.ty(cx, substs));
                            let r = self.check_type_for_ffi(cache, field_ty);
                            match r {
                                FfiSafe => {}
                                FfiBadStruct(..) | FfiBadUnion(..) | FfiBadEnum(..) => {
                                    return r;
                                }
                                FfiUnsafe(s) => {
                                    return FfiBadStruct(def.did, s);
                                }
                            }
                        }
                        FfiSafe
                    }
                    AdtKind::Union => {
                        if !cx.lookup_repr_hints(def.did).contains(&attr::ReprExtern) {
                            return FfiUnsafe("found union without foreign-function-safe \
                                              representation annotation in foreign module, \
                                              consider adding a #[repr(C)] attribute to the type");
                        }

                        for field in &def.struct_variant().fields {
                            let field_ty = cx.normalize_associated_type(&field.ty(cx, substs));
                            let r = self.check_type_for_ffi(cache, field_ty);
                            match r {
                                FfiSafe => {}
                                FfiBadStruct(..) | FfiBadUnion(..) | FfiBadEnum(..) => {
                                    return r;
                                }
                                FfiUnsafe(s) => {
                                    return FfiBadUnion(def.did, s);
                                }
                            }
                        }
                        FfiSafe
                    }
                    AdtKind::Enum => {
                        if def.variants.is_empty() {
                            // Empty enums are okay... although sort of useless.
                            return FfiSafe;
                        }

                        // Check for a repr() attribute to specify the size of the
                        // discriminant.
                        let repr_hints = cx.lookup_repr_hints(def.did);
                        match &repr_hints[..] {
                            &[] => {
                                // Special-case types like `Option<extern fn()>`.
                                if !is_repr_nullable_ptr(cx, def, substs) {
                                    return FfiUnsafe("found enum without foreign-function-safe \
                                                      representation annotation in foreign \
                                                      module, consider adding a #[repr(...)] \
                                                      attribute to the type");
                                }
                            }
                            &[ref hint] => {
                                if !hint.is_ffi_safe() {
                                    // FIXME: This shouldn't be reachable: we should check
                                    // this earlier.
                                    return FfiUnsafe("enum has unexpected #[repr(...)] attribute");
                                }

                                // Enum with an explicitly sized discriminant; either
                                // a C-style enum or a discriminated union.

                                // The layout of enum variants is implicitly repr(C).
                                // FIXME: Is that correct?
                            }
                            _ => {
                                // FIXME: This shouldn't be reachable: we should check
                                // this earlier.
                                return FfiUnsafe("enum has too many #[repr(...)] attributes");
                            }
                        }

                        // Check the contained variants.
                        for variant in &def.variants {
                            for field in &variant.fields {
                                let arg = cx.normalize_associated_type(&field.ty(cx, substs));
                                let r = self.check_type_for_ffi(cache, arg);
                                match r {
                                    FfiSafe => {}
                                    FfiBadStruct(..) | FfiBadUnion(..) | FfiBadEnum(..) => {
                                        return r;
                                    }
                                    FfiUnsafe(s) => {
                                        return FfiBadEnum(def.did, s);
                                    }
                                }
                            }
                        }
                        FfiSafe
                    }
                }
            }

            ty::TyChar => {
                FfiUnsafe("found Rust type `char` in foreign module, while \
                           `u32` or `libc::wchar_t` should be used")
            }

            // Primitive types with a stable representation.
            ty::TyBool | ty::TyInt(..) | ty::TyUint(..) | ty::TyFloat(..) | ty::TyNever => FfiSafe,

            ty::TyBox(..) => {
                FfiUnsafe("found Rust type Box<_> in foreign module, \
                           consider using a raw pointer instead")
            }

            ty::TySlice(_) => {
                FfiUnsafe("found Rust slice type in foreign module, \
                           consider using a raw pointer instead")
            }

            ty::TyDynamic(..) => {
                FfiUnsafe("found Rust trait type in foreign module, \
                           consider using a raw pointer instead")
            }

            ty::TyStr => {
                FfiUnsafe("found Rust type `str` in foreign module; \
                           consider using a `*const libc::c_char`")
            }

            ty::TyTuple(_) => {
                FfiUnsafe("found Rust tuple type in foreign module; \
                           consider using a struct instead")
            }

            ty::TyRawPtr(ref m) |
            ty::TyRef(_, ref m) => self.check_type_for_ffi(cache, m.ty),

            ty::TyArray(ty, _) => self.check_type_for_ffi(cache, ty),

            ty::TyFnPtr(bare_fn) => {
                match bare_fn.abi {
                    Abi::Rust | Abi::RustIntrinsic | Abi::PlatformIntrinsic | Abi::RustCall => {
                        return FfiUnsafe("found function pointer with Rust calling convention in \
                                          foreign module; consider using an `extern` function \
                                          pointer")
                    }
                    _ => {}
                }

                let sig = cx.erase_late_bound_regions(&bare_fn.sig);
                if !sig.output().is_nil() {
                    let r = self.check_type_for_ffi(cache, sig.output());
                    match r {
                        FfiSafe => {}
                        _ => {
                            return r;
                        }
                    }
                }
                for arg in sig.inputs() {
                    let r = self.check_type_for_ffi(cache, arg);
                    match r {
                        FfiSafe => {}
                        _ => {
                            return r;
                        }
                    }
                }
                FfiSafe
            }

            ty::TyParam(..) |
            ty::TyInfer(..) |
            ty::TyError |
            ty::TyClosure(..) |
            ty::TyProjection(..) |
            ty::TyAnon(..) |
            ty::TyFnDef(..) => bug!("Unexpected type in foreign function"),
        }
    }

    fn check_type_for_ffi_and_report_errors(&mut self, sp: Span, ty: Ty<'tcx>) {
        // it is only OK to use this function because extern fns cannot have
        // any generic types right now:
        let ty = self.cx.tcx.normalize_associated_type(&ty);

        match self.check_type_for_ffi(&mut FxHashSet(), ty) {
            FfiResult::FfiSafe => {}
            FfiResult::FfiUnsafe(s) => {
                self.cx.span_lint(IMPROPER_CTYPES, sp, s);
            }
            FfiResult::FfiBadStruct(_, s) => {
                // FIXME: This diagnostic is difficult to read, and doesn't
                // point at the relevant field.
                self.cx.span_lint(IMPROPER_CTYPES,
                                  sp,
                                  &format!("found non-foreign-function-safe member in struct \
                                            marked #[repr(C)]: {}",
                                           s));
            }
            FfiResult::FfiBadUnion(_, s) => {
                // FIXME: This diagnostic is difficult to read, and doesn't
                // point at the relevant field.
                self.cx.span_lint(IMPROPER_CTYPES,
                                  sp,
                                  &format!("found non-foreign-function-safe member in union \
                                            marked #[repr(C)]: {}",
                                           s));
            }
            FfiResult::FfiBadEnum(_, s) => {
                // FIXME: This diagnostic is difficult to read, and doesn't
                // point at the relevant variant.
                self.cx.span_lint(IMPROPER_CTYPES,
                                  sp,
                                  &format!("found non-foreign-function-safe member in enum: {}",
                                           s));
            }
        }
    }

    fn check_foreign_fn(&mut self, id: ast::NodeId, decl: &hir::FnDecl) {
        let def_id = self.cx.tcx.map.local_def_id(id);
        let sig = self.cx.tcx.item_type(def_id).fn_sig();
        let sig = self.cx.tcx.erase_late_bound_regions(&sig);

        for (input_ty, input_hir) in sig.inputs().iter().zip(&decl.inputs) {
            self.check_type_for_ffi_and_report_errors(input_hir.span, input_ty);
        }

        if let hir::Return(ref ret_hir) = decl.output {
            let ret_ty = sig.output();
            if !ret_ty.is_nil() {
                self.check_type_for_ffi_and_report_errors(ret_hir.span, ret_ty);
            }
        }
    }

    fn check_foreign_static(&mut self, id: ast::NodeId, span: Span) {
        let def_id = self.cx.tcx.map.local_def_id(id);
        let ty = self.cx.tcx.item_type(def_id);
        self.check_type_for_ffi_and_report_errors(span, ty);
    }
}

#[derive(Copy, Clone)]
pub struct ImproperCTypes;

impl LintPass for ImproperCTypes {
    fn get_lints(&self) -> LintArray {
        lint_array!(IMPROPER_CTYPES)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ImproperCTypes {
    fn check_item(&mut self, cx: &LateContext, it: &hir::Item) {
        let mut vis = ImproperCTypesVisitor { cx: cx };
        if let hir::ItemForeignMod(ref nmod) = it.node {
            if nmod.abi != Abi::RustIntrinsic && nmod.abi != Abi::PlatformIntrinsic {
                for ni in &nmod.items {
                    match ni.node {
                        hir::ForeignItemFn(ref decl, _, _) => {
                            vis.check_foreign_fn(ni.id, decl);
                        }
                        hir::ForeignItemStatic(ref ty, _) => {
                            vis.check_foreign_static(ni.id, ty.span);
                        }
                    }
                }
            }
        }
    }
}

pub struct VariantSizeDifferences;

impl LintPass for VariantSizeDifferences {
    fn get_lints(&self) -> LintArray {
        lint_array!(VARIANT_SIZE_DIFFERENCES)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for VariantSizeDifferences {
    fn check_item(&mut self, cx: &LateContext, it: &hir::Item) {
        if let hir::ItemEnum(ref enum_definition, ref gens) = it.node {
            if gens.ty_params.is_empty() {
                // sizes only make sense for non-generic types
                let t = cx.tcx.item_type(cx.tcx.map.local_def_id(it.id));
                let layout = cx.tcx.infer_ctxt((), Reveal::All).enter(|infcx| {
                    let ty = cx.tcx.erase_regions(&t);
                    ty.layout(&infcx).unwrap_or_else(|e| {
                        bug!("failed to get layout for `{}`: {}", t, e)
                    })
                });

                if let Layout::General { ref variants, ref size, discr, .. } = *layout {
                    let discr_size = Primitive::Int(discr).size(&cx.tcx.data_layout).bytes();

                    debug!("enum `{}` is {} bytes large with layout:\n{:#?}",
                      t, size.bytes(), layout);

                    let (largest, slargest, largest_index) = enum_definition.variants
                        .iter()
                        .zip(variants)
                        .map(|(variant, variant_layout)| {
                            // Subtract the size of the enum discriminant
                            let bytes = variant_layout.min_size
                                .bytes()
                                .saturating_sub(discr_size);

                            debug!("- variant `{}` is {} bytes large", variant.node.name, bytes);
                            bytes
                        })
                        .enumerate()
                        .fold((0, 0, 0), |(l, s, li), (idx, size)| if size > l {
                            (size, l, idx)
                        } else if size > s {
                            (l, size, li)
                        } else {
                            (l, s, li)
                        });

                    // we only warn if the largest variant is at least thrice as large as
                    // the second-largest.
                    if largest > slargest * 3 && slargest > 0 {
                        cx.span_lint(VARIANT_SIZE_DIFFERENCES,
                                     enum_definition.variants[largest_index].span,
                                     &format!("enum variant is more than three times larger \
                                               ({} bytes) than the next largest",
                                              largest));
                    }
                }
            }
        }
    }
}
