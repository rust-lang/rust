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
use rustc::ty::{self, Ty, TyCtxt};
use middle::const_val::ConstVal;
use rustc_const_eval::eval_const_expr_partial;
use rustc_const_eval::EvalHint::ExprTypeChecked;
use util::common::slice_pat;
use util::nodemap::{FnvHashSet};
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

register_long_diagnostics! {
E0519: r##"
It is not allowed to negate an unsigned integer.
You can negate a signed integer and cast it to an
unsigned integer or use the `!` operator.

```
let x: usize = -1isize as usize;
let y: usize = !0;
assert_eq!(x, y);
```

Alternatively you can use the `Wrapping` newtype
or the `wrapping_neg` operation that all
integral types support:

```
use std::num::Wrapping;
let x: Wrapping<usize> = -Wrapping(1);
let Wrapping(x) = x;
let y: usize = 1.wrapping_neg();
assert_eq!(x, y);
```

"##
}

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
                if let hir::ExprLit(ref lit) = expr.node {
                    match lit.node {
                        ast::LitKind::Int(_, ast::LitIntType::Unsigned(_)) => {
                            forbid_unsigned_negation(cx, e.span);
                        },
                        ast::LitKind::Int(_, ast::LitIntType::Unsuffixed) => {
                            if let ty::TyUint(_) = cx.tcx.node_id_to_type(e.id).sty {
                                forbid_unsigned_negation(cx, e.span);
                            }
                        },
                        _ => ()
                    }
                } else {
                    let t = cx.tcx.node_id_to_type(expr.id);
                    if let ty::TyUint(_) = t.sty {
                        forbid_unsigned_negation(cx, e.span);
                    }
                }
                // propagate negation, if the negation itself isn't negated
                if self.negated_expr_id != e.id {
                    self.negated_expr_id = expr.id;
                }
            },
            hir::ExprBinary(binop, ref l, ref r) => {
                if is_comparison(binop) && !check_limits(cx.tcx, binop, &l, &r) {
                    cx.span_lint(UNUSED_COMPARISONS, e.span,
                                 "comparison is useless due to type limits");
                }

                if binop.node.is_shift() {
                    let opt_ty_bits = match cx.tcx.node_id_to_type(l.id).sty {
                        ty::TyInt(t) => Some(int_ty_bits(t, cx.sess().target.int_type)),
                        ty::TyUint(t) => Some(uint_ty_bits(t, cx.sess().target.uint_type)),
                        _ => None
                    };

                    if let Some(bits) = opt_ty_bits {
                        let exceeding = if let hir::ExprLit(ref lit) = r.node {
                            if let ast::LitKind::Int(shift, _) = lit.node { shift >= bits }
                            else { false }
                        } else {
                            match eval_const_expr_partial(cx.tcx, &r, ExprTypeChecked, None) {
                                Ok(ConstVal::Integral(i)) => {
                                    i.is_negative() || i.to_u64()
                                                        .map(|i| i >= bits)
                                                        .unwrap_or(true)
                                },
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
                            ast::LitKind::Int(v, ast::LitIntType::Signed(_)) |
                            ast::LitKind::Int(v, ast::LitIntType::Unsuffixed) => {
                                let int_type = if let ast::IntTy::Is = t {
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
                                                 &format!("literal out of range for {:?}", t));
                                    return;
                                }
                            }
                            _ => bug!()
                        };
                    },
                    ty::TyUint(t) => {
                        let uint_type = if let ast::UintTy::Us = t {
                            cx.sess().target.uint_type
                        } else {
                            t
                        };
                        let (min, max) = uint_ty_range(uint_type);
                        let lit_val: u64 = match lit.node {
                            // _v is u8, within range by definition
                            ast::LitKind::Byte(_v) => return,
                            ast::LitKind::Int(v, _) => v,
                            _ => bug!()
                        };
                        if lit_val < min || lit_val > max {
                            cx.span_lint(OVERFLOWING_LITERALS, e.span,
                                         &format!("literal out of range for {:?}", t));
                        }
                    },
                    ty::TyFloat(t) => {
                        let (min, max) = float_ty_range(t);
                        let lit_val: f64 = match lit.node {
                            ast::LitKind::Float(ref v, _) |
                            ast::LitKind::FloatUnsuffixed(ref v) => {
                                match v.parse() {
                                    Ok(f) => f,
                                    Err(_) => return
                                }
                            }
                            _ => bug!()
                        };
                        if lit_val < min || lit_val > max {
                            cx.span_lint(OVERFLOWING_LITERALS, e.span,
                                         &format!("literal out of range for {:?}", t));
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
                _ => bug!()
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
                ast::IntTy::Is => (i64::MIN,        i64::MAX),
                ast::IntTy::I8 =>    (i8::MIN  as i64, i8::MAX  as i64),
                ast::IntTy::I16 =>   (i16::MIN as i64, i16::MAX as i64),
                ast::IntTy::I32 =>   (i32::MIN as i64, i32::MAX as i64),
                ast::IntTy::I64 =>   (i64::MIN,        i64::MAX)
            }
        }

        fn uint_ty_range(uint_ty: ast::UintTy) -> (u64, u64) {
            match uint_ty {
                ast::UintTy::Us => (u64::MIN,         u64::MAX),
                ast::UintTy::U8 =>    (u8::MIN   as u64, u8::MAX   as u64),
                ast::UintTy::U16 =>   (u16::MIN  as u64, u16::MAX  as u64),
                ast::UintTy::U32 =>   (u32::MIN  as u64, u32::MAX  as u64),
                ast::UintTy::U64 =>   (u64::MIN,         u64::MAX)
            }
        }

        fn float_ty_range(float_ty: ast::FloatTy) -> (f64, f64) {
            match float_ty {
                ast::FloatTy::F32 => (f32::MIN as f64, f32::MAX as f64),
                ast::FloatTy::F64 => (f64::MIN,        f64::MAX)
            }
        }

        fn int_ty_bits(int_ty: ast::IntTy, target_int_ty: ast::IntTy) -> u64 {
            match int_ty {
                ast::IntTy::Is => int_ty_bits(target_int_ty, target_int_ty),
                ast::IntTy::I8 => 8,
                ast::IntTy::I16 => 16 as u64,
                ast::IntTy::I32 => 32,
                ast::IntTy::I64 => 64,
            }
        }

        fn uint_ty_bits(uint_ty: ast::UintTy, target_uint_ty: ast::UintTy) -> u64 {
            match uint_ty {
                ast::UintTy::Us => uint_ty_bits(target_uint_ty, target_uint_ty),
                ast::UintTy::U8 => 8,
                ast::UintTy::U16 => 16,
                ast::UintTy::U32 => 32,
                ast::UintTy::U64 => 64,
            }
        }

        fn check_limits<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  binop: hir::BinOp,
                                  l: &hir::Expr,
                                  r: &hir::Expr) -> bool {
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
                            ast::LitKind::Int(v, ast::LitIntType::Signed(_)) |
                            ast::LitKind::Int(v, ast::LitIntType::Unsuffixed) => v as i64,
                            _ => return true
                        },
                        _ => bug!()
                    };
                    is_valid(norm_binop, lit_val, min, max)
                }
                ty::TyUint(uint_ty) => {
                    let (min, max): (u64, u64) = uint_ty_range(uint_ty);
                    let lit_val: u64 = match lit.node {
                        hir::ExprLit(ref li) => match li.node {
                            ast::LitKind::Int(v, _) => v,
                            _ => return true
                        },
                        _ => bug!()
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

        fn forbid_unsigned_negation(cx: &LateContext, span: Span) {
            cx.sess()
              .struct_span_err_with_code(span, "unary negation of unsigned integer", "E0519")
              .span_help(span, "use a cast or the `!` operator")
              .emit();
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
fn is_repr_nullable_ptr<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
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
                ty::TyFnPtr(_) => { return true; }
                ty::TyRef(..) => { return true; }
                _ => { }
            }
        }
    }
    false
}

impl<'a, 'tcx> ImproperCTypesVisitor<'a, 'tcx> {
    /// Check if the given type is "ffi-safe" (has a stable, well-defined
    /// representation which can be exported to C code).
    fn check_type_for_ffi(&self,
                          cache: &mut FnvHashSet<Ty<'tcx>>,
                          ty: Ty<'tcx>)
                          -> FfiResult {
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
                    let field_ty = cx.normalize_associated_type(&field.ty(cx, substs));
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
                match slice_pat(&&**repr_hints) {
                    &[] => {
                        // Special-case types like `Option<extern fn()>`.
                        if !is_repr_nullable_ptr(cx, def, substs) {
                            return FfiUnsafe(
                                "found enum without foreign-function-safe \
                                 representation annotation in foreign module, \
                                 consider adding a #[repr(...)] attribute to \
                                 the type")
                        }
                    }
                    &[ref hint] => {
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
                        let arg = cx.normalize_associated_type(&field.ty(cx, substs));
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

            ty::TyFnPtr(bare_fn) => {
                match bare_fn.abi {
                    Abi::Rust |
                    Abi::RustIntrinsic |
                    Abi::PlatformIntrinsic |
                    Abi::RustCall => {
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
            ty::TyFnDef(..) => {
                bug!("Unexpected type in foreign function")
            }
        }
    }

    fn check_type_for_ffi_and_report_errors(&mut self, sp: Span, ty: Ty<'tcx>) {
        // it is only OK to use this function because extern fns cannot have
        // any generic types right now:
        let ty = self.cx.tcx.normalize_associated_type(&ty);

        match self.check_type_for_ffi(&mut FnvHashSet(), ty) {
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

    fn check_foreign_fn(&mut self, id: ast::NodeId, decl: &hir::FnDecl) {
        let def_id = self.cx.tcx.map.local_def_id(id);
        let scheme = self.cx.tcx.lookup_item_type(def_id);
        let sig = scheme.ty.fn_sig();
        let sig = self.cx.tcx.erase_late_bound_regions(&sig);

        for (&input_ty, input_hir) in sig.inputs.iter().zip(&decl.inputs) {
            self.check_type_for_ffi_and_report_errors(input_hir.ty.span, &input_ty);
        }

        if let hir::Return(ref ret_hir) = decl.output {
            let ret_ty = sig.output.unwrap();
            if !ret_ty.is_nil() {
                self.check_type_for_ffi_and_report_errors(ret_hir.span, ret_ty);
            }
        }
    }

    fn check_foreign_static(&mut self, id: ast::NodeId, span: Span) {
        let def_id = self.cx.tcx.map.local_def_id(id);
        let scheme = self.cx.tcx.lookup_item_type(def_id);
        self.check_type_for_ffi_and_report_errors(span, scheme.ty);
    }
}

#[derive(Copy, Clone)]
pub struct ImproperCTypes;

impl LintPass for ImproperCTypes {
    fn get_lints(&self) -> LintArray {
        lint_array!(IMPROPER_CTYPES)
    }
}

impl LateLintPass for ImproperCTypes {
    fn check_item(&mut self, cx: &LateContext, it: &hir::Item) {
        let mut vis = ImproperCTypesVisitor { cx: cx };
        if let hir::ItemForeignMod(ref nmod) = it.node {
            if nmod.abi != Abi::RustIntrinsic && nmod.abi != Abi::PlatformIntrinsic {
                for ni in &nmod.items {
                    match ni.node {
                        hir::ForeignItemFn(ref decl, _) => {
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
