// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Lints in the Rust compiler.
//!
//! This contains lints which can feasibly be implemented as their own
//! AST visitor. Also see `rustc::lint::builtin`, which contains the
//! definitions of lints that are emitted directly inside the main
//! compiler.
//!
//! To add a new lint to rustc, declare it here using `declare_lint!()`.
//! Then add code to emit the new lint in the appropriate circumstances.
//! You can do that in an existing `LintPass` if it makes sense, or in a
//! new `LintPass`, or using `Session::add_lint` elsewhere in the
//! compiler. Only do the latter if the check can't be written cleanly as a
//! `LintPass` (also, note that such lints will need to be defined in
//! `rustc::lint::builtin`, not here).
//!
//! If you define a new `LintPass`, you will also need to add it to the
//! `add_builtin!` or `add_builtin_with_new!` invocation in `lib.rs`.
//! Use the former for unit-like structs and the latter for structs with
//! a `pub fn new()`.

use metadata::{csearch, decoder};
use middle::{cfg, def, infer, pat_util, stability, traits};
use middle::def_id::DefId;
use middle::subst::Substs;
use middle::ty::{self, Ty};
use middle::const_eval::{eval_const_expr_partial, ConstVal};
use middle::const_eval::EvalHint::ExprTypeChecked;
use rustc::ast_map;
use util::nodemap::{FnvHashMap, FnvHashSet, NodeSet};
use lint::{Level, Context, LintPass, LintArray, Lint};

use std::collections::HashSet;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::{cmp, slice};
use std::{i8, i16, i32, i64, u8, u16, u32, u64, f32, f64};

use syntax::{abi, ast};
use syntax::ast_util::is_shift_binop;
use syntax::attr::{self, AttrMetaMethods};
use syntax::codemap::{self, Span};
use syntax::feature_gate::{KNOWN_ATTRIBUTES, AttributeType};
use syntax::ast::{TyIs, TyUs, TyI8, TyU8, TyI16, TyU16, TyI32, TyU32, TyI64, TyU64};
use syntax::ptr::P;
use syntax::visit::{self, Visitor};

// hardwired lints from librustc
pub use lint::builtin::*;

declare_lint! {
    WHILE_TRUE,
    Warn,
    "suggest using `loop { }` instead of `while true { }`"
}

#[derive(Copy, Clone)]
pub struct WhileTrue;

impl LintPass for WhileTrue {
    fn get_lints(&self) -> LintArray {
        lint_array!(WHILE_TRUE)
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        if let ast::ExprWhile(ref cond, _, _) = e.node {
            if let ast::ExprLit(ref lit) = cond.node {
                if let ast::LitBool(true) = lit.node {
                    cx.span_lint(WHILE_TRUE, e.span,
                                 "denote infinite loops with loop { ... }");
                }
            }
        }
    }
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

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        match e.node {
            ast::ExprUnary(ast::UnNeg, ref expr) => {
                match expr.node  {
                    ast::ExprLit(ref lit) => {
                        match lit.node {
                            ast::LitInt(_, ast::UnsignedIntLit(_)) => {
                                check_unsigned_negation_feature(cx, e.span);
                            },
                            ast::LitInt(_, ast::UnsuffixedIntLit(_)) => {
                                if let ty::TyUint(_) = cx.tcx.expr_ty(e).sty {
                                    check_unsigned_negation_feature(cx, e.span);
                                }
                            },
                            _ => ()
                        }
                    },
                    _ => {
                        let t = cx.tcx.expr_ty(&**expr);
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
            ast::ExprParen(ref expr) if self.negated_expr_id == e.id => {
                self.negated_expr_id = expr.id;
            },
            ast::ExprBinary(binop, ref l, ref r) => {
                if is_comparison(binop) && !check_limits(cx.tcx, binop, &**l, &**r) {
                    cx.span_lint(UNUSED_COMPARISONS, e.span,
                                 "comparison is useless due to type limits");
                }

                if is_shift_binop(binop.node) {
                    let opt_ty_bits = match cx.tcx.expr_ty(&**l).sty {
                        ty::TyInt(t) => Some(int_ty_bits(t, cx.sess().target.int_type)),
                        ty::TyUint(t) => Some(uint_ty_bits(t, cx.sess().target.uint_type)),
                        _ => None
                    };

                    if let Some(bits) = opt_ty_bits {
                        let exceeding = if let ast::ExprLit(ref lit) = r.node {
                            if let ast::LitInt(shift, _) = lit.node { shift >= bits }
                            else { false }
                        } else {
                            match eval_const_expr_partial(cx.tcx, &**r, ExprTypeChecked) {
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
            ast::ExprLit(ref lit) => {
                match cx.tcx.expr_ty(e).sty {
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

        fn is_valid<T:cmp::PartialOrd>(binop: ast::BinOp, v: T,
                                min: T, max: T) -> bool {
            match binop.node {
                ast::BiLt => v >  min && v <= max,
                ast::BiLe => v >= min && v <  max,
                ast::BiGt => v >= min && v <  max,
                ast::BiGe => v >  min && v <= max,
                ast::BiEq | ast::BiNe => v >= min && v <= max,
                _ => panic!()
            }
        }

        fn rev_binop(binop: ast::BinOp) -> ast::BinOp {
            codemap::respan(binop.span, match binop.node {
                ast::BiLt => ast::BiGt,
                ast::BiLe => ast::BiGe,
                ast::BiGt => ast::BiLt,
                ast::BiGe => ast::BiLe,
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

        fn check_limits(tcx: &ty::ctxt, binop: ast::BinOp,
                        l: &ast::Expr, r: &ast::Expr) -> bool {
            let (lit, expr, swap) = match (&l.node, &r.node) {
                (&ast::ExprLit(_), _) => (l, r, true),
                (_, &ast::ExprLit(_)) => (r, l, false),
                _ => return true
            };
            // Normalize the binop so that the literal is always on the RHS in
            // the comparison
            let norm_binop = if swap {
                rev_binop(binop)
            } else {
                binop
            };
            match tcx.expr_ty(expr).sty {
                ty::TyInt(int_ty) => {
                    let (min, max) = int_ty_range(int_ty);
                    let lit_val: i64 = match lit.node {
                        ast::ExprLit(ref li) => match li.node {
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
                        ast::ExprLit(ref li) => match li.node {
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

        fn is_comparison(binop: ast::BinOp) -> bool {
            match binop.node {
                ast::BiEq | ast::BiLt | ast::BiLe |
                ast::BiNe | ast::BiGe | ast::BiGt => true,
                _ => false
            }
        }

        fn check_unsigned_negation_feature(cx: &Context, span: Span) {
            if !cx.sess().features.borrow().negate_unsigned {
                // FIXME(#27141): change this to syntax::feature_gate::emit_feature_err…
                cx.sess().span_warn(span,
                    "unary negation of unsigned integers will be feature gated in the future");
                // …and remove following two expressions.
                if option_env!("CFG_DISABLE_UNSTABLE_FEATURES").is_some() { return; }
                cx.sess().fileline_help(span, "add #![feature(negate_unsigned)] to the \
                                               crate attributes to enable the gate in advance");
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
    cx: &'a Context<'a, 'tcx>
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

            ty::TyInt(ast::TyIs) => {
                FfiUnsafe("found Rust type `isize` in foreign module, while \
                          `libc::c_int` or `libc::c_long` should be used")
            }
            ty::TyUint(ast::TyUs) => {
                FfiUnsafe("found Rust type `usize` in foreign module, while \
                          `libc::c_uint` or `libc::c_ulong` should be used")
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
    fn visit_ty(&mut self, ty: &ast::Ty) {
        match ty.node {
            ast::TyPath(..) |
            ast::TyBareFn(..) => self.check_def(ty.span, ty.id),
            ast::TyVec(..) => {
                self.cx.span_lint(IMPROPER_CTYPES, ty.span,
                    "found Rust slice type in foreign module, consider \
                     using a raw pointer instead");
            }
            ast::TyFixedLengthVec(ref ty, _) => self.visit_ty(ty),
            ast::TyTup(..) => {
                self.cx.span_lint(IMPROPER_CTYPES, ty.span,
                    "found Rust tuple type in foreign module; \
                     consider using a struct instead`")
            }
            _ => visit::walk_ty(self, ty)
        }
    }
}

#[derive(Copy, Clone)]
pub struct ImproperCTypes;

impl LintPass for ImproperCTypes {
    fn get_lints(&self) -> LintArray {
        lint_array!(IMPROPER_CTYPES)
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        fn check_ty(cx: &Context, ty: &ast::Ty) {
            let mut vis = ImproperCTypesVisitor { cx: cx };
            vis.visit_ty(ty);
        }

        fn check_foreign_fn(cx: &Context, decl: &ast::FnDecl) {
            for input in &decl.inputs {
                check_ty(cx, &*input.ty);
            }
            if let ast::Return(ref ret_ty) = decl.output {
                let tty = ast_ty_to_normalized(cx.tcx, ret_ty.id);
                if !tty.is_nil() {
                    check_ty(cx, &ret_ty);
                }
            }
        }

        match it.node {
            ast::ItemForeignMod(ref nmod)
                if nmod.abi != abi::RustIntrinsic &&
                   nmod.abi != abi::PlatformIntrinsic => {
                for ni in &nmod.items {
                    match ni.node {
                        ast::ForeignItemFn(ref decl, _) => check_foreign_fn(cx, &**decl),
                        ast::ForeignItemStatic(ref t, _) => check_ty(cx, &**t)
                    }
                }
            }
            _ => (),
        }
    }
}

declare_lint! {
    BOX_POINTERS,
    Allow,
    "use of owned (Box type) heap memory"
}

#[derive(Copy, Clone)]
pub struct BoxPointers;

impl BoxPointers {
    fn check_heap_type<'a, 'tcx>(&self, cx: &Context<'a, 'tcx>,
                                 span: Span, ty: Ty<'tcx>) {
        for leaf_ty in ty.walk() {
            if let ty::TyBox(_) = leaf_ty.sty {
                let m = format!("type uses owned (Box type) pointers: {}", ty);
                cx.span_lint(BOX_POINTERS, span, &m);
            }
        }
    }
}

impl LintPass for BoxPointers {
    fn get_lints(&self) -> LintArray {
        lint_array!(BOX_POINTERS)
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        match it.node {
            ast::ItemFn(..) |
            ast::ItemTy(..) |
            ast::ItemEnum(..) |
            ast::ItemStruct(..) =>
                self.check_heap_type(cx, it.span,
                                     cx.tcx.node_id_to_type(it.id)),
            _ => ()
        }

        // If it's a struct, we also have to check the fields' types
        match it.node {
            ast::ItemStruct(ref struct_def, _) => {
                for struct_field in &struct_def.fields {
                    self.check_heap_type(cx, struct_field.span,
                                         cx.tcx.node_id_to_type(struct_field.node.id));
                }
            }
            _ => ()
        }
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        let ty = cx.tcx.expr_ty(e);
        self.check_heap_type(cx, e.span, ty);
    }
}

declare_lint! {
    RAW_POINTER_DERIVE,
    Warn,
    "uses of #[derive] with raw pointers are rarely correct"
}

struct RawPtrDeriveVisitor<'a, 'tcx: 'a> {
    cx: &'a Context<'a, 'tcx>
}

impl<'a, 'tcx, 'v> Visitor<'v> for RawPtrDeriveVisitor<'a, 'tcx> {
    fn visit_ty(&mut self, ty: &ast::Ty) {
        const MSG: &'static str = "use of `#[derive]` with a raw pointer";
        if let ast::TyPtr(..) = ty.node {
            self.cx.span_lint(RAW_POINTER_DERIVE, ty.span, MSG);
        }
        visit::walk_ty(self, ty);
    }
    // explicit override to a no-op to reduce code bloat
    fn visit_expr(&mut self, _: &ast::Expr) {}
    fn visit_block(&mut self, _: &ast::Block) {}
}

pub struct RawPointerDerive {
    checked_raw_pointers: NodeSet,
}

impl RawPointerDerive {
    pub fn new() -> RawPointerDerive {
        RawPointerDerive {
            checked_raw_pointers: NodeSet(),
        }
    }
}

impl LintPass for RawPointerDerive {
    fn get_lints(&self) -> LintArray {
        lint_array!(RAW_POINTER_DERIVE)
    }

    fn check_item(&mut self, cx: &Context, item: &ast::Item) {
        if !attr::contains_name(&item.attrs, "automatically_derived") {
            return;
        }
        let did = match item.node {
            ast::ItemImpl(_, _, _, ref t_ref_opt, _, _) => {
                // Deriving the Copy trait does not cause a warning
                if let &Some(ref trait_ref) = t_ref_opt {
                    let def_id = cx.tcx.trait_ref_to_def_id(trait_ref);
                    if Some(def_id) == cx.tcx.lang_items.copy_trait() {
                        return;
                    }
                }

                match cx.tcx.node_id_to_type(item.id).sty {
                    ty::TyEnum(def, _) => def.did,
                    ty::TyStruct(def, _) => def.did,
                    _ => return,
                }
            }
            _ => return,
        };
        if !did.is_local() {
            return;
        }
        let item = match cx.tcx.map.find(did.node) {
            Some(ast_map::NodeItem(item)) => item,
            _ => return,
        };
        if !self.checked_raw_pointers.insert(item.id) {
            return;
        }
        match item.node {
            ast::ItemStruct(..) | ast::ItemEnum(..) => {
                let mut visitor = RawPtrDeriveVisitor { cx: cx };
                visit::walk_item(&mut visitor, &*item);
            }
            _ => {}
        }
    }
}

declare_lint! {
    UNUSED_ATTRIBUTES,
    Warn,
    "detects attributes that were not used by the compiler"
}

#[derive(Copy, Clone)]
pub struct UnusedAttributes;

impl LintPass for UnusedAttributes {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_ATTRIBUTES)
    }

    fn check_attribute(&mut self, cx: &Context, attr: &ast::Attribute) {
        // Note that check_name() marks the attribute as used if it matches.
        for &(ref name, ty) in KNOWN_ATTRIBUTES {
            match ty {
                AttributeType::Whitelisted
                | AttributeType::Gated(_, _) if attr.check_name(name) => {
                    break;
                },
                _ => ()
            }
        }

        let plugin_attributes = cx.sess().plugin_attributes.borrow_mut();
        for &(ref name, ty) in plugin_attributes.iter() {
            if ty == AttributeType::Whitelisted && attr.check_name(&*name) {
                break;
            }
        }

        if !attr::is_used(attr) {
            cx.span_lint(UNUSED_ATTRIBUTES, attr.span, "unused attribute");
            // Is it a builtin attribute that must be used at the crate level?
            let known_crate = KNOWN_ATTRIBUTES.contains(&(&attr.name(),
                                                          AttributeType::CrateLevel));
            // Has a plugin registered this attribute as one which must be used at
            // the crate level?
            let plugin_crate = plugin_attributes.iter()
                                                .find(|&&(ref x, t)| {
                                                        &*attr.name() == &*x &&
                                                        AttributeType::CrateLevel == t
                                                    }).is_some();
            if  known_crate || plugin_crate {
                let msg = match attr.node.style {
                    ast::AttrOuter => "crate-level attribute should be an inner \
                                       attribute: add an exclamation mark: #![foo]",
                    ast::AttrInner => "crate-level attribute should be in the \
                                       root module",
                };
                cx.span_lint(UNUSED_ATTRIBUTES, attr.span, msg);
            }
        }
    }
}

declare_lint! {
    pub PATH_STATEMENTS,
    Warn,
    "path statements with no effect"
}

#[derive(Copy, Clone)]
pub struct PathStatements;

impl LintPass for PathStatements {
    fn get_lints(&self) -> LintArray {
        lint_array!(PATH_STATEMENTS)
    }

    fn check_stmt(&mut self, cx: &Context, s: &ast::Stmt) {
        match s.node {
            ast::StmtSemi(ref expr, _) => {
                match expr.node {
                    ast::ExprPath(..) => cx.span_lint(PATH_STATEMENTS, s.span,
                                                      "path statement with no effect"),
                    _ => ()
                }
            }
            _ => ()
        }
    }
}

declare_lint! {
    pub UNUSED_MUST_USE,
    Warn,
    "unused result of a type flagged as #[must_use]"
}

declare_lint! {
    pub UNUSED_RESULTS,
    Allow,
    "unused result of an expression in a statement"
}

#[derive(Copy, Clone)]
pub struct UnusedResults;

impl LintPass for UnusedResults {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_MUST_USE, UNUSED_RESULTS)
    }

    fn check_stmt(&mut self, cx: &Context, s: &ast::Stmt) {
        let expr = match s.node {
            ast::StmtSemi(ref expr, _) => &**expr,
            _ => return
        };

        if let ast::ExprRet(..) = expr.node {
            return;
        }

        let t = cx.tcx.expr_ty(expr);
        let warned = match t.sty {
            ty::TyTuple(ref tys) if tys.is_empty() => return,
            ty::TyBool => return,
            ty::TyStruct(def, _) |
            ty::TyEnum(def, _) => {
                if def.did.is_local() {
                    if let ast_map::NodeItem(it) = cx.tcx.map.get(def.did.node) {
                        check_must_use(cx, &it.attrs, s.span)
                    } else {
                        false
                    }
                } else {
                    let attrs = csearch::get_item_attrs(&cx.sess().cstore, def.did);
                    check_must_use(cx, &attrs[..], s.span)
                }
            }
            _ => false,
        };
        if !warned {
            cx.span_lint(UNUSED_RESULTS, s.span, "unused result");
        }

        fn check_must_use(cx: &Context, attrs: &[ast::Attribute], sp: Span) -> bool {
            for attr in attrs {
                if attr.check_name("must_use") {
                    let mut msg = "unused result which must be used".to_string();
                    // check for #[must_use="..."]
                    match attr.value_str() {
                        None => {}
                        Some(s) => {
                            msg.push_str(": ");
                            msg.push_str(&s);
                        }
                    }
                    cx.span_lint(UNUSED_MUST_USE, sp, &msg);
                    return true;
                }
            }
            false
        }
    }
}

declare_lint! {
    pub NON_CAMEL_CASE_TYPES,
    Warn,
    "types, variants, traits and type parameters should have camel case names"
}

#[derive(Copy, Clone)]
pub struct NonCamelCaseTypes;

impl NonCamelCaseTypes {
    fn check_case(&self, cx: &Context, sort: &str, ident: ast::Ident, span: Span) {
        fn is_camel_case(ident: ast::Ident) -> bool {
            let ident = ident.name.as_str();
            if ident.is_empty() {
                return true;
            }
            let ident = ident.trim_matches('_');

            // start with a non-lowercase letter rather than non-uppercase
            // ones (some scripts don't have a concept of upper/lowercase)
            !ident.is_empty() && !ident.char_at(0).is_lowercase() && !ident.contains('_')
        }

        fn to_camel_case(s: &str) -> String {
            s.split('_').flat_map(|word| word.chars().enumerate().map(|(i, c)|
                if i == 0 {
                    c.to_uppercase().collect::<String>()
                } else {
                    c.to_lowercase().collect()
                }
            )).collect::<Vec<_>>().concat()
        }

        let s = ident.name.as_str();

        if !is_camel_case(ident) {
            let c = to_camel_case(&s);
            let m = if c.is_empty() {
                format!("{} `{}` should have a camel case name such as `CamelCase`", sort, s)
            } else {
                format!("{} `{}` should have a camel case name such as `{}`", sort, s, c)
            };
            cx.span_lint(NON_CAMEL_CASE_TYPES, span, &m[..]);
        }
    }
}

impl LintPass for NonCamelCaseTypes {
    fn get_lints(&self) -> LintArray {
        lint_array!(NON_CAMEL_CASE_TYPES)
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        let extern_repr_count = it.attrs.iter().filter(|attr| {
            attr::find_repr_attrs(cx.tcx.sess.diagnostic(), attr).iter()
                .any(|r| r == &attr::ReprExtern)
        }).count();
        let has_extern_repr = extern_repr_count > 0;

        if has_extern_repr {
            return;
        }

        match it.node {
            ast::ItemTy(..) | ast::ItemStruct(..) => {
                self.check_case(cx, "type", it.ident, it.span)
            }
            ast::ItemTrait(..) => {
                self.check_case(cx, "trait", it.ident, it.span)
            }
            ast::ItemEnum(ref enum_definition, _) => {
                if has_extern_repr {
                    return;
                }
                self.check_case(cx, "type", it.ident, it.span);
                for variant in &enum_definition.variants {
                    self.check_case(cx, "variant", variant.node.name, variant.span);
                }
            }
            _ => ()
        }
    }

    fn check_generics(&mut self, cx: &Context, it: &ast::Generics) {
        for gen in it.ty_params.iter() {
            self.check_case(cx, "type parameter", gen.ident, gen.span);
        }
    }
}

#[derive(PartialEq)]
enum MethodContext {
    TraitDefaultImpl,
    TraitImpl,
    PlainImpl
}

fn method_context(cx: &Context, id: ast::NodeId, span: Span) -> MethodContext {
    match cx.tcx.impl_or_trait_items.borrow().get(&DefId::local(id)) {
        None => cx.sess().span_bug(span, "missing method descriptor?!"),
        Some(item) => match item.container() {
            ty::TraitContainer(..) => MethodContext::TraitDefaultImpl,
            ty::ImplContainer(cid) => {
                match cx.tcx.impl_trait_ref(cid) {
                    Some(_) => MethodContext::TraitImpl,
                    None => MethodContext::PlainImpl
                }
            }
        }
    }
}

declare_lint! {
    pub NON_SNAKE_CASE,
    Warn,
    "methods, functions, lifetime parameters and modules should have snake case names"
}

#[derive(Copy, Clone)]
pub struct NonSnakeCase;

impl NonSnakeCase {
    fn to_snake_case(mut str: &str) -> String {
        let mut words = vec![];
        // Preserve leading underscores
        str = str.trim_left_matches(|c: char| {
            if c == '_' {
                words.push(String::new());
                true
            } else {
                false
            }
        });
        for s in str.split('_') {
            let mut last_upper = false;
            let mut buf = String::new();
            if s.is_empty() {
                continue;
            }
            for ch in s.chars() {
                if !buf.is_empty() && buf != "'"
                                   && ch.is_uppercase()
                                   && !last_upper {
                    words.push(buf);
                    buf = String::new();
                }
                last_upper = ch.is_uppercase();
                buf.extend(ch.to_lowercase());
            }
            words.push(buf);
        }
        words.join("_")
    }

    fn check_snake_case(&self, cx: &Context, sort: &str, name: &str, span: Option<Span>) {
        fn is_snake_case(ident: &str) -> bool {
            if ident.is_empty() {
                return true;
            }
            let ident = ident.trim_left_matches('\'');
            let ident = ident.trim_matches('_');

            let mut allow_underscore = true;
            ident.chars().all(|c| {
                allow_underscore = match c {
                    '_' if !allow_underscore => return false,
                    '_' => false,
                    // It would be more obvious to use `c.is_lowercase()`,
                    // but some characters do not have a lowercase form
                    c if !c.is_uppercase() => true,
                    _ => return false,
                };
                true
            })
        }

        if !is_snake_case(name) {
            let sc = NonSnakeCase::to_snake_case(name);
            let msg = if sc != name {
                format!("{} `{}` should have a snake case name such as `{}`",
                        sort, name, sc)
            } else {
                format!("{} `{}` should have a snake case name",
                        sort, name)
            };
            match span {
                Some(span) => cx.span_lint(NON_SNAKE_CASE, span, &msg),
                None => cx.lint(NON_SNAKE_CASE, &msg),
            }
        }
    }
}

impl LintPass for NonSnakeCase {
    fn get_lints(&self) -> LintArray {
        lint_array!(NON_SNAKE_CASE)
    }

    fn check_crate(&mut self, cx: &Context, cr: &ast::Crate) {
        let attr_crate_name = cr.attrs.iter().find(|at| at.check_name("crate_name"))
                                      .and_then(|at| at.value_str().map(|s| (at, s)));
        if let Some(ref name) = cx.tcx.sess.opts.crate_name {
            self.check_snake_case(cx, "crate", name, None);
        } else if let Some((attr, ref name)) = attr_crate_name {
            self.check_snake_case(cx, "crate", name, Some(attr.span));
        }
    }

    fn check_fn(&mut self, cx: &Context,
                fk: visit::FnKind, _: &ast::FnDecl,
                _: &ast::Block, span: Span, id: ast::NodeId) {
        match fk {
            visit::FkMethod(ident, _, _) => match method_context(cx, id, span) {
                MethodContext::PlainImpl => {
                    self.check_snake_case(cx, "method", &ident.name.as_str(), Some(span))
                },
                MethodContext::TraitDefaultImpl => {
                    self.check_snake_case(cx, "trait method", &ident.name.as_str(), Some(span))
                },
                _ => (),
            },
            visit::FkItemFn(ident, _, _, _, _, _) => {
                self.check_snake_case(cx, "function", &ident.name.as_str(), Some(span))
            },
            _ => (),
        }
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        if let ast::ItemMod(_) = it.node {
            self.check_snake_case(cx, "module", &it.ident.name.as_str(), Some(it.span));
        }
    }

    fn check_trait_item(&mut self, cx: &Context, trait_item: &ast::TraitItem) {
        if let ast::MethodTraitItem(_, None) = trait_item.node {
            self.check_snake_case(cx, "trait method", &trait_item.ident.name.as_str(),
                                  Some(trait_item.span));
        }
    }

    fn check_lifetime_def(&mut self, cx: &Context, t: &ast::LifetimeDef) {
        self.check_snake_case(cx, "lifetime", &t.lifetime.name.as_str(),
                              Some(t.lifetime.span));
    }

    fn check_pat(&mut self, cx: &Context, p: &ast::Pat) {
        if let &ast::PatIdent(_, ref path1, _) = &p.node {
            let def = cx.tcx.def_map.borrow().get(&p.id).map(|d| d.full_def());
            if let Some(def::DefLocal(_)) = def {
                self.check_snake_case(cx, "variable", &path1.node.name.as_str(), Some(p.span));
            }
        }
    }

    fn check_struct_def(&mut self, cx: &Context, s: &ast::StructDef,
                        _: ast::Ident, _: &ast::Generics, _: ast::NodeId) {
        for sf in &s.fields {
            if let ast::StructField_ { kind: ast::NamedField(ident, _), .. } = sf.node {
                self.check_snake_case(cx, "structure field", &ident.name.as_str(),
                                      Some(sf.span));
            }
        }
    }
}

declare_lint! {
    pub NON_UPPER_CASE_GLOBALS,
    Warn,
    "static constants should have uppercase identifiers"
}

#[derive(Copy, Clone)]
pub struct NonUpperCaseGlobals;

impl NonUpperCaseGlobals {
    fn check_upper_case(cx: &Context, sort: &str, ident: ast::Ident, span: Span) {
        let s = ident.name.as_str();

        if s.chars().any(|c| c.is_lowercase()) {
            let uc = NonSnakeCase::to_snake_case(&s).to_uppercase();
            if uc != &s[..] {
                cx.span_lint(NON_UPPER_CASE_GLOBALS, span,
                    &format!("{} `{}` should have an upper case name such as `{}`",
                             sort, s, uc));
            } else {
                cx.span_lint(NON_UPPER_CASE_GLOBALS, span,
                    &format!("{} `{}` should have an upper case name",
                             sort, s));
            }
        }
    }
}

impl LintPass for NonUpperCaseGlobals {
    fn get_lints(&self) -> LintArray {
        lint_array!(NON_UPPER_CASE_GLOBALS)
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        match it.node {
            // only check static constants
            ast::ItemStatic(_, ast::MutImmutable, _) => {
                NonUpperCaseGlobals::check_upper_case(cx, "static constant", it.ident, it.span);
            }
            ast::ItemConst(..) => {
                NonUpperCaseGlobals::check_upper_case(cx, "constant", it.ident, it.span);
            }
            _ => {}
        }
    }

    fn check_trait_item(&mut self, cx: &Context, ti: &ast::TraitItem) {
        match ti.node {
            ast::ConstTraitItem(..) => {
                NonUpperCaseGlobals::check_upper_case(cx, "associated constant",
                                                      ti.ident, ti.span);
            }
            _ => {}
        }
    }

    fn check_impl_item(&mut self, cx: &Context, ii: &ast::ImplItem) {
        match ii.node {
            ast::ConstImplItem(..) => {
                NonUpperCaseGlobals::check_upper_case(cx, "associated constant",
                                                      ii.ident, ii.span);
            }
            _ => {}
        }
    }

    fn check_pat(&mut self, cx: &Context, p: &ast::Pat) {
        // Lint for constants that look like binding identifiers (#7526)
        match (&p.node, cx.tcx.def_map.borrow().get(&p.id).map(|d| d.full_def())) {
            (&ast::PatIdent(_, ref path1, _), Some(def::DefConst(..))) => {
                NonUpperCaseGlobals::check_upper_case(cx, "constant in pattern",
                                                      path1.node, p.span);
            }
            _ => {}
        }
    }
}

declare_lint! {
    UNUSED_PARENS,
    Warn,
    "`if`, `match`, `while` and `return` do not need parentheses"
}

#[derive(Copy, Clone)]
pub struct UnusedParens;

impl UnusedParens {
    fn check_unused_parens_core(&self, cx: &Context, value: &ast::Expr, msg: &str,
                                struct_lit_needs_parens: bool) {
        if let ast::ExprParen(ref inner) = value.node {
            let necessary = struct_lit_needs_parens && contains_exterior_struct_lit(&**inner);
            if !necessary {
                cx.span_lint(UNUSED_PARENS, value.span,
                             &format!("unnecessary parentheses around {}", msg))
            }
        }

        /// Expressions that syntactically contain an "exterior" struct
        /// literal i.e. not surrounded by any parens or other
        /// delimiters, e.g. `X { y: 1 }`, `X { y: 1 }.method()`, `foo
        /// == X { y: 1 }` and `X { y: 1 } == foo` all do, but `(X {
        /// y: 1 }) == foo` does not.
        fn contains_exterior_struct_lit(value: &ast::Expr) -> bool {
            match value.node {
                ast::ExprStruct(..) => true,

                ast::ExprAssign(ref lhs, ref rhs) |
                ast::ExprAssignOp(_, ref lhs, ref rhs) |
                ast::ExprBinary(_, ref lhs, ref rhs) => {
                    // X { y: 1 } + X { y: 2 }
                    contains_exterior_struct_lit(&**lhs) ||
                        contains_exterior_struct_lit(&**rhs)
                }
                ast::ExprUnary(_, ref x) |
                ast::ExprCast(ref x, _) |
                ast::ExprField(ref x, _) |
                ast::ExprTupField(ref x, _) |
                ast::ExprIndex(ref x, _) => {
                    // &X { y: 1 }, X { y: 1 }.y
                    contains_exterior_struct_lit(&**x)
                }

                ast::ExprMethodCall(_, _, ref exprs) => {
                    // X { y: 1 }.bar(...)
                    contains_exterior_struct_lit(&*exprs[0])
                }

                _ => false
            }
        }
    }
}

impl LintPass for UnusedParens {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_PARENS)
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        let (value, msg, struct_lit_needs_parens) = match e.node {
            ast::ExprIf(ref cond, _, _) => (cond, "`if` condition", true),
            ast::ExprWhile(ref cond, _, _) => (cond, "`while` condition", true),
            ast::ExprMatch(ref head, _, source) => match source {
                ast::MatchSource::Normal => (head, "`match` head expression", true),
                ast::MatchSource::IfLetDesugar { .. } => (head, "`if let` head expression", true),
                ast::MatchSource::WhileLetDesugar => (head, "`while let` head expression", true),
                ast::MatchSource::ForLoopDesugar => (head, "`for` head expression", true),
            },
            ast::ExprRet(Some(ref value)) => (value, "`return` value", false),
            ast::ExprAssign(_, ref value) => (value, "assigned value", false),
            ast::ExprAssignOp(_, _, ref value) => (value, "assigned value", false),
            _ => return
        };
        self.check_unused_parens_core(cx, &**value, msg, struct_lit_needs_parens);
    }

    fn check_stmt(&mut self, cx: &Context, s: &ast::Stmt) {
        let (value, msg) = match s.node {
            ast::StmtDecl(ref decl, _) => match decl.node {
                ast::DeclLocal(ref local) => match local.init {
                    Some(ref value) => (value, "assigned value"),
                    None => return
                },
                _ => return
            },
            _ => return
        };
        self.check_unused_parens_core(cx, &**value, msg, false);
    }
}

declare_lint! {
    UNUSED_IMPORT_BRACES,
    Allow,
    "unnecessary braces around an imported item"
}

#[derive(Copy, Clone)]
pub struct UnusedImportBraces;

impl LintPass for UnusedImportBraces {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_IMPORT_BRACES)
    }

    fn check_item(&mut self, cx: &Context, item: &ast::Item) {
        if let ast::ItemUse(ref view_path) = item.node {
            if let ast::ViewPathList(_, ref items) = view_path.node {
                if items.len() == 1 {
                    if let ast::PathListIdent {ref name, ..} = items[0].node {
                        let m = format!("braces around {} is unnecessary",
                                        name);
                        cx.span_lint(UNUSED_IMPORT_BRACES, item.span,
                                     &m[..]);
                    }
                }
            }
        }
    }
}

declare_lint! {
    NON_SHORTHAND_FIELD_PATTERNS,
    Warn,
    "using `Struct { x: x }` instead of `Struct { x }`"
}

#[derive(Copy, Clone)]
pub struct NonShorthandFieldPatterns;

impl LintPass for NonShorthandFieldPatterns {
    fn get_lints(&self) -> LintArray {
        lint_array!(NON_SHORTHAND_FIELD_PATTERNS)
    }

    fn check_pat(&mut self, cx: &Context, pat: &ast::Pat) {
        let def_map = cx.tcx.def_map.borrow();
        if let ast::PatStruct(_, ref v, _) = pat.node {
            let field_pats = v.iter().filter(|fieldpat| {
                if fieldpat.node.is_shorthand {
                    return false;
                }
                let def = def_map.get(&fieldpat.node.pat.id).map(|d| d.full_def());
                def == Some(def::DefLocal(fieldpat.node.pat.id))
            });
            for fieldpat in field_pats {
                if let ast::PatIdent(_, ident, None) = fieldpat.node.pat.node {
                    if ident.node.name == fieldpat.node.ident.name {
                        // FIXME: should this comparison really be done on the name?
                        // doing it on the ident will fail during compilation of libcore
                        cx.span_lint(NON_SHORTHAND_FIELD_PATTERNS, fieldpat.span,
                                     &format!("the `{}:` in this pattern is redundant and can \
                                              be removed", ident.node))
                    }
                }
            }
        }
    }
}

declare_lint! {
    pub UNUSED_UNSAFE,
    Warn,
    "unnecessary use of an `unsafe` block"
}

#[derive(Copy, Clone)]
pub struct UnusedUnsafe;

impl LintPass for UnusedUnsafe {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_UNSAFE)
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        if let ast::ExprBlock(ref blk) = e.node {
            // Don't warn about generated blocks, that'll just pollute the output.
            if blk.rules == ast::UnsafeBlock(ast::UserProvided) &&
                !cx.tcx.used_unsafe.borrow().contains(&blk.id) {
                    cx.span_lint(UNUSED_UNSAFE, blk.span, "unnecessary `unsafe` block");
            }
        }
    }
}

declare_lint! {
    UNSAFE_CODE,
    Allow,
    "usage of `unsafe` code"
}

#[derive(Copy, Clone)]
pub struct UnsafeCode;

impl LintPass for UnsafeCode {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNSAFE_CODE)
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        if let ast::ExprBlock(ref blk) = e.node {
            // Don't warn about generated blocks, that'll just pollute the output.
            if blk.rules == ast::UnsafeBlock(ast::UserProvided) {
                cx.span_lint(UNSAFE_CODE, blk.span, "usage of an `unsafe` block");
            }
        }
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        match it.node {
            ast::ItemTrait(ast::Unsafety::Unsafe, _, _, _) =>
                cx.span_lint(UNSAFE_CODE, it.span, "declaration of an `unsafe` trait"),

            ast::ItemImpl(ast::Unsafety::Unsafe, _, _, _, _, _) =>
                cx.span_lint(UNSAFE_CODE, it.span, "implementation of an `unsafe` trait"),

            _ => return,
        }
    }

    fn check_fn(&mut self, cx: &Context, fk: visit::FnKind, _: &ast::FnDecl,
                _: &ast::Block, span: Span, _: ast::NodeId) {
        match fk {
            visit::FkItemFn(_, _, ast::Unsafety::Unsafe, _, _, _) =>
                cx.span_lint(UNSAFE_CODE, span, "declaration of an `unsafe` function"),

            visit::FkMethod(_, sig, _) => {
                if sig.unsafety == ast::Unsafety::Unsafe {
                    cx.span_lint(UNSAFE_CODE, span, "implementation of an `unsafe` method")
                }
            },

            _ => (),
        }
    }

    fn check_trait_item(&mut self, cx: &Context, trait_item: &ast::TraitItem) {
        if let ast::MethodTraitItem(ref sig, None) = trait_item.node {
            if sig.unsafety == ast::Unsafety::Unsafe {
                cx.span_lint(UNSAFE_CODE, trait_item.span,
                             "declaration of an `unsafe` method")
            }
        }
    }
}

declare_lint! {
    pub UNUSED_MUT,
    Warn,
    "detect mut variables which don't need to be mutable"
}

#[derive(Copy, Clone)]
pub struct UnusedMut;

impl UnusedMut {
    fn check_unused_mut_pat(&self, cx: &Context, pats: &[P<ast::Pat>]) {
        // collect all mutable pattern and group their NodeIDs by their Identifier to
        // avoid false warnings in match arms with multiple patterns

        let mut mutables = FnvHashMap();
        for p in pats {
            pat_util::pat_bindings(&cx.tcx.def_map, &**p, |mode, id, _, path1| {
                let ident = path1.node;
                if let ast::BindByValue(ast::MutMutable) = mode {
                    if !ident.name.as_str().starts_with("_") {
                        match mutables.entry(ident.name.usize()) {
                            Vacant(entry) => { entry.insert(vec![id]); },
                            Occupied(mut entry) => { entry.get_mut().push(id); },
                        }
                    }
                }
            });
        }

        let used_mutables = cx.tcx.used_mut_nodes.borrow();
        for (_, v) in &mutables {
            if !v.iter().any(|e| used_mutables.contains(e)) {
                cx.span_lint(UNUSED_MUT, cx.tcx.map.span(v[0]),
                             "variable does not need to be mutable");
            }
        }
    }
}

impl LintPass for UnusedMut {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_MUT)
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        if let ast::ExprMatch(_, ref arms, _) = e.node {
            for a in arms {
                self.check_unused_mut_pat(cx, &a.pats)
            }
        }
    }

    fn check_stmt(&mut self, cx: &Context, s: &ast::Stmt) {
        if let ast::StmtDecl(ref d, _) = s.node {
            if let ast::DeclLocal(ref l) = d.node {
                self.check_unused_mut_pat(cx, slice::ref_slice(&l.pat));
            }
        }
    }

    fn check_fn(&mut self, cx: &Context,
                _: visit::FnKind, decl: &ast::FnDecl,
                _: &ast::Block, _: Span, _: ast::NodeId) {
        for a in &decl.inputs {
            self.check_unused_mut_pat(cx, slice::ref_slice(&a.pat));
        }
    }
}

declare_lint! {
    UNUSED_ALLOCATION,
    Warn,
    "detects unnecessary allocations that can be eliminated"
}

#[derive(Copy, Clone)]
pub struct UnusedAllocation;

impl LintPass for UnusedAllocation {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_ALLOCATION)
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        match e.node {
            ast::ExprUnary(ast::UnUniq, _) => (),
            _ => return
        }

        if let Some(adjustment) = cx.tcx.tables.borrow().adjustments.get(&e.id) {
            if let ty::AdjustDerefRef(ty::AutoDerefRef { ref autoref, .. }) = *adjustment {
                match autoref {
                    &Some(ty::AutoPtr(_, ast::MutImmutable)) => {
                        cx.span_lint(UNUSED_ALLOCATION, e.span,
                                     "unnecessary allocation, use & instead");
                    }
                    &Some(ty::AutoPtr(_, ast::MutMutable)) => {
                        cx.span_lint(UNUSED_ALLOCATION, e.span,
                                     "unnecessary allocation, use &mut instead");
                    }
                    _ => ()
                }
            }
        }
    }
}

declare_lint! {
    MISSING_DOCS,
    Allow,
    "detects missing documentation for public members"
}

pub struct MissingDoc {
    /// Stack of IDs of struct definitions.
    struct_def_stack: Vec<ast::NodeId>,

    /// True if inside variant definition
    in_variant: bool,

    /// Stack of whether #[doc(hidden)] is set
    /// at each level which has lint attributes.
    doc_hidden_stack: Vec<bool>,

    /// Private traits or trait items that leaked through. Don't check their methods.
    private_traits: HashSet<ast::NodeId>,
}

impl MissingDoc {
    pub fn new() -> MissingDoc {
        MissingDoc {
            struct_def_stack: vec!(),
            in_variant: false,
            doc_hidden_stack: vec!(false),
            private_traits: HashSet::new(),
        }
    }

    fn doc_hidden(&self) -> bool {
        *self.doc_hidden_stack.last().expect("empty doc_hidden_stack")
    }

    fn check_missing_docs_attrs(&self,
                               cx: &Context,
                               id: Option<ast::NodeId>,
                               attrs: &[ast::Attribute],
                               sp: Span,
                               desc: &'static str) {
        // If we're building a test harness, then warning about
        // documentation is probably not really relevant right now.
        if cx.sess().opts.test {
            return;
        }

        // `#[doc(hidden)]` disables missing_docs check.
        if self.doc_hidden() {
            return;
        }

        // Only check publicly-visible items, using the result from the privacy pass.
        // It's an option so the crate root can also use this function (it doesn't
        // have a NodeId).
        if let Some(ref id) = id {
            if !cx.exported_items.contains(id) {
                return;
            }
        }

        let has_doc = attrs.iter().any(|a| {
            match a.node.value.node {
                ast::MetaNameValue(ref name, _) if *name == "doc" => true,
                _ => false
            }
        });
        if !has_doc {
            cx.span_lint(MISSING_DOCS, sp,
                         &format!("missing documentation for {}", desc));
        }
    }
}

impl LintPass for MissingDoc {
    fn get_lints(&self) -> LintArray {
        lint_array!(MISSING_DOCS)
    }

    fn enter_lint_attrs(&mut self, _: &Context, attrs: &[ast::Attribute]) {
        let doc_hidden = self.doc_hidden() || attrs.iter().any(|attr| {
            attr.check_name("doc") && match attr.meta_item_list() {
                None => false,
                Some(l) => attr::contains_name(&l[..], "hidden"),
            }
        });
        self.doc_hidden_stack.push(doc_hidden);
    }

    fn exit_lint_attrs(&mut self, _: &Context, _: &[ast::Attribute]) {
        self.doc_hidden_stack.pop().expect("empty doc_hidden_stack");
    }

    fn check_struct_def(&mut self, _: &Context, _: &ast::StructDef,
                        _: ast::Ident, _: &ast::Generics, id: ast::NodeId) {
        self.struct_def_stack.push(id);
    }

    fn check_struct_def_post(&mut self, _: &Context, _: &ast::StructDef,
                             _: ast::Ident, _: &ast::Generics, id: ast::NodeId) {
        let popped = self.struct_def_stack.pop().expect("empty struct_def_stack");
        assert!(popped == id);
    }

    fn check_crate(&mut self, cx: &Context, krate: &ast::Crate) {
        self.check_missing_docs_attrs(cx, None, &krate.attrs, krate.span, "crate");
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        let desc = match it.node {
            ast::ItemFn(..) => "a function",
            ast::ItemMod(..) => "a module",
            ast::ItemEnum(..) => "an enum",
            ast::ItemStruct(..) => "a struct",
            ast::ItemTrait(_, _, _, ref items) => {
                // Issue #11592, traits are always considered exported, even when private.
                if it.vis == ast::Visibility::Inherited {
                    self.private_traits.insert(it.id);
                    for itm in items {
                        self.private_traits.insert(itm.id);
                    }
                    return
                }
                "a trait"
            },
            ast::ItemTy(..) => "a type alias",
            ast::ItemImpl(_, _, _, Some(ref trait_ref), _, ref impl_items) => {
                // If the trait is private, add the impl items to private_traits so they don't get
                // reported for missing docs.
                let real_trait = cx.tcx.trait_ref_to_def_id(trait_ref);
                match cx.tcx.map.find(real_trait.node) {
                    Some(ast_map::NodeItem(item)) => if item.vis == ast::Visibility::Inherited {
                        for itm in impl_items {
                            self.private_traits.insert(itm.id);
                        }
                    },
                    _ => { }
                }
                return
            },
            ast::ItemConst(..) => "a constant",
            ast::ItemStatic(..) => "a static",
            _ => return
        };

        self.check_missing_docs_attrs(cx, Some(it.id), &it.attrs, it.span, desc);
    }

    fn check_trait_item(&mut self, cx: &Context, trait_item: &ast::TraitItem) {
        if self.private_traits.contains(&trait_item.id) { return }

        let desc = match trait_item.node {
            ast::ConstTraitItem(..) => "an associated constant",
            ast::MethodTraitItem(..) => "a trait method",
            ast::TypeTraitItem(..) => "an associated type",
        };

        self.check_missing_docs_attrs(cx, Some(trait_item.id),
                                      &trait_item.attrs,
                                      trait_item.span, desc);
    }

    fn check_impl_item(&mut self, cx: &Context, impl_item: &ast::ImplItem) {
        // If the method is an impl for a trait, don't doc.
        if method_context(cx, impl_item.id, impl_item.span) == MethodContext::TraitImpl {
            return;
        }

        let desc = match impl_item.node {
            ast::ConstImplItem(..) => "an associated constant",
            ast::MethodImplItem(..) => "a method",
            ast::TypeImplItem(_) => "an associated type",
            ast::MacImplItem(_) => "an impl item macro",
        };
        self.check_missing_docs_attrs(cx, Some(impl_item.id),
                                      &impl_item.attrs,
                                      impl_item.span, desc);
    }

    fn check_struct_field(&mut self, cx: &Context, sf: &ast::StructField) {
        if let ast::NamedField(_, vis) = sf.node.kind {
            if vis == ast::Public || self.in_variant {
                let cur_struct_def = *self.struct_def_stack.last()
                    .expect("empty struct_def_stack");
                self.check_missing_docs_attrs(cx, Some(cur_struct_def),
                                              &sf.node.attrs, sf.span,
                                              "a struct field")
            }
        }
    }

    fn check_variant(&mut self, cx: &Context, v: &ast::Variant, _: &ast::Generics) {
        self.check_missing_docs_attrs(cx, Some(v.node.id), &v.node.attrs, v.span, "a variant");
        assert!(!self.in_variant);
        self.in_variant = true;
    }

    fn check_variant_post(&mut self, _: &Context, _: &ast::Variant, _: &ast::Generics) {
        assert!(self.in_variant);
        self.in_variant = false;
    }
}

declare_lint! {
    pub MISSING_COPY_IMPLEMENTATIONS,
    Allow,
    "detects potentially-forgotten implementations of `Copy`"
}

#[derive(Copy, Clone)]
pub struct MissingCopyImplementations;

impl LintPass for MissingCopyImplementations {
    fn get_lints(&self) -> LintArray {
        lint_array!(MISSING_COPY_IMPLEMENTATIONS)
    }

    fn check_item(&mut self, cx: &Context, item: &ast::Item) {
        if !cx.exported_items.contains(&item.id) {
            return;
        }
        let (def, ty) = match item.node {
            ast::ItemStruct(_, ref ast_generics) => {
                if ast_generics.is_parameterized() {
                    return;
                }
                let def = cx.tcx.lookup_adt_def(DefId::local(item.id));
                (def, cx.tcx.mk_struct(def,
                                       cx.tcx.mk_substs(Substs::empty())))
            }
            ast::ItemEnum(_, ref ast_generics) => {
                if ast_generics.is_parameterized() {
                    return;
                }
                let def = cx.tcx.lookup_adt_def(DefId::local(item.id));
                (def, cx.tcx.mk_enum(def,
                                     cx.tcx.mk_substs(Substs::empty())))
            }
            _ => return,
        };
        if def.has_dtor() { return; }
        let parameter_environment = cx.tcx.empty_parameter_environment();
        // FIXME (@jroesch) should probably inver this so that the parameter env still impls this
        // method
        if !ty.moves_by_default(&parameter_environment, item.span) {
            return;
        }
        if parameter_environment.can_type_implement_copy(ty, item.span).is_ok() {
            cx.span_lint(MISSING_COPY_IMPLEMENTATIONS,
                         item.span,
                         "type could implement `Copy`; consider adding `impl \
                          Copy`")
        }
    }
}

declare_lint! {
    MISSING_DEBUG_IMPLEMENTATIONS,
    Allow,
    "detects missing implementations of fmt::Debug"
}

pub struct MissingDebugImplementations {
    impling_types: Option<NodeSet>,
}

impl MissingDebugImplementations {
    pub fn new() -> MissingDebugImplementations {
        MissingDebugImplementations {
            impling_types: None,
        }
    }
}

impl LintPass for MissingDebugImplementations {
    fn get_lints(&self) -> LintArray {
        lint_array!(MISSING_DEBUG_IMPLEMENTATIONS)
    }

    fn check_item(&mut self, cx: &Context, item: &ast::Item) {
        if !cx.exported_items.contains(&item.id) {
            return;
        }

        match item.node {
            ast::ItemStruct(..) | ast::ItemEnum(..) => {},
            _ => return,
        }

        let debug = match cx.tcx.lang_items.debug_trait() {
            Some(debug) => debug,
            None => return,
        };

        if self.impling_types.is_none() {
            let debug_def = cx.tcx.lookup_trait_def(debug);
            let mut impls = NodeSet();
            debug_def.for_each_impl(cx.tcx, |d| {
                if d.is_local() {
                    if let Some(ty_def) = cx.tcx.node_id_to_type(d.node).ty_to_def_id() {
                        impls.insert(ty_def.node);
                    }
                }
            });

            self.impling_types = Some(impls);
            debug!("{:?}", self.impling_types);
        }

        if !self.impling_types.as_ref().unwrap().contains(&item.id) {
            cx.span_lint(MISSING_DEBUG_IMPLEMENTATIONS,
                         item.span,
                         "type does not implement `fmt::Debug`; consider adding #[derive(Debug)] \
                          or a manual implementation")
        }
    }
}

declare_lint! {
    DEPRECATED,
    Warn,
    "detects use of #[deprecated] items"
}

/// Checks for use of items with `#[deprecated]` attributes
#[derive(Copy, Clone)]
pub struct Stability;

impl Stability {
    fn lint(&self, cx: &Context, _id: DefId,
            span: Span, stability: &Option<&attr::Stability>) {
        // Deprecated attributes apply in-crate and cross-crate.
        let (lint, label) = match *stability {
            Some(&attr::Stability { deprecated_since: Some(_), .. }) =>
                (DEPRECATED, "deprecated"),
            _ => return
        };

        output(cx, span, stability, lint, label);

        fn output(cx: &Context, span: Span, stability: &Option<&attr::Stability>,
                  lint: &'static Lint, label: &'static str) {
            let msg = match *stability {
                Some(&attr::Stability { reason: Some(ref s), .. }) => {
                    format!("use of {} item: {}", label, *s)
                }
                _ => format!("use of {} item", label)
            };

            cx.span_lint(lint, span, &msg[..]);
        }
    }
}

impl LintPass for Stability {
    fn get_lints(&self) -> LintArray {
        lint_array!(DEPRECATED)
    }

    fn check_item(&mut self, cx: &Context, item: &ast::Item) {
        stability::check_item(cx.tcx, item, false,
                              &mut |id, sp, stab| self.lint(cx, id, sp, stab));
    }

    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) {
        stability::check_expr(cx.tcx, e,
                              &mut |id, sp, stab| self.lint(cx, id, sp, stab));
    }

    fn check_path(&mut self, cx: &Context, path: &ast::Path, id: ast::NodeId) {
        stability::check_path(cx.tcx, path, id,
                              &mut |id, sp, stab| self.lint(cx, id, sp, stab));
    }

    fn check_pat(&mut self, cx: &Context, pat: &ast::Pat) {
        stability::check_pat(cx.tcx, pat,
                             &mut |id, sp, stab| self.lint(cx, id, sp, stab))
    }
}

declare_lint! {
    pub UNCONDITIONAL_RECURSION,
    Warn,
    "functions that cannot return without calling themselves"
}

#[derive(Copy, Clone)]
pub struct UnconditionalRecursion;


impl LintPass for UnconditionalRecursion {
    fn get_lints(&self) -> LintArray {
        lint_array![UNCONDITIONAL_RECURSION]
    }

    fn check_fn(&mut self, cx: &Context, fn_kind: visit::FnKind, _: &ast::FnDecl,
                blk: &ast::Block, sp: Span, id: ast::NodeId) {
        type F = for<'tcx> fn(&ty::ctxt<'tcx>,
                              ast::NodeId, ast::NodeId, ast::Ident, ast::NodeId) -> bool;

        let method = match fn_kind {
            visit::FkItemFn(..) => None,
            visit::FkMethod(..) => {
                cx.tcx.impl_or_trait_item(DefId::local(id)).as_opt_method()
            }
            // closures can't recur, so they don't matter.
            visit::FkClosure => return
        };

        // Walk through this function (say `f`) looking to see if
        // every possible path references itself, i.e. the function is
        // called recursively unconditionally. This is done by trying
        // to find a path from the entry node to the exit node that
        // *doesn't* call `f` by traversing from the entry while
        // pretending that calls of `f` are sinks (i.e. ignoring any
        // exit edges from them).
        //
        // NB. this has an edge case with non-returning statements,
        // like `loop {}` or `panic!()`: control flow never reaches
        // the exit node through these, so one can have a function
        // that never actually calls itselfs but is still picked up by
        // this lint:
        //
        //     fn f(cond: bool) {
        //         if !cond { panic!() } // could come from `assert!(cond)`
        //         f(false)
        //     }
        //
        // In general, functions of that form may be able to call
        // itself a finite number of times and then diverge. The lint
        // considers this to be an error for two reasons, (a) it is
        // easier to implement, and (b) it seems rare to actually want
        // to have behaviour like the above, rather than
        // e.g. accidentally recurring after an assert.

        let cfg = cfg::CFG::new(cx.tcx, blk);

        let mut work_queue = vec![cfg.entry];
        let mut reached_exit_without_self_call = false;
        let mut self_call_spans = vec![];
        let mut visited = HashSet::new();

        while let Some(idx) = work_queue.pop() {
            if idx == cfg.exit {
                // found a path!
                reached_exit_without_self_call = true;
                break;
            }

            let cfg_id = idx.node_id();
            if visited.contains(&cfg_id) {
                // already done
                continue;
            }
            visited.insert(cfg_id);

            let node_id = cfg.graph.node_data(idx).id();

            // is this a recursive call?
            let self_recursive = if node_id != ast::DUMMY_NODE_ID {
                match method {
                    Some(ref method) => {
                        expr_refers_to_this_method(cx.tcx, method, node_id)
                    }
                    None => expr_refers_to_this_fn(cx.tcx, id, node_id)
                }
            } else {
                false
            };
            if self_recursive {
                self_call_spans.push(cx.tcx.map.span(node_id));
                // this is a self call, so we shouldn't explore past
                // this node in the CFG.
                continue;
            }
            // add the successors of this node to explore the graph further.
            for (_, edge) in cfg.graph.outgoing_edges(idx) {
                let target_idx = edge.target();
                let target_cfg_id = target_idx.node_id();
                if !visited.contains(&target_cfg_id) {
                    work_queue.push(target_idx)
                }
            }
        }

        // Check the number of self calls because a function that
        // doesn't return (e.g. calls a `-> !` function or `loop { /*
        // no break */ }`) shouldn't be linted unless it actually
        // recurs.
        if !reached_exit_without_self_call && !self_call_spans.is_empty() {
            cx.span_lint(UNCONDITIONAL_RECURSION, sp,
                         "function cannot return without recurring");

            // FIXME #19668: these could be span_lint_note's instead of this manual guard.
            if cx.current_level(UNCONDITIONAL_RECURSION) != Level::Allow {
                let sess = cx.sess();
                // offer some help to the programmer.
                for call in &self_call_spans {
                    sess.span_note(*call, "recursive call site")
                }
                sess.fileline_help(sp, "a `loop` may express intention \
                                        better if this is on purpose")
            }
        }

        // all done
        return;

        // Functions for identifying if the given Expr NodeId `id`
        // represents a call to the function `fn_id`/method `method`.

        fn expr_refers_to_this_fn(tcx: &ty::ctxt,
                                  fn_id: ast::NodeId,
                                  id: ast::NodeId) -> bool {
            match tcx.map.get(id) {
                ast_map::NodeExpr(&ast::Expr { node: ast::ExprCall(ref callee, _), .. }) => {
                    tcx.def_map.borrow().get(&callee.id)
                        .map_or(false, |def| def.def_id() == DefId::local(fn_id))
                }
                _ => false
            }
        }

        // Check if the expression `id` performs a call to `method`.
        fn expr_refers_to_this_method(tcx: &ty::ctxt,
                                      method: &ty::Method,
                                      id: ast::NodeId) -> bool {
            let tables = tcx.tables.borrow();

            // Check for method calls and overloaded operators.
            if let Some(m) = tables.method_map.get(&ty::MethodCall::expr(id)) {
                if method_call_refers_to_method(tcx, method, m.def_id, m.substs, id) {
                    return true;
                }
            }

            // Check for overloaded autoderef method calls.
            if let Some(&ty::AdjustDerefRef(ref adj)) = tables.adjustments.get(&id) {
                for i in 0..adj.autoderefs {
                    let method_call = ty::MethodCall::autoderef(id, i as u32);
                    if let Some(m) = tables.method_map.get(&method_call) {
                        if method_call_refers_to_method(tcx, method, m.def_id, m.substs, id) {
                            return true;
                        }
                    }
                }
            }

            // Check for calls to methods via explicit paths (e.g. `T::method()`).
            match tcx.map.get(id) {
                ast_map::NodeExpr(&ast::Expr { node: ast::ExprCall(ref callee, _), .. }) => {
                    match tcx.def_map.borrow().get(&callee.id).map(|d| d.full_def()) {
                        Some(def::DefMethod(def_id)) => {
                            let no_substs = &ty::ItemSubsts::empty();
                            let ts = tables.item_substs.get(&callee.id).unwrap_or(no_substs);
                            method_call_refers_to_method(tcx, method, def_id, &ts.substs, id)
                        }
                        _ => false
                    }
                }
                _ => false
            }
        }

        // Check if the method call to the method with the ID `callee_id`
        // and instantiated with `callee_substs` refers to method `method`.
        fn method_call_refers_to_method<'tcx>(tcx: &ty::ctxt<'tcx>,
                                              method: &ty::Method,
                                              callee_id: DefId,
                                              callee_substs: &Substs<'tcx>,
                                              expr_id: ast::NodeId) -> bool {
            let callee_item = tcx.impl_or_trait_item(callee_id);

            match callee_item.container() {
                // This is an inherent method, so the `def_id` refers
                // directly to the method definition.
                ty::ImplContainer(_) => {
                    callee_id == method.def_id
                }

                // A trait method, from any number of possible sources.
                // Attempt to select a concrete impl before checking.
                ty::TraitContainer(trait_def_id) => {
                    let trait_substs = callee_substs.clone().method_to_trait();
                    let trait_substs = tcx.mk_substs(trait_substs);
                    let trait_ref = ty::TraitRef::new(trait_def_id, trait_substs);
                    let trait_ref = ty::Binder(trait_ref);
                    let span = tcx.map.span(expr_id);
                    let obligation =
                        traits::Obligation::new(traits::ObligationCause::misc(span, expr_id),
                                                trait_ref.to_poly_trait_predicate());

                    let param_env = ty::ParameterEnvironment::for_item(tcx, method.def_id.node);
                    let infcx = infer::new_infer_ctxt(tcx, &tcx.tables, Some(param_env), false);
                    let mut selcx = traits::SelectionContext::new(&infcx);
                    match selcx.select(&obligation) {
                        // The method comes from a `T: Trait` bound.
                        // If `T` is `Self`, then this call is inside
                        // a default method definition.
                        Ok(Some(traits::VtableParam(_))) => {
                            let self_ty = callee_substs.self_ty();
                            let on_self = self_ty.map_or(false, |t| t.is_self());
                            // We can only be recurring in a default
                            // method if we're being called literally
                            // on the `Self` type.
                            on_self && callee_id == method.def_id
                        }

                        // The `impl` is known, so we check that with a
                        // special case:
                        Ok(Some(traits::VtableImpl(vtable_impl))) => {
                            let container = ty::ImplContainer(vtable_impl.impl_def_id);
                            // It matches if it comes from the same impl,
                            // and has the same method name.
                            container == method.container
                                && callee_item.name() == method.name
                        }

                        // There's no way to know if this call is
                        // recursive, so we assume it's not.
                        _ => return false
                    }
                }
            }
        }
    }
}

declare_lint! {
    PLUGIN_AS_LIBRARY,
    Warn,
    "compiler plugin used as ordinary library in non-plugin crate"
}

#[derive(Copy, Clone)]
pub struct PluginAsLibrary;

impl LintPass for PluginAsLibrary {
    fn get_lints(&self) -> LintArray {
        lint_array![PLUGIN_AS_LIBRARY]
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        if cx.sess().plugin_registrar_fn.get().is_some() {
            // We're compiling a plugin; it's fine to link other plugins.
            return;
        }

        match it.node {
            ast::ItemExternCrate(..) => (),
            _ => return,
        };

        let md = match cx.sess().cstore.find_extern_mod_stmt_cnum(it.id) {
            Some(cnum) => cx.sess().cstore.get_crate_data(cnum),
            None => {
                // Probably means we aren't linking the crate for some reason.
                //
                // Not sure if / when this could happen.
                return;
            }
        };

        if decoder::get_plugin_registrar_fn(md.data()).is_some() {
            cx.span_lint(PLUGIN_AS_LIBRARY, it.span,
                         "compiler plugin used as an ordinary library");
        }
    }
}

declare_lint! {
    PRIVATE_NO_MANGLE_FNS,
    Warn,
    "functions marked #[no_mangle] should be exported"
}

declare_lint! {
    PRIVATE_NO_MANGLE_STATICS,
    Warn,
    "statics marked #[no_mangle] should be exported"
}

declare_lint! {
    NO_MANGLE_CONST_ITEMS,
    Deny,
    "const items will not have their symbols exported"
}

#[derive(Copy, Clone)]
pub struct InvalidNoMangleItems;

impl LintPass for InvalidNoMangleItems {
    fn get_lints(&self) -> LintArray {
        lint_array!(PRIVATE_NO_MANGLE_FNS,
                    PRIVATE_NO_MANGLE_STATICS,
                    NO_MANGLE_CONST_ITEMS)
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        match it.node {
            ast::ItemFn(..) => {
                if attr::contains_name(&it.attrs, "no_mangle") &&
                       !cx.exported_items.contains(&it.id) {
                    let msg = format!("function {} is marked #[no_mangle], but not exported",
                                      it.ident);
                    cx.span_lint(PRIVATE_NO_MANGLE_FNS, it.span, &msg);
                }
            },
            ast::ItemStatic(..) => {
                if attr::contains_name(&it.attrs, "no_mangle") &&
                       !cx.exported_items.contains(&it.id) {
                    let msg = format!("static {} is marked #[no_mangle], but not exported",
                                      it.ident);
                    cx.span_lint(PRIVATE_NO_MANGLE_STATICS, it.span, &msg);
                }
            },
            ast::ItemConst(..) => {
                if attr::contains_name(&it.attrs, "no_mangle") {
                    // Const items do not refer to a particular location in memory, and therefore
                    // don't have anything to attach a symbol to
                    let msg = "const items should never be #[no_mangle], consider instead using \
                               `pub static`";
                    cx.span_lint(NO_MANGLE_CONST_ITEMS, it.span, msg);
                }
            }
            _ => {},
        }
    }
}

#[derive(Clone, Copy)]
pub struct MutableTransmutes;

declare_lint! {
    MUTABLE_TRANSMUTES,
    Deny,
    "mutating transmuted &mut T from &T may cause undefined behavior"
}

impl LintPass for MutableTransmutes {
    fn get_lints(&self) -> LintArray {
        lint_array!(MUTABLE_TRANSMUTES)
    }

    fn check_expr(&mut self, cx: &Context, expr: &ast::Expr) {
        use syntax::abi::RustIntrinsic;
        let msg = "mutating transmuted &mut T from &T may cause undefined behavior,\
                   consider instead using an UnsafeCell";
        match get_transmute_from_to(cx, expr) {
            Some((&ty::TyRef(_, from_mt), &ty::TyRef(_, to_mt))) => {
                if to_mt.mutbl == ast::Mutability::MutMutable
                    && from_mt.mutbl == ast::Mutability::MutImmutable {
                    cx.span_lint(MUTABLE_TRANSMUTES, expr.span, msg);
                }
            }
            _ => ()
        }

        fn get_transmute_from_to<'a, 'tcx>(cx: &Context<'a, 'tcx>, expr: &ast::Expr)
            -> Option<(&'tcx ty::TypeVariants<'tcx>, &'tcx ty::TypeVariants<'tcx>)> {
            match expr.node {
                ast::ExprPath(..) => (),
                _ => return None
            }
            if let def::DefFn(did, _) = cx.tcx.resolve_expr(expr) {
                if !def_id_is_transmute(cx, did) {
                    return None;
                }
                let typ = cx.tcx.node_id_to_type(expr.id);
                match typ.sty {
                    ty::TyBareFn(_, ref bare_fn) if bare_fn.abi == RustIntrinsic => {
                        if let ty::FnConverging(to) = bare_fn.sig.0.output {
                            let from = bare_fn.sig.0.inputs[0];
                            return Some((&from.sty, &to.sty));
                        }
                    },
                    _ => ()
                }
            }
            None
        }

        fn def_id_is_transmute(cx: &Context, def_id: DefId) -> bool {
            match cx.tcx.lookup_item_type(def_id).ty.sty {
                ty::TyBareFn(_, ref bfty) if bfty.abi == RustIntrinsic => (),
                _ => return false
            }
            cx.tcx.with_path(def_id, |path| match path.last() {
                Some(ref last) => last.name().as_str() == "transmute",
                _ => false
            })
        }
    }
}

/// Forbids using the `#[feature(...)]` attribute
#[derive(Copy, Clone)]
pub struct UnstableFeatures;

declare_lint! {
    UNSTABLE_FEATURES,
    Allow,
    "enabling unstable features (deprecated. do not use)"
}

impl LintPass for UnstableFeatures {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNSTABLE_FEATURES)
    }
    fn check_attribute(&mut self, ctx: &Context, attr: &ast::Attribute) {
        if attr::contains_name(&[attr.node.value.clone()], "feature") {
            if let Some(items) = attr.node.value.meta_item_list() {
                for item in items {
                    ctx.span_lint(UNSTABLE_FEATURES, item.span, "unstable feature");
                }
            }
        }
    }
}

/// Lints for attempts to impl Drop on types that have `#[repr(C)]`
/// attribute (see issue #24585).
#[derive(Copy, Clone)]
pub struct DropWithReprExtern;

declare_lint! {
    DROP_WITH_REPR_EXTERN,
    Warn,
    "use of #[repr(C)] on a type that implements Drop"
}

impl LintPass for DropWithReprExtern {
    fn get_lints(&self) -> LintArray {
        lint_array!(DROP_WITH_REPR_EXTERN)
    }
    fn check_crate(&mut self, ctx: &Context, _: &ast::Crate) {
        for dtor_did in ctx.tcx.destructors.borrow().iter() {
            let (drop_impl_did, dtor_self_type) =
                if dtor_did.is_local() {
                    let impl_did = ctx.tcx.map.get_parent_did(dtor_did.node);
                    let ty = ctx.tcx.lookup_item_type(impl_did).ty;
                    (impl_did, ty)
                } else {
                    continue;
                };

            match dtor_self_type.sty {
                ty::TyEnum(self_type_def, _) |
                ty::TyStruct(self_type_def, _) => {
                    let self_type_did = self_type_def.did;
                    let hints = ctx.tcx.lookup_repr_hints(self_type_did);
                    if hints.iter().any(|attr| *attr == attr::ReprExtern) &&
                        self_type_def.dtor_kind().has_drop_flag() {
                        let drop_impl_span = ctx.tcx.map.def_id_span(drop_impl_did,
                                                                     codemap::DUMMY_SP);
                        let self_defn_span = ctx.tcx.map.def_id_span(self_type_did,
                                                                     codemap::DUMMY_SP);
                        ctx.span_lint(DROP_WITH_REPR_EXTERN,
                                      drop_impl_span,
                                      "implementing Drop adds hidden state to types, \
                                       possibly conflicting with `#[repr(C)]`");
                        // FIXME #19668: could be span_lint_note instead of manual guard.
                        if ctx.current_level(DROP_WITH_REPR_EXTERN) != Level::Allow {
                            ctx.sess().span_note(self_defn_span,
                                               "the `#[repr(C)]` attribute is attached here");
                        }
                    }
                }
                _ => {}
            }
        }
    }
}
