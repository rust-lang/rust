#![allow(non_snake_case)]

use crate::{LateContext, LateLintPass, LintContext};
use rustc::mir::interpret::{sign_extend, truncate};
use rustc::ty::layout::{self, IntegerExt, LayoutOf, SizeSkeleton, VariantIdx};
use rustc::ty::subst::SubstsRef;
use rustc::ty::{self, AdtKind, ParamEnv, Ty, TyCtxt};
use rustc_attr as attr;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::{is_range_literal, ExprKind, Node};
use rustc_index::vec::Idx;
use rustc_span::source_map;
use rustc_span::symbol::sym;
use rustc_span::Span;
use rustc_target::spec::abi::Abi;
use syntax::ast;

use log::debug;
use std::cmp;
use std::{f32, f64, i16, i32, i64, i8, u16, u32, u64, u8};

declare_lint! {
    UNUSED_COMPARISONS,
    Warn,
    "comparisons made useless by limits of the types involved"
}

declare_lint! {
    OVERFLOWING_LITERALS,
    Deny,
    "literal out of range for its type"
}

declare_lint! {
    VARIANT_SIZE_DIFFERENCES,
    Allow,
    "detects enums with widely varying variant sizes"
}

#[derive(Copy, Clone)]
pub struct TypeLimits {
    /// Id of the last visited negated expression
    negated_expr_id: hir::HirId,
}

impl_lint_pass!(TypeLimits => [UNUSED_COMPARISONS, OVERFLOWING_LITERALS]);

impl TypeLimits {
    pub fn new() -> TypeLimits {
        TypeLimits { negated_expr_id: hir::DUMMY_HIR_ID }
    }
}

/// Attempts to special-case the overflowing literal lint when it occurs as a range endpoint.
/// Returns `true` iff the lint was overridden.
fn lint_overflowing_range_endpoint<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    lit: &hir::Lit,
    lit_val: u128,
    max: u128,
    expr: &'tcx hir::Expr<'tcx>,
    parent_expr: &'tcx hir::Expr<'tcx>,
    ty: &str,
) -> bool {
    // We only want to handle exclusive (`..`) ranges,
    // which are represented as `ExprKind::Struct`.
    let mut overwritten = false;
    if let ExprKind::Struct(_, eps, _) = &parent_expr.kind {
        if eps.len() != 2 {
            return false;
        }
        // We can suggest using an inclusive range
        // (`..=`) instead only if it is the `end` that is
        // overflowing and only by 1.
        if eps[1].expr.hir_id == expr.hir_id && lit_val - 1 == max {
            cx.struct_span_lint(OVERFLOWING_LITERALS, parent_expr.span, |lint| {
                let mut err = lint.build(&format!("range endpoint is out of range for `{}`", ty));
                if let Ok(start) = cx.sess().source_map().span_to_snippet(eps[0].span) {
                    use ast::{LitIntType, LitKind};
                    // We need to preserve the literal's suffix,
                    // as it may determine typing information.
                    let suffix = match lit.node {
                        LitKind::Int(_, LitIntType::Signed(s)) => format!("{}", s.name_str()),
                        LitKind::Int(_, LitIntType::Unsigned(s)) => format!("{}", s.name_str()),
                        LitKind::Int(_, LitIntType::Unsuffixed) => "".to_owned(),
                        _ => bug!(),
                    };
                    let suggestion = format!("{}..={}{}", start, lit_val - 1, suffix);
                    err.span_suggestion(
                        parent_expr.span,
                        &"use an inclusive range instead",
                        suggestion,
                        Applicability::MachineApplicable,
                    );
                    err.emit();
                    overwritten = true;
                }
            });
        }
    }
    overwritten
}

// For `isize` & `usize`, be conservative with the warnings, so that the
// warnings are consistent between 32- and 64-bit platforms.
fn int_ty_range(int_ty: ast::IntTy) -> (i128, i128) {
    match int_ty {
        ast::IntTy::Isize => (i64::min_value() as i128, i64::max_value() as i128),
        ast::IntTy::I8 => (i8::min_value() as i64 as i128, i8::max_value() as i128),
        ast::IntTy::I16 => (i16::min_value() as i64 as i128, i16::max_value() as i128),
        ast::IntTy::I32 => (i32::min_value() as i64 as i128, i32::max_value() as i128),
        ast::IntTy::I64 => (i64::min_value() as i128, i64::max_value() as i128),
        ast::IntTy::I128 => (i128::min_value() as i128, i128::max_value()),
    }
}

fn uint_ty_range(uint_ty: ast::UintTy) -> (u128, u128) {
    match uint_ty {
        ast::UintTy::Usize => (u64::min_value() as u128, u64::max_value() as u128),
        ast::UintTy::U8 => (u8::min_value() as u128, u8::max_value() as u128),
        ast::UintTy::U16 => (u16::min_value() as u128, u16::max_value() as u128),
        ast::UintTy::U32 => (u32::min_value() as u128, u32::max_value() as u128),
        ast::UintTy::U64 => (u64::min_value() as u128, u64::max_value() as u128),
        ast::UintTy::U128 => (u128::min_value(), u128::max_value()),
    }
}

fn get_bin_hex_repr(cx: &LateContext<'_, '_>, lit: &hir::Lit) -> Option<String> {
    let src = cx.sess().source_map().span_to_snippet(lit.span).ok()?;
    let firstch = src.chars().next()?;

    if firstch == '0' {
        match src.chars().nth(1) {
            Some('x') | Some('b') => return Some(src),
            _ => return None,
        }
    }

    None
}

fn report_bin_hex_error(
    cx: &LateContext<'_, '_>,
    expr: &hir::Expr<'_>,
    ty: attr::IntType,
    repr_str: String,
    val: u128,
    negative: bool,
) {
    let size = layout::Integer::from_attr(&cx.tcx, ty).size();
    cx.struct_span_lint(OVERFLOWING_LITERALS, expr.span, |lint| {
        let (t, actually) = match ty {
            attr::IntType::SignedInt(t) => {
                let actually = sign_extend(val, size) as i128;
                (t.name_str(), actually.to_string())
            }
            attr::IntType::UnsignedInt(t) => {
                let actually = truncate(val, size);
                (t.name_str(), actually.to_string())
            }
        };
        let mut err = lint.build(&format!("literal out of range for {}", t));
        err.note(&format!(
            "the literal `{}` (decimal `{}`) does not fit into \
                    an `{}` and will become `{}{}`",
            repr_str, val, t, actually, t
        ));
        if let Some(sugg_ty) = get_type_suggestion(&cx.tables.node_type(expr.hir_id), val, negative)
        {
            if let Some(pos) = repr_str.chars().position(|c| c == 'i' || c == 'u') {
                let (sans_suffix, _) = repr_str.split_at(pos);
                err.span_suggestion(
                    expr.span,
                    &format!("consider using `{}` instead", sugg_ty),
                    format!("{}{}", sans_suffix, sugg_ty),
                    Applicability::MachineApplicable,
                );
            } else {
                err.help(&format!("consider using `{}` instead", sugg_ty));
            }
        }
        err.emit();
    });
}

// This function finds the next fitting type and generates a suggestion string.
// It searches for fitting types in the following way (`X < Y`):
//  - `iX`: if literal fits in `uX` => `uX`, else => `iY`
//  - `-iX` => `iY`
//  - `uX` => `uY`
//
// No suggestion for: `isize`, `usize`.
fn get_type_suggestion(t: Ty<'_>, val: u128, negative: bool) -> Option<&'static str> {
    use syntax::ast::IntTy::*;
    use syntax::ast::UintTy::*;
    macro_rules! find_fit {
        ($ty:expr, $val:expr, $negative:expr,
         $($type:ident => [$($utypes:expr),*] => [$($itypes:expr),*]),+) => {
            {
                let _neg = if negative { 1 } else { 0 };
                match $ty {
                    $($type => {
                        $(if !negative && val <= uint_ty_range($utypes).1 {
                            return Some($utypes.name_str())
                        })*
                        $(if val <= int_ty_range($itypes).1 as u128 + _neg {
                            return Some($itypes.name_str())
                        })*
                        None
                    },)+
                    _ => None
                }
            }
        }
    }
    match t.kind {
        ty::Int(i) => find_fit!(i, val, negative,
                      I8 => [U8] => [I16, I32, I64, I128],
                      I16 => [U16] => [I32, I64, I128],
                      I32 => [U32] => [I64, I128],
                      I64 => [U64] => [I128],
                      I128 => [U128] => []),
        ty::Uint(u) => find_fit!(u, val, negative,
                      U8 => [U8, U16, U32, U64, U128] => [],
                      U16 => [U16, U32, U64, U128] => [],
                      U32 => [U32, U64, U128] => [],
                      U64 => [U64, U128] => [],
                      U128 => [U128] => []),
        _ => None,
    }
}

fn lint_int_literal<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    type_limits: &TypeLimits,
    e: &'tcx hir::Expr<'tcx>,
    lit: &hir::Lit,
    t: ast::IntTy,
    v: u128,
) {
    let int_type = t.normalize(cx.sess().target.ptr_width);
    let (_, max) = int_ty_range(int_type);
    let max = max as u128;
    let negative = type_limits.negated_expr_id == e.hir_id;

    // Detect literal value out of range [min, max] inclusive
    // avoiding use of -min to prevent overflow/panic
    if (negative && v > max + 1) || (!negative && v > max) {
        if let Some(repr_str) = get_bin_hex_repr(cx, lit) {
            report_bin_hex_error(cx, e, attr::IntType::SignedInt(t), repr_str, v, negative);
            return;
        }

        let par_id = cx.tcx.hir().get_parent_node(e.hir_id);
        if let Node::Expr(par_e) = cx.tcx.hir().get(par_id) {
            if let hir::ExprKind::Struct(..) = par_e.kind {
                if is_range_literal(cx.sess().source_map(), par_e)
                    && lint_overflowing_range_endpoint(cx, lit, v, max, e, par_e, t.name_str())
                {
                    // The overflowing literal lint was overridden.
                    return;
                }
            }
        }

        cx.struct_span_lint(OVERFLOWING_LITERALS, e.span, |lint| {
            lint.build(&format!("literal out of range for `{}`", t.name_str())).emit()
        });
    }
}

fn lint_uint_literal<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    e: &'tcx hir::Expr<'tcx>,
    lit: &hir::Lit,
    t: ast::UintTy,
) {
    let uint_type = t.normalize(cx.sess().target.ptr_width);
    let (min, max) = uint_ty_range(uint_type);
    let lit_val: u128 = match lit.node {
        // _v is u8, within range by definition
        ast::LitKind::Byte(_v) => return,
        ast::LitKind::Int(v, _) => v,
        _ => bug!(),
    };
    if lit_val < min || lit_val > max {
        let parent_id = cx.tcx.hir().get_parent_node(e.hir_id);
        if let Node::Expr(par_e) = cx.tcx.hir().get(parent_id) {
            match par_e.kind {
                hir::ExprKind::Cast(..) => {
                    if let ty::Char = cx.tables.expr_ty(par_e).kind {
                        cx.struct_span_lint(OVERFLOWING_LITERALS, par_e.span, |lint| {
                            lint.build("only `u8` can be cast into `char`")
                                .span_suggestion(
                                    par_e.span,
                                    &"use a `char` literal instead",
                                    format!("'\\u{{{:X}}}'", lit_val),
                                    Applicability::MachineApplicable,
                                )
                                .emit();
                        });
                        return;
                    }
                }
                hir::ExprKind::Struct(..) if is_range_literal(cx.sess().source_map(), par_e) => {
                    let t = t.name_str();
                    if lint_overflowing_range_endpoint(cx, lit, lit_val, max, e, par_e, t) {
                        // The overflowing literal lint was overridden.
                        return;
                    }
                }
                _ => {}
            }
        }
        if let Some(repr_str) = get_bin_hex_repr(cx, lit) {
            report_bin_hex_error(cx, e, attr::IntType::UnsignedInt(t), repr_str, lit_val, false);
            return;
        }
        cx.struct_span_lint(OVERFLOWING_LITERALS, e.span, |lint| {
            lint.build(&format!("literal out of range for `{}`", t.name_str())).emit()
        });
    }
}

fn lint_literal<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    type_limits: &TypeLimits,
    e: &'tcx hir::Expr<'tcx>,
    lit: &hir::Lit,
) {
    match cx.tables.node_type(e.hir_id).kind {
        ty::Int(t) => {
            match lit.node {
                ast::LitKind::Int(v, ast::LitIntType::Signed(_))
                | ast::LitKind::Int(v, ast::LitIntType::Unsuffixed) => {
                    lint_int_literal(cx, type_limits, e, lit, t, v)
                }
                _ => bug!(),
            };
        }
        ty::Uint(t) => lint_uint_literal(cx, e, lit, t),
        ty::Float(t) => {
            let is_infinite = match lit.node {
                ast::LitKind::Float(v, _) => match t {
                    ast::FloatTy::F32 => v.as_str().parse().map(f32::is_infinite),
                    ast::FloatTy::F64 => v.as_str().parse().map(f64::is_infinite),
                },
                _ => bug!(),
            };
            if is_infinite == Ok(true) {
                cx.struct_span_lint(OVERFLOWING_LITERALS, e.span, |lint| {
                    lint.build(&format!("literal out of range for `{}`", t.name_str())).emit()
                });
            }
        }
        _ => {}
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for TypeLimits {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx hir::Expr<'tcx>) {
        match e.kind {
            hir::ExprKind::Unary(hir::UnOp::UnNeg, ref expr) => {
                // propagate negation, if the negation itself isn't negated
                if self.negated_expr_id != e.hir_id {
                    self.negated_expr_id = expr.hir_id;
                }
            }
            hir::ExprKind::Binary(binop, ref l, ref r) => {
                if is_comparison(binop) && !check_limits(cx, binop, &l, &r) {
                    cx.struct_span_lint(UNUSED_COMPARISONS, e.span, |lint| {
                        lint.build("comparison is useless due to type limits").emit()
                    });
                }
            }
            hir::ExprKind::Lit(ref lit) => lint_literal(cx, self, e, lit),
            _ => {}
        };

        fn is_valid<T: cmp::PartialOrd>(binop: hir::BinOp, v: T, min: T, max: T) -> bool {
            match binop.node {
                hir::BinOpKind::Lt => v > min && v <= max,
                hir::BinOpKind::Le => v >= min && v < max,
                hir::BinOpKind::Gt => v >= min && v < max,
                hir::BinOpKind::Ge => v > min && v <= max,
                hir::BinOpKind::Eq | hir::BinOpKind::Ne => v >= min && v <= max,
                _ => bug!(),
            }
        }

        fn rev_binop(binop: hir::BinOp) -> hir::BinOp {
            source_map::respan(
                binop.span,
                match binop.node {
                    hir::BinOpKind::Lt => hir::BinOpKind::Gt,
                    hir::BinOpKind::Le => hir::BinOpKind::Ge,
                    hir::BinOpKind::Gt => hir::BinOpKind::Lt,
                    hir::BinOpKind::Ge => hir::BinOpKind::Le,
                    _ => return binop,
                },
            )
        }

        fn check_limits(
            cx: &LateContext<'_, '_>,
            binop: hir::BinOp,
            l: &hir::Expr<'_>,
            r: &hir::Expr<'_>,
        ) -> bool {
            let (lit, expr, swap) = match (&l.kind, &r.kind) {
                (&hir::ExprKind::Lit(_), _) => (l, r, true),
                (_, &hir::ExprKind::Lit(_)) => (r, l, false),
                _ => return true,
            };
            // Normalize the binop so that the literal is always on the RHS in
            // the comparison
            let norm_binop = if swap { rev_binop(binop) } else { binop };
            match cx.tables.node_type(expr.hir_id).kind {
                ty::Int(int_ty) => {
                    let (min, max) = int_ty_range(int_ty);
                    let lit_val: i128 = match lit.kind {
                        hir::ExprKind::Lit(ref li) => match li.node {
                            ast::LitKind::Int(v, ast::LitIntType::Signed(_))
                            | ast::LitKind::Int(v, ast::LitIntType::Unsuffixed) => v as i128,
                            _ => return true,
                        },
                        _ => bug!(),
                    };
                    is_valid(norm_binop, lit_val, min, max)
                }
                ty::Uint(uint_ty) => {
                    let (min, max): (u128, u128) = uint_ty_range(uint_ty);
                    let lit_val: u128 = match lit.kind {
                        hir::ExprKind::Lit(ref li) => match li.node {
                            ast::LitKind::Int(v, _) => v,
                            _ => return true,
                        },
                        _ => bug!(),
                    };
                    is_valid(norm_binop, lit_val, min, max)
                }
                _ => true,
            }
        }

        fn is_comparison(binop: hir::BinOp) -> bool {
            match binop.node {
                hir::BinOpKind::Eq
                | hir::BinOpKind::Lt
                | hir::BinOpKind::Le
                | hir::BinOpKind::Ne
                | hir::BinOpKind::Ge
                | hir::BinOpKind::Gt => true,
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

declare_lint_pass!(ImproperCTypes => [IMPROPER_CTYPES]);

struct ImproperCTypesVisitor<'a, 'tcx> {
    cx: &'a LateContext<'a, 'tcx>,
}

enum FfiResult<'tcx> {
    FfiSafe,
    FfiPhantom(Ty<'tcx>),
    FfiUnsafe { ty: Ty<'tcx>, reason: &'static str, help: Option<&'static str> },
}

fn is_zst<'tcx>(tcx: TyCtxt<'tcx>, did: DefId, ty: Ty<'tcx>) -> bool {
    tcx.layout_of(tcx.param_env(did).and(ty)).map(|layout| layout.is_zst()).unwrap_or(false)
}

fn ty_is_known_nonnull<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.kind {
        ty::FnPtr(_) => true,
        ty::Ref(..) => true,
        ty::Adt(field_def, substs) if field_def.repr.transparent() && !field_def.is_union() => {
            for field in field_def.all_fields() {
                let field_ty =
                    tcx.normalize_erasing_regions(ParamEnv::reveal_all(), field.ty(tcx, substs));
                if is_zst(tcx, field.did, field_ty) {
                    continue;
                }

                let attrs = tcx.get_attrs(field_def.did);
                if attrs.iter().any(|a| a.check_name(sym::rustc_nonnull_optimization_guaranteed))
                    || ty_is_known_nonnull(tcx, field_ty)
                {
                    return true;
                }
            }

            false
        }
        _ => false,
    }
}

/// Check if this enum can be safely exported based on the
/// "nullable pointer optimization". Currently restricted
/// to function pointers, references, core::num::NonZero*,
/// core::ptr::NonNull, and #[repr(transparent)] newtypes.
/// FIXME: This duplicates code in codegen.
fn is_repr_nullable_ptr<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    ty_def: &'tcx ty::AdtDef,
    substs: SubstsRef<'tcx>,
) -> bool {
    if ty_def.variants.len() != 2 {
        return false;
    }

    let get_variant_fields = |index| &ty_def.variants[VariantIdx::new(index)].fields;
    let variant_fields = [get_variant_fields(0), get_variant_fields(1)];
    let fields = if variant_fields[0].is_empty() {
        &variant_fields[1]
    } else if variant_fields[1].is_empty() {
        &variant_fields[0]
    } else {
        return false;
    };

    if fields.len() != 1 {
        return false;
    }

    let field_ty = fields[0].ty(tcx, substs);
    if !ty_is_known_nonnull(tcx, field_ty) {
        return false;
    }

    // At this point, the field's type is known to be nonnull and the parent enum is Option-like.
    // If the computed size for the field and the enum are different, the nonnull optimization isn't
    // being applied (and we've got a problem somewhere).
    let compute_size_skeleton = |t| SizeSkeleton::compute(t, tcx, ParamEnv::reveal_all()).unwrap();
    if !compute_size_skeleton(ty).same_size(compute_size_skeleton(field_ty)) {
        bug!("improper_ctypes: Option nonnull optimization not applied?");
    }

    true
}

impl<'a, 'tcx> ImproperCTypesVisitor<'a, 'tcx> {
    /// Check if the type is array and emit an unsafe type lint.
    fn check_for_array_ty(&mut self, sp: Span, ty: Ty<'tcx>) -> bool {
        if let ty::Array(..) = ty.kind {
            self.emit_ffi_unsafe_type_lint(
                ty,
                sp,
                "passing raw arrays by value is not FFI-safe",
                Some("consider passing a pointer to the array"),
            );
            true
        } else {
            false
        }
    }

    /// Checks if the given type is "ffi-safe" (has a stable, well-defined
    /// representation which can be exported to C code).
    fn check_type_for_ffi(&self, cache: &mut FxHashSet<Ty<'tcx>>, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        use FfiResult::*;

        let cx = self.cx.tcx;

        // Protect against infinite recursion, for example
        // `struct S(*mut S);`.
        // FIXME: A recursion limit is necessary as well, for irregular
        // recursive types.
        if !cache.insert(ty) {
            return FfiSafe;
        }

        match ty.kind {
            ty::Adt(def, substs) => {
                if def.is_phantom_data() {
                    return FfiPhantom(ty);
                }
                match def.adt_kind() {
                    AdtKind::Struct => {
                        if !def.repr.c() && !def.repr.transparent() {
                            return FfiUnsafe {
                                ty,
                                reason: "this struct has unspecified layout",
                                help: Some(
                                    "consider adding a `#[repr(C)]` or \
                                            `#[repr(transparent)]` attribute to this struct",
                                ),
                            };
                        }

                        let is_non_exhaustive =
                            def.non_enum_variant().is_field_list_non_exhaustive();
                        if is_non_exhaustive && !def.did.is_local() {
                            return FfiUnsafe {
                                ty,
                                reason: "this struct is non-exhaustive",
                                help: None,
                            };
                        }

                        if def.non_enum_variant().fields.is_empty() {
                            return FfiUnsafe {
                                ty,
                                reason: "this struct has no fields",
                                help: Some("consider adding a member to this struct"),
                            };
                        }

                        // We can't completely trust repr(C) and repr(transparent) markings;
                        // make sure the fields are actually safe.
                        let mut all_phantom = true;
                        for field in &def.non_enum_variant().fields {
                            let field_ty = cx.normalize_erasing_regions(
                                ParamEnv::reveal_all(),
                                field.ty(cx, substs),
                            );
                            // repr(transparent) types are allowed to have arbitrary ZSTs, not just
                            // PhantomData -- skip checking all ZST fields
                            if def.repr.transparent() && is_zst(cx, field.did, field_ty) {
                                continue;
                            }
                            let r = self.check_type_for_ffi(cache, field_ty);
                            match r {
                                FfiSafe => {
                                    all_phantom = false;
                                }
                                FfiPhantom(..) => {}
                                FfiUnsafe { .. } => {
                                    return r;
                                }
                            }
                        }

                        if all_phantom { FfiPhantom(ty) } else { FfiSafe }
                    }
                    AdtKind::Union => {
                        if !def.repr.c() && !def.repr.transparent() {
                            return FfiUnsafe {
                                ty,
                                reason: "this union has unspecified layout",
                                help: Some(
                                    "consider adding a `#[repr(C)]` or \
                                            `#[repr(transparent)]` attribute to this union",
                                ),
                            };
                        }

                        if def.non_enum_variant().fields.is_empty() {
                            return FfiUnsafe {
                                ty,
                                reason: "this union has no fields",
                                help: Some("consider adding a field to this union"),
                            };
                        }

                        let mut all_phantom = true;
                        for field in &def.non_enum_variant().fields {
                            let field_ty = cx.normalize_erasing_regions(
                                ParamEnv::reveal_all(),
                                field.ty(cx, substs),
                            );
                            // repr(transparent) types are allowed to have arbitrary ZSTs, not just
                            // PhantomData -- skip checking all ZST fields.
                            if def.repr.transparent() && is_zst(cx, field.did, field_ty) {
                                continue;
                            }
                            let r = self.check_type_for_ffi(cache, field_ty);
                            match r {
                                FfiSafe => {
                                    all_phantom = false;
                                }
                                FfiPhantom(..) => {}
                                FfiUnsafe { .. } => {
                                    return r;
                                }
                            }
                        }

                        if all_phantom { FfiPhantom(ty) } else { FfiSafe }
                    }
                    AdtKind::Enum => {
                        if def.variants.is_empty() {
                            // Empty enums are okay... although sort of useless.
                            return FfiSafe;
                        }

                        // Check for a repr() attribute to specify the size of the
                        // discriminant.
                        if !def.repr.c() && !def.repr.transparent() && def.repr.int.is_none() {
                            // Special-case types like `Option<extern fn()>`.
                            if !is_repr_nullable_ptr(cx, ty, def, substs) {
                                return FfiUnsafe {
                                    ty,
                                    reason: "enum has no representation hint",
                                    help: Some(
                                        "consider adding a `#[repr(C)]`, \
                                                `#[repr(transparent)]`, or integer `#[repr(...)]` \
                                                attribute to this enum",
                                    ),
                                };
                            }
                        }

                        if def.is_variant_list_non_exhaustive() && !def.did.is_local() {
                            return FfiUnsafe {
                                ty,
                                reason: "this enum is non-exhaustive",
                                help: None,
                            };
                        }

                        // Check the contained variants.
                        for variant in &def.variants {
                            let is_non_exhaustive = variant.is_field_list_non_exhaustive();
                            if is_non_exhaustive && !variant.def_id.is_local() {
                                return FfiUnsafe {
                                    ty,
                                    reason: "this enum has non-exhaustive variants",
                                    help: None,
                                };
                            }

                            for field in &variant.fields {
                                let field_ty = cx.normalize_erasing_regions(
                                    ParamEnv::reveal_all(),
                                    field.ty(cx, substs),
                                );
                                // repr(transparent) types are allowed to have arbitrary ZSTs, not
                                // just PhantomData -- skip checking all ZST fields.
                                if def.repr.transparent() && is_zst(cx, field.did, field_ty) {
                                    continue;
                                }
                                let r = self.check_type_for_ffi(cache, field_ty);
                                match r {
                                    FfiSafe => {}
                                    FfiUnsafe { .. } => {
                                        return r;
                                    }
                                    FfiPhantom(..) => {
                                        return FfiUnsafe {
                                            ty,
                                            reason: "this enum contains a PhantomData field",
                                            help: None,
                                        };
                                    }
                                }
                            }
                        }
                        FfiSafe
                    }
                }
            }

            ty::Char => FfiUnsafe {
                ty,
                reason: "the `char` type has no C equivalent",
                help: Some("consider using `u32` or `libc::wchar_t` instead"),
            },

            ty::Int(ast::IntTy::I128) | ty::Uint(ast::UintTy::U128) => FfiUnsafe {
                ty,
                reason: "128-bit integers don't currently have a known stable ABI",
                help: None,
            },

            // Primitive types with a stable representation.
            ty::Bool | ty::Int(..) | ty::Uint(..) | ty::Float(..) | ty::Never => FfiSafe,

            ty::Slice(_) => FfiUnsafe {
                ty,
                reason: "slices have no C equivalent",
                help: Some("consider using a raw pointer instead"),
            },

            ty::Dynamic(..) => {
                FfiUnsafe { ty, reason: "trait objects have no C equivalent", help: None }
            }

            ty::Str => FfiUnsafe {
                ty,
                reason: "string slices have no C equivalent",
                help: Some("consider using `*const u8` and a length instead"),
            },

            ty::Tuple(..) => FfiUnsafe {
                ty,
                reason: "tuples have unspecified layout",
                help: Some("consider using a struct instead"),
            },

            ty::RawPtr(ty::TypeAndMut { ty, .. }) | ty::Ref(_, ty, _) => {
                self.check_type_for_ffi(cache, ty)
            }

            ty::Array(inner_ty, _) => self.check_type_for_ffi(cache, inner_ty),

            ty::FnPtr(sig) => {
                match sig.abi() {
                    Abi::Rust | Abi::RustIntrinsic | Abi::PlatformIntrinsic | Abi::RustCall => {
                        return FfiUnsafe {
                            ty,
                            reason: "this function pointer has Rust-specific calling convention",
                            help: Some(
                                "consider using an `extern fn(...) -> ...` \
                                        function pointer instead",
                            ),
                        };
                    }
                    _ => {}
                }

                let sig = cx.erase_late_bound_regions(&sig);
                if !sig.output().is_unit() {
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

            ty::Foreign(..) => FfiSafe,

            ty::Param(..)
            | ty::Infer(..)
            | ty::Bound(..)
            | ty::Error
            | ty::Closure(..)
            | ty::Generator(..)
            | ty::GeneratorWitness(..)
            | ty::Placeholder(..)
            | ty::UnnormalizedProjection(..)
            | ty::Projection(..)
            | ty::Opaque(..)
            | ty::FnDef(..) => bug!("unexpected type in foreign function: {:?}", ty),
        }
    }

    fn emit_ffi_unsafe_type_lint(
        &mut self,
        ty: Ty<'tcx>,
        sp: Span,
        note: &str,
        help: Option<&str>,
    ) {
        self.cx.struct_span_lint(IMPROPER_CTYPES, sp, |lint| {
            let mut diag =
                lint.build(&format!("`extern` block uses type `{}`, which is not FFI-safe", ty));
            diag.span_label(sp, "not FFI-safe");
            if let Some(help) = help {
                diag.help(help);
            }
            diag.note(note);
            if let ty::Adt(def, _) = ty.kind {
                if let Some(sp) = self.cx.tcx.hir().span_if_local(def.did) {
                    diag.span_note(sp, "the type is defined here");
                }
            }
            diag.emit();
        });
    }

    fn check_for_opaque_ty(&mut self, sp: Span, ty: Ty<'tcx>) -> bool {
        use crate::rustc::ty::TypeFoldable;

        struct ProhibitOpaqueTypes<'tcx> {
            ty: Option<Ty<'tcx>>,
        };

        impl<'tcx> ty::fold::TypeVisitor<'tcx> for ProhibitOpaqueTypes<'tcx> {
            fn visit_ty(&mut self, ty: Ty<'tcx>) -> bool {
                if let ty::Opaque(..) = ty.kind {
                    self.ty = Some(ty);
                    true
                } else {
                    ty.super_visit_with(self)
                }
            }
        }

        let mut visitor = ProhibitOpaqueTypes { ty: None };
        ty.visit_with(&mut visitor);
        if let Some(ty) = visitor.ty {
            self.emit_ffi_unsafe_type_lint(ty, sp, "opaque types have no C equivalent", None);
            true
        } else {
            false
        }
    }

    fn check_type_for_ffi_and_report_errors(&mut self, sp: Span, ty: Ty<'tcx>, is_static: bool) {
        // We have to check for opaque types before `normalize_erasing_regions`,
        // which will replace opaque types with their underlying concrete type.
        if self.check_for_opaque_ty(sp, ty) {
            // We've already emitted an error due to an opaque type.
            return;
        }

        // it is only OK to use this function because extern fns cannot have
        // any generic types right now:
        let ty = self.cx.tcx.normalize_erasing_regions(ParamEnv::reveal_all(), ty);
        // C doesn't really support passing arrays by value.
        // The only way to pass an array by value is through a struct.
        // So we first test that the top level isn't an array,
        // and then recursively check the types inside.
        if !is_static && self.check_for_array_ty(sp, ty) {
            return;
        }

        match self.check_type_for_ffi(&mut FxHashSet::default(), ty) {
            FfiResult::FfiSafe => {}
            FfiResult::FfiPhantom(ty) => {
                self.emit_ffi_unsafe_type_lint(ty, sp, "composed only of `PhantomData`", None);
            }
            FfiResult::FfiUnsafe { ty, reason, help } => {
                self.emit_ffi_unsafe_type_lint(ty, sp, reason, help);
            }
        }
    }

    fn check_foreign_fn(&mut self, id: hir::HirId, decl: &hir::FnDecl<'_>) {
        let def_id = self.cx.tcx.hir().local_def_id(id);
        let sig = self.cx.tcx.fn_sig(def_id);
        let sig = self.cx.tcx.erase_late_bound_regions(&sig);

        for (input_ty, input_hir) in sig.inputs().iter().zip(decl.inputs) {
            self.check_type_for_ffi_and_report_errors(input_hir.span, input_ty, false);
        }

        if let hir::FunctionRetTy::Return(ref ret_hir) = decl.output {
            let ret_ty = sig.output();
            if !ret_ty.is_unit() {
                self.check_type_for_ffi_and_report_errors(ret_hir.span, ret_ty, false);
            }
        }
    }

    fn check_foreign_static(&mut self, id: hir::HirId, span: Span) {
        let def_id = self.cx.tcx.hir().local_def_id(id);
        let ty = self.cx.tcx.type_of(def_id);
        self.check_type_for_ffi_and_report_errors(span, ty, true);
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ImproperCTypes {
    fn check_foreign_item(&mut self, cx: &LateContext<'_, '_>, it: &hir::ForeignItem<'_>) {
        let mut vis = ImproperCTypesVisitor { cx };
        let abi = cx.tcx.hir().get_foreign_abi(it.hir_id);
        if let Abi::Rust | Abi::RustCall | Abi::RustIntrinsic | Abi::PlatformIntrinsic = abi {
            // Don't worry about types in internal ABIs.
        } else {
            match it.kind {
                hir::ForeignItemKind::Fn(ref decl, _, _) => {
                    vis.check_foreign_fn(it.hir_id, decl);
                }
                hir::ForeignItemKind::Static(ref ty, _) => {
                    vis.check_foreign_static(it.hir_id, ty.span);
                }
                hir::ForeignItemKind::Type => (),
            }
        }
    }
}

declare_lint_pass!(VariantSizeDifferences => [VARIANT_SIZE_DIFFERENCES]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for VariantSizeDifferences {
    fn check_item(&mut self, cx: &LateContext<'_, '_>, it: &hir::Item<'_>) {
        if let hir::ItemKind::Enum(ref enum_definition, _) = it.kind {
            let item_def_id = cx.tcx.hir().local_def_id(it.hir_id);
            let t = cx.tcx.type_of(item_def_id);
            let ty = cx.tcx.erase_regions(&t);
            let layout = match cx.layout_of(ty) {
                Ok(layout) => layout,
                Err(ty::layout::LayoutError::Unknown(_)) => return,
                Err(err @ ty::layout::LayoutError::SizeOverflow(_)) => {
                    bug!("failed to get layout for `{}`: {}", t, err);
                }
            };
            let (variants, tag) = match layout.variants {
                layout::Variants::Multiple {
                    discr_kind: layout::DiscriminantKind::Tag,
                    ref discr,
                    ref variants,
                    ..
                } => (variants, discr),
                _ => return,
            };

            let discr_size = tag.value.size(&cx.tcx).bytes();

            debug!(
                "enum `{}` is {} bytes large with layout:\n{:#?}",
                t,
                layout.size.bytes(),
                layout
            );

            let (largest, slargest, largest_index) = enum_definition
                .variants
                .iter()
                .zip(variants)
                .map(|(variant, variant_layout)| {
                    // Subtract the size of the enum discriminant.
                    let bytes = variant_layout.size.bytes().saturating_sub(discr_size);

                    debug!("- variant `{}` is {} bytes large", variant.ident, bytes);
                    bytes
                })
                .enumerate()
                .fold((0, 0, 0), |(l, s, li), (idx, size)| {
                    if size > l {
                        (size, l, idx)
                    } else if size > s {
                        (l, size, li)
                    } else {
                        (l, s, li)
                    }
                });

            // We only warn if the largest variant is at least thrice as large as
            // the second-largest.
            if largest > slargest * 3 && slargest > 0 {
                cx.struct_span_lint(
                    VARIANT_SIZE_DIFFERENCES,
                    enum_definition.variants[largest_index].span,
                    |lint| {
                        lint.build(&format!(
                            "enum variant is more than three times \
                                          larger ({} bytes) than the next largest",
                            largest
                        ))
                        .emit()
                    },
                );
            }
        }
    }
}
