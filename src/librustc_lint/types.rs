#![allow(non_snake_case)]

use rustc::hir::Node;
use rustc::ty::subst::Substs;
use rustc::ty::{self, AdtKind, ParamEnv, Ty, TyCtxt};
use rustc::ty::layout::{self, IntegerExt, LayoutOf, VariantIdx};
use rustc::{lint, util};
use rustc_data_structures::indexed_vec::Idx;
use util::nodemap::FxHashSet;
use lint::{LateContext, LintContext, LintArray};
use lint::{LintPass, LateLintPass};

use std::cmp;
use std::{i8, i16, i32, i64, u8, u16, u32, u64, f32, f64};

use syntax::{ast, attr};
use syntax::errors::Applicability;
use rustc_target::spec::abi::Abi;
use syntax::edition::Edition;
use syntax_pos::Span;
use syntax::source_map;

use rustc::hir;

use rustc::mir::interpret::{sign_extend, truncate};

use log::debug;

declare_lint! {
    UNUSED_COMPARISONS,
    Warn,
    "comparisons made useless by limits of the types involved"
}

declare_lint! {
    OVERFLOWING_LITERALS,
    Warn,
    "literal out of range for its type",
    Edition::Edition2018 => Deny
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
    fn name(&self) -> &'static str {
        "TypeLimits"
    }

    fn get_lints(&self) -> LintArray {
        lint_array!(UNUSED_COMPARISONS,
                    OVERFLOWING_LITERALS)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for TypeLimits {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx hir::Expr) {
        match e.node {
            hir::ExprKind::Unary(hir::UnNeg, ref expr) => {
                // propagate negation, if the negation itself isn't negated
                if self.negated_expr_id != e.id {
                    self.negated_expr_id = expr.id;
                }
            }
            hir::ExprKind::Binary(binop, ref l, ref r) => {
                if is_comparison(binop) && !check_limits(cx, binop, &l, &r) {
                    cx.span_lint(UNUSED_COMPARISONS,
                                 e.span,
                                 "comparison is useless due to type limits");
                }
            }
            hir::ExprKind::Lit(ref lit) => {
                match cx.tables.node_type(e.hir_id).sty {
                    ty::Int(t) => {
                        match lit.node {
                            ast::LitKind::Int(v, ast::LitIntType::Signed(_)) |
                            ast::LitKind::Int(v, ast::LitIntType::Unsuffixed) => {
                                let int_type = if let ast::IntTy::Isize = t {
                                    cx.sess().target.isize_ty
                                } else {
                                    t
                                };
                                let (_, max) = int_ty_range(int_type);
                                let max = max as u128;
                                let negative = self.negated_expr_id == e.id;

                                // Detect literal value out of range [min, max] inclusive
                                // avoiding use of -min to prevent overflow/panic
                                if (negative && v > max + 1) || (!negative && v > max) {
                                    if let Some(repr_str) = get_bin_hex_repr(cx, lit) {
                                        report_bin_hex_error(
                                            cx,
                                            e,
                                            ty::Int(t),
                                            repr_str,
                                            v,
                                            negative,
                                        );
                                        return;
                                    }
                                    cx.span_lint(
                                        OVERFLOWING_LITERALS,
                                        e.span,
                                        &format!("literal out of range for {:?}", t),
                                    );
                                    return;
                                }
                            }
                            _ => bug!(),
                        };
                    }
                    ty::Uint(t) => {
                        let uint_type = if let ast::UintTy::Usize = t {
                            cx.sess().target.usize_ty
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
                            let parent_id = cx.tcx.hir().get_parent_node(e.id);
                            if let Node::Expr(parent_expr) = cx.tcx.hir().get(parent_id) {
                                if let hir::ExprKind::Cast(..) = parent_expr.node {
                                    if let ty::Char = cx.tables.expr_ty(parent_expr).sty {
                                        let mut err = cx.struct_span_lint(
                                                             OVERFLOWING_LITERALS,
                                                             parent_expr.span,
                                                             "only u8 can be cast into char");
                                        err.span_suggestion(
                                            parent_expr.span,
                                            &"use a char literal instead",
                                            format!("'\\u{{{:X}}}'", lit_val),
                                            Applicability::MachineApplicable
                                        );
                                        err.emit();
                                        return
                                    }
                                }
                            }
                            if let Some(repr_str) = get_bin_hex_repr(cx, lit) {
                                report_bin_hex_error(
                                    cx,
                                    e,
                                    ty::Uint(t),
                                    repr_str,
                                    lit_val,
                                    false,
                                );
                                return;
                            }
                            cx.span_lint(
                                OVERFLOWING_LITERALS,
                                e.span,
                                &format!("literal out of range for {:?}", t),
                            );
                        }
                    }
                    ty::Float(t) => {
                        let is_infinite = match lit.node {
                            ast::LitKind::Float(v, _) |
                            ast::LitKind::FloatUnsuffixed(v) => {
                                match t {
                                    ast::FloatTy::F32 => v.as_str().parse().map(f32::is_infinite),
                                    ast::FloatTy::F64 => v.as_str().parse().map(f64::is_infinite),
                                }
                            }
                            _ => bug!(),
                        };
                        if is_infinite == Ok(true) {
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
                hir::BinOpKind::Lt => v > min && v <= max,
                hir::BinOpKind::Le => v >= min && v < max,
                hir::BinOpKind::Gt => v >= min && v < max,
                hir::BinOpKind::Ge => v > min && v <= max,
                hir::BinOpKind::Eq | hir::BinOpKind::Ne => v >= min && v <= max,
                _ => bug!(),
            }
        }

        fn rev_binop(binop: hir::BinOp) -> hir::BinOp {
            source_map::respan(binop.span,
                            match binop.node {
                                hir::BinOpKind::Lt => hir::BinOpKind::Gt,
                                hir::BinOpKind::Le => hir::BinOpKind::Ge,
                                hir::BinOpKind::Gt => hir::BinOpKind::Lt,
                                hir::BinOpKind::Ge => hir::BinOpKind::Le,
                                _ => return binop,
                            })
        }

        // for isize & usize, be conservative with the warnings, so that the
        // warnings are consistent between 32- and 64-bit platforms
        fn int_ty_range(int_ty: ast::IntTy) -> (i128, i128) {
            match int_ty {
                ast::IntTy::Isize => (i64::min_value() as i128, i64::max_value() as i128),
                ast::IntTy::I8 => (i8::min_value() as i64 as i128, i8::max_value() as i128),
                ast::IntTy::I16 => (i16::min_value() as i64 as i128, i16::max_value() as i128),
                ast::IntTy::I32 => (i32::min_value() as i64 as i128, i32::max_value() as i128),
                ast::IntTy::I64 => (i64::min_value() as i128, i64::max_value() as i128),
                ast::IntTy::I128 =>(i128::min_value() as i128, i128::max_value()),
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

        fn check_limits(cx: &LateContext<'_, '_>,
                        binop: hir::BinOp,
                        l: &hir::Expr,
                        r: &hir::Expr)
                        -> bool {
            let (lit, expr, swap) = match (&l.node, &r.node) {
                (&hir::ExprKind::Lit(_), _) => (l, r, true),
                (_, &hir::ExprKind::Lit(_)) => (r, l, false),
                _ => return true,
            };
            // Normalize the binop so that the literal is always on the RHS in
            // the comparison
            let norm_binop = if swap { rev_binop(binop) } else { binop };
            match cx.tables.node_type(expr.hir_id).sty {
                ty::Int(int_ty) => {
                    let (min, max) = int_ty_range(int_ty);
                    let lit_val: i128 = match lit.node {
                        hir::ExprKind::Lit(ref li) => {
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
                ty::Uint(uint_ty) => {
                    let (min, max) :(u128, u128) = uint_ty_range(uint_ty);
                    let lit_val: u128 = match lit.node {
                        hir::ExprKind::Lit(ref li) => {
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
                hir::BinOpKind::Eq |
                hir::BinOpKind::Lt |
                hir::BinOpKind::Le |
                hir::BinOpKind::Ne |
                hir::BinOpKind::Ge |
                hir::BinOpKind::Gt => true,
                _ => false,
            }
        }

        fn get_bin_hex_repr(cx: &LateContext<'_, '_>, lit: &ast::Lit) -> Option<String> {
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

        // This function finds the next fitting type and generates a suggestion string.
        // It searches for fitting types in the following way (`X < Y`):
        //  - `iX`: if literal fits in `uX` => `uX`, else => `iY`
        //  - `-iX` => `iY`
        //  - `uX` => `uY`
        //
        // No suggestion for: `isize`, `usize`.
        fn get_type_suggestion<'a>(
            t: &ty::TyKind<'_>,
            val: u128,
            negative: bool,
        ) -> Option<String> {
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
                                    return Some(format!("{:?}", $utypes))
                                })*
                                $(if val <= int_ty_range($itypes).1 as u128 + _neg {
                                    return Some(format!("{:?}", $itypes))
                                })*
                                None
                            },)*
                            _ => None
                        }
                    }
                }
            }
            match t {
                &ty::Int(i) => find_fit!(i, val, negative,
                              I8 => [U8] => [I16, I32, I64, I128],
                              I16 => [U16] => [I32, I64, I128],
                              I32 => [U32] => [I64, I128],
                              I64 => [U64] => [I128],
                              I128 => [U128] => []),
                &ty::Uint(u) => find_fit!(u, val, negative,
                              U8 => [U8, U16, U32, U64, U128] => [],
                              U16 => [U16, U32, U64, U128] => [],
                              U32 => [U32, U64, U128] => [],
                              U64 => [U64, U128] => [],
                              U128 => [U128] => []),
                _ => None,
            }
        }

        fn report_bin_hex_error(
            cx: &LateContext<'_, '_>,
            expr: &hir::Expr,
            ty: ty::TyKind<'_>,
            repr_str: String,
            val: u128,
            negative: bool,
        ) {
            let (t, actually) = match ty {
                ty::Int(t) => {
                    let ity = attr::IntType::SignedInt(t);
                    let size = layout::Integer::from_attr(&cx.tcx, ity).size();
                    let actually = sign_extend(val, size) as i128;
                    (format!("{:?}", t), actually.to_string())
                }
                ty::Uint(t) => {
                    let ity = attr::IntType::UnsignedInt(t);
                    let size = layout::Integer::from_attr(&cx.tcx, ity).size();
                    let actually = truncate(val, size);
                    (format!("{:?}", t), actually.to_string())
                }
                _ => bug!(),
            };
            let mut err = cx.struct_span_lint(
                OVERFLOWING_LITERALS,
                expr.span,
                &format!("literal out of range for {}", t),
            );
            err.note(&format!(
                "the literal `{}` (decimal `{}`) does not fit into \
                 an `{}` and will become `{}{}`",
                repr_str, val, t, actually, t
            ));
            if let Some(sugg_ty) =
                get_type_suggestion(&cx.tables.node_type(expr.hir_id).sty, val, negative)
            {
                if let Some(pos) = repr_str.chars().position(|c| c == 'i' || c == 'u') {
                    let (sans_suffix, _) = repr_str.split_at(pos);
                    err.span_suggestion(
                        expr.span,
                        &format!("consider using `{}` instead", sugg_ty),
                        format!("{}{}", sans_suffix, sugg_ty),
                        Applicability::MachineApplicable
                    );
                } else {
                    err.help(&format!("consider using `{}` instead", sugg_ty));
                }
            }

            err.emit();
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

enum FfiResult<'tcx> {
    FfiSafe,
    FfiPhantom(Ty<'tcx>),
    FfiUnsafe {
        ty: Ty<'tcx>,
        reason: &'static str,
        help: Option<&'static str>,
    },
}

/// Check if this enum can be safely exported based on the
/// "nullable pointer optimization". Currently restricted
/// to function pointers and references, but could be
/// expanded to cover NonZero raw pointers and newtypes.
/// FIXME: This duplicates code in codegen.
fn is_repr_nullable_ptr<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  def: &'tcx ty::AdtDef,
                                  substs: &Substs<'tcx>)
                                  -> bool {
    if def.variants.len() == 2 {
        let data_idx;

        let zero = VariantIdx::new(0);
        let one = VariantIdx::new(1);

        if def.variants[zero].fields.is_empty() {
            data_idx = one;
        } else if def.variants[one].fields.is_empty() {
            data_idx = zero;
        } else {
            return false;
        }

        if def.variants[data_idx].fields.len() == 1 {
            match def.variants[data_idx].fields[0].ty(tcx, substs).sty {
                ty::FnPtr(_) => {
                    return true;
                }
                ty::Ref(..) => {
                    return true;
                }
                _ => {}
            }
        }
    }
    false
}

impl<'a, 'tcx> ImproperCTypesVisitor<'a, 'tcx> {
    /// Checks if the given type is "ffi-safe" (has a stable, well-defined
    /// representation which can be exported to C code).
    fn check_type_for_ffi(&self,
                          cache: &mut FxHashSet<Ty<'tcx>>,
                          ty: Ty<'tcx>) -> FfiResult<'tcx> {
        use FfiResult::*;

        let cx = self.cx.tcx;

        // Protect against infinite recursion, for example
        // `struct S(*mut S);`.
        // FIXME: A recursion limit is necessary as well, for irregular
        // recursive types.
        if !cache.insert(ty) {
            return FfiSafe;
        }

        match ty.sty {
            ty::Adt(def, substs) => {
                if def.is_phantom_data() {
                    return FfiPhantom(ty);
                }
                match def.adt_kind() {
                    AdtKind::Struct => {
                        if !def.repr.c() && !def.repr.transparent() {
                            return FfiUnsafe {
                                ty: ty,
                                reason: "this struct has unspecified layout",
                                help: Some("consider adding a #[repr(C)] or #[repr(transparent)] \
                                            attribute to this struct"),
                            };
                        }

                        if def.non_enum_variant().fields.is_empty() {
                            return FfiUnsafe {
                                ty: ty,
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
                            if def.repr.transparent() {
                                let is_zst = cx
                                    .layout_of(cx.param_env(field.did).and(field_ty))
                                    .map(|layout| layout.is_zst())
                                    .unwrap_or(false);
                                if is_zst {
                                    continue;
                                }
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
                        if !def.repr.c() {
                            return FfiUnsafe {
                                ty: ty,
                                reason: "this union has unspecified layout",
                                help: Some("consider adding a #[repr(C)] attribute to this union"),
                            };
                        }

                        if def.non_enum_variant().fields.is_empty() {
                            return FfiUnsafe {
                                ty: ty,
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
                        if !def.repr.c() && def.repr.int.is_none() {
                            // Special-case types like `Option<extern fn()>`.
                            if !is_repr_nullable_ptr(cx, def, substs) {
                                return FfiUnsafe {
                                    ty: ty,
                                    reason: "enum has no representation hint",
                                    help: Some("consider adding a #[repr(...)] attribute \
                                                to this enum"),
                                };
                            }
                        }

                        // Check the contained variants.
                        for variant in &def.variants {
                            for field in &variant.fields {
                                let arg = cx.normalize_erasing_regions(
                                    ParamEnv::reveal_all(),
                                    field.ty(cx, substs),
                                );
                                let r = self.check_type_for_ffi(cache, arg);
                                match r {
                                    FfiSafe => {}
                                    FfiUnsafe { .. } => {
                                        return r;
                                    }
                                    FfiPhantom(..) => {
                                        return FfiUnsafe {
                                            ty: ty,
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
                ty: ty,
                reason: "the `char` type has no C equivalent",
                help: Some("consider using `u32` or `libc::wchar_t` instead"),
            },

            ty::Int(ast::IntTy::I128) | ty::Uint(ast::UintTy::U128) => FfiUnsafe {
                ty: ty,
                reason: "128-bit integers don't currently have a known stable ABI",
                help: None,
            },

            // Primitive types with a stable representation.
            ty::Bool | ty::Int(..) | ty::Uint(..) | ty::Float(..) | ty::Never => FfiSafe,

            ty::Slice(_) => FfiUnsafe {
                ty: ty,
                reason: "slices have no C equivalent",
                help: Some("consider using a raw pointer instead"),
            },

            ty::Dynamic(..) => FfiUnsafe {
                ty: ty,
                reason: "trait objects have no C equivalent",
                help: None,
            },

            ty::Str => FfiUnsafe {
                ty: ty,
                reason: "string slices have no C equivalent",
                help: Some("consider using `*const u8` and a length instead"),
            },

            ty::Tuple(..) => FfiUnsafe {
                ty: ty,
                reason: "tuples have unspecified layout",
                help: Some("consider using a struct instead"),
            },

            ty::RawPtr(ty::TypeAndMut { ty, .. }) |
            ty::Ref(_, ty, _) => self.check_type_for_ffi(cache, ty),

            ty::Array(ty, _) => self.check_type_for_ffi(cache, ty),

            ty::FnPtr(sig) => {
                match sig.abi() {
                    Abi::Rust | Abi::RustIntrinsic | Abi::PlatformIntrinsic | Abi::RustCall => {
                        return FfiUnsafe {
                            ty: ty,
                            reason: "this function pointer has Rust-specific calling convention",
                            help: Some("consider using an `extern fn(...) -> ...` \
                                        function pointer instead"),
                        }
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

            ty::Param(..) |
            ty::Infer(..) |
            ty::Bound(..) |
            ty::Error |
            ty::Closure(..) |
            ty::Generator(..) |
            ty::GeneratorWitness(..) |
            ty::Placeholder(..) |
            ty::UnnormalizedProjection(..) |
            ty::Projection(..) |
            ty::Opaque(..) |
            ty::FnDef(..) => bug!("Unexpected type in foreign function"),
        }
    }

    fn check_type_for_ffi_and_report_errors(&mut self, sp: Span, ty: Ty<'tcx>) {
        // it is only OK to use this function because extern fns cannot have
        // any generic types right now:
        let ty = self.cx.tcx.normalize_erasing_regions(ParamEnv::reveal_all(), ty);

        match self.check_type_for_ffi(&mut FxHashSet::default(), ty) {
            FfiResult::FfiSafe => {}
            FfiResult::FfiPhantom(ty) => {
                self.cx.span_lint(IMPROPER_CTYPES,
                                  sp,
                                  &format!("`extern` block uses type `{}` which is not FFI-safe: \
                                            composed only of PhantomData", ty));
            }
            FfiResult::FfiUnsafe { ty: unsafe_ty, reason, help } => {
                let msg = format!("`extern` block uses type `{}` which is not FFI-safe: {}",
                                  unsafe_ty, reason);
                let mut diag = self.cx.struct_span_lint(IMPROPER_CTYPES, sp, &msg);
                if let Some(s) = help {
                    diag.help(s);
                }
                if let ty::Adt(def, _) = unsafe_ty.sty {
                    if let Some(sp) = self.cx.tcx.hir().span_if_local(def.did) {
                        diag.span_note(sp, "type defined here");
                    }
                }
                diag.emit();
            }
        }
    }

    fn check_foreign_fn(&mut self, id: ast::NodeId, decl: &hir::FnDecl) {
        let def_id = self.cx.tcx.hir().local_def_id(id);
        let sig = self.cx.tcx.fn_sig(def_id);
        let sig = self.cx.tcx.erase_late_bound_regions(&sig);

        for (input_ty, input_hir) in sig.inputs().iter().zip(&decl.inputs) {
            self.check_type_for_ffi_and_report_errors(input_hir.span, input_ty);
        }

        if let hir::Return(ref ret_hir) = decl.output {
            let ret_ty = sig.output();
            if !ret_ty.is_unit() {
                self.check_type_for_ffi_and_report_errors(ret_hir.span, ret_ty);
            }
        }
    }

    fn check_foreign_static(&mut self, id: ast::NodeId, span: Span) {
        let def_id = self.cx.tcx.hir().local_def_id(id);
        let ty = self.cx.tcx.type_of(def_id);
        self.check_type_for_ffi_and_report_errors(span, ty);
    }
}

#[derive(Copy, Clone)]
pub struct ImproperCTypes;

impl LintPass for ImproperCTypes {
    fn name(&self) -> &'static str {
        "ImproperCTypes"
    }

    fn get_lints(&self) -> LintArray {
        lint_array!(IMPROPER_CTYPES)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ImproperCTypes {
    fn check_foreign_item(&mut self, cx: &LateContext<'_, '_>, it: &hir::ForeignItem) {
        let mut vis = ImproperCTypesVisitor { cx };
        let abi = cx.tcx.hir().get_foreign_abi(it.id);
        if abi != Abi::RustIntrinsic && abi != Abi::PlatformIntrinsic {
            match it.node {
                hir::ForeignItemKind::Fn(ref decl, _, _) => {
                    vis.check_foreign_fn(it.id, decl);
                }
                hir::ForeignItemKind::Static(ref ty, _) => {
                    vis.check_foreign_static(it.id, ty.span);
                }
                hir::ForeignItemKind::Type => ()
            }
        }
    }
}

pub struct VariantSizeDifferences;

impl LintPass for VariantSizeDifferences {
    fn name(&self) -> &'static str {
        "VariantSizeDifferences"
    }

    fn get_lints(&self) -> LintArray {
        lint_array!(VARIANT_SIZE_DIFFERENCES)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for VariantSizeDifferences {
    fn check_item(&mut self, cx: &LateContext<'_, '_>, it: &hir::Item) {
        if let hir::ItemKind::Enum(ref enum_definition, _) = it.node {
            let item_def_id = cx.tcx.hir().local_def_id(it.id);
            let t = cx.tcx.type_of(item_def_id);
            let ty = cx.tcx.erase_regions(&t);
            match cx.layout_of(ty) {
                Ok(layout) => {
                    let variants = &layout.variants;
                    if let layout::Variants::Tagged { ref variants, ref tag, .. } = variants {
                        let discr_size = tag.value.size(&cx.tcx).bytes();

                        debug!("enum `{}` is {} bytes large with layout:\n{:#?}",
                               t, layout.size.bytes(), layout);

                        let (largest, slargest, largest_index) = enum_definition.variants
                            .iter()
                            .zip(variants)
                            .map(|(variant, variant_layout)| {
                                // Subtract the size of the enum discriminant.
                                let bytes = variant_layout.size.bytes().saturating_sub(discr_size);

                                debug!("- variant `{}` is {} bytes large",
                                       variant.node.ident,
                                       bytes);
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

                        // We only warn if the largest variant is at least thrice as large as
                        // the second-largest.
                        if largest > slargest * 3 && slargest > 0 {
                            cx.span_lint(VARIANT_SIZE_DIFFERENCES,
                                            enum_definition.variants[largest_index].span,
                                            &format!("enum variant is more than three times \
                                                      larger ({} bytes) than the next largest",
                                                     largest));
                        }
                    }
                }
                Err(ty::layout::LayoutError::Unknown(_)) => return,
                Err(err @ ty::layout::LayoutError::SizeOverflow(_)) => {
                    bug!("failed to get layout for `{}`: {}", t, err);
                }
            }
        }
    }
}
