use crate::{LateContext, LateLintPass, LintContext};
use rustc_ast as ast;
use rustc_attr as attr;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::{is_range_literal, ExprKind, Node};
use rustc_index::vec::Idx;
use rustc_middle::ty::layout::{IntegerExt, SizeSkeleton};
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, AdtKind, Ty, TyCtxt, TypeFoldable};
use rustc_span::source_map;
use rustc_span::symbol::sym;
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::Abi;
use rustc_target::abi::{Integer, LayoutOf, TagEncoding, VariantIdx, Variants};
use rustc_target::spec::abi::Abi as SpecAbi;

use std::cmp;
use std::ops::ControlFlow;
use tracing::debug;

declare_lint! {
    /// The `unused_comparisons` lint detects comparisons made useless by
    /// limits of the types involved.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn foo(x: u8) {
    ///     x >= 0;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// A useless comparison may indicate a mistake, and should be fixed or
    /// removed.
    UNUSED_COMPARISONS,
    Warn,
    "comparisons made useless by limits of the types involved"
}

declare_lint! {
    /// The `overflowing_literals` lint detects literal out of range for its
    /// type.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// let x: u8 = 1000;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// It is usually a mistake to use a literal that overflows the type where
    /// it is used. Either use a literal that is within range, or change the
    /// type to be within the range of the literal.
    OVERFLOWING_LITERALS,
    Deny,
    "literal out of range for its type"
}

declare_lint! {
    /// The `variant_size_differences` lint detects enums with widely varying
    /// variant sizes.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(variant_size_differences)]
    /// enum En {
    ///     V0(u8),
    ///     VBig([u8; 1024]),
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// It can be a mistake to add a variant to an enum that is much larger
    /// than the other variants, bloating the overall size required for all
    /// variants. This can impact performance and memory usage. This is
    /// triggered if one variant is more than 3 times larger than the
    /// second-largest variant.
    ///
    /// Consider placing the large variant's contents on the heap (for example
    /// via [`Box`]) to keep the overall size of the enum itself down.
    ///
    /// This lint is "allow" by default because it can be noisy, and may not be
    /// an actual problem. Decisions about this should be guided with
    /// profiling and benchmarking.
    ///
    /// [`Box`]: https://doc.rust-lang.org/std/boxed/index.html
    VARIANT_SIZE_DIFFERENCES,
    Allow,
    "detects enums with widely varying variant sizes"
}

#[derive(Copy, Clone)]
pub struct TypeLimits {
    /// Id of the last visited negated expression
    negated_expr_id: Option<hir::HirId>,
}

impl_lint_pass!(TypeLimits => [UNUSED_COMPARISONS, OVERFLOWING_LITERALS]);

impl TypeLimits {
    pub fn new() -> TypeLimits {
        TypeLimits { negated_expr_id: None }
    }
}

/// Attempts to special-case the overflowing literal lint when it occurs as a range endpoint.
/// Returns `true` iff the lint was overridden.
fn lint_overflowing_range_endpoint<'tcx>(
    cx: &LateContext<'tcx>,
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
                        LitKind::Int(_, LitIntType::Signed(s)) => s.name_str(),
                        LitKind::Int(_, LitIntType::Unsigned(s)) => s.name_str(),
                        LitKind::Int(_, LitIntType::Unsuffixed) => "",
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
        ast::IntTy::Isize => (i64::MIN.into(), i64::MAX.into()),
        ast::IntTy::I8 => (i8::MIN.into(), i8::MAX.into()),
        ast::IntTy::I16 => (i16::MIN.into(), i16::MAX.into()),
        ast::IntTy::I32 => (i32::MIN.into(), i32::MAX.into()),
        ast::IntTy::I64 => (i64::MIN.into(), i64::MAX.into()),
        ast::IntTy::I128 => (i128::MIN, i128::MAX),
    }
}

fn uint_ty_range(uint_ty: ast::UintTy) -> (u128, u128) {
    let max = match uint_ty {
        ast::UintTy::Usize => u64::MAX.into(),
        ast::UintTy::U8 => u8::MAX.into(),
        ast::UintTy::U16 => u16::MAX.into(),
        ast::UintTy::U32 => u32::MAX.into(),
        ast::UintTy::U64 => u64::MAX.into(),
        ast::UintTy::U128 => u128::MAX,
    };
    (0, max)
}

fn get_bin_hex_repr(cx: &LateContext<'_>, lit: &hir::Lit) -> Option<String> {
    let src = cx.sess().source_map().span_to_snippet(lit.span).ok()?;
    let firstch = src.chars().next()?;

    if firstch == '0' {
        match src.chars().nth(1) {
            Some('x' | 'b') => return Some(src),
            _ => return None,
        }
    }

    None
}

fn report_bin_hex_error(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    ty: attr::IntType,
    repr_str: String,
    val: u128,
    negative: bool,
) {
    let size = Integer::from_attr(&cx.tcx, ty).size();
    cx.struct_span_lint(OVERFLOWING_LITERALS, expr.span, |lint| {
        let (t, actually) = match ty {
            attr::IntType::SignedInt(t) => {
                let actually = size.sign_extend(val) as i128;
                (t.name_str(), actually.to_string())
            }
            attr::IntType::UnsignedInt(t) => {
                let actually = size.truncate(val);
                (t.name_str(), actually.to_string())
            }
        };
        let mut err = lint.build(&format!("literal out of range for {}", t));
        err.note(&format!(
            "the literal `{}` (decimal `{}`) does not fit into \
             the type `{}` and will become `{}{}`",
            repr_str, val, t, actually, t
        ));
        if let Some(sugg_ty) =
            get_type_suggestion(&cx.typeck_results().node_type(expr.hir_id), val, negative)
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
    use rustc_ast::IntTy::*;
    use rustc_ast::UintTy::*;
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
    match t.kind() {
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

fn lint_int_literal<'tcx>(
    cx: &LateContext<'tcx>,
    type_limits: &TypeLimits,
    e: &'tcx hir::Expr<'tcx>,
    lit: &hir::Lit,
    t: ast::IntTy,
    v: u128,
) {
    let int_type = t.normalize(cx.sess().target.pointer_width);
    let (min, max) = int_ty_range(int_type);
    let max = max as u128;
    let negative = type_limits.negated_expr_id == Some(e.hir_id);

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
                if is_range_literal(par_e)
                    && lint_overflowing_range_endpoint(cx, lit, v, max, e, par_e, t.name_str())
                {
                    // The overflowing literal lint was overridden.
                    return;
                }
            }
        }

        cx.struct_span_lint(OVERFLOWING_LITERALS, e.span, |lint| {
            lint.build(&format!("literal out of range for `{}`", t.name_str()))
                .note(&format!(
                    "the literal `{}` does not fit into the type `{}` whose range is `{}..={}`",
                    cx.sess()
                        .source_map()
                        .span_to_snippet(lit.span)
                        .expect("must get snippet from literal"),
                    t.name_str(),
                    min,
                    max,
                ))
                .emit();
        });
    }
}

fn lint_uint_literal<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx hir::Expr<'tcx>,
    lit: &hir::Lit,
    t: ast::UintTy,
) {
    let uint_type = t.normalize(cx.sess().target.pointer_width);
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
                    if let ty::Char = cx.typeck_results().expr_ty(par_e).kind() {
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
                hir::ExprKind::Struct(..) if is_range_literal(par_e) => {
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
            lint.build(&format!("literal out of range for `{}`", t.name_str()))
                .note(&format!(
                    "the literal `{}` does not fit into the type `{}` whose range is `{}..={}`",
                    cx.sess()
                        .source_map()
                        .span_to_snippet(lit.span)
                        .expect("must get snippet from literal"),
                    t.name_str(),
                    min,
                    max,
                ))
                .emit()
        });
    }
}

fn lint_literal<'tcx>(
    cx: &LateContext<'tcx>,
    type_limits: &TypeLimits,
    e: &'tcx hir::Expr<'tcx>,
    lit: &hir::Lit,
) {
    match *cx.typeck_results().node_type(e.hir_id).kind() {
        ty::Int(t) => {
            match lit.node {
                ast::LitKind::Int(v, ast::LitIntType::Signed(_) | ast::LitIntType::Unsuffixed) => {
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
                    lint.build(&format!("literal out of range for `{}`", t.name_str()))
                        .note(&format!(
                            "the literal `{}` does not fit into the type `{}` and will be converted to `{}::INFINITY`",
                            cx.sess()
                                .source_map()
                                .span_to_snippet(lit.span)
                                .expect("must get snippet from literal"),
                            t.name_str(),
                            t.name_str(),
                        ))
                        .emit();
                });
            }
        }
        _ => {}
    }
}

impl<'tcx> LateLintPass<'tcx> for TypeLimits {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx hir::Expr<'tcx>) {
        match e.kind {
            hir::ExprKind::Unary(hir::UnOp::UnNeg, ref expr) => {
                // propagate negation, if the negation itself isn't negated
                if self.negated_expr_id != Some(e.hir_id) {
                    self.negated_expr_id = Some(expr.hir_id);
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
            cx: &LateContext<'_>,
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
            match *cx.typeck_results().node_type(expr.hir_id).kind() {
                ty::Int(int_ty) => {
                    let (min, max) = int_ty_range(int_ty);
                    let lit_val: i128 = match lit.kind {
                        hir::ExprKind::Lit(ref li) => match li.node {
                            ast::LitKind::Int(
                                v,
                                ast::LitIntType::Signed(_) | ast::LitIntType::Unsuffixed,
                            ) => v as i128,
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
            matches!(
                binop.node,
                hir::BinOpKind::Eq
                    | hir::BinOpKind::Lt
                    | hir::BinOpKind::Le
                    | hir::BinOpKind::Ne
                    | hir::BinOpKind::Ge
                    | hir::BinOpKind::Gt
            )
        }
    }
}

declare_lint! {
    /// The `improper_ctypes` lint detects incorrect use of types in foreign
    /// modules.
    ///
    /// ### Example
    ///
    /// ```rust
    /// extern "C" {
    ///     static STATIC: String;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The compiler has several checks to verify that types used in `extern`
    /// blocks are safe and follow certain rules to ensure proper
    /// compatibility with the foreign interfaces. This lint is issued when it
    /// detects a probable mistake in a definition. The lint usually should
    /// provide a description of the issue, along with possibly a hint on how
    /// to resolve it.
    IMPROPER_CTYPES,
    Warn,
    "proper use of libc types in foreign modules"
}

declare_lint_pass!(ImproperCTypesDeclarations => [IMPROPER_CTYPES]);

declare_lint! {
    /// The `improper_ctypes_definitions` lint detects incorrect use of
    /// [`extern` function] definitions.
    ///
    /// [`extern` function]: https://doc.rust-lang.org/reference/items/functions.html#extern-function-qualifier
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// pub extern "C" fn str_type(p: &str) { }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// There are many parameter and return types that may be specified in an
    /// `extern` function that are not compatible with the given ABI. This
    /// lint is an alert that these types should not be used. The lint usually
    /// should provide a description of the issue, along with possibly a hint
    /// on how to resolve it.
    IMPROPER_CTYPES_DEFINITIONS,
    Warn,
    "proper use of libc types in foreign item definitions"
}

declare_lint_pass!(ImproperCTypesDefinitions => [IMPROPER_CTYPES_DEFINITIONS]);

#[derive(Clone, Copy)]
crate enum CItemKind {
    Declaration,
    Definition,
}

struct ImproperCTypesVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    mode: CItemKind,
}

enum FfiResult<'tcx> {
    FfiSafe,
    FfiPhantom(Ty<'tcx>),
    FfiUnsafe { ty: Ty<'tcx>, reason: String, help: Option<String> },
}

crate fn nonnull_optimization_guaranteed<'tcx>(tcx: TyCtxt<'tcx>, def: &ty::AdtDef) -> bool {
    tcx.get_attrs(def.did)
        .iter()
        .any(|a| tcx.sess.check_name(a, sym::rustc_nonnull_optimization_guaranteed))
}

/// `repr(transparent)` structs can have a single non-ZST field, this function returns that
/// field.
pub fn transparent_newtype_field<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    variant: &'a ty::VariantDef,
) -> Option<&'a ty::FieldDef> {
    let param_env = tcx.param_env(variant.def_id);
    for field in &variant.fields {
        let field_ty = tcx.type_of(field.did);
        let is_zst =
            tcx.layout_of(param_env.and(field_ty)).map(|layout| layout.is_zst()).unwrap_or(false);

        if !is_zst {
            return Some(field);
        }
    }

    None
}

/// Is type known to be non-null?
crate fn ty_is_known_nonnull<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, mode: CItemKind) -> bool {
    let tcx = cx.tcx;
    match ty.kind() {
        ty::FnPtr(_) => true,
        ty::Ref(..) => true,
        ty::Adt(def, _) if def.is_box() && matches!(mode, CItemKind::Definition) => true,
        ty::Adt(def, substs) if def.repr.transparent() && !def.is_union() => {
            let marked_non_null = nonnull_optimization_guaranteed(tcx, &def);

            if marked_non_null {
                return true;
            }

            for variant in &def.variants {
                if let Some(field) = transparent_newtype_field(cx.tcx, variant) {
                    if ty_is_known_nonnull(cx, field.ty(tcx, substs), mode) {
                        return true;
                    }
                }
            }

            false
        }
        _ => false,
    }
}

/// Given a non-null scalar (or transparent) type `ty`, return the nullable version of that type.
/// If the type passed in was not scalar, returns None.
fn get_nullable_type<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    let tcx = cx.tcx;
    Some(match *ty.kind() {
        ty::Adt(field_def, field_substs) => {
            let inner_field_ty = {
                let first_non_zst_ty =
                    field_def.variants.iter().filter_map(|v| transparent_newtype_field(cx.tcx, v));
                debug_assert_eq!(
                    first_non_zst_ty.clone().count(),
                    1,
                    "Wrong number of fields for transparent type"
                );
                first_non_zst_ty
                    .last()
                    .expect("No non-zst fields in transparent type.")
                    .ty(tcx, field_substs)
            };
            return get_nullable_type(cx, inner_field_ty);
        }
        ty::Int(ty) => tcx.mk_mach_int(ty),
        ty::Uint(ty) => tcx.mk_mach_uint(ty),
        ty::RawPtr(ty_mut) => tcx.mk_ptr(ty_mut),
        // As these types are always non-null, the nullable equivalent of
        // Option<T> of these types are their raw pointer counterparts.
        ty::Ref(_region, ty, mutbl) => tcx.mk_ptr(ty::TypeAndMut { ty, mutbl }),
        ty::FnPtr(..) => {
            // There is no nullable equivalent for Rust's function pointers -- you
            // must use an Option<fn(..) -> _> to represent it.
            ty
        }

        // We should only ever reach this case if ty_is_known_nonnull is extended
        // to other types.
        ref unhandled => {
            debug!(
                "get_nullable_type: Unhandled scalar kind: {:?} while checking {:?}",
                unhandled, ty
            );
            return None;
        }
    })
}

/// Check if this enum can be safely exported based on the "nullable pointer optimization". If it
/// can, return the type that `ty` can be safely converted to, otherwise return `None`.
/// Currently restricted to function pointers, boxes, references, `core::num::NonZero*`,
/// `core::ptr::NonNull`, and `#[repr(transparent)]` newtypes.
/// FIXME: This duplicates code in codegen.
crate fn repr_nullable_ptr<'tcx>(
    cx: &LateContext<'tcx>,
    ty: Ty<'tcx>,
    ckind: CItemKind,
) -> Option<Ty<'tcx>> {
    debug!("is_repr_nullable_ptr(cx, ty = {:?})", ty);
    if let ty::Adt(ty_def, substs) = ty.kind() {
        if ty_def.variants.len() != 2 {
            return None;
        }

        let get_variant_fields = |index| &ty_def.variants[VariantIdx::new(index)].fields;
        let variant_fields = [get_variant_fields(0), get_variant_fields(1)];
        let fields = if variant_fields[0].is_empty() {
            &variant_fields[1]
        } else if variant_fields[1].is_empty() {
            &variant_fields[0]
        } else {
            return None;
        };

        if fields.len() != 1 {
            return None;
        }

        let field_ty = fields[0].ty(cx.tcx, substs);
        if !ty_is_known_nonnull(cx, field_ty, ckind) {
            return None;
        }

        // At this point, the field's type is known to be nonnull and the parent enum is Option-like.
        // If the computed size for the field and the enum are different, the nonnull optimization isn't
        // being applied (and we've got a problem somewhere).
        let compute_size_skeleton = |t| SizeSkeleton::compute(t, cx.tcx, cx.param_env).unwrap();
        if !compute_size_skeleton(ty).same_size(compute_size_skeleton(field_ty)) {
            bug!("improper_ctypes: Option nonnull optimization not applied?");
        }

        // Return the nullable type this Option-like enum can be safely represented with.
        let field_ty_abi = &cx.layout_of(field_ty).unwrap().abi;
        if let Abi::Scalar(field_ty_scalar) = field_ty_abi {
            match (field_ty_scalar.valid_range.start(), field_ty_scalar.valid_range.end()) {
                (0, _) => unreachable!("Non-null optimisation extended to a non-zero value."),
                (1, _) => {
                    return Some(get_nullable_type(cx, field_ty).unwrap());
                }
                (start, end) => unreachable!("Unhandled start and end range: ({}, {})", start, end),
            };
        }
    }
    None
}

impl<'a, 'tcx> ImproperCTypesVisitor<'a, 'tcx> {
    /// Check if the type is array and emit an unsafe type lint.
    fn check_for_array_ty(&mut self, sp: Span, ty: Ty<'tcx>) -> bool {
        if let ty::Array(..) = ty.kind() {
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

    /// Checks if the given field's type is "ffi-safe".
    fn check_field_type_for_ffi(
        &self,
        cache: &mut FxHashSet<Ty<'tcx>>,
        field: &ty::FieldDef,
        substs: SubstsRef<'tcx>,
    ) -> FfiResult<'tcx> {
        let field_ty = field.ty(self.cx.tcx, substs);
        if field_ty.has_opaque_types() {
            self.check_type_for_ffi(cache, field_ty)
        } else {
            let field_ty = self.cx.tcx.normalize_erasing_regions(self.cx.param_env, field_ty);
            self.check_type_for_ffi(cache, field_ty)
        }
    }

    /// Checks if the given `VariantDef`'s field types are "ffi-safe".
    fn check_variant_for_ffi(
        &self,
        cache: &mut FxHashSet<Ty<'tcx>>,
        ty: Ty<'tcx>,
        def: &ty::AdtDef,
        variant: &ty::VariantDef,
        substs: SubstsRef<'tcx>,
    ) -> FfiResult<'tcx> {
        use FfiResult::*;

        if def.repr.transparent() {
            // Can assume that only one field is not a ZST, so only check
            // that field's type for FFI-safety.
            if let Some(field) = transparent_newtype_field(self.cx.tcx, variant) {
                self.check_field_type_for_ffi(cache, field, substs)
            } else {
                bug!("malformed transparent type");
            }
        } else {
            // We can't completely trust repr(C) markings; make sure the fields are
            // actually safe.
            let mut all_phantom = !variant.fields.is_empty();
            for field in &variant.fields {
                match self.check_field_type_for_ffi(cache, &field, substs) {
                    FfiSafe => {
                        all_phantom = false;
                    }
                    FfiPhantom(..) if def.is_enum() => {
                        return FfiUnsafe {
                            ty,
                            reason: "this enum contains a PhantomData field".into(),
                            help: None,
                        };
                    }
                    FfiPhantom(..) => {}
                    r => return r,
                }
            }

            if all_phantom { FfiPhantom(ty) } else { FfiSafe }
        }
    }

    /// Checks if the given type is "ffi-safe" (has a stable, well-defined
    /// representation which can be exported to C code).
    fn check_type_for_ffi(&self, cache: &mut FxHashSet<Ty<'tcx>>, ty: Ty<'tcx>) -> FfiResult<'tcx> {
        use FfiResult::*;

        let tcx = self.cx.tcx;

        // Protect against infinite recursion, for example
        // `struct S(*mut S);`.
        // FIXME: A recursion limit is necessary as well, for irregular
        // recursive types.
        if !cache.insert(ty) {
            return FfiSafe;
        }

        match *ty.kind() {
            ty::Adt(def, _) if def.is_box() && matches!(self.mode, CItemKind::Definition) => {
                FfiSafe
            }

            ty::Adt(def, substs) => {
                if def.is_phantom_data() {
                    return FfiPhantom(ty);
                }
                match def.adt_kind() {
                    AdtKind::Struct | AdtKind::Union => {
                        let kind = if def.is_struct() { "struct" } else { "union" };

                        if !def.repr.c() && !def.repr.transparent() {
                            return FfiUnsafe {
                                ty,
                                reason: format!("this {} has unspecified layout", kind),
                                help: Some(format!(
                                    "consider adding a `#[repr(C)]` or \
                                             `#[repr(transparent)]` attribute to this {}",
                                    kind
                                )),
                            };
                        }

                        let is_non_exhaustive =
                            def.non_enum_variant().is_field_list_non_exhaustive();
                        if is_non_exhaustive && !def.did.is_local() {
                            return FfiUnsafe {
                                ty,
                                reason: format!("this {} is non-exhaustive", kind),
                                help: None,
                            };
                        }

                        if def.non_enum_variant().fields.is_empty() {
                            return FfiUnsafe {
                                ty,
                                reason: format!("this {} has no fields", kind),
                                help: Some(format!("consider adding a member to this {}", kind)),
                            };
                        }

                        self.check_variant_for_ffi(cache, ty, def, def.non_enum_variant(), substs)
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
                            if repr_nullable_ptr(self.cx, ty, self.mode).is_none() {
                                return FfiUnsafe {
                                    ty,
                                    reason: "enum has no representation hint".into(),
                                    help: Some(
                                        "consider adding a `#[repr(C)]`, \
                                                `#[repr(transparent)]`, or integer `#[repr(...)]` \
                                                attribute to this enum"
                                            .into(),
                                    ),
                                };
                            }
                        }

                        if def.is_variant_list_non_exhaustive() && !def.did.is_local() {
                            return FfiUnsafe {
                                ty,
                                reason: "this enum is non-exhaustive".into(),
                                help: None,
                            };
                        }

                        // Check the contained variants.
                        for variant in &def.variants {
                            let is_non_exhaustive = variant.is_field_list_non_exhaustive();
                            if is_non_exhaustive && !variant.def_id.is_local() {
                                return FfiUnsafe {
                                    ty,
                                    reason: "this enum has non-exhaustive variants".into(),
                                    help: None,
                                };
                            }

                            match self.check_variant_for_ffi(cache, ty, def, variant, substs) {
                                FfiSafe => (),
                                r => return r,
                            }
                        }

                        FfiSafe
                    }
                }
            }

            ty::Char => FfiUnsafe {
                ty,
                reason: "the `char` type has no C equivalent".into(),
                help: Some("consider using `u32` or `libc::wchar_t` instead".into()),
            },

            ty::Int(ast::IntTy::I128) | ty::Uint(ast::UintTy::U128) => FfiUnsafe {
                ty,
                reason: "128-bit integers don't currently have a known stable ABI".into(),
                help: None,
            },

            // Primitive types with a stable representation.
            ty::Bool | ty::Int(..) | ty::Uint(..) | ty::Float(..) | ty::Never => FfiSafe,

            ty::Slice(_) => FfiUnsafe {
                ty,
                reason: "slices have no C equivalent".into(),
                help: Some("consider using a raw pointer instead".into()),
            },

            ty::Dynamic(..) => {
                FfiUnsafe { ty, reason: "trait objects have no C equivalent".into(), help: None }
            }

            ty::Str => FfiUnsafe {
                ty,
                reason: "string slices have no C equivalent".into(),
                help: Some("consider using `*const u8` and a length instead".into()),
            },

            ty::Tuple(..) => FfiUnsafe {
                ty,
                reason: "tuples have unspecified layout".into(),
                help: Some("consider using a struct instead".into()),
            },

            ty::RawPtr(ty::TypeAndMut { ty, .. }) | ty::Ref(_, ty, _)
                if {
                    matches!(self.mode, CItemKind::Definition)
                        && ty.is_sized(self.cx.tcx.at(DUMMY_SP), self.cx.param_env)
                } =>
            {
                FfiSafe
            }

            ty::RawPtr(ty::TypeAndMut { ty, .. }) | ty::Ref(_, ty, _) => {
                self.check_type_for_ffi(cache, ty)
            }

            ty::Array(inner_ty, _) => self.check_type_for_ffi(cache, inner_ty),

            ty::FnPtr(sig) => {
                if self.is_internal_abi(sig.abi()) {
                    return FfiUnsafe {
                        ty,
                        reason: "this function pointer has Rust-specific calling convention".into(),
                        help: Some(
                            "consider using an `extern fn(...) -> ...` \
                                    function pointer instead"
                                .into(),
                        ),
                    };
                }

                let sig = tcx.erase_late_bound_regions(sig);
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

            // While opaque types are checked for earlier, if a projection in a struct field
            // normalizes to an opaque type, then it will reach this branch.
            ty::Opaque(..) => {
                FfiUnsafe { ty, reason: "opaque types have no C equivalent".into(), help: None }
            }

            // `extern "C" fn` functions can have type parameters, which may or may not be FFI-safe,
            //  so they are currently ignored for the purposes of this lint.
            ty::Param(..) | ty::Projection(..) if matches!(self.mode, CItemKind::Definition) => {
                FfiSafe
            }

            ty::Param(..)
            | ty::Projection(..)
            | ty::Infer(..)
            | ty::Bound(..)
            | ty::Error(_)
            | ty::Closure(..)
            | ty::Generator(..)
            | ty::GeneratorWitness(..)
            | ty::Placeholder(..)
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
        let lint = match self.mode {
            CItemKind::Declaration => IMPROPER_CTYPES,
            CItemKind::Definition => IMPROPER_CTYPES_DEFINITIONS,
        };

        self.cx.struct_span_lint(lint, sp, |lint| {
            let item_description = match self.mode {
                CItemKind::Declaration => "block",
                CItemKind::Definition => "fn",
            };
            let mut diag = lint.build(&format!(
                "`extern` {} uses type `{}`, which is not FFI-safe",
                item_description, ty
            ));
            diag.span_label(sp, "not FFI-safe");
            if let Some(help) = help {
                diag.help(help);
            }
            diag.note(note);
            if let ty::Adt(def, _) = ty.kind() {
                if let Some(sp) = self.cx.tcx.hir().span_if_local(def.did) {
                    diag.span_note(sp, "the type is defined here");
                }
            }
            diag.emit();
        });
    }

    fn check_for_opaque_ty(&mut self, sp: Span, ty: Ty<'tcx>) -> bool {
        struct ProhibitOpaqueTypes<'a, 'tcx> {
            cx: &'a LateContext<'tcx>,
            ty: Option<Ty<'tcx>>,
        };

        impl<'a, 'tcx> ty::fold::TypeVisitor<'tcx> for ProhibitOpaqueTypes<'a, 'tcx> {
            fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<()> {
                match ty.kind() {
                    ty::Opaque(..) => {
                        self.ty = Some(ty);
                        ControlFlow::BREAK
                    }
                    // Consider opaque types within projections FFI-safe if they do not normalize
                    // to more opaque types.
                    ty::Projection(..) => {
                        let ty = self.cx.tcx.normalize_erasing_regions(self.cx.param_env, ty);

                        // If `ty` is a opaque type directly then `super_visit_with` won't invoke
                        // this function again.
                        if ty.has_opaque_types() {
                            self.visit_ty(ty)
                        } else {
                            ControlFlow::CONTINUE
                        }
                    }
                    _ => ty.super_visit_with(self),
                }
            }
        }

        let mut visitor = ProhibitOpaqueTypes { cx: self.cx, ty: None };
        ty.visit_with(&mut visitor);
        if let Some(ty) = visitor.ty {
            self.emit_ffi_unsafe_type_lint(ty, sp, "opaque types have no C equivalent", None);
            true
        } else {
            false
        }
    }

    fn check_type_for_ffi_and_report_errors(
        &mut self,
        sp: Span,
        ty: Ty<'tcx>,
        is_static: bool,
        is_return_type: bool,
    ) {
        // We have to check for opaque types before `normalize_erasing_regions`,
        // which will replace opaque types with their underlying concrete type.
        if self.check_for_opaque_ty(sp, ty) {
            // We've already emitted an error due to an opaque type.
            return;
        }

        // it is only OK to use this function because extern fns cannot have
        // any generic types right now:
        let ty = self.cx.tcx.normalize_erasing_regions(self.cx.param_env, ty);

        // C doesn't really support passing arrays by value - the only way to pass an array by value
        // is through a struct. So, first test that the top level isn't an array, and then
        // recursively check the types inside.
        if !is_static && self.check_for_array_ty(sp, ty) {
            return;
        }

        // Don't report FFI errors for unit return types. This check exists here, and not in
        // `check_foreign_fn` (where it would make more sense) so that normalization has definitely
        // happened.
        if is_return_type && ty.is_unit() {
            return;
        }

        match self.check_type_for_ffi(&mut FxHashSet::default(), ty) {
            FfiResult::FfiSafe => {}
            FfiResult::FfiPhantom(ty) => {
                self.emit_ffi_unsafe_type_lint(ty, sp, "composed only of `PhantomData`", None);
            }
            // If `ty` is a `repr(transparent)` newtype, and the non-zero-sized type is a generic
            // argument, which after substitution, is `()`, then this branch can be hit.
            FfiResult::FfiUnsafe { ty, .. } if is_return_type && ty.is_unit() => {}
            FfiResult::FfiUnsafe { ty, reason, help } => {
                self.emit_ffi_unsafe_type_lint(ty, sp, &reason, help.as_deref());
            }
        }
    }

    fn check_foreign_fn(&mut self, id: hir::HirId, decl: &hir::FnDecl<'_>) {
        let def_id = self.cx.tcx.hir().local_def_id(id);
        let sig = self.cx.tcx.fn_sig(def_id);
        let sig = self.cx.tcx.erase_late_bound_regions(sig);

        for (input_ty, input_hir) in sig.inputs().iter().zip(decl.inputs) {
            self.check_type_for_ffi_and_report_errors(input_hir.span, input_ty, false, false);
        }

        if let hir::FnRetTy::Return(ref ret_hir) = decl.output {
            let ret_ty = sig.output();
            self.check_type_for_ffi_and_report_errors(ret_hir.span, ret_ty, false, true);
        }
    }

    fn check_foreign_static(&mut self, id: hir::HirId, span: Span) {
        let def_id = self.cx.tcx.hir().local_def_id(id);
        let ty = self.cx.tcx.type_of(def_id);
        self.check_type_for_ffi_and_report_errors(span, ty, true, false);
    }

    fn is_internal_abi(&self, abi: SpecAbi) -> bool {
        matches!(
            abi,
            SpecAbi::Rust | SpecAbi::RustCall | SpecAbi::RustIntrinsic | SpecAbi::PlatformIntrinsic
        )
    }
}

impl<'tcx> LateLintPass<'tcx> for ImproperCTypesDeclarations {
    fn check_foreign_item(&mut self, cx: &LateContext<'_>, it: &hir::ForeignItem<'_>) {
        let mut vis = ImproperCTypesVisitor { cx, mode: CItemKind::Declaration };
        let abi = cx.tcx.hir().get_foreign_abi(it.hir_id);

        if !vis.is_internal_abi(abi) {
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

impl<'tcx> LateLintPass<'tcx> for ImproperCTypesDefinitions {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: hir::intravisit::FnKind<'tcx>,
        decl: &'tcx hir::FnDecl<'_>,
        _: &'tcx hir::Body<'_>,
        _: Span,
        hir_id: hir::HirId,
    ) {
        use hir::intravisit::FnKind;

        let abi = match kind {
            FnKind::ItemFn(_, _, header, ..) => header.abi,
            FnKind::Method(_, sig, ..) => sig.header.abi,
            _ => return,
        };

        let mut vis = ImproperCTypesVisitor { cx, mode: CItemKind::Definition };
        if !vis.is_internal_abi(abi) {
            vis.check_foreign_fn(hir_id, decl);
        }
    }
}

declare_lint_pass!(VariantSizeDifferences => [VARIANT_SIZE_DIFFERENCES]);

impl<'tcx> LateLintPass<'tcx> for VariantSizeDifferences {
    fn check_item(&mut self, cx: &LateContext<'_>, it: &hir::Item<'_>) {
        if let hir::ItemKind::Enum(ref enum_definition, _) = it.kind {
            let item_def_id = cx.tcx.hir().local_def_id(it.hir_id);
            let t = cx.tcx.type_of(item_def_id);
            let ty = cx.tcx.erase_regions(t);
            let layout = match cx.layout_of(ty) {
                Ok(layout) => layout,
                Err(
                    ty::layout::LayoutError::Unknown(_) | ty::layout::LayoutError::SizeOverflow(_),
                ) => return,
            };
            let (variants, tag) = match layout.variants {
                Variants::Multiple {
                    tag_encoding: TagEncoding::Direct,
                    ref tag,
                    ref variants,
                    ..
                } => (variants, tag),
                _ => return,
            };

            let tag_size = tag.value.size(&cx.tcx).bytes();

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
                    // Subtract the size of the enum tag.
                    let bytes = variant_layout.size.bytes().saturating_sub(tag_size);

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
