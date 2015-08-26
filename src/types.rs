use rustc::lint::*;
use syntax::ast;
use syntax::ast::*;
use syntax::ast_util::{is_comparison_binop, binop_to_string};
use syntax::codemap::Span;
use syntax::visit::{FnKind, Visitor, walk_ty};
use rustc::middle::ty;

use utils::{match_type, snippet, span_lint, span_help_and_lint, in_external_macro};
use utils::{LL_PATH, VEC_PATH};

/// Handles all the linting of funky types
#[allow(missing_copy_implementations)]
pub struct TypePass;

declare_lint!(pub BOX_VEC, Warn,
              "usage of `Box<Vec<T>>`, vector elements are already on the heap");
declare_lint!(pub LINKEDLIST, Warn,
              "usage of LinkedList, usually a vector is faster, or a more specialized data \
               structure like a RingBuf");

impl LintPass for TypePass {
    fn get_lints(&self) -> LintArray {
        lint_array!(BOX_VEC, LINKEDLIST)
    }

    fn check_ty(&mut self, cx: &Context, ast_ty: &ast::Ty) {
        if let Some(ty) = cx.tcx.ast_ty_to_ty_cache.borrow().get(&ast_ty.id) {
            if let ty::TyBox(ref inner) = ty.sty {
                if match_type(cx, inner, &VEC_PATH) {
                    span_help_and_lint(
                        cx, BOX_VEC, ast_ty.span,
                        "you seem to be trying to use `Box<Vec<T>>`. Did you mean to use `Vec<T>`?",
                        "`Vec<T>` is already on the heap, `Box<Vec<T>>` makes an extra allocation. \
                         for further information see https://github.com/\
                         Manishearth/rust-clippy/wiki#box_vec");
                }
            }
            else if match_type(cx, ty, &LL_PATH) {
                span_help_and_lint(
                    cx, LINKEDLIST, ast_ty.span,
                    "I see you're using a LinkedList! Perhaps you meant some other data structure?",
                    "a RingBuf might work; for further information see \
                     https://github.com/Manishearth/rust-clippy/wiki#ineffective_bit_mask");
            }
        }
    }
}

#[allow(missing_copy_implementations)]
pub struct LetPass;

declare_lint!(pub LET_UNIT_VALUE, Warn,
              "creating a let binding to a value of unit type, which usually can't be used afterwards");


fn check_let_unit(cx: &Context, decl: &Decl) {
    if let DeclLocal(ref local) = decl.node {
        let bindtype = &cx.tcx.pat_ty(&local.pat).sty;
        if *bindtype == ty::TyTuple(vec![]) {
            if in_external_macro(cx, decl.span) { return; }
            span_lint(cx, LET_UNIT_VALUE, decl.span, &format!(
                "this let-binding has unit value. Consider omitting `let {} =`",
                snippet(cx, local.pat.span, "..")));
        }
    }
}

impl LintPass for LetPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(LET_UNIT_VALUE)
    }

    fn check_decl(&mut self, cx: &Context, decl: &Decl) {
        check_let_unit(cx, decl)
    }
}

declare_lint!(pub UNIT_CMP, Warn,
              "comparing unit values (which is always `true` or `false`, respectively)");

#[allow(missing_copy_implementations)]
pub struct UnitCmp;

impl LintPass for UnitCmp {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNIT_CMP)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let ExprBinary(ref cmp, ref left, _) = expr.node {
            let op = cmp.node;
            let sty = &cx.tcx.expr_ty(left).sty;
            if *sty == ty::TyTuple(vec![]) && is_comparison_binop(op) {
                let result = match op {
                    BiEq | BiLe | BiGe => "true",
                    _ => "false"
                };
                span_lint(cx, UNIT_CMP, expr.span, &format!(
                    "{}-comparison of unit values detected. This will always be {}",
                    binop_to_string(op), result));
            }
        }
    }
}

pub struct CastPass;

declare_lint!(pub CAST_PRECISION_LOSS, Allow,
              "casts that cause loss of precision, e.g `x as f32` where `x: u64`");
declare_lint!(pub CAST_SIGN_LOSS, Allow,
              "casts from signed types to unsigned types, e.g `x as u32` where `x: i32`");
declare_lint!(pub CAST_POSSIBLE_TRUNCATION, Allow,
              "casts that may cause truncation of the value, e.g `x as u8` where `x: u32`, or `x as i32` where `x: f32`");
declare_lint!(pub CAST_POSSIBLE_WRAP, Allow,
              "casts that may cause wrapping around the value, e.g `x as i32` where `x: u32` and `x > i32::MAX`");

/// Returns the size in bits of an integral type.
/// Will return 0 if the type is not an int or uint variant
fn int_ty_to_nbits(typ: &ty::TyS) -> usize {
    let n = match typ.sty {
        ty::TyInt(i) =>  4 << (i as usize),
        ty::TyUint(u) => 4 << (u as usize),
        _ => 0
    };
    // n == 4 is the usize/isize case
    if n == 4 { ::std::usize::BITS } else { n }
}

fn is_isize_or_usize(typ: &ty::TyS) -> bool {
    match typ.sty {
        ty::TyInt(ast::TyIs) | ty::TyUint(ast::TyUs) => true,
        _ => false
    }
}

fn span_precision_loss_lint(cx: &Context, expr: &Expr, cast_from: &ty::TyS, cast_to_f64: bool) {
    let mantissa_nbits = if cast_to_f64 {52} else {23};
    let arch_dependent = is_isize_or_usize(cast_from) && cast_to_f64;
    let arch_dependent_str = "on targets with 64-bit wide pointers ";
    let from_nbits_str = if arch_dependent {"64".to_owned()}
                         else if is_isize_or_usize(cast_from) {"32 or 64".to_owned()}
                         else {int_ty_to_nbits(cast_from).to_string()};
    span_help_and_lint(cx, CAST_PRECISION_LOSS, expr.span,
        &format!("casting {0} to {1} causes a loss of precision {2}\
            ({0} is {3} bits wide, but {1}'s mantissa is only {4} bits wide)",
            cast_from, if cast_to_f64 {"f64"} else {"f32"},
            if arch_dependent {arch_dependent_str} else {""},
            from_nbits_str,
            mantissa_nbits),
        "for further information see https://github.com/\
        Manishearth/rust-clippy/wiki#cast_precision_loss");
}

enum ArchSuffix {
    _32, _64, None
}

fn check_truncation_and_wrapping(cx: &Context, expr: &Expr, cast_from: &ty::TyS, cast_to: &ty::TyS) {
    let arch_64_suffix = " on targets with 64-bit wide pointers";
    let arch_32_suffix = " on targets with 32-bit wide pointers";
    let cast_unsigned_to_signed = !cast_from.is_signed() && cast_to.is_signed();
    let (from_nbits, to_nbits) = (int_ty_to_nbits(cast_from), int_ty_to_nbits(cast_to));
    let (span_truncation, suffix_truncation, span_wrap, suffix_wrap) =
        match (is_isize_or_usize(cast_from), is_isize_or_usize(cast_to)) {
            (true, true) | (false, false) => (
                to_nbits < from_nbits,
                ArchSuffix::None,
                to_nbits == from_nbits && cast_unsigned_to_signed,
                ArchSuffix::None
                ),
            (true, false) => (
                to_nbits <= 32,
                if to_nbits == 32 {ArchSuffix::_64} else {ArchSuffix::None},
                to_nbits <= 32 && cast_unsigned_to_signed,
                ArchSuffix::_32
                ),
            (false, true) => (
                from_nbits == 64,
                ArchSuffix::_32,
                cast_unsigned_to_signed,
                if from_nbits == 64 {ArchSuffix::_64} else {ArchSuffix::_32}
                ),
        };
    if span_truncation {
        span_help_and_lint(cx, CAST_POSSIBLE_TRUNCATION, expr.span,
            &format!("casting {} to {} may truncate the value{}",
               cast_from, cast_to,
               match suffix_truncation {
                   ArchSuffix::_32 => arch_32_suffix,
                   ArchSuffix::_64 => arch_64_suffix,
                   ArchSuffix::None => "" }),
            "for further information see https://github.com/\
            Manishearth/rust-clippy/wiki#cast_possible_truncation");
    }
    if span_wrap {
        span_help_and_lint(cx, CAST_POSSIBLE_WRAP, expr.span,
            &format!("casting {} to {} may wrap around the value{}",
                cast_from, cast_to,
                match suffix_wrap {
                    ArchSuffix::_32 => arch_32_suffix,
                    ArchSuffix::_64 => arch_64_suffix,
                    ArchSuffix::None => "" }),
            "for further information see https://github.com/\
            Manishearth/rust-clippy/wiki#cast_possible_wrap");
    }
}

impl LintPass for CastPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(CAST_PRECISION_LOSS,
                    CAST_SIGN_LOSS,
                    CAST_POSSIBLE_TRUNCATION,
                    CAST_POSSIBLE_WRAP)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let ExprCast(ref ex, _) = expr.node {
            let (cast_from, cast_to) = (cx.tcx.expr_ty(ex), cx.tcx.expr_ty(expr));
            if cast_from.is_numeric() && cast_to.is_numeric() && !in_external_macro(cx, expr.span) {
                match (cast_from.is_integral(), cast_to.is_integral()) {
                    (true, false) => {
                        let from_nbits = int_ty_to_nbits(cast_from);
                        let to_nbits = if let ty::TyFloat(ast::TyF32) = cast_to.sty {32} else {64};
                        if is_isize_or_usize(cast_from) || from_nbits >= to_nbits {
                            span_precision_loss_lint(cx, expr, cast_from, to_nbits == 64);
                        }
                    },
                    (false, true) => {
                        span_help_and_lint(cx, CAST_POSSIBLE_TRUNCATION, expr.span,
                            &format!("casting {} to {} may truncate the value", 
                                  cast_from, cast_to),
                            "for further information see https://github.com/\
                            Manishearth/rust-clippy/wiki#cast_possible_truncation");
                        if !cast_to.is_signed() {
                            span_help_and_lint(cx, CAST_SIGN_LOSS, expr.span,
                                &format!("casting {} to {} may lose the sign of the value", 
                                    cast_from, cast_to),
                                "for further information see https://github.com/\
                                Manishearth/rust-clippy/wiki#cast_sign_loss");
                        }
                    },
                    (true, true) => {
                        if cast_from.is_signed() && !cast_to.is_signed() {
                            span_help_and_lint(cx, CAST_SIGN_LOSS, expr.span,
                                &format!("casting {} to {} may lose the sign of the value", 
                                    cast_from, cast_to),
                                "for further information see https://github.com/\
                                Manishearth/rust-clippy/wiki#cast_sign_loss");
                        }
                        check_truncation_and_wrapping(cx, expr, cast_from, cast_to);
                    }
                    (false, false) => {
                        if let (&ty::TyFloat(ast::TyF64),
                                &ty::TyFloat(ast::TyF32)) = (&cast_from.sty, &cast_to.sty) {
                            span_help_and_lint(cx, CAST_POSSIBLE_TRUNCATION, 
                                expr.span, 
                                "casting f64 to f32 may truncate the value",
                                "for further information see https://github.com/\
                                Manishearth/rust-clippy/wiki#cast_possible_truncation");
                        }
                    }
                }
            }
        }
    }
}

declare_lint!(pub TYPE_COMPLEXITY, Warn,
              "usage of very complex types; recommends factoring out parts into `type` definitions");

#[allow(missing_copy_implementations)]
pub struct TypeComplexityPass;

impl LintPass for TypeComplexityPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(TYPE_COMPLEXITY)
    }

    fn check_fn(&mut self, cx: &Context, _: FnKind, decl: &FnDecl, _: &Block, _: Span, _: NodeId) {
        check_fndecl(cx, decl);
    }

    fn check_struct_field(&mut self, cx: &Context, field: &StructField) {
        check_type(cx, &field.node.ty);
    }

    fn check_variant(&mut self, cx: &Context, var: &Variant, _: &Generics) {
        // StructVariant is covered by check_struct_field
        if let TupleVariantKind(ref args) = var.node.kind {
            for arg in args {
                check_type(cx, &arg.ty);
            }
        }
    }

    fn check_item(&mut self, cx: &Context, item: &Item) {
        match item.node {
            ItemStatic(ref ty, _, _) |
            ItemConst(ref ty, _) => check_type(cx, ty),
            // functions, enums, structs, impls and traits are covered
            _ => ()
        }
    }

    fn check_trait_item(&mut self, cx: &Context, item: &TraitItem) {
        match item.node {
            ConstTraitItem(ref ty, _) |
            TypeTraitItem(_, Some(ref ty)) => check_type(cx, ty),
            MethodTraitItem(MethodSig { ref decl, .. }, None) => check_fndecl(cx, decl),
            // methods with default impl are covered by check_fn
            _ => ()
        }
    }

    fn check_impl_item(&mut self, cx: &Context, item: &ImplItem) {
        match item.node {
            ConstImplItem(ref ty, _) |
            TypeImplItem(ref ty) => check_type(cx, ty),
            // methods are covered by check_fn
            _ => ()
        }
    }

    fn check_local(&mut self, cx: &Context, local: &Local) {
        if let Some(ref ty) = local.ty {
            check_type(cx, ty);
        }
    }
}

fn check_fndecl(cx: &Context, decl: &FnDecl) {
    for arg in &decl.inputs {
        check_type(cx, &arg.ty);
    }
    if let Return(ref ty) = decl.output {
        check_type(cx, ty);
    }
}

fn check_type(cx: &Context, ty: &ast::Ty) {
    if in_external_macro(cx, ty.span) { return; }
    let score = {
        let mut visitor = TypeComplexityVisitor { score: 0, nest: 1 };
        visitor.visit_ty(ty);
        visitor.score
    };
    // println!("{:?} --> {}", ty, score);
    if score > 250 {
        span_lint(cx, TYPE_COMPLEXITY, ty.span, &format!(
            "very complex type used. Consider factoring parts into `type` definitions"));
    }
}

/// Walks a type and assigns a complexity score to it.
struct TypeComplexityVisitor {
    /// total complexity score of the type
    score: u32,
    /// current nesting level
    nest: u32,
}

impl<'v> Visitor<'v> for TypeComplexityVisitor {
    fn visit_ty(&mut self, ty: &'v ast::Ty) {
        let (add_score, sub_nest) = match ty.node {
            // _, &x and *x have only small overhead; don't mess with nesting level
            TyInfer |
            TyPtr(..) |
            TyRptr(..) => (1, 0),

            // the "normal" components of a type: named types, arrays/tuples
            TyPath(..) |
            TyVec(..) |
            TyTup(..) |
            TyFixedLengthVec(..) => (10 * self.nest, 1),

            // "Sum" of trait bounds
            TyObjectSum(..) => (20 * self.nest, 0),

            // function types and "for<...>" bring a lot of overhead
            TyBareFn(..) |
            TyPolyTraitRef(..) => (50 * self.nest, 1),

            _ => (0, 0)
        };
        self.score += add_score;
        self.nest += sub_nest;
        walk_ty(self, ty);
        self.nest -= sub_nest;
    }
}
