use rustc::lint::*;
use syntax::ast;
use syntax::ast::*;
use syntax::ast_util::{is_comparison_binop, binop_to_string};
use syntax::codemap::Span;
use syntax::visit::{FnKind, Visitor, walk_ty};
use rustc::middle::ty;
use syntax::codemap::ExpnInfo;

use utils::{in_macro, match_type, snippet, span_lint, span_help_and_lint, in_external_macro};
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
                        "`Vec<T>` is already on the heap, `Box<Vec<T>>` makes an extra allocation");
                }
            }
            else if match_type(cx, ty, &LL_PATH) {
                span_help_and_lint(
                    cx, LINKEDLIST, ast_ty.span,
                    "I see you're using a LinkedList! Perhaps you meant some other data structure?",
                    "a RingBuf might work");
            }
        }
    }
}

#[allow(missing_copy_implementations)]
pub struct LetPass;

declare_lint!(pub LET_UNIT_VALUE, Warn,
              "creating a let binding to a value of unit type, which usually can't be used afterwards");


fn check_let_unit(cx: &Context, decl: &Decl, info: Option<&ExpnInfo>) {
    if in_macro(cx, info) { return; }
    if let DeclLocal(ref local) = decl.node {
        let bindtype = &cx.tcx.pat_ty(&*local.pat).sty;
        if *bindtype == ty::TyTuple(vec![]) {
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
        cx.sess().codemap().with_expn_info(
            decl.span.expn_id,
            |info| check_let_unit(cx, decl, info));
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

impl LintPass for CastPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(CAST_PRECISION_LOSS,
                    CAST_SIGN_LOSS,
                    CAST_POSSIBLE_TRUNCATION,
                    CAST_POSSIBLE_WRAP)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let ExprCast(ref ex, _) = expr.node {
            let (cast_from, cast_to) = (cx.tcx.expr_ty(&*ex), cx.tcx.expr_ty(expr));
            if cast_from.is_numeric() && cast_to.is_numeric() && !in_external_macro(cx, expr.span) {
                match (cast_from.is_integral(), cast_to.is_integral()) {
                    (true, false) => {
                        let from_nbits = int_ty_to_nbits(cast_from);
                        let to_nbits : usize = match cast_to.sty {
                            ty::TyFloat(ast::TyF32) => 32,
                            ty::TyFloat(ast::TyF64) => 64,
                            _ => 0
                        };
                        if from_nbits != 0 {
                            // When casting to f32, precision loss would occur regardless of the arch
                            if is_isize_or_usize(cast_from) {
                                if to_nbits == 64 {
                                    span_lint(cx, CAST_PRECISION_LOSS, expr.span,
                                              &format!("casting {0} to f64 causes a loss of precision on targets with 64-bit wide pointers \
                                        	  			({0} is 64 bits wide, but f64's mantissa is only 52 bits wide)",
                                                       cast_from));
                                }
                                else {
                                    span_lint(cx, CAST_PRECISION_LOSS, expr.span,
                                              &format!("casting {0} to f32 causes a loss of precision \
                                        	  			({0} is 32 or 64 bits wide, but f32's mantissa is only 23 bits wide)",
                                                       cast_from));
                                }
                            }
                            else if from_nbits >= to_nbits {
                                span_lint(cx, CAST_PRECISION_LOSS, expr.span,
                                          &format!("casting {0} to {1} causes a loss of precision \
                                          	    ({0} is {2} bits wide, but {1}'s mantissa is only {3} bits wide)",
                                                   cast_from, cast_to, from_nbits, if to_nbits == 64 {52} else {23} ));
                            }
                        }
                    },
                    (false, true) => {
                        span_lint(cx, CAST_POSSIBLE_TRUNCATION, expr.span,
                                  &format!("casting {} to {} may truncate the value", cast_from, cast_to));
                        if !cast_to.is_signed() {
                            span_lint(cx, CAST_SIGN_LOSS, expr.span,
                                      &format!("casting {} to {} may lose the sign of the value", cast_from, cast_to));
                        }
                    },
                    (true, true) => {
                        let from_nbits = int_ty_to_nbits(cast_from);
                        let to_nbits   = int_ty_to_nbits(cast_to);
                        if cast_from.is_signed() && !cast_to.is_signed() {
                            span_lint(cx, CAST_SIGN_LOSS, expr.span,
                                      &format!("casting {} to {} may lose the sign of the value", cast_from, cast_to));
                        }
                        match (is_isize_or_usize(cast_from), is_isize_or_usize(cast_to)) {
                            (true, true) | (false, false) =>
                                if to_nbits < from_nbits {
                                    span_lint(cx, CAST_POSSIBLE_TRUNCATION, expr.span,
                                              &format!("casting {} to {} may truncate the value", cast_from, cast_to));
                                }
                                else if !cast_from.is_signed() && cast_to.is_signed() && to_nbits == from_nbits {
                                    span_lint(cx, CAST_POSSIBLE_WRAP, expr.span,
                                              &format!("casting {} to {} may wrap around the value", cast_from, cast_to));
                                },
                            (true, false) =>
                                if to_nbits == 32 {
                                    span_lint(cx, CAST_POSSIBLE_TRUNCATION, expr.span,
                                              &format!("casting {} to {} may truncate the value on targets with 64-bit wide pointers",
                                                       cast_from, cast_to));
                                    if !cast_from.is_signed() && cast_to.is_signed() {
                                        span_lint(cx, CAST_POSSIBLE_WRAP, expr.span,
                                                  &format!("casting {} to {} may wrap around the value on targets with 32-bit wide pointers",
                                                           cast_from, cast_to));
                                    }
                                }
                                else if to_nbits < 32 {
                                    span_lint(cx, CAST_POSSIBLE_TRUNCATION, expr.span,
                                              &format!("casting {} to {} may truncate the value", cast_from, cast_to));
                                },
                            (false, true) =>
                                if from_nbits == 64 {
                                    span_lint(cx, CAST_POSSIBLE_TRUNCATION, expr.span,
                                              &format!("casting {} to {} may truncate the value on targets with 32-bit wide pointers",
                                                       cast_from, cast_to));
                                    if !cast_from.is_signed() && cast_to.is_signed() {
                                        span_lint(cx, CAST_POSSIBLE_WRAP, expr.span,
                                                  &format!("casting {} to {} may wrap around the value on targets with 64-bit wide pointers",
                                                           cast_from, cast_to));
                                    }
                                }
                                else {
                                    if !cast_from.is_signed() && cast_to.is_signed() {
                                        span_lint(cx, CAST_POSSIBLE_WRAP, expr.span,
                                                  &format!("casting {} to {} may wrap around the value on targets with 32-bit wide pointers",
                                                           cast_from, cast_to));
                                    }
                                }
                        }
                    }
                    (false, false) => {
                        if let (&ty::TyFloat(ast::TyF64),
                                &ty::TyFloat(ast::TyF32)) = (&cast_from.sty, &cast_to.sty) {
                            span_lint(cx, CAST_POSSIBLE_TRUNCATION, expr.span, "casting f64 to f32 may truncate the value");
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
        check_type(cx, &*field.node.ty);
    }

    fn check_variant(&mut self, cx: &Context, var: &Variant, _: &Generics) {
        // StructVariant is covered by check_struct_field
        if let TupleVariantKind(ref args) = var.node.kind {
            for arg in args {
                check_type(cx, &*arg.ty);
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
        check_type(cx, &*arg.ty);
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
