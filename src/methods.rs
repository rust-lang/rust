use syntax::ast::*;
use rustc::lint::*;
use rustc::middle::ty;

use utils::{span_lint, match_path, match_type, walk_ptrs_ty};
use utils::{OPTION_PATH, RESULT_PATH, STRING_PATH};

use self::SelfKind::*;
use self::OutType::*;

#[derive(Copy,Clone)]
pub struct MethodsPass;

declare_lint!(pub OPTION_UNWRAP_USED, Allow,
              "using `Option.unwrap()`, which should at least get a better message using `expect()`");
declare_lint!(pub RESULT_UNWRAP_USED, Allow,
              "using `Result.unwrap()`, which might be better handled");
declare_lint!(pub STR_TO_STRING, Warn,
              "using `to_string()` on a str, which should be `to_owned()`");
declare_lint!(pub STRING_TO_STRING, Warn,
              "calling `String.to_string()` which is a no-op");
declare_lint!(pub SHOULD_IMPLEMENT_TRAIT, Warn,
              "defining a method that should be implementing a std trait");

impl LintPass for MethodsPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(OPTION_UNWRAP_USED, RESULT_UNWRAP_USED, STR_TO_STRING, STRING_TO_STRING,
                    SHOULD_IMPLEMENT_TRAIT)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let ExprMethodCall(ref ident, _, ref args) = expr.node {
            let obj_ty = walk_ptrs_ty(cx.tcx.expr_ty(&*args[0]));
            if ident.node.name == "unwrap" {
                if match_type(cx, obj_ty, &OPTION_PATH) {
                    span_lint(cx, OPTION_UNWRAP_USED, expr.span,
                              "used unwrap() on an Option value. If you don't want \
                               to handle the None case gracefully, consider using \
                               expect() to provide a better panic message");
                } else if match_type(cx, obj_ty, &RESULT_PATH) {
                    span_lint(cx, RESULT_UNWRAP_USED, expr.span,
                              "used unwrap() on a Result value. Graceful handling \
                               of Err values is preferred");
                }
            }
            else if ident.node.name == "to_string" {
                if obj_ty.sty == ty::TyStr {
                    span_lint(cx, STR_TO_STRING, expr.span, "`str.to_owned()` is faster");
                } else if match_type(cx, obj_ty, &STRING_PATH) {
                    span_lint(cx, STRING_TO_STRING, expr.span, "`String.to_string()` is a no-op");
                }
            }
        }
    }

    fn check_item(&mut self, cx: &Context, item: &Item) {
        if let ItemImpl(_, _, _, None, _, ref items) = item.node {
            for item in items {
                let name = item.ident.name;
                for &(method_name, n_args, self_kind, out_type, trait_name) in &TRAIT_METHODS {
                    if_let_chain! {
                        [
                            name == method_name,
                            let MethodImplItem(ref sig, _) = item.node,
                            sig.decl.inputs.len() == n_args,
                            out_type.matches(&sig.decl.output),
                            self_kind.matches(&sig.explicit_self.node)
                        ], {
                            span_lint(cx, SHOULD_IMPLEMENT_TRAIT, item.span, &format!(
                                "defining a method called `{}` on this type; consider implementing \
                                 the `{}` trait or choosing a less ambiguous name", name, trait_name));
                        }
                    }
                }
            }
        }
    }
}

const TRAIT_METHODS: [(&'static str, usize, SelfKind, OutType, &'static str); 30] = [
    ("add",        2, ValueSelf,  AnyType,  "std::ops::Add`"),
    ("sub",        2, ValueSelf,  AnyType,  "std::ops::Sub"),
    ("mul",        2, ValueSelf,  AnyType,  "std::ops::Mul"),
    ("div",        2, ValueSelf,  AnyType,  "std::ops::Div"),
    ("rem",        2, ValueSelf,  AnyType,  "std::ops::Rem"),
    ("shl",        2, ValueSelf,  AnyType,  "std::ops::Shl"),
    ("shr",        2, ValueSelf,  AnyType,  "std::ops::Shr"),
    ("bitand",     2, ValueSelf,  AnyType,  "std::ops::BitAnd"),
    ("bitor",      2, ValueSelf,  AnyType,  "std::ops::BitOr"),
    ("bitxor",     2, ValueSelf,  AnyType,  "std::ops::BitXor"),
    ("neg",        1, ValueSelf,  AnyType,  "std::ops::Neg"),
    ("not",        1, ValueSelf,  AnyType,  "std::ops::Not"),
    ("drop",       1, RefMutSelf, UnitType, "std::ops::Drop"),
    ("index",      2, RefSelf,    RefType,  "std::ops::Index"),
    ("index_mut",  2, RefMutSelf, RefType,  "std::ops::IndexMut"),
    ("deref",      1, RefSelf,    RefType,  "std::ops::Deref"),
    ("deref_mut",  1, RefMutSelf, RefType,  "std::ops::DerefMut"),
    ("clone",      1, RefSelf,    AnyType,  "std::clone::Clone"),
    ("borrow",     1, RefSelf,    RefType,  "std::borrow::Borrow"),
    ("borrow_mut", 1, RefMutSelf, RefType,  "std::borrow::BorrowMut"),
    ("as_ref",     1, RefSelf,    RefType,  "std::convert::AsRef"),
    ("as_mut",     1, RefMutSelf, RefType,  "std::convert::AsMut"),
    ("eq",         2, RefSelf,    BoolType, "std::cmp::PartialEq"),
    ("cmp",        2, RefSelf,    AnyType,  "std::cmp::Ord"),
    ("default",    0, NoSelf,     AnyType,  "std::default::Default"),
    ("hash",       2, RefSelf,    UnitType, "std::hash::Hash"),
    ("next",       1, RefMutSelf, AnyType,  "std::iter::Iterator"),
    ("into_iter",  1, ValueSelf,  AnyType,  "std::iter::IntoIterator"),
    ("from_iter",  1, NoSelf,     AnyType,  "std::iter::FromIterator"),
    ("from_str",   1, NoSelf,     AnyType,  "std::str::FromStr"),
];

#[derive(Clone, Copy)]
enum SelfKind {
    ValueSelf,
    RefSelf,
    RefMutSelf,
    NoSelf
}

impl SelfKind {
    fn matches(&self, slf: &ExplicitSelf_) -> bool {
        match (self, slf) {
            (&ValueSelf, &SelfValue(_)) => true,
            (&RefSelf, &SelfRegion(_, Mutability::MutImmutable, _)) => true,
            (&RefMutSelf, &SelfRegion(_, Mutability::MutMutable, _)) => true,
            (&NoSelf, &SelfStatic) => true,
            _ => false
        }
    }
}

#[derive(Clone, Copy)]
enum OutType {
    UnitType,
    BoolType,
    AnyType,
    RefType,
}

impl OutType {
    fn matches(&self, ty: &FunctionRetTy) -> bool {
        match (self, ty) {
            (&UnitType, &DefaultReturn(_)) => true,
            (&UnitType, &Return(ref ty)) if ty.node == TyTup(vec![]) => true,
            (&BoolType, &Return(ref ty)) if is_bool(ty) => true,
            (&AnyType, &Return(ref ty)) if ty.node != TyTup(vec![])  => true,
            (&RefType, &Return(ref ty)) => {
                if let TyRptr(_, _) = ty.node { true } else { false }
            }
            _ => false
        }
    }
}

fn is_bool(ty: &Ty) -> bool {
    if let TyPath(None, ref p) = ty.node {
        if match_path(p, &["bool"]) {
            return true;
        }
    }
    false
}
