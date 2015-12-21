use rustc_front::hir::*;
use rustc::lint::*;
use rustc::middle::ty;
use rustc::middle::subst::{Subst, TypeSpace};
use std::iter;
use std::borrow::Cow;

use utils::{snippet, span_lint, span_note_and_lint, match_path, match_type, walk_ptrs_ty_depth,
    walk_ptrs_ty};
use utils::{OPTION_PATH, RESULT_PATH, STRING_PATH};

use self::SelfKind::*;
use self::OutType::*;

#[derive(Clone)]
pub struct MethodsPass;

/// **What it does:** This lint checks for `.unwrap()` calls on `Option`s. It is `Allow` by default.
///
/// **Why is this bad?** Usually it is better to handle the `None` case, or to at least call `.expect(_)` with a more helpful message. Still, for a lot of quick-and-dirty code, `unwrap` is a good choice, which is why this lint is `Allow` by default.
///
/// **Known problems:** None
///
/// **Example:** `x.unwrap()`
declare_lint!(pub OPTION_UNWRAP_USED, Allow,
              "using `Option.unwrap()`, which should at least get a better message using `expect()`");

/// **What it does:** This lint checks for `.unwrap()` calls on `Result`s. It is `Allow` by default.
///
/// **Why is this bad?** `result.unwrap()` will let the thread panic on `Err` values. Normally, you want to implement more sophisticated error handling, and propagate errors upwards with `try!`.
///
/// Even if you want to panic on errors, not all `Error`s implement good messages on display. Therefore it may be beneficial to look at the places where they may get displayed. Activate this lint to do just that.
///
/// **Known problems:** None
///
/// **Example:** `x.unwrap()`
declare_lint!(pub RESULT_UNWRAP_USED, Allow,
              "using `Result.unwrap()`, which might be better handled");

/// **What it does:** This lint checks for `.to_string()` method calls on values of type `&str`. It is `Warn` by default.
///
/// **Why is this bad?** This uses the whole formatting machinery just to clone a string. Using `.to_owned()` is lighter on resources. You can also consider using a [`Cow<'a, str>`](http://doc.rust-lang.org/std/borrow/enum.Cow.html) instead in some cases.
///
/// **Known problems:** None
///
/// **Example:** `s.to_string()` where `s: &str`
declare_lint!(pub STR_TO_STRING, Warn,
              "using `to_string()` on a str, which should be `to_owned()`");

/// **What it does:** This lint checks for `.to_string()` method calls on values of type `String`. It is `Warn` by default.
///
/// **Why is this bad?** As our string is already owned, this whole operation is basically a no-op, but still creates a clone of the string (which, if really wanted, should be done with `.clone()`).
///
/// **Known problems:** None
///
/// **Example:** `s.to_string()` where `s: String`
declare_lint!(pub STRING_TO_STRING, Warn,
              "calling `String.to_string()` which is a no-op");

/// **What it does:** This lint checks for methods that should live in a trait implementation of a `std` trait (see [llogiq's blog post](http://llogiq.github.io/2015/07/30/traits.html) for further information) instead of an inherent implementation. It is `Warn` by default.
///
/// **Why is this bad?** Implementing the traits improve ergonomics for users of the code, often with very little cost. Also people seeing a `mul(..)` method may expect `*` to work equally, so you should have good reason to disappoint them.
///
/// **Known problems:** None
///
/// **Example:**
/// ```
/// struct X;
/// impl X {
///    fn add(&self, other: &X) -> X { .. }
/// }
/// ```
declare_lint!(pub SHOULD_IMPLEMENT_TRAIT, Warn,
              "defining a method that should be implementing a std trait");

/// **What it does:** This lint checks for methods with certain name prefixes and `Warn`s (by default) if the prefix doesn't match how self is taken. The actual rules are:
///
/// |Prefix |`self` taken        |
/// |-------|--------------------|
/// |`as_`  |`&self` or &mut self|
/// |`from_`| none               |
/// |`into_`|`self`              |
/// |`is_`  |`&self` or none     |
/// |`to_`  |`&self`             |
///
/// **Why is this bad?** Consistency breeds readability. If you follow the conventions, your users won't be surprised that they e.g. need to supply a mutable reference to a `as_`.. function.
///
/// **Known problems:** None
///
/// **Example**
///
/// ```
/// impl X {
///     fn as_str(self) -> &str { .. }
/// }
/// ```
declare_lint!(pub WRONG_SELF_CONVENTION, Warn,
              "defining a method named with an established prefix (like \"into_\") that takes \
               `self` with the wrong convention");

/// **What it does:** This is the same as [`wrong_self_convention`](#wrong_self_convention), but for public items. This lint is `Allow` by default.
///
/// **Why is this bad?** See [`wrong_self_convention`](#wrong_self_convention).
///
/// **Known problems:** Actually *renaming* the function may break clients if the function is part of the public interface. In that case, be mindful of the stability guarantees you've given your users.
///
/// **Example:**
/// ```
/// impl X {
///     pub fn as_str(self) -> &str { .. }
/// }
/// ```
declare_lint!(pub WRONG_PUB_SELF_CONVENTION, Allow,
              "defining a public method named with an established prefix (like \"into_\") that takes \
               `self` with the wrong convention");

/// **What it does:** This lint `Warn`s on using `ok().expect(..)`.
///
/// **Why is this bad?** Because you usually call `expect()` on the `Result` directly to get a good error message.
///
/// **Known problems:** None.
///
/// **Example:** `x.ok().expect("why did I do this again?")`
declare_lint!(pub OK_EXPECT, Warn,
              "using `ok().expect()`, which gives worse error messages than \
               calling `expect` directly on the Result");

/// **What it does:** This lint `Warn`s on `_.map(_).unwrap_or(_)`.
///
/// **Why is this bad?** Readability, this can be written more concisely as `_.map_or(_, _)`.
///
/// **Known problems:** None.
///
/// **Example:** `x.map(|a| a + 1).unwrap_or(0)`
declare_lint!(pub OPTION_MAP_UNWRAP_OR, Warn,
              "using `Option.map(f).unwrap_or(a)`, which is more succinctly expressed as \
               `map_or(a, f)`)");

/// **What it does:** This lint `Warn`s on `_.map(_).unwrap_or_else(_)`.
///
/// **Why is this bad?** Readability, this can be written more concisely as `_.map_or_else(_, _)`.
///
/// **Known problems:** None.
///
/// **Example:** `x.map(|a| a + 1).unwrap_or_else(some_function)`
declare_lint!(pub OPTION_MAP_UNWRAP_OR_ELSE, Warn,
              "using `Option.map(f).unwrap_or_else(g)`, which is more succinctly expressed as \
               `map_or_else(g, f)`)");

impl LintPass for MethodsPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(OPTION_UNWRAP_USED, RESULT_UNWRAP_USED, STR_TO_STRING, STRING_TO_STRING,
                    SHOULD_IMPLEMENT_TRAIT, WRONG_SELF_CONVENTION, OK_EXPECT, OPTION_MAP_UNWRAP_OR,
                    OPTION_MAP_UNWRAP_OR_ELSE)
    }
}

impl LateLintPass for MethodsPass {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {

        if let ExprMethodCall(ref name, _, ref args) = expr.node {
            let (obj_ty, ptr_depth) = walk_ptrs_ty_depth(cx.tcx.expr_ty(&args[0]));
            match &*name.node.as_str() {
                "unwrap" if match_type(cx, obj_ty, &OPTION_PATH) => {
                    span_lint(cx, OPTION_UNWRAP_USED, expr.span,
                              "used unwrap() on an Option value. If you don't want \
                               to handle the None case gracefully, consider using \
                               expect() to provide a better panic message");
                },
                "unwrap" if match_type(cx, obj_ty, &RESULT_PATH) => {
                    span_lint(cx, RESULT_UNWRAP_USED, expr.span,
                              "used unwrap() on a Result value. Graceful handling \
                               of Err values is preferred");
                },
                "to_string" if obj_ty.sty == ty::TyStr => {
                    let mut arg_str = snippet(cx, args[0].span, "_");
                    if ptr_depth > 1 {
                        arg_str = Cow::Owned(format!(
                            "({}{})",
                            iter::repeat('*').take(ptr_depth - 1).collect::<String>(),
                            arg_str));
                    }
                    span_lint(cx, STR_TO_STRING, expr.span, &format!(
                        "`{}.to_owned()` is faster", arg_str));
                },
                "to_string" if match_type(cx, obj_ty, &STRING_PATH) => {
                    span_lint(cx, STRING_TO_STRING, expr.span, "`String.to_string()` is a no-op; use \
                                                                `clone()` to make a copy");
                },
                "expect" => if let ExprMethodCall(ref inner_name, _, ref inner_args) = args[0].node {
                    if inner_name.node.as_str() == "ok"
                            && match_type(cx, cx.tcx.expr_ty(&inner_args[0]), &RESULT_PATH) {
                        let result_type = cx.tcx.expr_ty(&inner_args[0]);
                        if let Some(error_type) = get_error_type(cx, result_type) {
                            if has_debug_impl(error_type, cx) {
                                span_lint(cx, OK_EXPECT, expr.span,
                                         "called `ok().expect()` on a Result \
                                          value. You can call `expect` directly \
                                          on the `Result`");
                            }
                        }
                    }
                },
                // check Option.map(_).unwrap_or(_)
                "unwrap_or" => if let ExprMethodCall(ref inner_name, _, ref inner_args) = args[0].node {
                    if inner_name.node.as_str() == "map"
                            && match_type(cx, cx.tcx.expr_ty(&inner_args[0]), &OPTION_PATH) {
                        // lint message
                        let msg =
                            "called `map(f).unwrap_or(a)` on an Option value. This can be done \
                             more directly by calling `map_or(a, f)` instead";
                        // get args to map() and unwrap_or()
                        let map_arg = snippet(cx, inner_args[1].span, "..");
                        let unwrap_arg = snippet(cx, args[1].span, "..");
                        // lint, with note if neither arg is > 1 line and both map() and
                        // unwrap_or() have the same span
                        let multiline = map_arg.lines().count() > 1
                                        || unwrap_arg.lines().count() > 1;
                        let same_span = inner_args[1].span.expn_id == args[1].span.expn_id;
                        if same_span && !multiline {
                            span_note_and_lint(
                                cx, OPTION_MAP_UNWRAP_OR, expr.span, msg, expr.span,
                                &format!("replace this with map_or({1}, {0})",
                                         map_arg, unwrap_arg)
                            );
                        }
                        else if same_span && multiline {
                            span_lint(cx, OPTION_MAP_UNWRAP_OR, expr.span, msg);
                        };
                    }
                },
                // check Option.map(_).unwrap_or_else(_)
                "unwrap_or_else" => if let ExprMethodCall(ref inner_name, _, ref inner_args) = args[0].node {
                    if inner_name.node.as_str() == "map"
                            && match_type(cx, cx.tcx.expr_ty(&inner_args[0]), &OPTION_PATH) {
                        // lint message
                        let msg =
                            "called `map(f).unwrap_or_else(g)` on an Option value. This can be \
                             done more directly by calling `map_or_else(g, f)` instead";
                        // get args to map() and unwrap_or_else()
                        let map_arg = snippet(cx, inner_args[1].span, "..");
                        let unwrap_arg = snippet(cx, args[1].span, "..");
                        // lint, with note if neither arg is > 1 line and both map() and
                        // unwrap_or_else() have the same span
                        let multiline = map_arg.lines().count() > 1
                                        || unwrap_arg.lines().count() > 1;
                        let same_span = inner_args[1].span.expn_id == args[1].span.expn_id;
                        if same_span && !multiline {
                            span_note_and_lint(
                                cx, OPTION_MAP_UNWRAP_OR_ELSE, expr.span, msg, expr.span,
                                &format!("replace this with map_or_else({1}, {0})",
                                         map_arg, unwrap_arg)
                            );
                        }
                        else if same_span && multiline {
                            span_lint(cx, OPTION_MAP_UNWRAP_OR_ELSE, expr.span, msg);
                        };
                    }
                },
                _ => {},
            }
        }
    }

    fn check_item(&mut self, cx: &LateContext, item: &Item) {
        if let ItemImpl(_, _, _, None, ref ty, ref items) = item.node {
            for implitem in items {
                let name = implitem.name;
                if let ImplItemKind::Method(ref sig, _) = implitem.node {
                    // check missing trait implementations
                    for &(method_name, n_args, self_kind, out_type, trait_name) in &TRAIT_METHODS {
                        if_let_chain! {
                            [
                                name.as_str() == method_name,
                                sig.decl.inputs.len() == n_args,
                                out_type.matches(&sig.decl.output),
                                self_kind.matches(&sig.explicit_self.node, false)
                            ], {
                                span_lint(cx, SHOULD_IMPLEMENT_TRAIT, implitem.span, &format!(
                                    "defining a method called `{}` on this type; consider implementing \
                                     the `{}` trait or choosing a less ambiguous name", name, trait_name));
                            }
                        }
                    }
                    // check conventions w.r.t. conversion method names and predicates
                    let is_copy = is_copy(cx, &ty, &item);
                    for &(prefix, self_kinds) in &CONVENTIONS {
                        if name.as_str().starts_with(prefix) &&
                                !self_kinds.iter().any(|k| k.matches(&sig.explicit_self.node, is_copy)) {
                            let lint = if item.vis == Visibility::Public {
                                WRONG_PUB_SELF_CONVENTION
                            } else {
                                WRONG_SELF_CONVENTION
                            };
                            span_lint(cx, lint, sig.explicit_self.span, &format!(
                                "methods called `{}*` usually take {}; consider choosing a less \
                                 ambiguous name", prefix,
                                &self_kinds.iter().map(|k| k.description()).collect::<Vec<_>>().join(" or ")));
                        }
                    }
                }
            }
        }
    }
}

// Given a `Result<T, E>` type, return its error type (`E`)
fn get_error_type<'a>(cx: &LateContext, ty: ty::Ty<'a>) -> Option<ty::Ty<'a>> {
    if !match_type(cx, ty, &RESULT_PATH) {
        return None;
    }
    if let ty::TyEnum(_, substs) = ty.sty {
        if let Some(err_ty) = substs.types.opt_get(TypeSpace, 1) {
            return Some(err_ty);
        }
    }
    None
}

// This checks whether a given type is known to implement Debug. It's
// conservative, i.e. it should not return false positives, but will return
// false negatives.
fn has_debug_impl<'a, 'b>(ty: ty::Ty<'a>, cx: &LateContext<'b, 'a>) -> bool {
    let no_ref_ty = walk_ptrs_ty(ty);
    let debug = match cx.tcx.lang_items.debug_trait() {
        Some(debug) => debug,
        None => return false
    };
    let debug_def = cx.tcx.lookup_trait_def(debug);
    let mut debug_impl_exists = false;
    debug_def.for_each_relevant_impl(cx.tcx, no_ref_ty, |d| {
        let self_ty = &cx.tcx.impl_trait_ref(d).and_then(|im| im.substs.self_ty());
        if let Some(self_ty) = *self_ty {
            if !self_ty.flags.get().contains(ty::TypeFlags::HAS_PARAMS) {
                debug_impl_exists = true;
            }
        }
    });
    debug_impl_exists
}

const CONVENTIONS: [(&'static str, &'static [SelfKind]); 5] = [
    ("into_", &[ValueSelf]),
    ("to_",   &[RefSelf]),
    ("as_",   &[RefSelf, RefMutSelf]),
    ("is_",   &[RefSelf, NoSelf]),
    ("from_", &[NoSelf]),
];

const TRAIT_METHODS: [(&'static str, usize, SelfKind, OutType, &'static str); 30] = [
    ("add",        2, ValueSelf,  AnyType,  "std::ops::Add"),
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
    NoSelf,
}

impl SelfKind {
    fn matches(&self, slf: &ExplicitSelf_, allow_value_for_ref: bool) -> bool {
        match (self, slf) {
            (&ValueSelf, &SelfValue(_)) => true,
            (&RefSelf, &SelfRegion(_, Mutability::MutImmutable, _)) => true,
            (&RefMutSelf, &SelfRegion(_, Mutability::MutMutable, _)) => true,
            (&RefSelf, &SelfValue(_)) => allow_value_for_ref,
            (&RefMutSelf, &SelfValue(_)) => allow_value_for_ref,
            (&NoSelf, &SelfStatic) => true,
            (_, &SelfExplicit(ref ty, _)) => self.matches_explicit_type(ty, allow_value_for_ref),
            _ => false
        }
    }

    fn matches_explicit_type(&self, ty: &Ty, allow_value_for_ref: bool) -> bool {
        match (self, &ty.node) {
            (&ValueSelf, &TyPath(..)) => true,
            (&RefSelf, &TyRptr(_, MutTy { mutbl: Mutability::MutImmutable, .. })) => true,
            (&RefMutSelf, &TyRptr(_, MutTy { mutbl: Mutability::MutMutable, .. })) => true,
            (&RefSelf, &TyPath(..)) => allow_value_for_ref,
            (&RefMutSelf, &TyPath(..)) => allow_value_for_ref,
            _ => false
        }
    }

    fn description(&self) -> &'static str {
        match *self {
            ValueSelf => "self by value",
            RefSelf => "self by reference",
            RefMutSelf => "self by mutable reference",
            NoSelf => "no self",
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
            (&UnitType, &Return(ref ty)) if ty.node == TyTup(vec![].into()) => true,
            (&BoolType, &Return(ref ty)) if is_bool(ty) => true,
            (&AnyType, &Return(ref ty)) if ty.node != TyTup(vec![].into())  => true,
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

fn is_copy(cx: &LateContext, ast_ty: &Ty, item: &Item) -> bool {
    match cx.tcx.ast_ty_to_ty_cache.borrow().get(&ast_ty.id) {
        None => false,
        Some(ty) => {
            let env = ty::ParameterEnvironment::for_item(cx.tcx, item.id);
            !ty.subst(cx.tcx, &env.free_substs).moves_by_default(&env, ast_ty.span)
        }
    }
}
