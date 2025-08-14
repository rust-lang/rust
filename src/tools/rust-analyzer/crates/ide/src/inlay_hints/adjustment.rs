//! Implementation of "adjustment" inlay hints:
//! ```no_run
//! let _: u32  = /* <never-to-any> */ loop {};
//! let _: &u32 = /* &* */ &mut 0;
//! ```
use std::ops::Not;

use either::Either;
use hir::{
    Adjust, Adjustment, AutoBorrow, DisplayTarget, HirDisplay, Mutability, OverloadedDeref,
    PointerCast, Safety,
};
use ide_db::famous_defs::FamousDefs;

use ide_db::text_edit::TextEditBuilder;
use syntax::ast::{self, AstNode, prec::ExprPrecedence};

use crate::{
    AdjustmentHints, AdjustmentHintsMode, InlayHint, InlayHintLabel, InlayHintLabelPart,
    InlayHintPosition, InlayHintsConfig, InlayKind, InlayTooltip,
};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    display_target: DisplayTarget,
    expr: &ast::Expr,
) -> Option<()> {
    if config.adjustment_hints_hide_outside_unsafe && !sema.is_inside_unsafe(expr) {
        return None;
    }

    if config.adjustment_hints == AdjustmentHints::Never {
        return None;
    }

    // ParenExpr resolve to their contained expressions HIR so they will dupe these hints
    if let ast::Expr::ParenExpr(_) = expr {
        return None;
    }
    if let ast::Expr::BlockExpr(b) = expr
        && !b.is_standalone()
    {
        return None;
    }

    let descended = sema.descend_node_into_attributes(expr.clone()).pop();
    let desc_expr = descended.as_ref().unwrap_or(expr);
    let adjustments = sema.expr_adjustments(desc_expr).filter(|it| !it.is_empty())?;

    if let ast::Expr::BlockExpr(_) | ast::Expr::IfExpr(_) | ast::Expr::MatchExpr(_) = desc_expr {
        // Don't show unnecessary reborrows for these, they will just repeat the inner ones again
        if matches!(
            &*adjustments,
            [Adjustment { kind: Adjust::Deref(_), source, .. }, Adjustment { kind: Adjust::Borrow(_), target, .. }]
            if source == target
        ) {
            return None;
        }
    }

    let (postfix, needs_outer_parens, needs_inner_parens) =
        mode_and_needs_parens_for_adjustment_hints(expr, config.adjustment_hints_mode);

    let range = expr.syntax().text_range();
    let mut pre = InlayHint {
        range,
        position: InlayHintPosition::Before,
        pad_left: false,
        pad_right: false,
        kind: InlayKind::Adjustment,
        label: InlayHintLabel::default(),
        text_edit: None,
        resolve_parent: Some(range),
    };
    let mut post = InlayHint {
        range,
        position: InlayHintPosition::After,
        pad_left: false,
        pad_right: false,
        kind: InlayKind::Adjustment,
        label: InlayHintLabel::default(),
        text_edit: None,
        resolve_parent: Some(range),
    };

    if needs_outer_parens || (postfix && needs_inner_parens) {
        pre.label.append_str("(");
    }

    if postfix && needs_inner_parens {
        post.label.append_str(")");
    }

    let mut iter = if postfix {
        Either::Left(adjustments.into_iter())
    } else {
        Either::Right(adjustments.into_iter().rev())
    };
    let iter: &mut dyn Iterator<Item = _> = iter.as_mut().either(|it| it as _, |it| it as _);

    let mut has_adjustments = false;
    let mut allow_edit = !postfix;
    for Adjustment { source, target, kind } in iter {
        if source == target {
            cov_mark::hit!(same_type_adjustment);
            continue;
        }
        has_adjustments = true;

        let (text, coercion, detailed_tooltip) = match kind {
            Adjust::NeverToAny if config.adjustment_hints == AdjustmentHints::Always => {
                allow_edit = false;
                (
                    "<never-to-any>",
                    "never to any",
                    "Coerces the never type `!` into any other type. This happens in code paths that never return, like after `panic!()` or `return`.",
                )
            }
            Adjust::Deref(None) => (
                "*",
                "dereference",
                "Built-in dereference of a reference to access the underlying value. The compiler inserts `*` to get the value from `&T`.",
            ),
            Adjust::Deref(Some(OverloadedDeref(Mutability::Shared))) => (
                "*",
                "`Deref` dereference",
                "Dereference via the `Deref` trait. Used for types like `Box<T>` or `Rc<T>` so they act like plain `T`.",
            ),
            Adjust::Deref(Some(OverloadedDeref(Mutability::Mut))) => (
                "*",
                "`DerefMut` dereference",
                "Mutable dereference using the `DerefMut` trait. Enables smart pointers to give mutable access to their inner values.",
            ),
            Adjust::Borrow(AutoBorrow::Ref(Mutability::Shared)) => (
                "&",
                "shared borrow",
                "Inserts `&` to create a shared reference. Lets you use a value without moving or cloning it.",
            ),
            Adjust::Borrow(AutoBorrow::Ref(Mutability::Mut)) => (
                "&mut ",
                "mutable borrow",
                "Inserts `&mut` to create a unique, mutable reference. Lets you modify a value without taking ownership.",
            ),
            Adjust::Borrow(AutoBorrow::RawPtr(Mutability::Shared)) => (
                "&raw const ",
                "const raw pointer",
                "Converts a reference to a raw const pointer `*const T`. Often used when working with FFI or unsafe code.",
            ),
            Adjust::Borrow(AutoBorrow::RawPtr(Mutability::Mut)) => (
                "&raw mut ",
                "mut raw pointer",
                "Converts a mutable reference to a raw mutable pointer `*mut T`. Allows mutation in unsafe contexts.",
            ),
            // some of these could be represented via `as` casts, but that's not too nice and
            // handling everything as a prefix expr makes the `(` and `)` insertion easier
            Adjust::Pointer(cast) if config.adjustment_hints == AdjustmentHints::Always => {
                allow_edit = false;
                match cast {
                    PointerCast::ReifyFnPointer => (
                        "<fn-item-to-fn-pointer>",
                        "fn item to fn pointer",
                        "Converts a named function to a function pointer `fn()`. Useful when passing functions as values.",
                    ),
                    PointerCast::UnsafeFnPointer => (
                        "<safe-fn-pointer-to-unsafe-fn-pointer>",
                        "safe fn pointer to unsafe fn pointer",
                        "Coerces a safe function pointer to an unsafe one. Allows calling it in an unsafe context.",
                    ),
                    PointerCast::ClosureFnPointer(Safety::Unsafe) => (
                        "<closure-to-unsafe-fn-pointer>",
                        "closure to unsafe fn pointer",
                        "Converts a non-capturing closure to an unsafe function pointer. Required for use in `extern` or unsafe APIs.",
                    ),
                    PointerCast::ClosureFnPointer(Safety::Safe) => (
                        "<closure-to-fn-pointer>",
                        "closure to fn pointer",
                        "Converts a non-capturing closure to a function pointer. Lets closures behave like plain functions.",
                    ),
                    PointerCast::MutToConstPointer => (
                        "<mut-ptr-to-const-ptr>",
                        "mut ptr to const ptr",
                        "Coerces `*mut T` to `*const T`. Safe because const pointers restrict what you can do.",
                    ),
                    PointerCast::ArrayToPointer => (
                        "<array-ptr-to-element-ptr>",
                        "array to pointer",
                        "Converts an array to a pointer to its first element. Similar to how arrays decay to pointers in C.",
                    ),
                    PointerCast::Unsize => (
                        "<unsize>",
                        "unsize coercion",
                        "Converts a sized type to an unsized one. Used for things like turning arrays into slices or concrete types into trait objects.",
                    ),
                }
            }
            _ => continue,
        };
        let label = InlayHintLabelPart {
            text: if postfix { format!(".{}", text.trim_end()) } else { text.to_owned() },
            linked_location: None,
            tooltip: Some(config.lazy_tooltip(|| {
                InlayTooltip::Markdown(format!(
                    "`{}` → `{}`\n\n**{}**\n\n{}",
                    source.display(sema.db, display_target),
                    target.display(sema.db, display_target),
                    coercion,
                    detailed_tooltip
                ))
            })),
        };
        if postfix { &mut post } else { &mut pre }.label.append_part(label);
    }
    if !has_adjustments {
        return None;
    }

    if !postfix && needs_inner_parens {
        pre.label.append_str("(");
    }
    if needs_outer_parens || (!postfix && needs_inner_parens) {
        post.label.append_str(")");
    }

    let mut pre = pre.label.parts.is_empty().not().then_some(pre);
    let mut post = post.label.parts.is_empty().not().then_some(post);
    if pre.is_none() && post.is_none() {
        return None;
    }
    if allow_edit {
        let edit = Some(config.lazy_text_edit(|| {
            let mut b = TextEditBuilder::default();
            if let Some(pre) = &pre {
                b.insert(
                    pre.range.start(),
                    pre.label.parts.iter().map(|part| &*part.text).collect::<String>(),
                );
            }
            if let Some(post) = &post {
                b.insert(
                    post.range.end(),
                    post.label.parts.iter().map(|part| &*part.text).collect::<String>(),
                );
            }
            b.finish()
        }));
        match (&mut pre, &mut post) {
            (Some(pre), Some(post)) => {
                pre.text_edit = edit.clone();
                post.text_edit = edit;
            }
            (Some(pre), None) => pre.text_edit = edit,
            (None, Some(post)) => post.text_edit = edit,
            (None, None) => (),
        }
    }
    acc.extend(pre);
    acc.extend(post);
    Some(())
}

/// Returns whatever the hint should be postfix and if we need to add parentheses on the inside and/or outside of `expr`,
/// if we are going to add (`postfix`) adjustments hints to it.
fn mode_and_needs_parens_for_adjustment_hints(
    expr: &ast::Expr,
    mode: AdjustmentHintsMode,
) -> (bool, bool, bool) {
    use {AdjustmentHintsMode::*, std::cmp::Ordering::*};

    match mode {
        Prefix | Postfix => {
            let postfix = matches!(mode, Postfix);
            let (inside, outside) = needs_parens_for_adjustment_hints(expr, postfix);
            (postfix, inside, outside)
        }
        PreferPrefix | PreferPostfix => {
            let prefer_postfix = matches!(mode, PreferPostfix);

            let (pre_inside, pre_outside) = needs_parens_for_adjustment_hints(expr, false);
            let prefix = (false, pre_inside, pre_outside);
            let pre_count = pre_inside as u8 + pre_outside as u8;

            let (post_inside, post_outside) = needs_parens_for_adjustment_hints(expr, true);
            let postfix = (true, post_inside, post_outside);
            let post_count = post_inside as u8 + post_outside as u8;

            match pre_count.cmp(&post_count) {
                Less => prefix,
                Greater => postfix,
                Equal if prefer_postfix => postfix,
                Equal => prefix,
            }
        }
    }
}

/// Returns whatever we need to add parentheses on the inside and/or outside of `expr`,
/// if we are going to add (`postfix`) adjustments hints to it.
fn needs_parens_for_adjustment_hints(expr: &ast::Expr, postfix: bool) -> (bool, bool) {
    let prec = expr.precedence();
    if postfix {
        let needs_inner_parens = prec.needs_parentheses_in(ExprPrecedence::Postfix);
        // given we are the higher precedence, no parent expression will have stronger requirements
        let needs_outer_parens = false;
        (needs_outer_parens, needs_inner_parens)
    } else {
        let needs_inner_parens = prec.needs_parentheses_in(ExprPrecedence::Prefix);
        let parent = expr
            .syntax()
            .parent()
            .and_then(ast::Expr::cast)
            // if we are already wrapped, great, no need to wrap again
            .filter(|it| !matches!(it, ast::Expr::ParenExpr(_)))
            .map(|it| it.precedence())
            .filter(|&prec| prec != ExprPrecedence::Unambiguous);

        // if we have no parent, we don't need outer parens to disambiguate
        // otherwise anything with higher precedence than what we insert needs to wrap us
        let needs_outer_parens = parent
            .is_some_and(|parent_prec| ExprPrecedence::Prefix.needs_parentheses_in(parent_prec));
        (needs_outer_parens, needs_inner_parens)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        AdjustmentHints, AdjustmentHintsMode, InlayHintsConfig,
        inlay_hints::tests::{DISABLED_CONFIG, check_with_config},
    };

    #[test]
    fn adjustment_hints_prefix() {
        check_with_config(
            InlayHintsConfig { adjustment_hints: AdjustmentHints::Always, ..DISABLED_CONFIG },
            r#"
//- minicore: coerce_unsized, fn, eq, index, dispatch_from_dyn
fn main() {
    let _: u32         = loop {};
                       //^^^^^^^<never-to-any>
    let _: &u32        = &mut 0;
                       //^^^^^^&*
    let _: &mut u32    = &mut 0;
                       //^^^^^^&mut *
    let _: *const u32  = &mut 0;
                       //^^^^^^&raw const *
    let _: *mut u32    = &mut 0;
                       //^^^^^^&raw mut *
    let _: fn()        = main;
                       //^^^^<fn-item-to-fn-pointer>
    let _: unsafe fn() = main;
                       //^^^^<safe-fn-pointer-to-unsafe-fn-pointer><fn-item-to-fn-pointer>
    let _: unsafe fn() = main as fn();
                       //^^^^^^^^^^^^<safe-fn-pointer-to-unsafe-fn-pointer>(
                       //^^^^^^^^^^^^)
                       //^^^^<fn-item-to-fn-pointer>
    let _: fn()        = || {};
                       //^^^^^<closure-to-fn-pointer>
    let _: unsafe fn() = || {};
                       //^^^^^<closure-to-unsafe-fn-pointer>
    let _: *const u32  = &mut 0u32 as *mut u32;
                       //^^^^^^^^^^^^^^^^^^^^^<mut-ptr-to-const-ptr>(
                       //^^^^^^^^^^^^^^^^^^^^^)
                       //^^^^^^^^^&raw mut *
    let _: &mut [_]    = &mut [0; 0];
                       //^^^^^^^^^^^<unsize>&mut *

    Struct.consume();
    Struct.by_ref();
  //^^^^^^(&
  //^^^^^^)
    Struct.by_ref_mut();
  //^^^^^^(&mut $
  //^^^^^^)

    (&Struct).consume();
   //^^^^^^^*
    (&Struct).by_ref();
   //^^^^^^^&*

    (&mut Struct).consume();
   //^^^^^^^^^^^*
    (&mut Struct).by_ref();
   //^^^^^^^^^^^&*
    (&mut Struct).by_ref_mut();
   //^^^^^^^^^^^&mut *

    // Check that block-like expressions don't duplicate hints
    let _: &mut [u32] = (&mut []);
                       //^^^^^^^<unsize>&mut *
    let _: &mut [u32] = { &mut [] };
                        //^^^^^^^<unsize>&mut *
    let _: &mut [u32] = unsafe { &mut [] };
                               //^^^^^^^<unsize>&mut *
    let _: &mut [u32] = if true {
        &mut []
      //^^^^^^^<unsize>&mut *
    } else {
        loop {}
      //^^^^^^^<never-to-any>
    };
    let _: &mut [u32] = match () { () => &mut [] };
                                       //^^^^^^^<unsize>&mut *

    let _: &mut dyn Fn() = &mut || ();
                         //^^^^^^^^^^<unsize>&mut *
    () == ();
 // ^^&
       // ^^&
    (()) == {()};
  // ^^&
         // ^^^^&
    let closure: dyn Fn = || ();
    closure();
  //^^^^^^^(&
  //^^^^^^^)
    Struct[0];
  //^^^^^^(&
  //^^^^^^)
    &mut Struct[0];
       //^^^^^^(&mut $
       //^^^^^^)
    let _: (&mut (),) = (&mut (),);
                       //^^^^^^^&mut *
}

#[derive(Copy, Clone)]
struct Struct;
impl Struct {
    fn consume(self) {}
    fn by_ref(&self) {}
    fn by_ref_mut(&mut self) {}
}
struct StructMut;
impl core::ops::Index<usize> for Struct {
    type Output = ();
}
impl core::ops::IndexMut for Struct {}
"#,
        );
    }

    #[test]
    fn adjustment_hints_postfix() {
        check_with_config(
            InlayHintsConfig {
                adjustment_hints: AdjustmentHints::Always,
                adjustment_hints_mode: AdjustmentHintsMode::Postfix,
                ..DISABLED_CONFIG
            },
            r#"
//- minicore: coerce_unsized, fn, eq, index, dispatch_from_dyn
fn main() {

    Struct.consume();
    Struct.by_ref();
  //^^^^^^.&
    Struct.by_ref_mut();
  //^^^^^^.&mut

    (&Struct).consume();
   //^^^^^^^(
   //^^^^^^^).*
    (&Struct).by_ref();
   //^^^^^^^(
   //^^^^^^^).*.&

    (&mut Struct).consume();
   //^^^^^^^^^^^(
   //^^^^^^^^^^^).*
    (&mut Struct).by_ref();
   //^^^^^^^^^^^(
   //^^^^^^^^^^^).*.&
    (&mut Struct).by_ref_mut();
   //^^^^^^^^^^^(
   //^^^^^^^^^^^).*.&mut

    // Check that block-like expressions don't duplicate hints
    let _: &mut [u32] = (&mut []);
                       //^^^^^^^(
                       //^^^^^^^).*.&mut.<unsize>
    let _: &mut [u32] = { &mut [] };
                        //^^^^^^^(
                        //^^^^^^^).*.&mut.<unsize>
    let _: &mut [u32] = unsafe { &mut [] };
                               //^^^^^^^(
                               //^^^^^^^).*.&mut.<unsize>
    let _: &mut [u32] = if true {
        &mut []
      //^^^^^^^(
      //^^^^^^^).*.&mut.<unsize>
    } else {
        loop {}
      //^^^^^^^.<never-to-any>
    };
    let _: &mut [u32] = match () { () => &mut [] };
                                       //^^^^^^^(
                                       //^^^^^^^).*.&mut.<unsize>

    let _: &mut dyn Fn() = &mut || ();
                         //^^^^^^^^^^(
                         //^^^^^^^^^^).*.&mut.<unsize>
    () == ();
 // ^^.&
       // ^^.&
    (()) == {()};
  // ^^.&
         // ^^^^.&
    let closure: dyn Fn = || ();
    closure();
  //^^^^^^^.&
    Struct[0];
  //^^^^^^.&
    &mut Struct[0];
       //^^^^^^.&mut
    let _: (&mut (),) = (&mut (),);
                       //^^^^^^^(
                       //^^^^^^^).*.&mut
}

#[derive(Copy, Clone)]
struct Struct;
impl Struct {
    fn consume(self) {}
    fn by_ref(&self) {}
    fn by_ref_mut(&mut self) {}
}
struct StructMut;
impl core::ops::Index<usize> for Struct {
    type Output = ();
}
impl core::ops::IndexMut for Struct {}
"#,
        );
    }

    #[test]
    fn adjustment_hints_prefer_prefix() {
        check_with_config(
            InlayHintsConfig {
                adjustment_hints: AdjustmentHints::Always,
                adjustment_hints_mode: AdjustmentHintsMode::PreferPrefix,
                ..DISABLED_CONFIG
            },
            r#"
fn main() {
    let _: u32         = loop {};
                       //^^^^^^^<never-to-any>

    Struct.by_ref();
  //^^^^^^.&

    let (): () = return ();
               //^^^^^^^^^<never-to-any>

    struct Struct;
    impl Struct { fn by_ref(&self) {} }
}
            "#,
        )
    }

    #[test]
    fn adjustment_hints_prefer_postfix() {
        check_with_config(
            InlayHintsConfig {
                adjustment_hints: AdjustmentHints::Always,
                adjustment_hints_mode: AdjustmentHintsMode::PreferPostfix,
                ..DISABLED_CONFIG
            },
            r#"
fn main() {
    let _: u32         = loop {};
                       //^^^^^^^.<never-to-any>

    Struct.by_ref();
  //^^^^^^.&

    let (): () = return ();
               //^^^^^^^^^<never-to-any>

    struct Struct;
    impl Struct { fn by_ref(&self) {} }
}
            "#,
        )
    }

    #[test]
    fn never_to_never_is_never_shown() {
        cov_mark::check!(same_type_adjustment);
        check_with_config(
            InlayHintsConfig { adjustment_hints: AdjustmentHints::Always, ..DISABLED_CONFIG },
            r#"
fn never() -> ! {
    return loop {};
}

fn or_else() {
    let () = () else { return };
}
            "#,
        )
    }

    #[test]
    fn adjustment_hints_unsafe_only() {
        check_with_config(
            InlayHintsConfig {
                adjustment_hints: AdjustmentHints::Always,
                adjustment_hints_hide_outside_unsafe: true,
                ..DISABLED_CONFIG
            },
            r#"
unsafe fn enabled() {
    f(&&());
    //^^^^&**
}

fn disabled() {
    f(&&());
}

fn mixed() {
    f(&&());

    unsafe {
        f(&&());
        //^^^^&**
    }
}

const _: () = {
    f(&&());

    unsafe {
        f(&&());
        //^^^^&**
    }
};

static STATIC: () = {
    f(&&());

    unsafe {
        f(&&());
        //^^^^&**
    }
};

enum E {
    Disable = { f(&&()); 0 },
    Enable = unsafe { f(&&()); 1 },
                      //^^^^&**
}

const fn f(_: &()) {}
            "#,
        )
    }

    #[test]
    fn adjustment_hints_unsafe_only_with_item() {
        check_with_config(
            InlayHintsConfig {
                adjustment_hints: AdjustmentHints::Always,
                adjustment_hints_hide_outside_unsafe: true,
                ..DISABLED_CONFIG
            },
            r#"
fn a() {
    struct Struct;
    impl Struct {
        fn by_ref(&self) {}
    }

    _ = Struct.by_ref();

    _ = unsafe { Struct.by_ref() };
               //^^^^^^(&
               //^^^^^^)
}
            "#,
        );
    }

    #[test]
    fn let_stmt_explicit_ty() {
        check_with_config(
            InlayHintsConfig { adjustment_hints: AdjustmentHints::Always, ..DISABLED_CONFIG },
            r#"
fn main() {
    let () = return;
           //^^^^^^<never-to-any>
    let (): () = return;
               //^^^^^^<never-to-any>
}
            "#,
        )
    }

    // regression test for a stackoverflow in hir display code
    #[test]
    fn adjustment_hints_method_call_on_impl_trait_self() {
        check_with_config(
            InlayHintsConfig { adjustment_hints: AdjustmentHints::Always, ..DISABLED_CONFIG },
            r#"
//- minicore: slice, coerce_unsized
trait T<RHS = Self> {}

fn hello(it: &&[impl T]) {
    it.len();
  //^^(&**
  //^^)
}
"#,
        );
    }
}
