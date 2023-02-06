//! Implementation of "adjustment" inlay hints:
//! ```no_run
//! let _: u32  = /* <never-to-any> */ loop {};
//! let _: &u32 = /* &* */ &mut 0;
//! ```
use hir::{Adjust, Adjustment, AutoBorrow, HirDisplay, Mutability, PointerCast, Safety, Semantics};
use ide_db::RootDatabase;

use stdx::never;
use syntax::{
    ast::{self, make, AstNode},
    ted,
};

use crate::{
    AdjustmentHints, AdjustmentHintsMode, InlayHint, InlayHintLabel, InlayHintsConfig, InlayKind,
    InlayTooltip,
};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<'_, RootDatabase>,
    config: &InlayHintsConfig,
    expr: &ast::Expr,
) -> Option<()> {
    if config.adjustment_hints_hide_outside_unsafe && !sema.is_inside_unsafe(expr) {
        return None;
    }

    if config.adjustment_hints == AdjustmentHints::Never {
        return None;
    }

    // These inherit from the inner expression which would result in duplicate hints
    if let ast::Expr::ParenExpr(_)
    | ast::Expr::IfExpr(_)
    | ast::Expr::BlockExpr(_)
    | ast::Expr::MatchExpr(_) = expr
    {
        return None;
    }

    let descended = sema.descend_node_into_attributes(expr.clone()).pop();
    let desc_expr = descended.as_ref().unwrap_or(expr);
    let adjustments = sema.expr_adjustments(desc_expr).filter(|it| !it.is_empty())?;

    let (postfix, needs_outer_parens, needs_inner_parens) =
        mode_and_needs_parens_for_adjustment_hints(expr, config.adjustment_hints_mode);

    if needs_outer_parens {
        acc.push(InlayHint::opening_paren(expr.syntax().text_range()));
    }

    if postfix && needs_inner_parens {
        acc.push(InlayHint::opening_paren(expr.syntax().text_range()));
        acc.push(InlayHint::closing_paren(expr.syntax().text_range()));
    }

    let (mut tmp0, mut tmp1);
    let iter: &mut dyn Iterator<Item = _> = if postfix {
        tmp0 = adjustments.into_iter();
        &mut tmp0
    } else {
        tmp1 = adjustments.into_iter().rev();
        &mut tmp1
    };

    for Adjustment { source, target, kind } in iter {
        if source == target {
            continue;
        }

        // FIXME: Add some nicer tooltips to each of these
        let (text, coercion) = match kind {
            Adjust::NeverToAny if config.adjustment_hints == AdjustmentHints::Always => {
                ("<never-to-any>", "never to any")
            }
            Adjust::Deref(_) => ("*", "dereference"),
            Adjust::Borrow(AutoBorrow::Ref(Mutability::Shared)) => ("&", "borrow"),
            Adjust::Borrow(AutoBorrow::Ref(Mutability::Mut)) => ("&mut ", "unique borrow"),
            Adjust::Borrow(AutoBorrow::RawPtr(Mutability::Shared)) => {
                ("&raw const ", "const pointer borrow")
            }
            Adjust::Borrow(AutoBorrow::RawPtr(Mutability::Mut)) => {
                ("&raw mut ", "mut pointer borrow")
            }
            // some of these could be represented via `as` casts, but that's not too nice and
            // handling everything as a prefix expr makes the `(` and `)` insertion easier
            Adjust::Pointer(cast) if config.adjustment_hints == AdjustmentHints::Always => {
                match cast {
                    PointerCast::ReifyFnPointer => {
                        ("<fn-item-to-fn-pointer>", "fn item to fn pointer")
                    }
                    PointerCast::UnsafeFnPointer => (
                        "<safe-fn-pointer-to-unsafe-fn-pointer>",
                        "safe fn pointer to unsafe fn pointer",
                    ),
                    PointerCast::ClosureFnPointer(Safety::Unsafe) => {
                        ("<closure-to-unsafe-fn-pointer>", "closure to unsafe fn pointer")
                    }
                    PointerCast::ClosureFnPointer(Safety::Safe) => {
                        ("<closure-to-fn-pointer>", "closure to fn pointer")
                    }
                    PointerCast::MutToConstPointer => {
                        ("<mut-ptr-to-const-ptr>", "mut ptr to const ptr")
                    }
                    PointerCast::ArrayToPointer => ("<array-ptr-to-element-ptr>", ""),
                    PointerCast::Unsize => ("<unsize>", "unsize"),
                }
            }
            _ => continue,
        };
        acc.push(InlayHint {
            range: expr.syntax().text_range(),
            kind: if postfix { InlayKind::AdjustmentPostfix } else { InlayKind::Adjustment },
            label: InlayHintLabel::simple(
                if postfix { format!(".{}", text.trim_end()) } else { text.to_owned() },
                Some(InlayTooltip::Markdown(format!(
                    "`{}` → `{}` ({coercion} coercion)",
                    source.display(sema.db),
                    target.display(sema.db),
                ))),
                None,
            ),
        });
    }
    if !postfix && needs_inner_parens {
        acc.push(InlayHint::opening_paren(expr.syntax().text_range()));
        acc.push(InlayHint::closing_paren(expr.syntax().text_range()));
    }
    if needs_outer_parens {
        acc.push(InlayHint::closing_paren(expr.syntax().text_range()));
    }
    Some(())
}

/// Returns whatever the hint should be postfix and if we need to add paretheses on the inside and/or outside of `expr`,
/// if we are going to add (`postfix`) adjustments hints to it.
fn mode_and_needs_parens_for_adjustment_hints(
    expr: &ast::Expr,
    mode: AdjustmentHintsMode,
) -> (bool, bool, bool) {
    use {std::cmp::Ordering::*, AdjustmentHintsMode::*};

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

/// Returns whatever we need to add paretheses on the inside and/or outside of `expr`,
/// if we are going to add (`postfix`) adjustments hints to it.
fn needs_parens_for_adjustment_hints(expr: &ast::Expr, postfix: bool) -> (bool, bool) {
    // This is a very miserable pile of hacks...
    //
    // `Expr::needs_parens_in` requires that the expression is the child of the other expression,
    // that is supposed to be its parent.
    //
    // But we want to check what would happen if we add `*`/`.*` to the inner expression.
    // To check for inner we need `` expr.needs_parens_in(`*expr`) ``,
    // to check for outer we need `` `*expr`.needs_parens_in(parent) ``,
    // where "expr" is the `expr` parameter, `*expr` is the editted `expr`,
    // and "parent" is the parent of the original expression...
    //
    // For this we utilize mutable mutable trees, which is a HACK, but it works.
    //
    // FIXME: comeup with a better API for `needs_parens_in`, so that we don't have to do *this*

    // Make `&expr`/`expr?`
    let dummy_expr = {
        // `make::*` function go through a string, so they parse wrongly.
        // for example `` make::expr_try(`|| a`) `` would result in a
        // `|| (a?)` and not `(|| a)?`.
        //
        // Thus we need dummy parens to preserve the relationship we want.
        // The parens are then simply ignored by the following code.
        let dummy_paren = make::expr_paren(expr.clone());
        if postfix {
            make::expr_try(dummy_paren)
        } else {
            make::expr_ref(dummy_paren, false)
        }
    };

    // Do the dark mutable tree magic.
    // This essentially makes `dummy_expr` and `expr` switch places (families),
    // so that `expr`'s parent is not `dummy_expr`'s parent.
    let dummy_expr = dummy_expr.clone_for_update();
    let expr = expr.clone_for_update();
    ted::replace(expr.syntax(), dummy_expr.syntax());

    let parent = dummy_expr.syntax().parent();
    let Some(expr) = (|| {
        if postfix {
            let ast::Expr::TryExpr(e) = &dummy_expr else { return None };
            let Some(ast::Expr::ParenExpr(e)) = e.expr() else { return None };

            e.expr()
        } else {
            let ast::Expr::RefExpr(e) = &dummy_expr else { return None };
            let Some(ast::Expr::ParenExpr(e)) = e.expr() else { return None };

            e.expr()
        }
    })() else {
        never!("broken syntax tree?\n{:?}\n{:?}", expr, dummy_expr);
        return (true, true)
    };

    // At this point
    // - `parent`     is the parrent of the original expression
    // - `dummy_expr` is the original expression wrapped in the operator we want (`*`/`.*`)
    // - `expr`       is the clone of the original expression (with `dummy_expr` as the parent)

    let needs_outer_parens = parent.map_or(false, |p| dummy_expr.needs_parens_in(p));
    let needs_inner_parens = expr.needs_parens_in(dummy_expr.syntax().clone());

    (needs_outer_parens, needs_inner_parens)
}

#[cfg(test)]
mod tests {
    use crate::{
        inlay_hints::tests::{check_with_config, DISABLED_CONFIG},
        AdjustmentHints, AdjustmentHintsMode, InlayHintsConfig,
    };

    #[test]
    fn adjustment_hints() {
        check_with_config(
            InlayHintsConfig { adjustment_hints: AdjustmentHints::Always, ..DISABLED_CONFIG },
            r#"
//- minicore: coerce_unsized, fn
fn main() {
    let _: u32         = loop {};
                       //^^^^^^^<never-to-any>
    let _: &u32        = &mut 0;
                       //^^^^^^&
                       //^^^^^^*
    let _: &mut u32    = &mut 0;
                       //^^^^^^&mut $
                       //^^^^^^*
    let _: *const u32  = &mut 0;
                       //^^^^^^&raw const $
                       //^^^^^^*
    let _: *mut u32    = &mut 0;
                       //^^^^^^&raw mut $
                       //^^^^^^*
    let _: fn()        = main;
                       //^^^^<fn-item-to-fn-pointer>
    let _: unsafe fn() = main;
                       //^^^^<safe-fn-pointer-to-unsafe-fn-pointer>
                       //^^^^<fn-item-to-fn-pointer>
    let _: unsafe fn() = main as fn();
                       //^^^^^^^^^^^^<safe-fn-pointer-to-unsafe-fn-pointer>
                       //^^^^^^^^^^^^(
                       //^^^^^^^^^^^^)
    let _: fn()        = || {};
                       //^^^^^<closure-to-fn-pointer>
    let _: unsafe fn() = || {};
                       //^^^^^<closure-to-unsafe-fn-pointer>
    let _: *const u32  = &mut 0u32 as *mut u32;
                       //^^^^^^^^^^^^^^^^^^^^^<mut-ptr-to-const-ptr>
                       //^^^^^^^^^^^^^^^^^^^^^(
                       //^^^^^^^^^^^^^^^^^^^^^)
    let _: &mut [_]    = &mut [0; 0];
                       //^^^^^^^^^^^<unsize>
                       //^^^^^^^^^^^&mut $
                       //^^^^^^^^^^^*

    Struct.consume();
    Struct.by_ref();
  //^^^^^^(
  //^^^^^^&
  //^^^^^^)
    Struct.by_ref_mut();
  //^^^^^^(
  //^^^^^^&mut $
  //^^^^^^)

    (&Struct).consume();
   //^^^^^^^*
    (&Struct).by_ref();

    (&mut Struct).consume();
   //^^^^^^^^^^^*
    (&mut Struct).by_ref();
   //^^^^^^^^^^^&
   //^^^^^^^^^^^*
    (&mut Struct).by_ref_mut();

    // Check that block-like expressions don't duplicate hints
    let _: &mut [u32] = (&mut []);
                       //^^^^^^^<unsize>
                       //^^^^^^^&mut $
                       //^^^^^^^*
    let _: &mut [u32] = { &mut [] };
                        //^^^^^^^<unsize>
                        //^^^^^^^&mut $
                        //^^^^^^^*
    let _: &mut [u32] = unsafe { &mut [] };
                               //^^^^^^^<unsize>
                               //^^^^^^^&mut $
                               //^^^^^^^*
    let _: &mut [u32] = if true {
        &mut []
      //^^^^^^^<unsize>
      //^^^^^^^&mut $
      //^^^^^^^*
    } else {
        loop {}
      //^^^^^^^<never-to-any>
    };
    let _: &mut [u32] = match () { () => &mut [] }
                                       //^^^^^^^<unsize>
                                       //^^^^^^^&mut $
                                       //^^^^^^^*

    let _: &mut dyn Fn() = &mut || ();
                         //^^^^^^^^^^<unsize>
                         //^^^^^^^^^^&mut $
                         //^^^^^^^^^^*
}

#[derive(Copy, Clone)]
struct Struct;
impl Struct {
    fn consume(self) {}
    fn by_ref(&self) {}
    fn by_ref_mut(&mut self) {}
}
"#,
        )
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
//- minicore: coerce_unsized, fn
fn main() {

    Struct.consume();
    Struct.by_ref();
  //^^^^^^.&
    Struct.by_ref_mut();
  //^^^^^^.&mut

    (&Struct).consume();
   //^^^^^^^(
   //^^^^^^^)
   //^^^^^^^.*
    (&Struct).by_ref();

    (&mut Struct).consume();
   //^^^^^^^^^^^(
   //^^^^^^^^^^^)
   //^^^^^^^^^^^.*
    (&mut Struct).by_ref();
   //^^^^^^^^^^^(
   //^^^^^^^^^^^)
   //^^^^^^^^^^^.*
   //^^^^^^^^^^^.&
    (&mut Struct).by_ref_mut();

    // Check that block-like expressions don't duplicate hints
    let _: &mut [u32] = (&mut []);
                       //^^^^^^^(
                       //^^^^^^^)
                       //^^^^^^^.*
                       //^^^^^^^.&mut
                       //^^^^^^^.<unsize>
    let _: &mut [u32] = { &mut [] };
                        //^^^^^^^(
                        //^^^^^^^)
                        //^^^^^^^.*
                        //^^^^^^^.&mut
                        //^^^^^^^.<unsize>
    let _: &mut [u32] = unsafe { &mut [] };
                               //^^^^^^^(
                               //^^^^^^^)
                               //^^^^^^^.*
                               //^^^^^^^.&mut
                               //^^^^^^^.<unsize>
    let _: &mut [u32] = if true {
        &mut []
      //^^^^^^^(
      //^^^^^^^)
      //^^^^^^^.*
      //^^^^^^^.&mut
      //^^^^^^^.<unsize>
    } else {
        loop {}
      //^^^^^^^.<never-to-any>
    };
    let _: &mut [u32] = match () { () => &mut [] }
                                       //^^^^^^^(
                                       //^^^^^^^)
                                       //^^^^^^^.*
                                       //^^^^^^^.&mut
                                       //^^^^^^^.<unsize>

    let _: &mut dyn Fn() = &mut || ();
                         //^^^^^^^^^^(
                         //^^^^^^^^^^)
                         //^^^^^^^^^^.*
                         //^^^^^^^^^^.&mut
                         //^^^^^^^^^^.<unsize>
}

#[derive(Copy, Clone)]
struct Struct;
impl Struct {
    fn consume(self) {}
    fn by_ref(&self) {}
    fn by_ref_mut(&mut self) {}
}
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
    //^^^^&
    //^^^^*
    //^^^^*
}

fn disabled() {
    f(&&());
}

fn mixed() {
    f(&&());

    unsafe {
        f(&&());
        //^^^^&
        //^^^^*
        //^^^^*
    }
}

const _: () = {
    f(&&());

    unsafe {
        f(&&());
        //^^^^&
        //^^^^*
        //^^^^*
    }
};

static STATIC: () = {
    f(&&());

    unsafe {
        f(&&());
        //^^^^&
        //^^^^*
        //^^^^*
    }
};

enum E {
    Disable = { f(&&()); 0 },
    Enable = unsafe { f(&&()); 1 },
                      //^^^^&
                      //^^^^*
                      //^^^^*
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
               //^^^^^^(
               //^^^^^^&
               //^^^^^^)
}
            "#,
        );
    }

    #[test]
    fn bug() {
        check_with_config(
            InlayHintsConfig { adjustment_hints: AdjustmentHints::Always, ..DISABLED_CONFIG },
            r#"
fn main() {
    // These should be identical, but they are not...

    let () = return;
    let (): () = return;
               //^^^^^^<never-to-any>
}
            "#,
        )
    }
}
