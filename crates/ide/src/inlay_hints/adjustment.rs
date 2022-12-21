//! Implementation of "adjustment" inlay hints:
//! ```no_run
//! let _: u32  = /* <never-to-any> */ loop {};
//! let _: &u32 = /* &* */ &mut 0;
//! ```
use either::Either;
use hir::{
    db::DefDatabase, Adjust, AutoBorrow, InFile, Mutability, OverloadedDeref, PointerCast, Safety,
    Semantics,
};
use ide_db::RootDatabase;

use syntax::{
    ast::{self, AstNode},
    SyntaxNode,
};

use crate::{AdjustmentHints, InlayHint, InlayHintsConfig, InlayKind};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<'_, RootDatabase>,
    config: &InlayHintsConfig,
    expr: &ast::Expr,
) -> Option<()> {
    if config.adjustment_hints_hide_outside_unsafe && !is_inside_unsafe(sema, expr.syntax()) {
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

    let parent = expr.syntax().parent().and_then(ast::Expr::cast);
    let descended = sema.descend_node_into_attributes(expr.clone()).pop();
    let desc_expr = descended.as_ref().unwrap_or(expr);
    let adjustments = sema.expr_adjustments(desc_expr).filter(|it| !it.is_empty())?;
    let needs_parens = match parent {
        Some(parent) => {
            match parent {
                ast::Expr::AwaitExpr(_)
                | ast::Expr::CallExpr(_)
                | ast::Expr::CastExpr(_)
                | ast::Expr::FieldExpr(_)
                | ast::Expr::MethodCallExpr(_)
                | ast::Expr::TryExpr(_) => true,
                // FIXME: shorthands need special casing, though not sure if adjustments are even valid there
                ast::Expr::RecordExpr(_) => false,
                ast::Expr::IndexExpr(index) => index.base().as_ref() == Some(expr),
                _ => false,
            }
        }
        None => false,
    };
    if needs_parens {
        acc.push(InlayHint {
            range: expr.syntax().text_range(),
            kind: InlayKind::OpeningParenthesis,
            label: "(".into(),
            tooltip: None,
        });
    }
    for adjustment in adjustments.into_iter().rev() {
        if adjustment.source == adjustment.target {
            continue;
        }

        // FIXME: Add some nicer tooltips to each of these
        let text = match adjustment.kind {
            Adjust::NeverToAny if config.adjustment_hints == AdjustmentHints::Always => {
                "<never-to-any>"
            }
            Adjust::Deref(None) => "*",
            Adjust::Deref(Some(OverloadedDeref(Mutability::Mut))) => "*",
            Adjust::Deref(Some(OverloadedDeref(Mutability::Shared))) => "*",
            Adjust::Borrow(AutoBorrow::Ref(Mutability::Shared)) => "&",
            Adjust::Borrow(AutoBorrow::Ref(Mutability::Mut)) => "&mut ",
            Adjust::Borrow(AutoBorrow::RawPtr(Mutability::Shared)) => "&raw const ",
            Adjust::Borrow(AutoBorrow::RawPtr(Mutability::Mut)) => "&raw mut ",
            // some of these could be represented via `as` casts, but that's not too nice and
            // handling everything as a prefix expr makes the `(` and `)` insertion easier
            Adjust::Pointer(cast) if config.adjustment_hints == AdjustmentHints::Always => {
                match cast {
                    PointerCast::ReifyFnPointer => "<fn-item-to-fn-pointer>",
                    PointerCast::UnsafeFnPointer => "<safe-fn-pointer-to-unsafe-fn-pointer>",
                    PointerCast::ClosureFnPointer(Safety::Unsafe) => {
                        "<closure-to-unsafe-fn-pointer>"
                    }
                    PointerCast::ClosureFnPointer(Safety::Safe) => "<closure-to-fn-pointer>",
                    PointerCast::MutToConstPointer => "<mut-ptr-to-const-ptr>",
                    PointerCast::ArrayToPointer => "<array-ptr-to-element-ptr>",
                    PointerCast::Unsize => "<unsize>",
                }
            }
            _ => continue,
        };
        acc.push(InlayHint {
            range: expr.syntax().text_range(),
            kind: InlayKind::AdjustmentHint,
            label: text.into(),
            tooltip: None,
        });
    }
    if needs_parens {
        acc.push(InlayHint {
            range: expr.syntax().text_range(),
            kind: InlayKind::ClosingParenthesis,
            label: ")".into(),
            tooltip: None,
        });
    }
    Some(())
}

fn is_inside_unsafe(sema: &Semantics<'_, RootDatabase>, node: &SyntaxNode) -> bool {
    let item_or_variant = |ancestor: SyntaxNode| {
        if ast::Item::can_cast(ancestor.kind()) {
            ast::Item::cast(ancestor).map(Either::Left)
        } else {
            ast::Variant::cast(ancestor).map(Either::Right)
        }
    };
    let Some(enclosing_item) = node.ancestors().find_map(item_or_variant) else { return false };

    let def = match &enclosing_item {
        Either::Left(ast::Item::Fn(it)) => {
            sema.to_def(it).map(<_>::into).map(hir::DefWithBodyId::FunctionId)
        }
        Either::Left(ast::Item::Const(it)) => {
            sema.to_def(it).map(<_>::into).map(hir::DefWithBodyId::ConstId)
        }
        Either::Left(ast::Item::Static(it)) => {
            sema.to_def(it).map(<_>::into).map(hir::DefWithBodyId::StaticId)
        }
        Either::Left(_) => None,
        Either::Right(it) => sema.to_def(it).map(<_>::into).map(hir::DefWithBodyId::VariantId),
    };
    let Some(def) = def else { return false };
    let enclosing_node = enclosing_item.as_ref().either(|i| i.syntax(), |v| v.syntax());

    if ast::Fn::cast(enclosing_node.clone()).and_then(|f| f.unsafe_token()).is_some() {
        return true;
    }

    let (body, source_map) = sema.db.body_with_source_map(def);

    let file_id = sema.hir_file_for(node);

    let Some(mut parent) = node.parent() else { return false };
    loop {
        if &parent == enclosing_node {
            break false;
        }

        if let Some(parent) = ast::Expr::cast(parent.clone()) {
            if let Some(expr_id) = source_map.node_expr(InFile { file_id, value: &parent }) {
                if let hir::Expr::Unsafe { .. } = body[expr_id] {
                    break true;
                }
            }
        }

        let Some(parent_) = parent.parent() else { break false };
        parent = parent_;
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        inlay_hints::tests::{check_with_config, DISABLED_CONFIG},
        AdjustmentHints, InlayHintsConfig,
    };

    #[test]
    fn adjustment_hints() {
        check_with_config(
            InlayHintsConfig { adjustment_hints: AdjustmentHints::Always, ..DISABLED_CONFIG },
            r#"
//- minicore: coerce_unsized
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
    let _: fn()        = || {};
                       //^^^^^<closure-to-fn-pointer>
    let _: unsafe fn() = || {};
                       //^^^^^<closure-to-unsafe-fn-pointer>
    let _: *const u32  = &mut 0u32 as *mut u32;
                       //^^^^^^^^^^^^^^^^^^^^^<mut-ptr-to-const-ptr>
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
}

#[derive(Copy, Clone)]
struct Struct;
impl Struct {
    fn consume(self) {}
    fn by_ref(&self) {}
    fn by_ref_mut(&mut self) {}
}
trait Trait {}
impl Trait for Struct {}
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
}
