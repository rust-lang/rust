//! Implementation of "closure return type" inlay hints.
//!
//! Tests live in [`bind_pat`][super::bind_pat] module.
use hir::DisplayTarget;
use ide_db::{famous_defs::FamousDefs, text_edit::TextEditBuilder};
use syntax::ast::{self, AstNode};

use crate::{
    ClosureReturnTypeHints, InlayHint, InlayHintPosition, InlayHintsConfig, InlayKind,
    inlay_hints::{closure_has_block_body, label_of_ty, ty_to_text_edit},
};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    famous_defs @ FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    display_target: DisplayTarget,
    closure: ast::ClosureExpr,
) -> Option<()> {
    if config.closure_return_type_hints == ClosureReturnTypeHints::Never {
        return None;
    }

    let ret_type = closure.ret_type().map(|rt| (rt.thin_arrow_token(), rt.ty().is_some()));
    let arrow = match ret_type {
        Some((_, true)) => return None,
        Some((arrow, _)) => arrow,
        None => None,
    };

    let has_block_body = closure_has_block_body(&closure);
    if !has_block_body && config.closure_return_type_hints == ClosureReturnTypeHints::WithBlock {
        return None;
    }

    let param_list = closure.param_list()?;

    let resolve_parent = Some(closure.syntax().text_range());
    let descended_closure = sema.descend_node_into_attributes(closure.clone()).pop()?;
    let ty = sema.type_of_expr(&ast::Expr::ClosureExpr(descended_closure.clone()))?.adjusted();
    let callable = ty.as_callable(sema.db)?;
    let ty = callable.return_type();
    if arrow.is_none() && ty.is_unit() {
        return None;
    }

    let mut label = label_of_ty(famous_defs, config, &ty, display_target)?;

    if arrow.is_none() {
        label.prepend_str(" -> ");
    }

    let offset_to_insert_ty =
        arrow.as_ref().map_or_else(|| param_list.syntax().text_range(), |t| t.text_range()).end();

    // Insert braces if necessary
    let insert_braces = |builder: &mut TextEditBuilder| {
        if !has_block_body && let Some(range) = closure.body().map(|b| b.syntax().text_range()) {
            builder.insert(range.start(), "{ ".to_owned());
            builder.insert(range.end(), " }".to_owned());
        }
    };

    let text_edit = ty_to_text_edit(
        sema,
        config,
        descended_closure.syntax(),
        &ty,
        offset_to_insert_ty,
        &insert_braces,
        if arrow.is_none() { " -> " } else { "" },
    );

    acc.push(InlayHint {
        range: param_list.syntax().text_range(),
        kind: InlayKind::Type,
        label,
        text_edit,
        position: InlayHintPosition::After,
        pad_left: false,
        pad_right: false,
        resolve_parent,
    });
    Some(())
}

#[cfg(test)]
mod tests {
    use crate::inlay_hints::tests::{DISABLED_CONFIG, check_with_config};

    use super::*;

    #[test]
    fn return_type_hints_for_closure_without_block() {
        check_with_config(
            InlayHintsConfig {
                closure_return_type_hints: ClosureReturnTypeHints::Always,
                ..DISABLED_CONFIG
            },
            r#"
fn main() {
    let a = || { 0 };
          //^^ -> i32
    let b = || 0;
          //^^ -> i32
}"#,
        );
    }
}
