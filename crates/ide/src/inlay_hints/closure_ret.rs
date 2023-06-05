//! Implementation of "closure return type" inlay hints.
//!
//! Tests live in [`bind_pat`][super::bind_pat] module.
use ide_db::{base_db::FileId, famous_defs::FamousDefs};
use syntax::ast::{self, AstNode};

use crate::{
    inlay_hints::{closure_has_block_body, label_of_ty, ty_to_text_edit},
    ClosureReturnTypeHints, InlayHint, InlayHintPosition, InlayHintsConfig, InlayKind,
};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    famous_defs @ FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    _file_id: FileId,
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

    let closure = sema.descend_node_into_attributes(closure).pop()?;
    let ty = sema.type_of_expr(&ast::Expr::ClosureExpr(closure.clone()))?.adjusted();
    let callable = ty.as_callable(sema.db)?;
    let ty = callable.return_type();
    if arrow.is_none() && ty.is_unit() {
        return None;
    }

    let mut label = label_of_ty(famous_defs, config, &ty)?;

    if arrow.is_none() {
        label.prepend_str(" -> ");
    }
    // FIXME?: We could provide text edit to insert braces for closures with non-block body.
    let text_edit = if has_block_body {
        ty_to_text_edit(
            sema,
            closure.syntax(),
            &ty,
            arrow
                .as_ref()
                .map_or_else(|| param_list.syntax().text_range(), |t| t.text_range())
                .end(),
            if arrow.is_none() { String::from(" -> ") } else { String::new() },
        )
    } else {
        None
    };

    acc.push(InlayHint {
        range: param_list.syntax().text_range(),
        kind: InlayKind::Type,
        label,
        text_edit,
        position: InlayHintPosition::After,
        pad_left: false,
        pad_right: false,
    });
    Some(())
}

#[cfg(test)]
mod tests {
    use crate::inlay_hints::tests::{check_with_config, DISABLED_CONFIG};

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
