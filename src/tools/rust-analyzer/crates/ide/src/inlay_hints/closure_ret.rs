//! Implementation of "closure return type" inlay hints.
//!
//! Tests live in [`bind_pat`][super::bind_pat] module.
use hir::{DisplayTarget, HirDisplay};
use ide_db::{famous_defs::FamousDefs, text_edit::TextEdit};
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
    let text_edit = if has_block_body {
        ty_to_text_edit(
            sema,
            config,
            descended_closure.syntax(),
            &ty,
            arrow
                .as_ref()
                .map_or_else(|| param_list.syntax().text_range(), |t| t.text_range())
                .end(),
            if arrow.is_none() { " -> " } else { "" },
        )
    } else {
        Some(config.lazy_text_edit(|| {
            let body = closure.body();
            let body_range = match body {
                Some(body) => body.syntax().text_range(),
                None => return TextEdit::builder().finish(),
            };
            let mut builder = TextEdit::builder();
            let insert_pos = param_list.syntax().text_range().end();

            let rendered = match sema.scope(descended_closure.syntax()).and_then(|scope| {
                ty.display_source_code(scope.db, scope.module().into(), false).ok()
            }) {
                Some(rendered) => rendered,
                None => return TextEdit::builder().finish(),
            };

            let arrow_text = if arrow.is_none() { " -> ".to_owned() } else { "".to_owned() };
            builder.insert(insert_pos, arrow_text);
            builder.insert(insert_pos, rendered);
            builder.insert(body_range.start(), "{ ".to_owned());
            builder.insert(body_range.end(), " }".to_owned());

            builder.finish()
        }))
    };

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
