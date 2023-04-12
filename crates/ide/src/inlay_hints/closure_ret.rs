//! Implementation of "closure return type" inlay hints.
//!
//! Tests live in [`bind_pat`][super::bind_pat] module.
use ide_db::{base_db::FileId, famous_defs::FamousDefs};
use syntax::ast::{self, AstNode};

use crate::{
    inlay_hints::{closure_has_block_body, label_of_ty, ty_to_text_edit},
    ClosureReturnTypeHints, InlayHint, InlayHintsConfig, InlayKind,
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

    if closure.ret_type().is_some() {
        return None;
    }

    let has_block_body = closure_has_block_body(&closure);
    if !has_block_body && config.closure_return_type_hints == ClosureReturnTypeHints::WithBlock {
        return None;
    }

    let param_list = closure.param_list()?;

    let closure = sema.descend_node_into_attributes(closure).pop()?;
    let ty = sema.type_of_expr(&ast::Expr::ClosureExpr(closure.clone()))?.adjusted();
    let callable = ty.as_callable(sema.db)?;
    let ty = callable.return_type();
    if ty.is_unit() {
        return None;
    }

    // FIXME?: We could provide text edit to insert braces for closures with non-block body.
    let text_edit = if has_block_body {
        ty_to_text_edit(
            sema,
            closure.syntax(),
            &ty,
            param_list.syntax().text_range().end(),
            String::from(" -> "),
        )
    } else {
        None
    };

    acc.push(InlayHint {
        range: param_list.syntax().text_range(),
        kind: InlayKind::ClosureReturnType,
        label: label_of_ty(famous_defs, config, ty)?,
        text_edit,
    });
    Some(())
}
