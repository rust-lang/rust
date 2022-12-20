//! Implementation of "closure return type" inlay hints.
use hir::{HirDisplay, Semantics};
use ide_db::{base_db::FileId, famous_defs::FamousDefs, RootDatabase};
use syntax::ast::{self, AstNode};

use crate::{
    inlay_hints::{closure_has_block_body, hint_iterator},
    ClosureReturnTypeHints, InlayHint, InlayHintsConfig, InlayKind, InlayTooltip,
};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<'_, RootDatabase>,
    famous_defs: &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    file_id: FileId,
    closure: ast::ClosureExpr,
) -> Option<()> {
    if config.closure_return_type_hints == ClosureReturnTypeHints::Never {
        return None;
    }

    if closure.ret_type().is_some() {
        return None;
    }

    if !closure_has_block_body(&closure)
        && config.closure_return_type_hints == ClosureReturnTypeHints::WithBlock
    {
        return None;
    }

    let param_list = closure.param_list()?;

    let closure = sema.descend_node_into_attributes(closure.clone()).pop()?;
    let ty = sema.type_of_expr(&ast::Expr::ClosureExpr(closure))?.adjusted();
    let callable = ty.as_callable(sema.db)?;
    let ty = callable.return_type();
    if ty.is_unit() {
        return None;
    }
    acc.push(InlayHint {
        range: param_list.syntax().text_range(),
        kind: InlayKind::ClosureReturnTypeHint,
        label: hint_iterator(sema, &famous_defs, config, &ty)
            .unwrap_or_else(|| ty.display_truncated(sema.db, config.max_length).to_string())
            .into(),
        tooltip: Some(InlayTooltip::HoverRanged(file_id, param_list.syntax().text_range())),
    });
    Some(())
}
