//! Implementation of "closure return type" inlay hints.
use ide_db::{base_db::FileId, famous_defs::FamousDefs};
use syntax::ast::{self, AstNode};

use crate::{
    inlay_hints::closure_has_block_body, ClosureReturnTypeHints, InlayHint, InlayHintsConfig,
    InlayKind,
};

use super::label_of_ty;

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

    if !closure_has_block_body(&closure)
        && config.closure_return_type_hints == ClosureReturnTypeHints::WithBlock
    {
        return None;
    }

    let param_list = closure.param_list()?;

    let closure = sema.descend_node_into_attributes(closure).pop()?;
    let ty = sema.type_of_expr(&ast::Expr::ClosureExpr(closure))?.adjusted();
    let callable = ty.as_callable(sema.db)?;
    let ty = callable.return_type();
    if ty.is_unit() {
        return None;
    }
    acc.push(InlayHint {
        range: param_list.syntax().text_range(),
        kind: InlayKind::ClosureReturnType,
        label: label_of_ty(famous_defs, config, ty)?,
    });
    Some(())
}
