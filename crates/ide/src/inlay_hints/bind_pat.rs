use hir::{HirDisplay, Semantics, TypeInfo};
use ide_db::{base_db::FileId, famous_defs::FamousDefs, RootDatabase};

use itertools::Itertools;
use syntax::{
    ast::{self, AstNode, HasName},
    match_ast,
};

use crate::{
    inlay_hints::{closure_has_block_body, hint_iterator},
    InlayHint, InlayHintsConfig, InlayKind, InlayTooltip,
};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<'_, RootDatabase>,
    config: &InlayHintsConfig,
    file_id: FileId,
    pat: &ast::IdentPat,
) -> Option<()> {
    if !config.type_hints {
        return None;
    }

    let descended = sema.descend_node_into_attributes(pat.clone()).pop();
    let desc_pat = descended.as_ref().unwrap_or(pat);
    let ty = sema.type_of_pat(&desc_pat.clone().into())?.original;

    if should_not_display_type_hint(sema, config, pat, &ty) {
        return None;
    }

    let krate = sema.scope(desc_pat.syntax())?.krate();
    let famous_defs = FamousDefs(sema, krate);
    let label = hint_iterator(sema, &famous_defs, config, &ty);

    let label = match label {
        Some(label) => label,
        None => {
            let ty_name = ty.display_truncated(sema.db, config.max_length).to_string();
            if config.hide_named_constructor_hints
                && is_named_constructor(sema, pat, &ty_name).is_some()
            {
                return None;
            }
            ty_name
        }
    };

    acc.push(InlayHint {
        range: match pat.name() {
            Some(name) => name.syntax().text_range(),
            None => pat.syntax().text_range(),
        },
        kind: InlayKind::TypeHint,
        label: label.into(),
        tooltip: pat
            .name()
            .map(|it| it.syntax().text_range())
            .map(|it| InlayTooltip::HoverRanged(file_id, it)),
    });

    Some(())
}

fn should_not_display_type_hint(
    sema: &Semantics<'_, RootDatabase>,
    config: &InlayHintsConfig,
    bind_pat: &ast::IdentPat,
    pat_ty: &hir::Type,
) -> bool {
    let db = sema.db;

    if pat_ty.is_unknown() {
        return true;
    }

    if let Some(hir::Adt::Struct(s)) = pat_ty.as_adt() {
        if s.fields(db).is_empty() && s.name(db).to_smol_str() == bind_pat.to_string() {
            return true;
        }
    }

    if config.hide_closure_initialization_hints {
        if let Some(parent) = bind_pat.syntax().parent() {
            if let Some(it) = ast::LetStmt::cast(parent.clone()) {
                if let Some(ast::Expr::ClosureExpr(closure)) = it.initializer() {
                    if closure_has_block_body(&closure) {
                        return true;
                    }
                }
            }
        }
    }

    for node in bind_pat.syntax().ancestors() {
        match_ast! {
            match node {
                ast::LetStmt(it) => return it.ty().is_some(),
                // FIXME: We might wanna show type hints in parameters for non-top level patterns as well
                ast::Param(it) => return it.ty().is_some(),
                ast::MatchArm(_) => return pat_is_enum_variant(db, bind_pat, pat_ty),
                ast::LetExpr(_) => return pat_is_enum_variant(db, bind_pat, pat_ty),
                ast::IfExpr(_) => return false,
                ast::WhileExpr(_) => return false,
                ast::ForExpr(it) => {
                    // We *should* display hint only if user provided "in {expr}" and we know the type of expr (and it's not unit).
                    // Type of expr should be iterable.
                    return it.in_token().is_none() ||
                        it.iterable()
                            .and_then(|iterable_expr| sema.type_of_expr(&iterable_expr))
                            .map(TypeInfo::original)
                            .map_or(true, |iterable_ty| iterable_ty.is_unknown() || iterable_ty.is_unit())
                },
                _ => (),
            }
        }
    }
    false
}

fn is_named_constructor(
    sema: &Semantics<'_, RootDatabase>,
    pat: &ast::IdentPat,
    ty_name: &str,
) -> Option<()> {
    let let_node = pat.syntax().parent()?;
    let expr = match_ast! {
        match let_node {
            ast::LetStmt(it) => it.initializer(),
            ast::LetExpr(it) => it.expr(),
            _ => None,
        }
    }?;

    let expr = sema.descend_node_into_attributes(expr.clone()).pop().unwrap_or(expr);
    // unwrap postfix expressions
    let expr = match expr {
        ast::Expr::TryExpr(it) => it.expr(),
        ast::Expr::AwaitExpr(it) => it.expr(),
        expr => Some(expr),
    }?;
    let expr = match expr {
        ast::Expr::CallExpr(call) => match call.expr()? {
            ast::Expr::PathExpr(path) => path,
            _ => return None,
        },
        ast::Expr::PathExpr(path) => path,
        _ => return None,
    };
    let path = expr.path()?;

    let callable = sema.type_of_expr(&ast::Expr::PathExpr(expr))?.original.as_callable(sema.db);
    let callable_kind = callable.map(|it| it.kind());
    let qual_seg = match callable_kind {
        Some(hir::CallableKind::Function(_) | hir::CallableKind::TupleEnumVariant(_)) => {
            path.qualifier()?.segment()
        }
        _ => path.segment(),
    }?;

    let ctor_name = match qual_seg.kind()? {
        ast::PathSegmentKind::Name(name_ref) => {
            match qual_seg.generic_arg_list().map(|it| it.generic_args()) {
                Some(generics) => format!("{}<{}>", name_ref, generics.format(", ")),
                None => name_ref.to_string(),
            }
        }
        ast::PathSegmentKind::Type { type_ref: Some(ty), trait_ref: None } => ty.to_string(),
        _ => return None,
    };
    (ctor_name == ty_name).then(|| ())
}

fn pat_is_enum_variant(db: &RootDatabase, bind_pat: &ast::IdentPat, pat_ty: &hir::Type) -> bool {
    if let Some(hir::Adt::Enum(enum_data)) = pat_ty.as_adt() {
        let pat_text = bind_pat.to_string();
        enum_data
            .variants(db)
            .into_iter()
            .map(|variant| variant.name(db).to_smol_str())
            .any(|enum_name| enum_name == pat_text)
    } else {
        false
    }
}
