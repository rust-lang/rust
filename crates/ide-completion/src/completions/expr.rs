//! Completion of names from the current scope in expression position.

use hir::ScopeDef;

use crate::{
    context::{PathCompletionCtx, PathKind, PathQualifierCtx},
    CompletionContext, Completions,
};

pub(crate) fn complete_expr_path(acc: &mut Completions, ctx: &CompletionContext) {
    let _p = profile::span("complete_expr_path");
    if ctx.is_path_disallowed() {
        return;
    }

    let (&is_absolute_path, qualifier) = match &ctx.path_context {
        Some(PathCompletionCtx {
            kind: Some(PathKind::Expr), is_absolute_path, qualifier, ..
        }) => (is_absolute_path, qualifier),
        _ => return,
    };

    match qualifier {
        Some(PathQualifierCtx { .. }) => return,
        None if is_absolute_path => acc.add_crate_roots(ctx),
        None => {
            acc.add_nameref_keywords_with_colon(ctx);
            if let Some(hir::Adt::Enum(e)) =
                ctx.expected_type.as_ref().and_then(|ty| ty.strip_references().as_adt())
            {
                super::enum_variants_with_paths(acc, ctx, e, |acc, ctx, variant, path| {
                    acc.add_qualified_enum_variant(ctx, variant, path)
                });
            }
            ctx.process_all_names(&mut |name, def| {
                use hir::{GenericParam::*, ModuleDef::*};
                let add_resolution = match def {
                    ScopeDef::GenericParam(LifetimeParam(_)) | ScopeDef::Label(_) => false,
                    // Don't suggest attribute macros and derives.
                    ScopeDef::ModuleDef(Macro(mac)) => mac.is_fn_like(ctx.db),
                    _ => true,
                };
                if add_resolution {
                    acc.add_resolution(ctx, name, def);
                }
            });
        }
    }
}
