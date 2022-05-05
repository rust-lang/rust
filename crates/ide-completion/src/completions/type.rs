//! Completion of names from the current scope in type position.

use hir::ScopeDef;
use syntax::{ast, AstNode};

use crate::{
    context::{PathCompletionCtx, PathKind, PathQualifierCtx},
    patterns::ImmediateLocation,
    CompletionContext, Completions,
};

pub(crate) fn complete_type_path(acc: &mut Completions, ctx: &CompletionContext) {
    let _p = profile::span("complete_type_path");
    if ctx.is_path_disallowed() {
        return;
    }

    let (&is_absolute_path, qualifier) = match &ctx.path_context {
        Some(PathCompletionCtx {
            kind: Some(PathKind::Type), is_absolute_path, qualifier, ..
        }) => (is_absolute_path, qualifier),
        _ => return,
    };

    match qualifier {
        Some(PathQualifierCtx { .. }) => return,
        None if is_absolute_path => acc.add_crate_roots(ctx),
        None => {
            acc.add_nameref_keywords_with_colon(ctx);
            if let Some(ImmediateLocation::TypeBound) = &ctx.completion_location {
                ctx.process_all_names(&mut |name, res| {
                    let add_resolution = match res {
                        ScopeDef::ModuleDef(hir::ModuleDef::Macro(mac)) => mac.is_fn_like(ctx.db),
                        ScopeDef::ModuleDef(
                            hir::ModuleDef::Trait(_) | hir::ModuleDef::Module(_),
                        ) => true,
                        _ => false,
                    };
                    if add_resolution {
                        acc.add_resolution(ctx, name, res);
                    }
                });
                return;
            }
            if let Some(ImmediateLocation::GenericArgList(arg_list)) = &ctx.completion_location {
                if let Some(path_seg) = arg_list.syntax().parent().and_then(ast::PathSegment::cast)
                {
                    if let Some(hir::PathResolution::Def(hir::ModuleDef::Trait(trait_))) =
                        ctx.sema.resolve_path(&path_seg.parent_path())
                    {
                        trait_.items(ctx.sema.db).into_iter().for_each(|it| {
                            if let hir::AssocItem::TypeAlias(alias) = it {
                                acc.add_type_alias_with_eq(ctx, alias)
                            }
                        });
                    }
                }
            }
            ctx.process_all_names(&mut |name, def| {
                use hir::{GenericParam::*, ModuleDef::*};
                let add_resolution = match def {
                    ScopeDef::GenericParam(LifetimeParam(_)) | ScopeDef::Label(_) => false,
                    // no values in type places
                    ScopeDef::ModuleDef(Function(_) | Variant(_) | Static(_))
                    | ScopeDef::Local(_) => false,
                    // unless its a constant in a generic arg list position
                    ScopeDef::ModuleDef(Const(_)) | ScopeDef::GenericParam(ConstParam(_)) => {
                        ctx.expects_generic_arg()
                    }
                    ScopeDef::ImplSelfType(_) => {
                        !ctx.previous_token_is(syntax::T![impl])
                            && !ctx.previous_token_is(syntax::T![for])
                    }
                    // Don't suggest attribute macros and derives.
                    ScopeDef::ModuleDef(Macro(mac)) => mac.is_fn_like(ctx.db),
                    // Type things are fine
                    ScopeDef::ModuleDef(
                        BuiltinType(_) | Adt(_) | Module(_) | Trait(_) | TypeAlias(_),
                    )
                    | ScopeDef::AdtSelfType(_)
                    | ScopeDef::Unknown
                    | ScopeDef::GenericParam(TypeParam(_)) => true,
                };
                if add_resolution {
                    acc.add_resolution(ctx, name, def);
                }
            });
        }
    }
}
