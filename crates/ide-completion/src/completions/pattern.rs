//! Completes constants and paths in unqualified patterns.

use hir::{db::DefDatabase, AssocItem, ScopeDef};
use syntax::ast::Pat;

use crate::{
    context::{PathCompletionCtx, PatternContext, PatternRefutability, Qualified},
    CompletionContext, Completions,
};

/// Completes constants and paths in unqualified patterns.
pub(crate) fn complete_pattern(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    pattern_ctx: &PatternContext,
) {
    match pattern_ctx.parent_pat.as_ref() {
        Some(Pat::RangePat(_) | Pat::BoxPat(_)) => (),
        Some(Pat::RefPat(r)) => {
            if r.mut_token().is_none() {
                acc.add_keyword(ctx, "mut");
            }
        }
        _ => {
            let tok = ctx.token.text_range().start();
            match (pattern_ctx.ref_token.as_ref(), pattern_ctx.mut_token.as_ref()) {
                (None, None) => {
                    acc.add_keyword(ctx, "ref");
                    acc.add_keyword(ctx, "mut");
                }
                (None, Some(m)) if tok < m.text_range().start() => {
                    acc.add_keyword(ctx, "ref");
                }
                (Some(r), None) if tok > r.text_range().end() => {
                    acc.add_keyword(ctx, "mut");
                }
                _ => (),
            }
        }
    }

    if pattern_ctx.record_pat.is_some() {
        return;
    }

    let refutable = pattern_ctx.refutability == PatternRefutability::Refutable;
    let single_variant_enum = |enum_: hir::Enum| ctx.db.enum_data(enum_.into()).variants.len() == 1;

    if let Some(hir::Adt::Enum(e)) =
        ctx.expected_type.as_ref().and_then(|ty| ty.strip_references().as_adt())
    {
        if refutable || single_variant_enum(e) {
            super::enum_variants_with_paths(
                acc,
                ctx,
                e,
                &pattern_ctx.impl_,
                |acc, ctx, variant, path| {
                    acc.add_qualified_variant_pat(ctx, pattern_ctx, variant, path);
                },
            );
        }
    }

    // FIXME: ideally, we should look at the type we are matching against and
    // suggest variants + auto-imports
    ctx.process_all_names(&mut |name, res, _| {
        let add_simple_path = match res {
            hir::ScopeDef::ModuleDef(def) => match def {
                hir::ModuleDef::Adt(hir::Adt::Struct(strukt)) => {
                    acc.add_struct_pat(ctx, pattern_ctx, strukt, Some(name.clone()));
                    true
                }
                hir::ModuleDef::Variant(variant)
                    if refutable || single_variant_enum(variant.parent_enum(ctx.db)) =>
                {
                    acc.add_variant_pat(ctx, pattern_ctx, None, variant, Some(name.clone()));
                    true
                }
                hir::ModuleDef::Adt(hir::Adt::Enum(e)) => refutable || single_variant_enum(e),
                hir::ModuleDef::Const(..) => refutable,
                hir::ModuleDef::Module(..) => true,
                hir::ModuleDef::Macro(mac) => mac.is_fn_like(ctx.db),
                _ => false,
            },
            hir::ScopeDef::ImplSelfType(impl_) => match impl_.self_ty(ctx.db).as_adt() {
                Some(hir::Adt::Struct(strukt)) => {
                    acc.add_struct_pat(ctx, pattern_ctx, strukt, Some(name.clone()));
                    true
                }
                Some(hir::Adt::Enum(e)) => refutable || single_variant_enum(e),
                Some(hir::Adt::Union(_)) => true,
                _ => false,
            },
            ScopeDef::GenericParam(hir::GenericParam::ConstParam(_)) => true,
            ScopeDef::GenericParam(_)
            | ScopeDef::AdtSelfType(_)
            | ScopeDef::Local(_)
            | ScopeDef::Label(_)
            | ScopeDef::Unknown => false,
        };
        if add_simple_path {
            acc.add_pattern_resolution(ctx, pattern_ctx, name, res);
        }
    });
}

pub(crate) fn complete_pattern_path(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx @ PathCompletionCtx { qualified, .. }: &PathCompletionCtx,
) {
    match qualified {
        Qualified::With { resolution: Some(resolution), super_chain_len, .. } => {
            acc.add_super_keyword(ctx, *super_chain_len);

            match resolution {
                hir::PathResolution::Def(hir::ModuleDef::Module(module)) => {
                    let module_scope = module.scope(ctx.db, Some(ctx.module));
                    for (name, def) in module_scope {
                        let add_resolution = match def {
                            ScopeDef::ModuleDef(hir::ModuleDef::Macro(mac)) => {
                                mac.is_fn_like(ctx.db)
                            }
                            ScopeDef::ModuleDef(_) => true,
                            _ => false,
                        };

                        if add_resolution {
                            acc.add_path_resolution(ctx, path_ctx, name, def, vec![]);
                        }
                    }
                }
                res => {
                    let ty = match res {
                        hir::PathResolution::TypeParam(param) => param.ty(ctx.db),
                        hir::PathResolution::SelfType(impl_def) => impl_def.self_ty(ctx.db),
                        hir::PathResolution::Def(hir::ModuleDef::Adt(hir::Adt::Struct(s))) => {
                            s.ty(ctx.db)
                        }
                        hir::PathResolution::Def(hir::ModuleDef::Adt(hir::Adt::Enum(e))) => {
                            e.ty(ctx.db)
                        }
                        hir::PathResolution::Def(hir::ModuleDef::Adt(hir::Adt::Union(u))) => {
                            u.ty(ctx.db)
                        }
                        hir::PathResolution::Def(hir::ModuleDef::BuiltinType(ty)) => ty.ty(ctx.db),
                        hir::PathResolution::Def(hir::ModuleDef::TypeAlias(ty)) => ty.ty(ctx.db),
                        _ => return,
                    };

                    if let Some(hir::Adt::Enum(e)) = ty.as_adt() {
                        acc.add_enum_variants(ctx, path_ctx, e);
                    }

                    ctx.iterate_path_candidates(&ty, |item| match item {
                        AssocItem::TypeAlias(ta) => acc.add_type_alias(ctx, ta),
                        AssocItem::Const(c) => acc.add_const(ctx, c),
                        _ => {}
                    });
                }
            }
        }
        Qualified::Absolute => acc.add_crate_roots(ctx, path_ctx),
        Qualified::No => {
            // this will only be hit if there are brackets or braces, otherwise this will be parsed as an ident pattern
            ctx.process_all_names(&mut |name, res, doc_aliases| {
                // FIXME: we should check what kind of pattern we are in and filter accordingly
                let add_completion = match res {
                    ScopeDef::ModuleDef(hir::ModuleDef::Macro(mac)) => mac.is_fn_like(ctx.db),
                    ScopeDef::ModuleDef(hir::ModuleDef::Adt(_)) => true,
                    ScopeDef::ModuleDef(hir::ModuleDef::Variant(_)) => true,
                    ScopeDef::ModuleDef(hir::ModuleDef::Module(_)) => true,
                    ScopeDef::ImplSelfType(_) => true,
                    _ => false,
                };
                if add_completion {
                    acc.add_path_resolution(ctx, path_ctx, name, res, doc_aliases);
                }
            });

            acc.add_nameref_keywords_with_colon(ctx);
        }
        Qualified::TypeAnchor { .. } | Qualified::With { .. } => {}
    }
}
