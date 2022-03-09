//! Completes constants and paths in unqualified patterns.

use hir::{db::DefDatabase, AssocItem, ScopeDef};
use rustc_hash::FxHashSet;
use syntax::ast::Pat;

use crate::{
    context::{PathCompletionCtx, PathQualifierCtx, PatternRefutability},
    CompletionContext, Completions,
};

/// Completes constants and paths in unqualified patterns.
pub(crate) fn complete_pattern(acc: &mut Completions, ctx: &CompletionContext) {
    let patctx = match &ctx.pattern_ctx {
        Some(ctx) => ctx,
        _ => return,
    };
    let refutable = patctx.refutability == PatternRefutability::Refutable;

    if let Some(path_ctx) = &ctx.path_context {
        pattern_path_completion(acc, ctx, path_ctx);
        return;
    }

    match patctx.parent_pat.as_ref() {
        Some(Pat::RangePat(_) | Pat::BoxPat(_)) => (),
        Some(Pat::RefPat(r)) => {
            if r.mut_token().is_none() {
                acc.add_keyword(ctx, "mut");
            }
        }
        _ => {
            let tok = ctx.token.text_range().start();
            match (patctx.ref_token.as_ref(), patctx.mut_token.as_ref()) {
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

    let single_variant_enum = |enum_: hir::Enum| ctx.db.enum_data(enum_.into()).variants.len() == 1;

    if let Some(hir::Adt::Enum(e)) =
        ctx.expected_type.as_ref().and_then(|ty| ty.strip_references().as_adt())
    {
        if refutable || single_variant_enum(e) {
            super::enum_variants_with_paths(acc, ctx, e, |acc, ctx, variant, path| {
                acc.add_qualified_variant_pat(ctx, variant, path.clone());
                acc.add_qualified_enum_variant(ctx, variant, path);
            });
        }
    }

    // FIXME: ideally, we should look at the type we are matching against and
    // suggest variants + auto-imports
    ctx.process_all_names(&mut |name, res| {
        let add_resolution = match res {
            hir::ScopeDef::ModuleDef(def) => match def {
                hir::ModuleDef::Adt(hir::Adt::Struct(strukt)) => {
                    acc.add_struct_pat(ctx, strukt, Some(name.clone()));
                    true
                }
                hir::ModuleDef::Variant(variant)
                    if refutable || single_variant_enum(variant.parent_enum(ctx.db)) =>
                {
                    acc.add_variant_pat(ctx, variant, Some(name.clone()));
                    true
                }
                hir::ModuleDef::Adt(hir::Adt::Enum(e)) => refutable || single_variant_enum(e),
                hir::ModuleDef::Const(..) | hir::ModuleDef::Module(..) => refutable,
                hir::ModuleDef::Macro(mac) => mac.is_fn_like(ctx.db),
                _ => false,
            },
            hir::ScopeDef::ImplSelfType(impl_) => match impl_.self_ty(ctx.db).as_adt() {
                Some(hir::Adt::Struct(strukt)) => {
                    acc.add_struct_pat(ctx, strukt, Some(name.clone()));
                    true
                }
                Some(hir::Adt::Enum(_)) => refutable,
                _ => true,
            },
            _ => false,
        };
        if add_resolution {
            acc.add_resolution(ctx, name, res);
        }
    });
}

fn pattern_path_completion(
    acc: &mut Completions,
    ctx: &CompletionContext,
    PathCompletionCtx { qualifier, is_absolute_path, .. }: &PathCompletionCtx,
) {
    match qualifier {
        Some(PathQualifierCtx { resolution, is_super_chain, .. }) => {
            if *is_super_chain {
                acc.add_keyword(ctx, "super::");
            }

            let resolution = match resolution {
                Some(it) => it,
                None => return,
            };

            match resolution {
                hir::PathResolution::Def(hir::ModuleDef::Module(module)) => {
                    let module_scope = module.scope(ctx.db, ctx.module);
                    for (name, def) in module_scope {
                        let add_resolution = match def {
                            ScopeDef::ModuleDef(hir::ModuleDef::Macro(mac)) => {
                                mac.is_fn_like(ctx.db)
                            }
                            ScopeDef::ModuleDef(_) => true,
                            _ => false,
                        };

                        if add_resolution {
                            acc.add_resolution(ctx, name, def);
                        }
                    }
                }
                hir::PathResolution::Def(hir::ModuleDef::Adt(hir::Adt::Enum(e))) => {
                    cov_mark::hit!(enum_plain_qualified_use_tree);
                    e.variants(ctx.db)
                        .into_iter()
                        .for_each(|variant| acc.add_enum_variant(ctx, variant, None));
                }
                res @ (hir::PathResolution::TypeParam(_) | hir::PathResolution::SelfType(_)) => {
                    let ty = match res {
                        hir::PathResolution::TypeParam(param) => param.ty(ctx.db),
                        hir::PathResolution::SelfType(impl_def) => impl_def.self_ty(ctx.db),
                        _ => return,
                    };

                    if let Some(hir::Adt::Enum(e)) = ty.as_adt() {
                        e.variants(ctx.db)
                            .into_iter()
                            .for_each(|variant| acc.add_enum_variant(ctx, variant, None));
                    }

                    let traits_in_scope = ctx.scope.visible_traits();
                    let mut seen = FxHashSet::default();
                    ty.iterate_path_candidates(
                        ctx.db,
                        &ctx.scope,
                        &traits_in_scope,
                        ctx.module,
                        None,
                        |item| {
                            // Note associated consts cannot be referenced in patterns
                            if let AssocItem::TypeAlias(ta) = item {
                                // We might iterate candidates of a trait multiple times here, so deduplicate them.
                                if seen.insert(item) {
                                    acc.add_type_alias(ctx, ta);
                                }
                            }
                            None::<()>
                        },
                    );
                }
                _ => {}
            }
        }
        // qualifier can only be none here if we are in a TuplePat or RecordPat in which case special characters have to follow the path
        None if *is_absolute_path => acc.add_crate_roots(ctx),
        None => {
            cov_mark::hit!(unqualified_path_only_modules_in_import);
            ctx.process_all_names(&mut |name, res| {
                if let ScopeDef::ModuleDef(hir::ModuleDef::Module(_)) = res {
                    acc.add_resolution(ctx, name, res);
                }
            });
            acc.add_nameref_keywords(ctx);
        }
    }
}
