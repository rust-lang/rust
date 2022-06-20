//! Completes constants and paths in unqualified patterns.

use hir::{db::DefDatabase, AssocItem, ScopeDef};
use ide_db::FxHashSet;
use syntax::ast::Pat;

use crate::{
    context::{PathCompletionCtx, PatternContext, PatternRefutability, Qualified},
    CompletionContext, Completions,
};

/// Completes constants and paths in unqualified patterns.
pub(crate) fn complete_pattern(
    acc: &mut Completions,
    ctx: &CompletionContext,
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
    ctx.process_all_names(&mut |name, res| {
        let add_simple_path = match res {
            hir::ScopeDef::ModuleDef(def) => match def {
                hir::ModuleDef::Adt(hir::Adt::Struct(strukt)) => {
                    acc.add_struct_pat(ctx, pattern_ctx, strukt, Some(name.clone()));
                    true
                }
                hir::ModuleDef::Variant(variant)
                    if refutable || single_variant_enum(variant.parent_enum(ctx.db)) =>
                {
                    acc.add_variant_pat(ctx, pattern_ctx, variant, Some(name.clone()));
                    true
                }
                hir::ModuleDef::Adt(hir::Adt::Enum(e)) => refutable || single_variant_enum(e),
                hir::ModuleDef::Const(..) => refutable,
                hir::ModuleDef::Module(..) => true,
                hir::ModuleDef::Macro(mac) if mac.is_fn_like(ctx.db) => {
                    return acc.add_macro(
                        ctx,
                        &PathCompletionCtx {
                            has_call_parens: false,
                            has_macro_bang: false,
                            qualified: Qualified::No,
                            // FIXME
                            path: syntax::ast::make::ext::ident_path("dummy__"),
                            parent: None,
                            kind: crate::context::PathKind::Pat { pat_ctx: pattern_ctx.clone() },
                            has_type_args: false,
                            use_tree_parent: false,
                        },
                        mac,
                        name,
                    );
                }
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
            acc.add_resolution_simple(ctx, name, res);
        }
    });
}

pub(crate) fn complete_pattern_path(
    acc: &mut Completions,
    ctx: &CompletionContext,
    path_ctx @ PathCompletionCtx { qualified, .. }: &PathCompletionCtx,
) {
    match qualified {
        Qualified::With { resolution: Some(resolution), is_super_chain, .. } => {
            if *is_super_chain {
                acc.add_keyword(ctx, "super::");
            }

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
                            acc.add_path_resolution(ctx, path_ctx, name, def);
                        }
                    }
                }
                res @ (hir::PathResolution::TypeParam(_)
                | hir::PathResolution::SelfType(_)
                | hir::PathResolution::Def(hir::ModuleDef::Adt(hir::Adt::Struct(_)))
                | hir::PathResolution::Def(hir::ModuleDef::Adt(hir::Adt::Enum(_)))
                | hir::PathResolution::Def(hir::ModuleDef::Adt(hir::Adt::Union(_)))
                | hir::PathResolution::Def(hir::ModuleDef::BuiltinType(_))) => {
                    let ty = match res {
                        hir::PathResolution::TypeParam(param) => param.ty(ctx.db),
                        hir::PathResolution::SelfType(impl_def) => impl_def.self_ty(ctx.db),
                        hir::PathResolution::Def(hir::ModuleDef::Adt(hir::Adt::Struct(s))) => {
                            s.ty(ctx.db)
                        }
                        hir::PathResolution::Def(hir::ModuleDef::Adt(hir::Adt::Enum(e))) => {
                            cov_mark::hit!(enum_plain_qualified_use_tree);
                            e.variants(ctx.db).into_iter().for_each(|variant| {
                                acc.add_enum_variant(ctx, path_ctx, variant, None)
                            });
                            e.ty(ctx.db)
                        }
                        hir::PathResolution::Def(hir::ModuleDef::Adt(hir::Adt::Union(u))) => {
                            u.ty(ctx.db)
                        }
                        hir::PathResolution::Def(hir::ModuleDef::BuiltinType(ty)) => ty.ty(ctx.db),
                        _ => return,
                    };

                    let mut seen = FxHashSet::default();
                    ty.iterate_path_candidates(
                        ctx.db,
                        &ctx.scope,
                        &ctx.scope.visible_traits().0,
                        Some(ctx.module),
                        None,
                        |item| {
                            match item {
                                AssocItem::TypeAlias(ta) => {
                                    // We might iterate candidates of a trait multiple times here, so deduplicate them.
                                    if seen.insert(item) {
                                        acc.add_type_alias(ctx, ta);
                                    }
                                }
                                AssocItem::Const(c) => {
                                    if seen.insert(item) {
                                        acc.add_const(ctx, c);
                                    }
                                }
                                _ => {}
                            }
                            None::<()>
                        },
                    );
                }
                _ => {}
            }
        }
        // qualifier can only be none here if we are in a TuplePat or RecordPat in which case special characters have to follow the path
        Qualified::Absolute => acc.add_crate_roots(ctx),
        Qualified::No => {
            ctx.process_all_names(&mut |name, res| {
                // FIXME: properly filter here
                if let ScopeDef::ModuleDef(_) = res {
                    acc.add_path_resolution(ctx, path_ctx, name, res);
                }
            });

            acc.add_nameref_keywords_with_colon(ctx);
        }
        Qualified::Infer | Qualified::With { .. } => {}
    }
}
