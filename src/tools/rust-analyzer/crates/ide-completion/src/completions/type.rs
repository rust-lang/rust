//! Completion of names from the current scope in type position.

use hir::{HirDisplay, ScopeDef};
use syntax::{AstNode, ast};

use crate::{
    CompletionContext, Completions,
    context::{PathCompletionCtx, Qualified, TypeAscriptionTarget, TypeLocation},
    render::render_type_inference,
};

pub(crate) fn complete_type_path(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx @ PathCompletionCtx { qualified, .. }: &PathCompletionCtx<'_>,
    location: &TypeLocation,
) {
    let _p = tracing::info_span!("complete_type_path").entered();

    let scope_def_applicable = |def| {
        use hir::{GenericParam::*, ModuleDef::*};
        match def {
            ScopeDef::GenericParam(LifetimeParam(_)) => location.complete_lifetimes(),
            ScopeDef::Label(_) => false,
            // no values in type places
            ScopeDef::ModuleDef(Function(_) | Variant(_) | Static(_)) | ScopeDef::Local(_) => false,
            // unless its a constant in a generic arg list position
            ScopeDef::ModuleDef(Const(_)) | ScopeDef::GenericParam(ConstParam(_)) => {
                location.complete_consts()
            }
            ScopeDef::ImplSelfType(_) => location.complete_self_type(),
            // Don't suggest attribute macros and derives.
            ScopeDef::ModuleDef(Macro(mac)) => mac.is_fn_like(ctx.db),
            ScopeDef::ModuleDef(Trait(_) | Module(_))
                if matches!(location, TypeLocation::ImplTrait) =>
            {
                true
            }
            // Type things are fine
            ScopeDef::ModuleDef(
                BuiltinType(_) | Adt(_) | Module(_) | Trait(_) | TraitAlias(_) | TypeAlias(_),
            )
            | ScopeDef::AdtSelfType(_)
            | ScopeDef::Unknown
            | ScopeDef::GenericParam(TypeParam(_)) => location.complete_types(),
        }
    };

    let add_assoc_item = |acc: &mut Completions, item| match item {
        hir::AssocItem::Const(ct) if matches!(location, TypeLocation::GenericArg { .. }) => {
            acc.add_const(ctx, ct)
        }
        hir::AssocItem::Function(_) | hir::AssocItem::Const(_) => (),
        hir::AssocItem::TypeAlias(ty) => acc.add_type_alias(ctx, ty),
    };

    match qualified {
        Qualified::TypeAnchor { ty: None, trait_: None } => ctx
            .traits_in_scope()
            .iter()
            .flat_map(|&it| hir::Trait::from(it).items(ctx.sema.db))
            .for_each(|item| add_assoc_item(acc, item)),
        Qualified::TypeAnchor { trait_: Some(trait_), .. } => {
            trait_.items(ctx.sema.db).into_iter().for_each(|item| add_assoc_item(acc, item))
        }
        Qualified::TypeAnchor { ty: Some(ty), trait_: None } => {
            ctx.iterate_path_candidates(ty, |item| {
                add_assoc_item(acc, item);
            });

            // Iterate assoc types separately
            ty.iterate_assoc_items(ctx.db, ctx.krate, |item| {
                if let hir::AssocItem::TypeAlias(ty) = item {
                    acc.add_type_alias(ctx, ty)
                }
                None::<()>
            });
        }
        Qualified::With { resolution: None, .. } => {}
        Qualified::With { resolution: Some(resolution), .. } => {
            // Add associated types on type parameters and `Self`.
            ctx.scope.assoc_type_shorthand_candidates(resolution, |_, alias| {
                acc.add_type_alias(ctx, alias);
                None::<()>
            });

            match resolution {
                hir::PathResolution::Def(hir::ModuleDef::Module(module)) => {
                    let module_scope = module.scope(ctx.db, Some(ctx.module));
                    for (name, def) in module_scope {
                        if scope_def_applicable(def) {
                            acc.add_path_resolution(ctx, path_ctx, name, def, vec![]);
                        }
                    }
                }
                hir::PathResolution::Def(
                    def @ (hir::ModuleDef::Adt(_)
                    | hir::ModuleDef::TypeAlias(_)
                    | hir::ModuleDef::BuiltinType(_)),
                ) => {
                    let ty = match def {
                        hir::ModuleDef::Adt(adt) => adt.ty(ctx.db),
                        hir::ModuleDef::TypeAlias(a) => a.ty(ctx.db),
                        hir::ModuleDef::BuiltinType(builtin) => builtin.ty(ctx.db),
                        _ => return,
                    };

                    // XXX: For parity with Rust bug #22519, this does not complete Ty::AssocType.
                    // (where AssocType is defined on a trait, not an inherent impl)

                    ctx.iterate_path_candidates(&ty, |item| {
                        add_assoc_item(acc, item);
                    });

                    // Iterate assoc types separately
                    ty.iterate_assoc_items(ctx.db, ctx.krate, |item| {
                        if let hir::AssocItem::TypeAlias(ty) = item {
                            acc.add_type_alias(ctx, ty)
                        }
                        None::<()>
                    });
                }
                hir::PathResolution::Def(hir::ModuleDef::Trait(t)) => {
                    // Handles `Trait::assoc` as well as `<Ty as Trait>::assoc`.
                    for item in t.items(ctx.db) {
                        add_assoc_item(acc, item);
                    }
                }
                hir::PathResolution::TypeParam(_) | hir::PathResolution::SelfType(_) => {
                    let ty = match resolution {
                        hir::PathResolution::TypeParam(param) => param.ty(ctx.db),
                        hir::PathResolution::SelfType(impl_def) => impl_def.self_ty(ctx.db),
                        _ => return,
                    };

                    ctx.iterate_path_candidates(&ty, |item| {
                        add_assoc_item(acc, item);
                    });
                }
                _ => (),
            }
        }
        Qualified::Absolute => acc.add_crate_roots(ctx, path_ctx),
        Qualified::No => {
            match location {
                TypeLocation::TypeBound => {
                    acc.add_nameref_keywords_with_colon(ctx);
                    ctx.process_all_names(&mut |name, res, doc_aliases| {
                        let add_resolution = match res {
                            ScopeDef::ModuleDef(hir::ModuleDef::Macro(mac)) => {
                                mac.is_fn_like(ctx.db)
                            }
                            ScopeDef::ModuleDef(
                                hir::ModuleDef::Trait(_) | hir::ModuleDef::Module(_),
                            ) => true,
                            _ => false,
                        };
                        if add_resolution {
                            acc.add_path_resolution(ctx, path_ctx, name, res, doc_aliases);
                        }
                    });
                    return;
                }
                TypeLocation::GenericArg {
                    args: Some(arg_list), of_trait: Some(trait_), ..
                } => {
                    if arg_list.syntax().ancestors().find_map(ast::TypeBound::cast).is_some() {
                        let arg_idx = arg_list
                            .generic_args()
                            .filter(|arg| {
                                arg.syntax().text_range().end()
                                    < ctx.original_token.text_range().start()
                            })
                            .count();

                        let n_required_params = trait_.type_or_const_param_count(ctx.sema.db, true);
                        if arg_idx >= n_required_params {
                            trait_.items_with_supertraits(ctx.sema.db).into_iter().for_each(|it| {
                                if let hir::AssocItem::TypeAlias(alias) = it {
                                    cov_mark::hit!(complete_assoc_type_in_generics_list);
                                    acc.add_type_alias_with_eq(ctx, alias);
                                }
                            });

                            let n_params = trait_.type_or_const_param_count(ctx.sema.db, false);
                            if arg_idx >= n_params {
                                return; // only show assoc types
                            }
                        }
                    }
                }
                TypeLocation::ImplTrait => {
                    acc.add_nameref_keywords_with_colon(ctx);
                    ctx.process_all_names(&mut |name, def, doc_aliases| {
                        let is_trait_or_module = matches!(
                            def,
                            ScopeDef::ModuleDef(
                                hir::ModuleDef::Module(_) | hir::ModuleDef::Trait(_)
                            )
                        );
                        if is_trait_or_module {
                            acc.add_path_resolution(ctx, path_ctx, name, def, doc_aliases);
                        }
                    });
                    return;
                }
                _ => {}
            };

            acc.add_nameref_keywords_with_colon(ctx);
            ctx.process_all_names(&mut |name, def, doc_aliases| {
                if scope_def_applicable(def) {
                    acc.add_path_resolution(ctx, path_ctx, name, def, doc_aliases);
                }
            });
        }
    }
}

pub(crate) fn complete_ascribed_type(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx: &PathCompletionCtx<'_>,
    ascription: &TypeAscriptionTarget,
) -> Option<()> {
    if !path_ctx.is_trivial_path() {
        return None;
    }
    let ty = match ascription {
        TypeAscriptionTarget::Let(pat) | TypeAscriptionTarget::FnParam(pat) => {
            ctx.sema.type_of_pat(pat.as_ref()?)
        }
        TypeAscriptionTarget::Const(exp) | TypeAscriptionTarget::RetType(exp) => {
            ctx.sema.type_of_expr(exp.as_ref()?)
        }
    }?
    .adjusted();
    if !ty.is_unknown() {
        let ty_string = ty.display_source_code(ctx.db, ctx.module.into(), true).ok()?;
        acc.add(render_type_inference(ty_string, ctx));
    }
    None
}
