//! Completion of names from the current scope in expression position.

use hir::ScopeDef;
use syntax::ast;

use crate::{
    completions::record::add_default_update,
    context::{ExprCtx, PathCompletionCtx, Qualified},
    CompletionContext, Completions,
};

pub(crate) fn complete_expr_path(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx @ PathCompletionCtx { qualified, .. }: &PathCompletionCtx,
    expr_ctx: &ExprCtx,
) {
    let _p = profile::span("complete_expr_path");
    if !ctx.qualifier_ctx.none() {
        return;
    }

    let &ExprCtx {
        in_block_expr,
        in_loop_body,
        after_if_expr,
        in_condition,
        incomplete_let,
        ref ref_expr_parent,
        ref is_func_update,
        ref innermost_ret_ty,
        ref impl_,
        in_match_guard,
        ..
    } = expr_ctx;

    let wants_mut_token =
        ref_expr_parent.as_ref().map(|it| it.mut_token().is_none()).unwrap_or(false);

    let scope_def_applicable = |def| match def {
        ScopeDef::GenericParam(hir::GenericParam::LifetimeParam(_)) | ScopeDef::Label(_) => false,
        ScopeDef::ModuleDef(hir::ModuleDef::Macro(mac)) => mac.is_fn_like(ctx.db),
        _ => true,
    };

    let add_assoc_item = |acc: &mut Completions, item| match item {
        hir::AssocItem::Function(func) => acc.add_function(ctx, path_ctx, func, None),
        hir::AssocItem::Const(ct) => acc.add_const(ctx, ct),
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
            if let Some(hir::Adt::Enum(e)) = ty.as_adt() {
                cov_mark::hit!(completes_variant_through_alias);
                acc.add_enum_variants(ctx, path_ctx, e);
            }

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
                            acc.add_path_resolution(
                                ctx,
                                path_ctx,
                                name,
                                def,
                                ctx.doc_aliases_in_scope(def),
                            );
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
                        hir::ModuleDef::BuiltinType(builtin) => {
                            cov_mark::hit!(completes_primitive_assoc_const);
                            builtin.ty(ctx.db)
                        }
                        _ => return,
                    };

                    if let Some(hir::Adt::Enum(e)) = ty.as_adt() {
                        cov_mark::hit!(completes_variant_through_alias);
                        acc.add_enum_variants(ctx, path_ctx, e);
                    }

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

                    if let Some(hir::Adt::Enum(e)) = ty.as_adt() {
                        cov_mark::hit!(completes_variant_through_self);
                        acc.add_enum_variants(ctx, path_ctx, e);
                    }

                    ctx.iterate_path_candidates(&ty, |item| {
                        add_assoc_item(acc, item);
                    });
                }
                _ => (),
            }
        }
        Qualified::Absolute => acc.add_crate_roots(ctx, path_ctx),
        Qualified::No => {
            acc.add_nameref_keywords_with_colon(ctx);
            if let Some(adt) =
                ctx.expected_type.as_ref().and_then(|ty| ty.strip_references().as_adt())
            {
                let self_ty = (|| ctx.sema.to_def(impl_.as_ref()?)?.self_ty(ctx.db).as_adt())();
                let complete_self = self_ty == Some(adt);

                match adt {
                    hir::Adt::Struct(strukt) => {
                        let path = ctx
                            .module
                            .find_use_path(
                                ctx.db,
                                hir::ModuleDef::from(strukt),
                                ctx.config.prefer_no_std,
                            )
                            .filter(|it| it.len() > 1);

                        acc.add_struct_literal(ctx, path_ctx, strukt, path, None);

                        if complete_self {
                            acc.add_struct_literal(
                                ctx,
                                path_ctx,
                                strukt,
                                None,
                                Some(hir::known::SELF_TYPE),
                            );
                        }
                    }
                    hir::Adt::Union(un) => {
                        let path = ctx
                            .module
                            .find_use_path(
                                ctx.db,
                                hir::ModuleDef::from(un),
                                ctx.config.prefer_no_std,
                            )
                            .filter(|it| it.len() > 1);

                        acc.add_union_literal(ctx, un, path, None);
                        if complete_self {
                            acc.add_union_literal(ctx, un, None, Some(hir::known::SELF_TYPE));
                        }
                    }
                    hir::Adt::Enum(e) => {
                        super::enum_variants_with_paths(
                            acc,
                            ctx,
                            e,
                            impl_,
                            |acc, ctx, variant, path| {
                                acc.add_qualified_enum_variant(ctx, path_ctx, variant, path)
                            },
                        );
                    }
                }
            }
            ctx.process_all_names(&mut |name, def, doc_aliases| match def {
                ScopeDef::ModuleDef(hir::ModuleDef::Trait(t)) => {
                    let assocs = t.items_with_supertraits(ctx.db);
                    match &*assocs {
                        // traits with no assoc items are unusable as expressions since
                        // there is no associated item path that can be constructed with them
                        [] => (),
                        // FIXME: Render the assoc item with the trait qualified
                        &[_item] => acc.add_path_resolution(ctx, path_ctx, name, def, doc_aliases),
                        // FIXME: Append `::` to the thing here, since a trait on its own won't work
                        [..] => acc.add_path_resolution(ctx, path_ctx, name, def, doc_aliases),
                    }
                }
                _ if scope_def_applicable(def) => {
                    acc.add_path_resolution(ctx, path_ctx, name, def, doc_aliases)
                }
                _ => (),
            });

            match is_func_update {
                Some(record_expr) => {
                    let ty = ctx.sema.type_of_expr(&ast::Expr::RecordExpr(record_expr.clone()));

                    match ty.as_ref().and_then(|t| t.original.as_adt()) {
                        Some(hir::Adt::Union(_)) => (),
                        _ => {
                            cov_mark::hit!(functional_update);
                            let missing_fields =
                                ctx.sema.record_literal_missing_fields(record_expr);
                            if !missing_fields.is_empty() {
                                add_default_update(acc, ctx, ty);
                            }
                        }
                    };
                }
                None => {
                    let mut add_keyword = |kw, snippet| {
                        acc.add_keyword_snippet_expr(ctx, incomplete_let, kw, snippet)
                    };

                    if !in_block_expr {
                        add_keyword("unsafe", "unsafe {\n    $0\n}");
                    }
                    add_keyword("match", "match $1 {\n    $0\n}");
                    add_keyword("while", "while $1 {\n    $0\n}");
                    add_keyword("while let", "while let $1 = $2 {\n    $0\n}");
                    add_keyword("loop", "loop {\n    $0\n}");
                    if in_match_guard {
                        add_keyword("if", "if $0");
                    } else {
                        add_keyword("if", "if $1 {\n    $0\n}");
                    }
                    add_keyword("if let", "if let $1 = $2 {\n    $0\n}");
                    add_keyword("for", "for $1 in $2 {\n    $0\n}");
                    add_keyword("true", "true");
                    add_keyword("false", "false");

                    if in_condition || in_block_expr {
                        add_keyword("let", "let");
                    }

                    if after_if_expr {
                        add_keyword("else", "else {\n    $0\n}");
                        add_keyword("else if", "else if $1 {\n    $0\n}");
                    }

                    if wants_mut_token {
                        add_keyword("mut", "mut ");
                    }

                    if in_loop_body {
                        if in_block_expr {
                            add_keyword("continue", "continue;");
                            add_keyword("break", "break;");
                        } else {
                            add_keyword("continue", "continue");
                            add_keyword("break", "break");
                        }
                    }

                    if let Some(ret_ty) = innermost_ret_ty {
                        add_keyword(
                            "return",
                            match (ret_ty.is_unit(), in_block_expr) {
                                (true, true) => {
                                    cov_mark::hit!(return_unit_block);
                                    "return;"
                                }
                                (true, false) => {
                                    cov_mark::hit!(return_unit_no_block);
                                    "return"
                                }
                                (false, true) => {
                                    cov_mark::hit!(return_value_block);
                                    "return $0;"
                                }
                                (false, false) => {
                                    cov_mark::hit!(return_value_no_block);
                                    "return $0"
                                }
                            },
                        );
                    }
                }
            }
        }
    }
}
