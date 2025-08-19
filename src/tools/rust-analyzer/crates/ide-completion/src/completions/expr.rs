//! Completion of names from the current scope in expression position.

use std::ops::ControlFlow;

use hir::{Complete, Name, PathCandidateCallback, ScopeDef, sym};
use ide_db::FxHashSet;
use syntax::ast;

use crate::{
    CompletionContext, Completions,
    completions::record::add_default_update,
    context::{BreakableKind, PathCompletionCtx, PathExprCtx, Qualified},
};

struct PathCallback<'a, F> {
    ctx: &'a CompletionContext<'a>,
    acc: &'a mut Completions,
    add_assoc_item: F,
    seen: FxHashSet<hir::AssocItem>,
}

impl<F> PathCandidateCallback for PathCallback<'_, F>
where
    F: FnMut(&mut Completions, hir::AssocItem),
{
    fn on_inherent_item(&mut self, item: hir::AssocItem) -> ControlFlow<()> {
        if self.seen.insert(item) {
            (self.add_assoc_item)(self.acc, item);
        }
        ControlFlow::Continue(())
    }

    fn on_trait_item(&mut self, item: hir::AssocItem) -> ControlFlow<()> {
        // The excluded check needs to come before the `seen` test, so that if we see the same method twice,
        // once as inherent and once not, we will include it.
        if item.container_trait(self.ctx.db).is_none_or(|trait_| {
            !self.ctx.exclude_traits.contains(&trait_)
                && trait_.complete(self.ctx.db) != Complete::IgnoreMethods
        }) && self.seen.insert(item)
        {
            (self.add_assoc_item)(self.acc, item);
        }
        ControlFlow::Continue(())
    }
}

pub(crate) fn complete_expr_path(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx @ PathCompletionCtx { qualified, .. }: &PathCompletionCtx<'_>,
    expr_ctx: &PathExprCtx<'_>,
) {
    let _p = tracing::info_span!("complete_expr_path").entered();
    if !ctx.qualifier_ctx.none() {
        return;
    }

    let &PathExprCtx {
        in_block_expr,
        in_breakable,
        after_if_expr,
        in_condition,
        incomplete_let,
        ref ref_expr_parent,
        after_amp,
        ref is_func_update,
        ref innermost_ret_ty,
        ref impl_,
        in_match_guard,
        ..
    } = expr_ctx;

    let (has_raw_token, has_const_token, has_mut_token) = ref_expr_parent
        .as_ref()
        .map(|it| (it.raw_token().is_some(), it.const_token().is_some(), it.mut_token().is_some()))
        .unwrap_or((false, false, false));

    let wants_raw_token = ref_expr_parent.is_some() && !has_raw_token && after_amp;
    let wants_const_token =
        ref_expr_parent.is_some() && has_raw_token && !has_const_token && !has_mut_token;
    let wants_mut_token = if ref_expr_parent.is_some() {
        if has_raw_token { !has_const_token && !has_mut_token } else { !has_mut_token }
    } else {
        false
    };

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
        // We exclude associated types/consts of excluded traits here together with methods,
        // even though we don't exclude them when completing in type position, because it's easier.
        Qualified::TypeAnchor { ty: None, trait_: None } => ctx
            .traits_in_scope()
            .iter()
            .copied()
            .map(hir::Trait::from)
            .filter(|it| {
                !ctx.exclude_traits.contains(it) && it.complete(ctx.db) != Complete::IgnoreMethods
            })
            .flat_map(|it| it.items(ctx.sema.db))
            .for_each(|item| add_assoc_item(acc, item)),
        Qualified::TypeAnchor { trait_: Some(trait_), .. } => {
            // Don't filter excluded traits here, user requested this specific trait.
            trait_.items(ctx.sema.db).into_iter().for_each(|item| add_assoc_item(acc, item))
        }
        Qualified::TypeAnchor { ty: Some(ty), trait_: None } => {
            if let Some(hir::Adt::Enum(e)) = ty.as_adt() {
                cov_mark::hit!(completes_variant_through_alias);
                acc.add_enum_variants(ctx, path_ctx, e);
            }

            ty.iterate_path_candidates_split_inherent(
                ctx.db,
                &ctx.scope,
                &ctx.traits_in_scope(),
                Some(ctx.module),
                None,
                PathCallback { ctx, acc, add_assoc_item, seen: FxHashSet::default() },
            );

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
                    let visible_from = if ctx.config.enable_private_editable {
                        // Set visible_from to None so private items are returned.
                        // They will be possibly filtered out in add_path_resolution()
                        // via def_is_visible().
                        None
                    } else {
                        Some(ctx.module)
                    };

                    let module_scope = module.scope(ctx.db, visible_from);
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

                    ty.iterate_path_candidates_split_inherent(
                        ctx.db,
                        &ctx.scope,
                        &ctx.traits_in_scope(),
                        Some(ctx.module),
                        None,
                        PathCallback { ctx, acc, add_assoc_item, seen: FxHashSet::default() },
                    );

                    // Iterate assoc types separately
                    ty.iterate_assoc_items(ctx.db, ctx.krate, |item| {
                        if let hir::AssocItem::TypeAlias(ty) = item {
                            acc.add_type_alias(ctx, ty)
                        }
                        None::<()>
                    });
                }
                hir::PathResolution::Def(hir::ModuleDef::Trait(t)) => {
                    // Don't filter excluded traits here, user requested this specific trait.
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

                    ty.iterate_path_candidates_split_inherent(
                        ctx.db,
                        &ctx.scope,
                        &ctx.traits_in_scope(),
                        Some(ctx.module),
                        None,
                        PathCallback { ctx, acc, add_assoc_item, seen: FxHashSet::default() },
                    );
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
                            .find_path(
                                ctx.db,
                                hir::ModuleDef::from(strukt),
                                ctx.config.import_path_config(ctx.is_nightly),
                            )
                            .filter(|it| it.len() > 1);

                        acc.add_struct_literal(ctx, path_ctx, strukt, path, None);

                        if complete_self {
                            acc.add_struct_literal(
                                ctx,
                                path_ctx,
                                strukt,
                                None,
                                Some(Name::new_symbol_root(sym::Self_)),
                            );
                        }
                    }
                    hir::Adt::Union(un) => {
                        let path = ctx
                            .module
                            .find_path(
                                ctx.db,
                                hir::ModuleDef::from(un),
                                ctx.config.import_path_config(ctx.is_nightly),
                            )
                            .filter(|it| it.len() > 1);

                        acc.add_union_literal(ctx, un, path, None);
                        if complete_self {
                            acc.add_union_literal(
                                ctx,
                                un,
                                None,
                                Some(Name::new_symbol_root(sym::Self_)),
                            );
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
                // synthetic names currently leak out as we lack synthetic hygiene, so filter them
                // out here
                ScopeDef::Local(_) => {
                    if !name.as_str().starts_with('<') {
                        acc.add_path_resolution(ctx, path_ctx, name, def, doc_aliases)
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
                        add_keyword("const", "const {\n    $0\n}");
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

                    if in_condition {
                        add_keyword("letm", "let mut $1 = $0");
                        add_keyword("let", "let $1 = $0");
                    }

                    if in_block_expr {
                        add_keyword("letm", "let mut $1 = $0;");
                        add_keyword("let", "let $1 = $0;");
                    }

                    if after_if_expr {
                        add_keyword("else", "else {\n    $0\n}");
                        add_keyword("else if", "else if $1 {\n    $0\n}");
                    }

                    if wants_raw_token {
                        add_keyword("raw", "raw ");
                    }
                    if wants_const_token {
                        add_keyword("const", "const ");
                    }
                    if wants_mut_token {
                        add_keyword("mut", "mut ");
                    }

                    if in_breakable != BreakableKind::None {
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

pub(crate) fn complete_expr(acc: &mut Completions, ctx: &CompletionContext<'_>) {
    let _p = tracing::info_span!("complete_expr").entered();

    if !ctx.config.enable_term_search {
        return;
    }

    if !ctx.qualifier_ctx.none() {
        return;
    }

    if let Some(ty) = &ctx.expected_type {
        // Ignore unit types as they are not very interesting
        if ty.is_unit() || ty.is_unknown() {
            return;
        }

        let term_search_ctx = hir::term_search::TermSearchCtx {
            sema: &ctx.sema,
            scope: &ctx.scope,
            goal: ty.clone(),
            config: hir::term_search::TermSearchConfig {
                enable_borrowcheck: false,
                many_alternatives_threshold: 1,
                fuel: 200,
            },
        };
        let exprs = hir::term_search::term_search(&term_search_ctx);
        for expr in exprs {
            // Expand method calls
            match expr {
                hir::term_search::Expr::Method { func, generics, target, params }
                    if target.is_many() =>
                {
                    let target_ty = target.ty(ctx.db);
                    let term_search_ctx =
                        hir::term_search::TermSearchCtx { goal: target_ty, ..term_search_ctx };
                    let target_exprs = hir::term_search::term_search(&term_search_ctx);

                    for expr in target_exprs {
                        let expanded_expr = hir::term_search::Expr::Method {
                            func,
                            generics: generics.clone(),
                            target: Box::new(expr),
                            params: params.clone(),
                        };

                        acc.add_expr(ctx, &expanded_expr)
                    }
                }
                _ => acc.add_expr(ctx, &expr),
            }
        }
    }
}
