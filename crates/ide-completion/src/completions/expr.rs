//! Completion of names from the current scope in expression position.

use hir::ScopeDef;
use ide_db::FxHashSet;

use crate::{
    context::{NameRefContext, NameRefKind, PathCompletionCtx, PathKind, PathQualifierCtx},
    CompletionContext, Completions,
};

pub(crate) fn complete_expr_path(acc: &mut Completions, ctx: &CompletionContext) {
    let _p = profile::span("complete_expr_path");

    let (
        is_absolute_path,
        qualifier,
        in_block_expr,
        in_loop_body,
        is_func_update,
        after_if_expr,
        wants_mut_token,
        in_condition,
    ) = match ctx.nameref_ctx() {
        Some(&NameRefContext {
            kind:
                Some(NameRefKind::Path(PathCompletionCtx {
                    kind:
                        PathKind::Expr {
                            in_block_expr,
                            in_loop_body,
                            after_if_expr,
                            in_condition,
                            ref ref_expr_parent,
                            ref is_func_update,
                        },
                    is_absolute_path,
                    ref qualifier,
                    ..
                })),
            ..
        }) if ctx.qualifier_ctx.none() => (
            is_absolute_path,
            qualifier,
            in_block_expr,
            in_loop_body,
            is_func_update.is_some(),
            after_if_expr,
            ref_expr_parent.as_ref().map(|it| it.mut_token().is_none()).unwrap_or(false),
            in_condition,
        ),
        _ => return,
    };

    let scope_def_applicable = |def| {
        use hir::{GenericParam::*, ModuleDef::*};
        match def {
            ScopeDef::GenericParam(LifetimeParam(_)) | ScopeDef::Label(_) => false,
            // Don't suggest attribute macros and derives.
            ScopeDef::ModuleDef(Macro(mac)) => mac.is_fn_like(ctx.db),
            _ => true,
        }
    };

    match qualifier {
        Some(PathQualifierCtx { is_infer_qualifier, resolution, .. }) => {
            if *is_infer_qualifier {
                ctx.traits_in_scope()
                    .0
                    .into_iter()
                    .flat_map(|it| hir::Trait::from(it).items(ctx.sema.db))
                    .for_each(|item| add_assoc_item(acc, ctx, item));
                return;
            }
            let resolution = match resolution {
                Some(it) => it,
                None => return,
            };
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
                            acc.add_resolution(ctx, name, def);
                        }
                    }
                }

                hir::PathResolution::Def(
                    def @ (hir::ModuleDef::Adt(_)
                    | hir::ModuleDef::TypeAlias(_)
                    | hir::ModuleDef::BuiltinType(_)),
                ) => {
                    if let &hir::ModuleDef::Adt(hir::Adt::Enum(e)) = def {
                        add_enum_variants(acc, ctx, e);
                    }
                    let ty = match def {
                        hir::ModuleDef::Adt(adt) => adt.ty(ctx.db),
                        hir::ModuleDef::TypeAlias(a) => {
                            let ty = a.ty(ctx.db);
                            if let Some(hir::Adt::Enum(e)) = ty.as_adt() {
                                cov_mark::hit!(completes_variant_through_alias);
                                add_enum_variants(acc, ctx, e);
                            }
                            ty
                        }
                        hir::ModuleDef::BuiltinType(builtin) => {
                            cov_mark::hit!(completes_primitive_assoc_const);
                            builtin.ty(ctx.db)
                        }
                        _ => unreachable!(),
                    };

                    // XXX: For parity with Rust bug #22519, this does not complete Ty::AssocType.
                    // (where AssocType is defined on a trait, not an inherent impl)

                    ty.iterate_path_candidates(
                        ctx.db,
                        &ctx.scope,
                        &ctx.traits_in_scope().0,
                        Some(ctx.module),
                        None,
                        |item| {
                            add_assoc_item(acc, ctx, item);
                            None::<()>
                        },
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
                    // Handles `Trait::assoc` as well as `<Ty as Trait>::assoc`.
                    for item in t.items(ctx.db) {
                        add_assoc_item(acc, ctx, item);
                    }
                }
                hir::PathResolution::TypeParam(_) | hir::PathResolution::SelfType(_) => {
                    let ty = match resolution {
                        hir::PathResolution::TypeParam(param) => param.ty(ctx.db),
                        hir::PathResolution::SelfType(impl_def) => impl_def.self_ty(ctx.db),
                        _ => return,
                    };

                    if let Some(hir::Adt::Enum(e)) = ty.as_adt() {
                        add_enum_variants(acc, ctx, e);
                    }
                    let mut seen = FxHashSet::default();
                    ty.iterate_path_candidates(
                        ctx.db,
                        &ctx.scope,
                        &ctx.traits_in_scope().0,
                        Some(ctx.module),
                        None,
                        |item| {
                            // We might iterate candidates of a trait multiple times here, so deduplicate
                            // them.
                            if seen.insert(item) {
                                add_assoc_item(acc, ctx, item);
                            }
                            None::<()>
                        },
                    );
                }
                _ => (),
            }
        }
        None if is_absolute_path => acc.add_crate_roots(ctx),
        None => {
            acc.add_nameref_keywords_with_colon(ctx);
            if let Some(adt) =
                ctx.expected_type.as_ref().and_then(|ty| ty.strip_references().as_adt())
            {
                let self_ty =
                    (|| ctx.sema.to_def(ctx.impl_def.as_ref()?)?.self_ty(ctx.db).as_adt())();
                let complete_self = self_ty == Some(adt);

                match adt {
                    hir::Adt::Struct(strukt) => {
                        let path = ctx
                            .module
                            .find_use_path(ctx.db, hir::ModuleDef::from(strukt))
                            .filter(|it| it.len() > 1);

                        acc.add_struct_literal(ctx, strukt, path, None);

                        if complete_self {
                            acc.add_struct_literal(ctx, strukt, None, Some(hir::known::SELF_TYPE));
                        }
                    }
                    hir::Adt::Union(un) => {
                        let path = ctx
                            .module
                            .find_use_path(ctx.db, hir::ModuleDef::from(un))
                            .filter(|it| it.len() > 1);

                        acc.add_union_literal(ctx, un, path, None);
                        if complete_self {
                            acc.add_union_literal(ctx, un, None, Some(hir::known::SELF_TYPE));
                        }
                    }
                    hir::Adt::Enum(e) => {
                        super::enum_variants_with_paths(acc, ctx, e, |acc, ctx, variant, path| {
                            acc.add_qualified_enum_variant(ctx, variant, path)
                        });
                    }
                }
            }
            ctx.process_all_names(&mut |name, def| {
                if scope_def_applicable(def) {
                    acc.add_resolution(ctx, name, def);
                }
            });

            if !is_func_update {
                let mut add_keyword = |kw, snippet| acc.add_keyword_snippet(ctx, kw, snippet);

                if !in_block_expr {
                    add_keyword("unsafe", "unsafe {\n    $0\n}");
                }
                add_keyword("match", "match $1 {\n    $0\n}");
                add_keyword("while", "while $1 {\n    $0\n}");
                add_keyword("while let", "while let $1 = $2 {\n    $0\n}");
                add_keyword("loop", "loop {\n    $0\n}");
                add_keyword("if", "if $1 {\n    $0\n}");
                add_keyword("if let", "if let $1 = $2 {\n    $0\n}");
                add_keyword("for", "for $1 in $2 {\n    $0\n}");
                add_keyword("true", "true");
                add_keyword("false", "false");

                if (in_condition && !is_absolute_path) || in_block_expr {
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

                if let Some(fn_def) = &ctx.function_def {
                    add_keyword(
                        "return",
                        match (in_block_expr, fn_def.ret_type().is_some()) {
                            (true, true) => "return ;",
                            (true, false) => "return;",
                            (false, true) => "return $0",
                            (false, false) => "return",
                        },
                    );
                }
            }
        }
    }
}

fn add_assoc_item(acc: &mut Completions, ctx: &CompletionContext, item: hir::AssocItem) {
    match item {
        hir::AssocItem::Function(func) => acc.add_function(ctx, func, None),
        hir::AssocItem::Const(ct) => acc.add_const(ctx, ct),
        hir::AssocItem::TypeAlias(ty) => acc.add_type_alias(ctx, ty),
    }
}

fn add_enum_variants(acc: &mut Completions, ctx: &CompletionContext, e: hir::Enum) {
    e.variants(ctx.db).into_iter().for_each(|variant| acc.add_enum_variant(ctx, variant, None));
}
