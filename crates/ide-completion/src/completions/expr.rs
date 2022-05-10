//! Completion of names from the current scope in expression position.

use hir::ScopeDef;
use ide_db::FxHashSet;
use syntax::T;

use crate::{
    context::{PathCompletionCtx, PathKind, PathQualifierCtx},
    CompletionContext, Completions,
};

pub(crate) fn complete_expr_path(acc: &mut Completions, ctx: &CompletionContext) {
    let _p = profile::span("complete_expr_path");
    if ctx.is_path_disallowed() {
        return;
    }

    let (is_absolute_path, qualifier, in_block_expr, in_loop_body, in_functional_update) =
        match ctx.path_context() {
            Some(&PathCompletionCtx {
                kind: PathKind::Expr { in_block_expr, in_loop_body, in_functional_update },
                is_absolute_path,
                ref qualifier,
                ..
            }) => (is_absolute_path, qualifier, in_block_expr, in_loop_body, in_functional_update),
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
            if let Some(hir::Adt::Enum(e)) =
                ctx.expected_type.as_ref().and_then(|ty| ty.strip_references().as_adt())
            {
                super::enum_variants_with_paths(acc, ctx, e, |acc, ctx, variant, path| {
                    acc.add_qualified_enum_variant(ctx, variant, path)
                });
            }
            ctx.process_all_names(&mut |name, def| {
                if scope_def_applicable(def) {
                    acc.add_resolution(ctx, name, def);
                }
            });

            if !in_functional_update {
                let mut add_keyword =
                    |kw, snippet| super::keyword::add_keyword(acc, ctx, kw, snippet);

                if ctx.expects_expression() {
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
                }

                if ctx.previous_token_is(T![if])
                    || ctx.previous_token_is(T![while])
                    || in_block_expr
                {
                    add_keyword("let", "let");
                }

                if ctx.after_if() {
                    add_keyword("else", "else {\n    $0\n}");
                    add_keyword("else if", "else if $1 {\n    $0\n}");
                }

                if ctx.expects_ident_ref_expr() {
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
