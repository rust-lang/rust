use std::ops::Not;

use crate::{
    assist_context::{AssistContext, Assists},
    utils::convert_param_list_to_arg_list,
};
use either::Either;
use hir::{HasVisibility, db::HirDatabase};
use ide_db::{
    FxHashMap, FxHashSet,
    assists::{AssistId, GroupLabel},
    path_transform::PathTransform,
    syntax_helpers::suggest_name,
};
use itertools::Itertools;
use syntax::{
    AstNode, Edition, NodeOrToken, SmolStr, SyntaxKind, ToSmolStr,
    ast::{
        self, AssocItem, GenericArgList, GenericParamList, HasAttrs, HasGenericArgs,
        HasGenericParams, HasName, HasTypeBounds, HasVisibility as astHasVisibility, Path,
        WherePred,
        edit::{self, AstNodeEdit},
        edit_in_place::AttrsOwnerEdit,
        make,
    },
    ted::{self, Position},
};

// Assist: generate_delegate_trait
//
// Generate delegate trait implementation for `StructField`s.
//
// ```
// trait SomeTrait {
//     type T;
//     fn fn_(arg: u32) -> u32;
//     fn method_(&mut self) -> bool;
// }
// struct A;
// impl SomeTrait for A {
//     type T = u32;
//
//     fn fn_(arg: u32) -> u32 {
//         42
//     }
//
//     fn method_(&mut self) -> bool {
//         false
//     }
// }
// struct B {
//     a$0: A,
// }
// ```
// ->
// ```
// trait SomeTrait {
//     type T;
//     fn fn_(arg: u32) -> u32;
//     fn method_(&mut self) -> bool;
// }
// struct A;
// impl SomeTrait for A {
//     type T = u32;
//
//     fn fn_(arg: u32) -> u32 {
//         42
//     }
//
//     fn method_(&mut self) -> bool {
//         false
//     }
// }
// struct B {
//     a: A,
// }
//
// impl SomeTrait for B {
//     type T = <A as SomeTrait>::T;
//
//     fn fn_(arg: u32) -> u32 {
//         <A as SomeTrait>::fn_(arg)
//     }
//
//     fn method_(&mut self) -> bool {
//         <A as SomeTrait>::method_(&mut self.a)
//     }
// }
// ```
pub(crate) fn generate_delegate_trait(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    if !ctx.config.code_action_grouping {
        return None;
    }

    let strukt = Struct::new(ctx.find_node_at_offset::<ast::Struct>()?)?;

    let field: Field = match ctx.find_node_at_offset::<ast::RecordField>() {
        Some(field) => Field::new(ctx, Either::Left(field))?,
        None => {
            let field = ctx.find_node_at_offset::<ast::TupleField>()?;
            let field_list = ctx.find_node_at_offset::<ast::TupleFieldList>()?;
            Field::new(ctx, either::Right((field, field_list)))?
        }
    };

    strukt.delegate(field, acc, ctx);
    Some(())
}

/// A utility object that represents a struct's field.
#[derive(Debug)]
struct Field {
    name: String,
    ty: ast::Type,
    range: syntax::TextRange,
    impls: Vec<Delegee>,
    edition: Edition,
}

impl Field {
    pub(crate) fn new(
        ctx: &AssistContext<'_>,
        f: Either<ast::RecordField, (ast::TupleField, ast::TupleFieldList)>,
    ) -> Option<Field> {
        let db = ctx.sema.db;

        let module = ctx.sema.file_to_module_def(ctx.vfs_file_id())?;
        let edition = module.krate().edition(ctx.db());

        let (name, range, ty) = match f {
            Either::Left(f) => {
                let name = f.name()?.to_string();
                (name, f.syntax().text_range(), f.ty()?)
            }
            Either::Right((f, l)) => {
                let name = l.fields().position(|it| it == f)?.to_string();
                (name, f.syntax().text_range(), f.ty()?)
            }
        };

        let hir_ty = ctx.sema.resolve_type(&ty)?;
        let type_impls = hir::Impl::all_for_type(db, hir_ty.clone());
        let mut impls = Vec::with_capacity(type_impls.len());

        if let Some(tp) = hir_ty.as_type_param(db) {
            for tb in tp.trait_bounds(db) {
                impls.push(Delegee::Bound(tb));
            }
        };

        for imp in type_impls {
            if let Some(tr) = imp.trait_(db).filter(|tr| tr.is_visible_from(db, module)) {
                impls.push(Delegee::Impls(tr, imp))
            }
        }

        Some(Field { name, ty, range, impls, edition })
    }
}

/// A field that we want to delegate can offer the enclosing struct
/// trait to implement in two ways. The first way is when the field
/// actually implements the trait and the second way is when the field
/// has a bound type parameter. We handle these cases in different ways
/// hence the enum.
#[derive(Debug)]
enum Delegee {
    Bound(hir::Trait),
    Impls(hir::Trait, hir::Impl),
}

impl Delegee {
    fn signature(&self, db: &dyn HirDatabase, edition: Edition) -> String {
        let mut s = String::new();

        let (Delegee::Bound(it) | Delegee::Impls(it, _)) = self;

        for m in it.module(db).path_to_root(db).iter().rev() {
            if let Some(name) = m.name(db) {
                s.push_str(&format!("{}::", name.display_no_db(edition).to_smolstr()));
            }
        }

        s.push_str(&it.name(db).display_no_db(edition).to_smolstr());
        s
    }
}

/// A utility struct that is used for the enclosing struct.
struct Struct {
    strukt: ast::Struct,
    name: ast::Name,
}

impl Struct {
    pub(crate) fn new(s: ast::Struct) -> Option<Self> {
        let name = s.name()?;
        Some(Struct { name, strukt: s })
    }

    pub(crate) fn delegate(&self, field: Field, acc: &mut Assists, ctx: &AssistContext<'_>) {
        let db = ctx.db();

        for (index, delegee) in field.impls.iter().enumerate() {
            let trait_ = match delegee {
                Delegee::Bound(b) => b,
                Delegee::Impls(i, _) => i,
            };

            // Skip trait that has `Self` type, which cannot be delegated
            //
            // See [`test_self_ty`]
            if has_self_type(*trait_, ctx).is_some() {
                continue;
            }

            // FIXME :  We can omit already implemented impl_traits
            // But we don't know what the &[hir::Type] argument should look like.
            // if self.hir_ty.impls_trait(db, trait_, &[]) {
            //     continue;
            // }
            let signature = delegee.signature(db, field.edition);

            let Some(delegate) =
                generate_impl(ctx, self, &field.ty, &field.name, delegee, field.edition)
            else {
                continue;
            };

            acc.add_group(
                &GroupLabel(format!("Generate delegate trait impls for field `{}`", field.name)),
                AssistId(
                    "generate_delegate_trait",
                    ide_db::assists::AssistKind::Generate,
                    Some(index),
                ),
                format!("Generate delegate trait impl `{}` for `{}`", signature, field.name),
                field.range,
                |builder| {
                    builder.insert(
                        self.strukt.syntax().text_range().end(),
                        format!("\n\n{}", delegate.syntax()),
                    );
                },
            );
        }
    }
}

fn generate_impl(
    ctx: &AssistContext<'_>,
    strukt: &Struct,
    field_ty: &ast::Type,
    field_name: &str,
    delegee: &Delegee,
    edition: Edition,
) -> Option<ast::Impl> {
    let db = ctx.db();
    let ast_strukt = &strukt.strukt;
    let strukt_ty = make::ty_path(make::ext::ident_path(&strukt.name.to_string()));
    let strukt_params = ast_strukt.generic_param_list();

    match delegee {
        Delegee::Bound(delegee) => {
            let bound_def = ctx.sema.source(delegee.to_owned())?.value;
            let bound_params = bound_def.generic_param_list();

            let delegate = make::impl_trait(
                delegee.is_unsafe(db),
                bound_params.clone(),
                bound_params.map(|params| params.to_generic_args()),
                strukt_params.clone(),
                strukt_params.map(|params| params.to_generic_args()),
                delegee.is_auto(db),
                make::ty(&delegee.name(db).display_no_db(edition).to_smolstr()),
                strukt_ty,
                bound_def.where_clause(),
                ast_strukt.where_clause(),
                None,
            )
            .clone_for_update();

            // Goto link : https://doc.rust-lang.org/reference/paths.html#qualified-paths
            let qualified_path_type =
                make::path_from_text(&format!("<{} as {}>", field_ty, delegate.trait_()?));

            let delegate_assoc_items = delegate.get_or_create_assoc_item_list();
            if let Some(ai) = bound_def.assoc_item_list() {
                ai.assoc_items()
                    .filter(|item| matches!(item, AssocItem::MacroCall(_)).not())
                    .for_each(|item| {
                        let assoc = process_assoc_item(
                            item.clone_for_update(),
                            qualified_path_type.clone(),
                            field_name,
                        );
                        if let Some(assoc) = assoc {
                            delegate_assoc_items.add_item(assoc);
                        }
                    });
            };

            let target_scope = ctx.sema.scope(strukt.strukt.syntax())?;
            let source_scope = ctx.sema.scope(bound_def.syntax())?;
            let transform = PathTransform::generic_transformation(&target_scope, &source_scope);
            ast::Impl::cast(transform.apply(delegate.syntax()))
        }
        Delegee::Impls(trait_, old_impl) => {
            let old_impl = ctx.sema.source(old_impl.to_owned())?.value;
            let old_impl_params = old_impl.generic_param_list();

            // 1) Resolve conflicts between generic parameters in old_impl and
            // those in strukt.
            //
            // These generics parameters will also be used in `field_ty` and
            // `where_clauses`, so we should substitute arguments in them as well.
            let strukt_params = resolve_name_conflicts(strukt_params, &old_impl_params);
            let (field_ty, ty_where_clause) = match &strukt_params {
                Some(strukt_params) => {
                    let args = strukt_params.to_generic_args();
                    let field_ty = rename_strukt_args(ctx, ast_strukt, field_ty, &args)?;
                    let where_clause = ast_strukt
                        .where_clause()
                        .and_then(|wc| rename_strukt_args(ctx, ast_strukt, &wc, &args));
                    (field_ty, where_clause)
                }
                None => (field_ty.clone_for_update(), None),
            };

            // 2) Handle instantiated generics in `field_ty`.

            // 2.1) Some generics used in `self_ty` may be instantiated, so they
            // are no longer generics, we should remove and instantiate those
            // generics in advance.

            // `old_trait_args` contains names of generic args for trait in `old_impl`
            let old_impl_trait_args = old_impl
                .trait_()?
                .generic_arg_list()
                .map(|l| l.generic_args().map(|arg| arg.to_string()))
                .map_or_else(FxHashSet::default, |it| it.collect());

            let trait_gen_params = remove_instantiated_params(
                &old_impl.self_ty()?,
                old_impl_params.clone(),
                &old_impl_trait_args,
            );

            // 2.2) Generate generic args applied on impl.
            let transform_args = generate_args_for_impl(
                old_impl_params,
                &old_impl.self_ty()?,
                &field_ty,
                &trait_gen_params,
                &old_impl_trait_args,
            );

            // 2.3) Instantiate generics with `transform_impl`, this step also
            // remove unused params.
            let trait_gen_args = old_impl.trait_()?.generic_arg_list().and_then(|trait_args| {
                let trait_args = &mut trait_args.clone_for_update();
                if let Some(new_args) = transform_impl(
                    ctx,
                    ast_strukt,
                    &old_impl,
                    &transform_args,
                    trait_args.clone_subtree(),
                ) {
                    *trait_args = new_args.clone_subtree();
                    Some(new_args)
                } else {
                    None
                }
            });

            let type_gen_args = strukt_params.clone().map(|params| params.to_generic_args());
            let path_type =
                make::ty(&trait_.name(db).display_no_db(edition).to_smolstr()).clone_for_update();
            let path_type = transform_impl(ctx, ast_strukt, &old_impl, &transform_args, path_type)?;
            // 3) Generate delegate trait impl
            let delegate = make::impl_trait(
                trait_.is_unsafe(db),
                trait_gen_params,
                trait_gen_args,
                strukt_params,
                type_gen_args,
                trait_.is_auto(db),
                path_type,
                strukt_ty,
                old_impl.where_clause().map(|wc| wc.clone_for_update()),
                ty_where_clause,
                None,
            )
            .clone_for_update();
            // Goto link : https://doc.rust-lang.org/reference/paths.html#qualified-paths
            let qualified_path_type =
                make::path_from_text(&format!("<{} as {}>", field_ty, delegate.trait_()?));

            // 4) Transform associated items in delegte trait impl
            let delegate_assoc_items = delegate.get_or_create_assoc_item_list();
            for item in old_impl
                .get_or_create_assoc_item_list()
                .assoc_items()
                .filter(|item| matches!(item, AssocItem::MacroCall(_)).not())
            {
                let item = item.clone_for_update();
                let item = transform_impl(ctx, ast_strukt, &old_impl, &transform_args, item)?;

                let assoc = process_assoc_item(item, qualified_path_type.clone(), field_name)?;
                delegate_assoc_items.add_item(assoc);
            }

            // 5) Remove useless where clauses
            if let Some(wc) = delegate.where_clause() {
                remove_useless_where_clauses(&delegate.trait_()?, &delegate.self_ty()?, wc);
            }
            Some(delegate)
        }
    }
}

fn transform_impl<N: ast::AstNode>(
    ctx: &AssistContext<'_>,
    strukt: &ast::Struct,
    old_impl: &ast::Impl,
    args: &Option<GenericArgList>,
    syntax: N,
) -> Option<N> {
    let source_scope = ctx.sema.scope(old_impl.self_ty()?.syntax())?;
    let target_scope = ctx.sema.scope(strukt.syntax())?;
    let hir_old_impl = ctx.sema.to_impl_def(old_impl)?;

    let transform = args.as_ref().map_or_else(
        || PathTransform::generic_transformation(&target_scope, &source_scope),
        |args| {
            PathTransform::impl_transformation(
                &target_scope,
                &source_scope,
                hir_old_impl,
                args.clone(),
            )
        },
    );

    N::cast(transform.apply(syntax.syntax()))
}

fn remove_instantiated_params(
    self_ty: &ast::Type,
    old_impl_params: Option<GenericParamList>,
    old_trait_args: &FxHashSet<String>,
) -> Option<GenericParamList> {
    match self_ty {
        ast::Type::PathType(path_type) => {
            old_impl_params.and_then(|gpl| {
                // Remove generic parameters in field_ty (which is instantiated).
                let new_gpl = gpl.clone_for_update();

                path_type
                    .path()?
                    .segments()
                    .filter_map(|seg| seg.generic_arg_list())
                    .flat_map(|it| it.generic_args())
                    // However, if the param is also used in the trait arguments,
                    // it shouldn't be removed now, which will be instantiated in
                    // later `path_transform`
                    .filter(|arg| !old_trait_args.contains(&arg.to_string()))
                    .for_each(|arg| new_gpl.remove_generic_arg(&arg));
                (new_gpl.generic_params().count() > 0).then_some(new_gpl)
            })
        }
        _ => old_impl_params,
    }
}

fn remove_useless_where_clauses(trait_ty: &ast::Type, self_ty: &ast::Type, wc: ast::WhereClause) {
    let live_generics = [trait_ty, self_ty]
        .into_iter()
        .flat_map(|ty| ty.generic_arg_list())
        .flat_map(|gal| gal.generic_args())
        .map(|x| x.to_string())
        .collect::<FxHashSet<_>>();

    // Keep where-clauses that have generics after substitution, and remove the
    // rest.
    let has_live_generics = |pred: &WherePred| {
        pred.syntax()
            .descendants_with_tokens()
            .filter_map(|e| e.into_token())
            .any(|e| e.kind() == SyntaxKind::IDENT && live_generics.contains(&e.to_string()))
            .not()
    };
    wc.predicates().filter(has_live_generics).for_each(|pred| wc.remove_predicate(pred));

    if wc.predicates().count() == 0 {
        // Remove useless whitespaces
        [syntax::Direction::Prev, syntax::Direction::Next]
            .into_iter()
            .flat_map(|dir| {
                wc.syntax()
                    .siblings_with_tokens(dir)
                    .skip(1)
                    .take_while(|node_or_tok| node_or_tok.kind() == SyntaxKind::WHITESPACE)
            })
            .for_each(ted::remove);

        ted::insert(
            ted::Position::after(wc.syntax()),
            NodeOrToken::Token(make::token(SyntaxKind::WHITESPACE)),
        );
        // Remove where clause
        ted::remove(wc.syntax());
    }
}

// Generate generic args that should be apply to current impl.
//
// For example, say we have implementation `impl<A, B, C> Trait for B<A>`,
// and `b: B<T>` in struct `S<T>`. Then the `A` should be instantiated to `T`.
// While the last two generic args `B` and `C` doesn't change, it remains
// `<B, C>`. So we apply `<T, B, C>` as generic arguments to impl.
fn generate_args_for_impl(
    old_impl_gpl: Option<GenericParamList>,
    self_ty: &ast::Type,
    field_ty: &ast::Type,
    trait_params: &Option<GenericParamList>,
    old_trait_args: &FxHashSet<String>,
) -> Option<ast::GenericArgList> {
    let old_impl_args = old_impl_gpl.map(|gpl| gpl.to_generic_args().generic_args())?;
    // Create pairs of the args of `self_ty` and corresponding `field_ty` to
    // form the substitution list
    let mut arg_substs = FxHashMap::default();

    if let field_ty @ ast::Type::PathType(_) = field_ty {
        let field_args = field_ty.generic_arg_list().map(|gal| gal.generic_args());
        let self_ty_args = self_ty.generic_arg_list().map(|gal| gal.generic_args());
        if let (Some(field_args), Some(self_ty_args)) = (field_args, self_ty_args) {
            self_ty_args.zip(field_args).for_each(|(self_ty_arg, field_arg)| {
                arg_substs.entry(self_ty_arg.to_string()).or_insert(field_arg);
            })
        }
    }

    let args = old_impl_args
        .map(|old_arg| {
            arg_substs.get(&old_arg.to_string()).map_or_else(
                || old_arg.clone(),
                |replace_with| {
                    // The old_arg will be replaced, so it becomes redundant
                    if trait_params.is_some() && old_trait_args.contains(&old_arg.to_string()) {
                        trait_params.as_ref().unwrap().remove_generic_arg(&old_arg)
                    }
                    replace_with.clone()
                },
            )
        })
        .collect_vec();
    args.is_empty().not().then(|| make::generic_arg_list(args))
}

fn rename_strukt_args<N>(
    ctx: &AssistContext<'_>,
    strukt: &ast::Struct,
    item: &N,
    args: &GenericArgList,
) -> Option<N>
where
    N: ast::AstNode,
{
    let hir_strukt = ctx.sema.to_struct_def(strukt)?;
    let hir_adt = hir::Adt::from(hir_strukt);

    let item = item.clone_for_update();
    let scope = ctx.sema.scope(item.syntax())?;

    let transform = PathTransform::adt_transformation(&scope, &scope, hir_adt, args.clone());
    N::cast(transform.apply(item.syntax()))
}

fn has_self_type(trait_: hir::Trait, ctx: &AssistContext<'_>) -> Option<()> {
    let trait_source = ctx.sema.source(trait_)?.value;
    trait_source
        .syntax()
        .descendants_with_tokens()
        .filter_map(|e| e.into_token())
        .find(|e| e.kind() == SyntaxKind::SELF_TYPE_KW)
        .map(|_| ())
}

fn resolve_name_conflicts(
    strukt_params: Option<ast::GenericParamList>,
    old_impl_params: &Option<ast::GenericParamList>,
) -> Option<ast::GenericParamList> {
    match (strukt_params, old_impl_params) {
        (Some(old_strukt_params), Some(old_impl_params)) => {
            let params = make::generic_param_list(std::iter::empty()).clone_for_update();

            for old_strukt_param in old_strukt_params.generic_params() {
                // Get old name from `strukt`
                let name = SmolStr::from(match &old_strukt_param {
                    ast::GenericParam::ConstParam(c) => c.name()?.to_string(),
                    ast::GenericParam::LifetimeParam(l) => {
                        l.lifetime()?.lifetime_ident_token()?.to_string()
                    }
                    ast::GenericParam::TypeParam(t) => t.name()?.to_string(),
                });

                // The new name cannot be conflicted with generics in trait, and the renamed names.
                let param_list_to_names = |param_list: &GenericParamList| {
                    param_list.generic_params().flat_map(|param| match param {
                        ast::GenericParam::TypeParam(t) => t.name().map(|name| name.to_string()),
                        p => Some(p.to_string()),
                    })
                };
                let existing_names = param_list_to_names(old_impl_params)
                    .chain(param_list_to_names(&params))
                    .collect_vec();
                let mut name_generator = suggest_name::NameGenerator::new_with_names(
                    existing_names.iter().map(|s| s.as_str()),
                );
                let name = name_generator.suggest_name(&name);
                match old_strukt_param {
                    ast::GenericParam::ConstParam(c) => {
                        if let Some(const_ty) = c.ty() {
                            let const_param = make::const_param(make::name(&name), const_ty);
                            params.add_generic_param(ast::GenericParam::ConstParam(
                                const_param.clone_for_update(),
                            ));
                        }
                    }
                    p @ ast::GenericParam::LifetimeParam(_) => {
                        params.add_generic_param(p.clone_for_update());
                    }
                    ast::GenericParam::TypeParam(t) => {
                        let type_bounds = t.type_bound_list();
                        let type_param = make::type_param(make::name(&name), type_bounds);
                        params.add_generic_param(ast::GenericParam::TypeParam(
                            type_param.clone_for_update(),
                        ));
                    }
                }
            }
            Some(params)
        }
        (Some(old_strukt_gpl), None) => Some(old_strukt_gpl),
        _ => None,
    }
}

fn process_assoc_item(
    item: syntax::ast::AssocItem,
    qual_path_ty: ast::Path,
    base_name: &str,
) -> Option<ast::AssocItem> {
    let attrs = item.attrs();
    let assoc = match item {
        AssocItem::Const(c) => const_assoc_item(c, qual_path_ty),
        AssocItem::Fn(f) => func_assoc_item(f, qual_path_ty, base_name),
        AssocItem::MacroCall(_) => {
            // FIXME : Handle MacroCall case.
            // macro_assoc_item(mac, qual_path_ty)
            None
        }
        AssocItem::TypeAlias(ta) => ty_assoc_item(ta, qual_path_ty),
    };
    if let Some(assoc) = &assoc {
        attrs.for_each(|attr| {
            assoc.add_attr(attr.clone());
            // fix indentations
            if let Some(tok) = attr.syntax().next_sibling_or_token() {
                let pos = Position::after(tok);
                ted::insert(pos, make::tokens::whitespace("    "));
            }
        })
    }
    assoc
}

fn const_assoc_item(item: syntax::ast::Const, qual_path_ty: ast::Path) -> Option<AssocItem> {
    let path_expr_segment = make::path_from_text(item.name()?.to_string().as_str());

    // We want rhs of the const assignment to be a qualified path
    // The general case for const assignment can be found [here](`https://doc.rust-lang.org/reference/items/constant-items.html`)
    // The qualified will have the following generic syntax :
    // <Base as Trait<GenArgs>>::ConstName;
    // FIXME : We can't rely on `make::path_qualified` for now but it would be nice to replace the following with it.
    // make::path_qualified(qual_path_ty, path_expr_segment.as_single_segment().unwrap());
    let qualified_path = qualified_path(qual_path_ty, path_expr_segment);
    let inner = make::item_const(
        item.visibility(),
        item.name()?,
        item.ty()?,
        make::expr_path(qualified_path),
    )
    .clone_for_update();

    Some(AssocItem::Const(inner))
}

fn func_assoc_item(
    item: syntax::ast::Fn,
    qual_path_ty: Path,
    base_name: &str,
) -> Option<AssocItem> {
    let path_expr_segment = make::path_from_text(item.name()?.to_string().as_str());
    let qualified_path = qualified_path(qual_path_ty, path_expr_segment);

    let call = match item.param_list() {
        // Methods and funcs should be handled separately.
        // We ask if the func has a `self` param.
        Some(l) => match l.self_param() {
            Some(slf) => {
                let mut self_kw = make::expr_path(make::path_from_text("self"));
                self_kw = make::expr_field(self_kw, base_name);

                let tail_expr_self = match slf.kind() {
                    ast::SelfParamKind::Owned => self_kw,
                    ast::SelfParamKind::Ref => make::expr_ref(self_kw, false),
                    ast::SelfParamKind::MutRef => make::expr_ref(self_kw, true),
                };

                let param_count = l.params().count();
                let args = convert_param_list_to_arg_list(l).clone_for_update();
                let pos_after_l_paren = Position::after(args.l_paren_token()?);
                if param_count > 0 {
                    // Add SelfParam and a TOKEN::COMMA
                    ted::insert_all_raw(
                        pos_after_l_paren,
                        vec![
                            NodeOrToken::Node(tail_expr_self.syntax().clone_for_update()),
                            NodeOrToken::Token(make::token(SyntaxKind::COMMA)),
                            NodeOrToken::Token(make::token(SyntaxKind::WHITESPACE)),
                        ],
                    );
                } else {
                    // Add SelfParam only
                    ted::insert_raw(
                        pos_after_l_paren,
                        NodeOrToken::Node(tail_expr_self.syntax().clone_for_update()),
                    );
                }

                make::expr_call(make::expr_path(qualified_path), args)
            }
            None => {
                make::expr_call(make::expr_path(qualified_path), convert_param_list_to_arg_list(l))
            }
        },
        None => make::expr_call(
            make::expr_path(qualified_path),
            convert_param_list_to_arg_list(make::param_list(None, Vec::new())),
        ),
    }
    .clone_for_update();

    let body = make::block_expr(vec![], Some(call.into())).clone_for_update();
    let func = make::fn_(
        item.visibility(),
        item.name()?,
        item.generic_param_list(),
        item.where_clause(),
        item.param_list()?,
        body,
        item.ret_type(),
        item.async_token().is_some(),
        item.const_token().is_some(),
        item.unsafe_token().is_some(),
        item.gen_token().is_some(),
    )
    .clone_for_update();

    Some(AssocItem::Fn(func.indent(edit::IndentLevel(1))))
}

fn ty_assoc_item(item: syntax::ast::TypeAlias, qual_path_ty: Path) -> Option<AssocItem> {
    let path_expr_segment = make::path_from_text(item.name()?.to_string().as_str());
    let qualified_path = qualified_path(qual_path_ty, path_expr_segment);
    let ty = make::ty_path(qualified_path);
    let ident = item.name()?.to_string();

    let alias = make::ty_alias(
        ident.as_str(),
        item.generic_param_list(),
        None,
        item.where_clause(),
        Some((ty, None)),
    )
    .clone_for_update();

    Some(AssocItem::TypeAlias(alias))
}

fn qualified_path(qual_path_ty: ast::Path, path_expr_seg: ast::Path) -> ast::Path {
    make::path_from_text(&format!("{qual_path_ty}::{path_expr_seg}"))
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::tests::{
        check_assist, check_assist_not_applicable, check_assist_not_applicable_no_grouping,
    };

    #[test]
    fn test_tuple_struct_basic() {
        check_assist(
            generate_delegate_trait,
            r#"
struct Base;
struct S(B$0ase);
trait Trait {}
impl Trait for Base {}
"#,
            r#"
struct Base;
struct S(Base);

impl Trait for S {}
trait Trait {}
impl Trait for Base {}
"#,
        );
    }

    #[test]
    fn test_self_ty() {
        // trait with `Self` type cannot be delegated
        //
        // See the function `fn f() -> Self`.
        // It should be `fn f() -> Base` in `Base`, and `fn f() -> S` in `S`
        check_assist_not_applicable(
            generate_delegate_trait,
            r#"
struct Base(());
struct S(B$0ase);
trait Trait {
    fn f() -> Self;
}
impl Trait for Base {
    fn f() -> Base {
        Base(())
    }
}
"#,
        );
    }

    #[test]
    fn test_struct_struct_basic() {
        check_assist(
            generate_delegate_trait,
            r#"
struct Base;
struct S {
    ba$0se : Base
}
trait Trait {}
impl Trait for Base {}
"#,
            r#"
struct Base;
struct S {
    base : Base
}

impl Trait for S {}
trait Trait {}
impl Trait for Base {}
"#,
        )
    }

    // Structs need to be by def populated with fields
    // However user can invoke this assist while still editing
    // We therefore assert its non-applicability
    #[test]
    fn test_yet_empty_struct() {
        check_assist_not_applicable(
            generate_delegate_trait,
            r#"
struct Base;
struct S {
    $0
}

impl Trait for S {}
trait Trait {}
impl Trait for Base {}
"#,
        )
    }

    #[test]
    fn test_yet_unspecified_field_type() {
        check_assist_not_applicable(
            generate_delegate_trait,
            r#"
struct Base;
struct S {
    ab$0c
}

impl Trait for S {}
trait Trait {}
impl Trait for Base {}
"#,
        );
    }

    #[test]
    fn test_unsafe_trait() {
        check_assist(
            generate_delegate_trait,
            r#"
struct Base;
struct S {
    ba$0se : Base
}
unsafe trait Trait {}
unsafe impl Trait for Base {}
"#,
            r#"
struct Base;
struct S {
    base : Base
}

unsafe impl Trait for S {}
unsafe trait Trait {}
unsafe impl Trait for Base {}
"#,
        );
    }

    #[test]
    fn test_unsafe_trait_with_unsafe_fn() {
        check_assist(
            generate_delegate_trait,
            r#"
struct Base;
struct S {
    ba$0se: Base,
}

unsafe trait Trait {
    unsafe fn a_func();
    unsafe fn a_method(&self);
}
unsafe impl Trait for Base {
    unsafe fn a_func() {}
    unsafe fn a_method(&self) {}
}
"#,
            r#"
struct Base;
struct S {
    base: Base,
}

unsafe impl Trait for S {
    unsafe fn a_func() {
        <Base as Trait>::a_func()
    }

    unsafe fn a_method(&self) {
        <Base as Trait>::a_method(&self.base)
    }
}

unsafe trait Trait {
    unsafe fn a_func();
    unsafe fn a_method(&self);
}
unsafe impl Trait for Base {
    unsafe fn a_func() {}
    unsafe fn a_method(&self) {}
}
"#,
        );
    }

    #[test]
    fn test_struct_with_where_clause() {
        check_assist(
            generate_delegate_trait,
            r#"
trait AnotherTrait {}
struct S<T>
where
    T: AnotherTrait,
{
    b$0 : T,
}"#,
            r#"
trait AnotherTrait {}
struct S<T>
where
    T: AnotherTrait,
{
    b : T,
}

impl<T> AnotherTrait for S<T>
where
    T: AnotherTrait,
{
}"#,
        );
    }

    #[test]
    fn test_fields_with_generics() {
        check_assist(
            generate_delegate_trait,
            r#"
struct B<T> {
    a: T
}

trait Trait<T> {
    fn f(&self, a: T) -> T;
}

impl<T1, T2> Trait<T1> for B<T2> {
    fn f(&self, a: T1) -> T1 { a }
}

struct A {}
struct S {
    b :$0 B<A>,
}
"#,
            r#"
struct B<T> {
    a: T
}

trait Trait<T> {
    fn f(&self, a: T) -> T;
}

impl<T1, T2> Trait<T1> for B<T2> {
    fn f(&self, a: T1) -> T1 { a }
}

struct A {}
struct S {
    b : B<A>,
}

impl<T1> Trait<T1> for S {
    fn f(&self, a: T1) -> T1 {
        <B<A> as Trait<T1>>::f(&self.b, a)
    }
}
"#,
        );
    }

    #[test]
    fn test_generics_with_conflict_names() {
        check_assist(
            generate_delegate_trait,
            r#"
struct B<T> {
    a: T
}

trait Trait<T> {
    fn f(&self, a: T) -> T;
}

impl<T, T0> Trait<T> for B<T0> {
    fn f(&self, a: T) -> T { a }
}

struct S<T> {
    b : $0B<T>,
}
"#,
            r#"
struct B<T> {
    a: T
}

trait Trait<T> {
    fn f(&self, a: T) -> T;
}

impl<T, T0> Trait<T> for B<T0> {
    fn f(&self, a: T) -> T { a }
}

struct S<T> {
    b : B<T>,
}

impl<T, T1> Trait<T> for S<T1> {
    fn f(&self, a: T) -> T {
        <B<T1> as Trait<T>>::f(&self.b, a)
    }
}
"#,
        );
    }

    #[test]
    fn test_lifetime_with_conflict_names() {
        check_assist(
            generate_delegate_trait,
            r#"
struct B<'a, T> {
    a: &'a T
}

trait Trait<T> {
    fn f(&self, a: T) -> T;
}

impl<'a, T, T0> Trait<T> for B<'a, T0> {
    fn f(&self, a: T) -> T { a }
}

struct S<'a, T> {
    b : $0B<'a, T>,
}
"#,
            r#"
struct B<'a, T> {
    a: &'a T
}

trait Trait<T> {
    fn f(&self, a: T) -> T;
}

impl<'a, T, T0> Trait<T> for B<'a, T0> {
    fn f(&self, a: T) -> T { a }
}

struct S<'a, T> {
    b : B<'a, T>,
}

impl<'a, T, T1> Trait<T> for S<'a, T1> {
    fn f(&self, a: T) -> T {
        <B<'a, T1> as Trait<T>>::f(&self.b, a)
    }
}
"#,
        );
    }

    #[test]
    fn test_multiple_generics() {
        check_assist(
            generate_delegate_trait,
            r#"
struct B<T1, T2> {
    a: T1,
    b: T2
}

trait Trait<T> {
    fn f(&self, a: T) -> T;
}

impl<T, T0> Trait<T> for B<T, T0> {
    fn f(&self, a: T) -> T { a }
}

struct S<T> {
    b :$0 B<i32, T>,
}
"#,
            r#"
struct B<T1, T2> {
    a: T1,
    b: T2
}

trait Trait<T> {
    fn f(&self, a: T) -> T;
}

impl<T, T0> Trait<T> for B<T, T0> {
    fn f(&self, a: T) -> T { a }
}

struct S<T> {
    b : B<i32, T>,
}

impl<T1> Trait<i32> for S<T1> {
    fn f(&self, a: i32) -> i32 {
        <B<i32, T1> as Trait<i32>>::f(&self.b, a)
    }
}
"#,
        );
    }

    #[test]
    fn test_generics_multiplex() {
        check_assist(
            generate_delegate_trait,
            r#"
struct B<T> {
    a: T
}

trait Trait<T> {
    fn f(&self, a: T) -> T;
}

impl<T> Trait<T> for B<T> {
    fn f(&self, a: T) -> T { a }
}

struct S<T> {
    b : $0B<T>,
}
"#,
            r#"
struct B<T> {
    a: T
}

trait Trait<T> {
    fn f(&self, a: T) -> T;
}

impl<T> Trait<T> for B<T> {
    fn f(&self, a: T) -> T { a }
}

struct S<T> {
    b : B<T>,
}

impl<T1> Trait<T1> for S<T1> {
    fn f(&self, a: T1) -> T1 {
        <B<T1> as Trait<T1>>::f(&self.b, a)
    }
}
"#,
        );
    }

    #[test]
    fn test_complex_without_where() {
        check_assist(
            generate_delegate_trait,
            r#"
trait Trait<'a, T, const C: usize> {
    type AssocType;
    const AssocConst: usize;
    fn assoc_fn(p: ());
    fn assoc_method(&self, p: ());
}

struct Base;
struct S {
    field$0: Base
}

impl<'a, T, const C: usize> Trait<'a, T, C> for Base {
    type AssocType = ();
    const AssocConst: usize = 0;
    fn assoc_fn(p: ()) {}
    fn assoc_method(&self, p: ()) {}
}
"#,
            r#"
trait Trait<'a, T, const C: usize> {
    type AssocType;
    const AssocConst: usize;
    fn assoc_fn(p: ());
    fn assoc_method(&self, p: ());
}

struct Base;
struct S {
    field: Base
}

impl<'a, T, const C: usize> Trait<'a, T, C> for S {
    type AssocType = <Base as Trait<'a, T, C>>::AssocType;

    const AssocConst: usize = <Base as Trait<'a, T, C>>::AssocConst;

    fn assoc_fn(p: ()) {
        <Base as Trait<'a, T, C>>::assoc_fn(p)
    }

    fn assoc_method(&self, p: ()) {
        <Base as Trait<'a, T, C>>::assoc_method(&self.field, p)
    }
}

impl<'a, T, const C: usize> Trait<'a, T, C> for Base {
    type AssocType = ();
    const AssocConst: usize = 0;
    fn assoc_fn(p: ()) {}
    fn assoc_method(&self, p: ()) {}
}
"#,
        );
    }

    #[test]
    fn test_complex_two() {
        check_assist(
            generate_delegate_trait,
            r"
trait AnotherTrait {}

trait Trait<'a, T, const C: usize> {
    type AssocType;
    const AssocConst: usize;
    fn assoc_fn(p: ());
    fn assoc_method(&self, p: ());
}

struct Base;
struct S {
    fi$0eld: Base,
}

impl<'b, C, const D: usize> Trait<'b, C, D> for Base
where
    C: AnotherTrait,
{
    type AssocType = ();
    const AssocConst: usize = 0;
    fn assoc_fn(p: ()) {}
    fn assoc_method(&self, p: ()) {}
}",
            r#"
trait AnotherTrait {}

trait Trait<'a, T, const C: usize> {
    type AssocType;
    const AssocConst: usize;
    fn assoc_fn(p: ());
    fn assoc_method(&self, p: ());
}

struct Base;
struct S {
    field: Base,
}

impl<'b, C, const D: usize> Trait<'b, C, D> for S
where
    C: AnotherTrait,
{
    type AssocType = <Base as Trait<'b, C, D>>::AssocType;

    const AssocConst: usize = <Base as Trait<'b, C, D>>::AssocConst;

    fn assoc_fn(p: ()) {
        <Base as Trait<'b, C, D>>::assoc_fn(p)
    }

    fn assoc_method(&self, p: ()) {
        <Base as Trait<'b, C, D>>::assoc_method(&self.field, p)
    }
}

impl<'b, C, const D: usize> Trait<'b, C, D> for Base
where
    C: AnotherTrait,
{
    type AssocType = ();
    const AssocConst: usize = 0;
    fn assoc_fn(p: ()) {}
    fn assoc_method(&self, p: ()) {}
}"#,
        )
    }

    #[test]
    fn test_complex_three() {
        check_assist(
            generate_delegate_trait,
            r#"
trait AnotherTrait {}
trait YetAnotherTrait {}

struct StructImplsAll();
impl AnotherTrait for StructImplsAll {}
impl YetAnotherTrait for StructImplsAll {}

trait Trait<'a, T, const C: usize> {
    type A;
    const ASSOC_CONST: usize = C;
    fn assoc_fn(p: ());
    fn assoc_method(&self, p: ());
}

struct Base;
struct S {
    fi$0eld: Base,
}

impl<'b, A: AnotherTrait + YetAnotherTrait, const B: usize> Trait<'b, A, B> for Base
where
    A: AnotherTrait,
{
    type A = i32;

    const ASSOC_CONST: usize = B;

    fn assoc_fn(p: ()) {}

    fn assoc_method(&self, p: ()) {}
}
"#,
            r#"
trait AnotherTrait {}
trait YetAnotherTrait {}

struct StructImplsAll();
impl AnotherTrait for StructImplsAll {}
impl YetAnotherTrait for StructImplsAll {}

trait Trait<'a, T, const C: usize> {
    type A;
    const ASSOC_CONST: usize = C;
    fn assoc_fn(p: ());
    fn assoc_method(&self, p: ());
}

struct Base;
struct S {
    field: Base,
}

impl<'b, A: AnotherTrait + YetAnotherTrait, const B: usize> Trait<'b, A, B> for S
where
    A: AnotherTrait,
{
    type A = <Base as Trait<'b, A, B>>::A;

    const ASSOC_CONST: usize = <Base as Trait<'b, A, B>>::ASSOC_CONST;

    fn assoc_fn(p: ()) {
        <Base as Trait<'b, A, B>>::assoc_fn(p)
    }

    fn assoc_method(&self, p: ()) {
        <Base as Trait<'b, A, B>>::assoc_method(&self.field, p)
    }
}

impl<'b, A: AnotherTrait + YetAnotherTrait, const B: usize> Trait<'b, A, B> for Base
where
    A: AnotherTrait,
{
    type A = i32;

    const ASSOC_CONST: usize = B;

    fn assoc_fn(p: ()) {}

    fn assoc_method(&self, p: ()) {}
}
"#,
        )
    }

    #[test]
    fn test_type_bound() {
        check_assist(
            generate_delegate_trait,
            r#"
trait AnotherTrait {}
struct S<T>
where
    T: AnotherTrait,
{
    b$0: T,
}"#,
            r#"
trait AnotherTrait {}
struct S<T>
where
    T: AnotherTrait,
{
    b: T,
}

impl<T> AnotherTrait for S<T>
where
    T: AnotherTrait,
{
}"#,
        );
    }

    #[test]
    fn test_type_bound_with_generics_1() {
        check_assist(
            generate_delegate_trait,
            r#"
trait AnotherTrait {}
struct B<T, T1>
where
    T1: AnotherTrait
{
    a: T,
    b: T1
}

trait Trait<T> {
    fn f(&self, a: T) -> T;
}

impl<T, T0, T1: AnotherTrait> Trait<T> for B<T0, T1> {
    fn f(&self, a: T) -> T { a }
}

struct S<T, T1>
where
    T1: AnotherTrait
{
    b : $0B<T, T1>,
}"#,
            r#"
trait AnotherTrait {}
struct B<T, T1>
where
    T1: AnotherTrait
{
    a: T,
    b: T1
}

trait Trait<T> {
    fn f(&self, a: T) -> T;
}

impl<T, T0, T1: AnotherTrait> Trait<T> for B<T0, T1> {
    fn f(&self, a: T) -> T { a }
}

struct S<T, T1>
where
    T1: AnotherTrait
{
    b : B<T, T1>,
}

impl<T, T2, T3> Trait<T> for S<T2, T3>
where
    T3: AnotherTrait
{
    fn f(&self, a: T) -> T {
        <B<T2, T3> as Trait<T>>::f(&self.b, a)
    }
}"#,
        );
    }

    #[test]
    fn test_type_bound_with_generics_2() {
        check_assist(
            generate_delegate_trait,
            r#"
trait AnotherTrait {}
struct B<T1>
where
    T1: AnotherTrait
{
    b: T1
}

trait Trait<T1> {
    fn f(&self, a: T1) -> T1;
}

impl<T, T1: AnotherTrait> Trait<T> for B<T1> {
    fn f(&self, a: T) -> T { a }
}

struct S<T>
where
    T: AnotherTrait
{
    b : $0B<T>,
}"#,
            r#"
trait AnotherTrait {}
struct B<T1>
where
    T1: AnotherTrait
{
    b: T1
}

trait Trait<T1> {
    fn f(&self, a: T1) -> T1;
}

impl<T, T1: AnotherTrait> Trait<T> for B<T1> {
    fn f(&self, a: T) -> T { a }
}

struct S<T>
where
    T: AnotherTrait
{
    b : B<T>,
}

impl<T, T2> Trait<T> for S<T2>
where
    T2: AnotherTrait
{
    fn f(&self, a: T) -> T {
        <B<T2> as Trait<T>>::f(&self.b, a)
    }
}"#,
        );
    }

    #[test]
    fn test_docstring_example() {
        check_assist(
            generate_delegate_trait,
            r#"
trait SomeTrait {
    type T;
    fn fn_(arg: u32) -> u32;
    fn method_(&mut self) -> bool;
}
struct A;
impl SomeTrait for A {
    type T = u32;
    fn fn_(arg: u32) -> u32 {
        42
    }
    fn method_(&mut self) -> bool {
        false
    }
}
struct B {
    a$0: A,
}
"#,
            r#"
trait SomeTrait {
    type T;
    fn fn_(arg: u32) -> u32;
    fn method_(&mut self) -> bool;
}
struct A;
impl SomeTrait for A {
    type T = u32;
    fn fn_(arg: u32) -> u32 {
        42
    }
    fn method_(&mut self) -> bool {
        false
    }
}
struct B {
    a: A,
}

impl SomeTrait for B {
    type T = <A as SomeTrait>::T;

    fn fn_(arg: u32) -> u32 {
        <A as SomeTrait>::fn_(arg)
    }

    fn method_(&mut self) -> bool {
        <A as SomeTrait>::method_(&mut self.a)
    }
}
"#,
        );
    }

    #[test]
    fn import_from_other_mod() {
        check_assist(
            generate_delegate_trait,
            r#"
mod some_module {
    pub trait SomeTrait {
        type T;
        fn fn_(arg: u32) -> u32;
        fn method_(&mut self) -> bool;
    }
    pub struct A;
    impl SomeTrait for A {
        type T = u32;

        fn fn_(arg: u32) -> u32 {
            42
        }

        fn method_(&mut self) -> bool {
            false
        }
    }
}

struct B {
    a$0: some_module::A,
}"#,
            r#"
mod some_module {
    pub trait SomeTrait {
        type T;
        fn fn_(arg: u32) -> u32;
        fn method_(&mut self) -> bool;
    }
    pub struct A;
    impl SomeTrait for A {
        type T = u32;

        fn fn_(arg: u32) -> u32 {
            42
        }

        fn method_(&mut self) -> bool {
            false
        }
    }
}

struct B {
    a: some_module::A,
}

impl some_module::SomeTrait for B {
    type T = <some_module::A as some_module::SomeTrait>::T;

    fn fn_(arg: u32) -> u32 {
        <some_module::A as some_module::SomeTrait>::fn_(arg)
    }

    fn method_(&mut self) -> bool {
        <some_module::A as some_module::SomeTrait>::method_(&mut self.a)
    }
}"#,
        )
    }

    #[test]
    fn test_fn_with_attrs() {
        check_assist(
            generate_delegate_trait,
            r#"
struct A;

trait T {
    #[cfg(test)]
    fn f(&self, a: u32);
    #[cfg(not(test))]
    fn f(&self, a: bool);
}

impl T for A {
    #[cfg(test)]
    fn f(&self, a: u32) {}
    #[cfg(not(test))]
    fn f(&self, a: bool) {}
}

struct B {
    a$0: A,
}
"#,
            r#"
struct A;

trait T {
    #[cfg(test)]
    fn f(&self, a: u32);
    #[cfg(not(test))]
    fn f(&self, a: bool);
}

impl T for A {
    #[cfg(test)]
    fn f(&self, a: u32) {}
    #[cfg(not(test))]
    fn f(&self, a: bool) {}
}

struct B {
    a: A,
}

impl T for B {
    #[cfg(test)]
    fn f(&self, a: u32) {
        <A as T>::f(&self.a, a)
    }

    #[cfg(not(test))]
    fn f(&self, a: bool) {
        <A as T>::f(&self.a, a)
    }
}
"#,
        );
    }

    #[test]
    fn assoc_items_attributes_mutably_cloned() {
        check_assist(
            generate_delegate_trait,
            r#"
pub struct A;
pub trait C<D> {
    #[allow(clippy::dead_code)]
    fn a_funk(&self) -> &D;
}

pub struct B<T: C<A>> {
    has_dr$0ain: T,
}
"#,
            r#"
pub struct A;
pub trait C<D> {
    #[allow(clippy::dead_code)]
    fn a_funk(&self) -> &D;
}

pub struct B<T: C<A>> {
    has_drain: T,
}

impl<D, T: C<A>> C<D> for B<T> {
    #[allow(clippy::dead_code)]
    fn a_funk(&self) -> &D {
        <T as C<D>>::a_funk(&self.has_drain)
    }
}
"#,
        )
    }

    #[test]
    fn delegate_trait_skipped_when_no_grouping() {
        check_assist_not_applicable_no_grouping(
            generate_delegate_trait,
            r#"
trait SomeTrait {
    type T;
    fn fn_(arg: u32) -> u32;
    fn method_(&mut self) -> bool;
}
struct A;
impl SomeTrait for A {
    type T = u32;

    fn fn_(arg: u32) -> u32 {
        42
    }

    fn method_(&mut self) -> bool {
        false
    }
}
struct B {
    a$0 : A,
}
"#,
        );
    }
}
