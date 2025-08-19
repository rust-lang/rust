//! See [`PathTransform`].

use crate::helpers::mod_path_to_ast;
use either::Either;
use hir::{
    AsAssocItem, HirDisplay, HirFileId, ImportPathConfig, ModuleDef, SemanticsScope,
    prettify_macro_expansion,
};
use itertools::Itertools;
use rustc_hash::FxHashMap;
use span::Edition;
use syntax::{
    NodeOrToken, SyntaxNode,
    ast::{self, AstNode, HasGenericArgs, make},
    syntax_editor::{self, SyntaxEditor},
};

#[derive(Default, Debug)]
struct AstSubsts {
    types_and_consts: Vec<TypeOrConst>,
    lifetimes: Vec<ast::LifetimeArg>,
}

#[derive(Debug)]
enum TypeOrConst {
    Either(ast::TypeArg), // indistinguishable type or const param
    Const(ast::ConstArg),
}

type LifetimeName = String;
type DefaultedParam = Either<hir::TypeParam, hir::ConstParam>;

/// `PathTransform` substitutes path in SyntaxNodes in bulk.
///
/// This is mostly useful for IDE code generation. If you paste some existing
/// code into a new context (for example, to add method overrides to an `impl`
/// block), you generally want to appropriately qualify the names, and sometimes
/// you might want to substitute generic parameters as well:
///
/// ```ignore
/// mod x {
///   pub struct A<V>;
///   pub trait T<U> { fn foo(&self, _: U) -> A<U>; }
/// }
///
/// mod y {
///   use x::T;
///
///   impl T<()> for () {
///      // If we invoke **Add Missing Members** here, we want to copy-paste `foo`.
///      // But we want a slightly-modified version of it:
///      fn foo(&self, _: ()) -> x::A<()> {}
///   }
/// }
/// ```
pub struct PathTransform<'a> {
    generic_def: Option<hir::GenericDef>,
    substs: AstSubsts,
    target_scope: &'a SemanticsScope<'a>,
    source_scope: &'a SemanticsScope<'a>,
}

impl<'a> PathTransform<'a> {
    pub fn trait_impl(
        target_scope: &'a SemanticsScope<'a>,
        source_scope: &'a SemanticsScope<'a>,
        trait_: hir::Trait,
        impl_: ast::Impl,
    ) -> PathTransform<'a> {
        PathTransform {
            source_scope,
            target_scope,
            generic_def: Some(trait_.into()),
            substs: get_syntactic_substs(impl_).unwrap_or_default(),
        }
    }

    pub fn function_call(
        target_scope: &'a SemanticsScope<'a>,
        source_scope: &'a SemanticsScope<'a>,
        function: hir::Function,
        generic_arg_list: ast::GenericArgList,
    ) -> PathTransform<'a> {
        PathTransform {
            source_scope,
            target_scope,
            generic_def: Some(function.into()),
            substs: get_type_args_from_arg_list(generic_arg_list).unwrap_or_default(),
        }
    }

    pub fn impl_transformation(
        target_scope: &'a SemanticsScope<'a>,
        source_scope: &'a SemanticsScope<'a>,
        impl_: hir::Impl,
        generic_arg_list: ast::GenericArgList,
    ) -> PathTransform<'a> {
        PathTransform {
            source_scope,
            target_scope,
            generic_def: Some(impl_.into()),
            substs: get_type_args_from_arg_list(generic_arg_list).unwrap_or_default(),
        }
    }

    pub fn adt_transformation(
        target_scope: &'a SemanticsScope<'a>,
        source_scope: &'a SemanticsScope<'a>,
        adt: hir::Adt,
        generic_arg_list: ast::GenericArgList,
    ) -> PathTransform<'a> {
        PathTransform {
            source_scope,
            target_scope,
            generic_def: Some(adt.into()),
            substs: get_type_args_from_arg_list(generic_arg_list).unwrap_or_default(),
        }
    }

    pub fn generic_transformation(
        target_scope: &'a SemanticsScope<'a>,
        source_scope: &'a SemanticsScope<'a>,
    ) -> PathTransform<'a> {
        PathTransform {
            source_scope,
            target_scope,
            generic_def: None,
            substs: AstSubsts::default(),
        }
    }

    #[must_use]
    pub fn apply(&self, syntax: &SyntaxNode) -> SyntaxNode {
        self.build_ctx().apply(syntax)
    }

    #[must_use]
    pub fn apply_all<'b>(
        &self,
        nodes: impl IntoIterator<Item = &'b SyntaxNode>,
    ) -> Vec<SyntaxNode> {
        let ctx = self.build_ctx();
        nodes.into_iter().map(|node| ctx.apply(&node.clone())).collect()
    }

    fn prettify_target_node(&self, node: SyntaxNode) -> SyntaxNode {
        match self.target_scope.file_id() {
            HirFileId::FileId(_) => node,
            HirFileId::MacroFile(file_id) => {
                let db = self.target_scope.db;
                prettify_macro_expansion(
                    db,
                    node,
                    &db.expansion_span_map(file_id),
                    self.target_scope.module().krate().into(),
                )
            }
        }
    }

    fn prettify_target_ast<N: AstNode>(&self, node: N) -> N {
        N::cast(self.prettify_target_node(node.syntax().clone())).unwrap()
    }

    fn build_ctx(&self) -> Ctx<'a> {
        let db = self.source_scope.db;
        let target_module = self.target_scope.module();
        let source_module = self.source_scope.module();
        let skip = match self.generic_def {
            // this is a trait impl, so we need to skip the first type parameter (i.e. Self) -- this is a bit hacky
            Some(hir::GenericDef::Trait(_)) => 1,
            _ => 0,
        };
        let mut type_substs: FxHashMap<hir::TypeParam, ast::Type> = Default::default();
        let mut const_substs: FxHashMap<hir::ConstParam, SyntaxNode> = Default::default();
        let mut defaulted_params: Vec<DefaultedParam> = Default::default();
        let target_edition = target_module.krate().edition(self.source_scope.db);
        self.generic_def
            .into_iter()
            .flat_map(|it| it.type_or_const_params(db))
            .skip(skip)
            // The actual list of trait type parameters may be longer than the one
            // used in the `impl` block due to trailing default type parameters.
            // For that case we extend the `substs` with an empty iterator so we
            // can still hit those trailing values and check if they actually have
            // a default type. If they do, go for that type from `hir` to `ast` so
            // the resulting change can be applied correctly.
            .zip(self.substs.types_and_consts.iter().map(Some).chain(std::iter::repeat(None)))
            .for_each(|(k, v)| match (k.split(db), v) {
                (Either::Right(k), Some(TypeOrConst::Either(v))) => {
                    if let Some(ty) = v.ty() {
                        type_substs.insert(k, self.prettify_target_ast(ty));
                    }
                }
                (Either::Right(k), None) => {
                    if let Some(default) = k.default(db)
                        && let Some(default) =
                            &default.display_source_code(db, source_module.into(), false).ok()
                    {
                        type_substs.insert(k, make::ty(default).clone_for_update());
                        defaulted_params.push(Either::Left(k));
                    }
                }
                (Either::Left(k), Some(TypeOrConst::Either(v))) => {
                    if let Some(ty) = v.ty() {
                        const_substs.insert(k, self.prettify_target_node(ty.syntax().clone()));
                    }
                }
                (Either::Left(k), Some(TypeOrConst::Const(v))) => {
                    if let Some(expr) = v.expr() {
                        // FIXME: expressions in curly brackets can cause ambiguity after insertion
                        // (e.g. `N * 2` -> `{1 + 1} * 2`; it's unclear whether `{1 + 1}`
                        // is a standalone statement or a part of another expression)
                        // and sometimes require slight modifications; see
                        // https://doc.rust-lang.org/reference/statements.html#expression-statements
                        // (default values in curly brackets can cause the same problem)
                        const_substs.insert(k, self.prettify_target_node(expr.syntax().clone()));
                    }
                }
                (Either::Left(k), None) => {
                    if let Some(default) =
                        k.default(db, target_module.krate().to_display_target(db))
                        && let Some(default) = default.expr()
                    {
                        const_substs.insert(k, default.syntax().clone_for_update());
                        defaulted_params.push(Either::Right(k));
                    }
                }
                _ => (), // ignore mismatching params
            });
        // No need to prettify lifetimes, there's nothing to prettify.
        let lifetime_substs: FxHashMap<_, _> = self
            .generic_def
            .into_iter()
            .flat_map(|it| it.lifetime_params(db))
            .zip(self.substs.lifetimes.clone())
            .filter_map(|(k, v)| {
                Some((k.name(db).display(db, target_edition).to_string(), v.lifetime()?))
            })
            .collect();
        let mut ctx = Ctx {
            type_substs,
            const_substs,
            lifetime_substs,
            target_module,
            source_scope: self.source_scope,
            same_self_type: self.target_scope.has_same_self_type(self.source_scope),
            target_edition,
        };
        ctx.transform_default_values(defaulted_params);
        ctx
    }
}

struct Ctx<'a> {
    type_substs: FxHashMap<hir::TypeParam, ast::Type>,
    const_substs: FxHashMap<hir::ConstParam, SyntaxNode>,
    lifetime_substs: FxHashMap<LifetimeName, ast::Lifetime>,
    target_module: hir::Module,
    source_scope: &'a SemanticsScope<'a>,
    same_self_type: bool,
    target_edition: Edition,
}

fn preorder_rev(item: &SyntaxNode) -> impl Iterator<Item = SyntaxNode> {
    let x = item
        .preorder()
        .filter_map(|event| match event {
            syntax::WalkEvent::Enter(node) => Some(node),
            syntax::WalkEvent::Leave(_) => None,
        })
        .collect_vec();
    x.into_iter().rev()
}

impl Ctx<'_> {
    fn apply(&self, item: &SyntaxNode) -> SyntaxNode {
        // `transform_path` may update a node's parent and that would break the
        // tree traversal. Thus all paths in the tree are collected into a vec
        // so that such operation is safe.
        let item = self.transform_path(item).clone_subtree();
        let mut editor = SyntaxEditor::new(item.clone());
        preorder_rev(&item).filter_map(ast::Lifetime::cast).for_each(|lifetime| {
            if let Some(subst) = self.lifetime_substs.get(&lifetime.syntax().text().to_string()) {
                editor
                    .replace(lifetime.syntax(), subst.clone_subtree().clone_for_update().syntax());
            }
        });

        editor.finish().new_root().clone()
    }

    fn transform_default_values(&mut self, defaulted_params: Vec<DefaultedParam>) {
        // By now the default values are simply copied from where they are declared
        // and should be transformed. As any value is allowed to refer to previous
        // generic (both type and const) parameters, they should be all iterated left-to-right.
        for param in defaulted_params {
            let value = match &param {
                Either::Left(k) => self.type_substs.get(k).unwrap().syntax(),
                Either::Right(k) => self.const_substs.get(k).unwrap(),
            };
            // `transform_path` may update a node's parent and that would break the
            // tree traversal. Thus all paths in the tree are collected into a vec
            // so that such operation is safe.
            let new_value = self.transform_path(value);
            match param {
                Either::Left(k) => {
                    self.type_substs.insert(k, ast::Type::cast(new_value.clone()).unwrap());
                }
                Either::Right(k) => {
                    self.const_substs.insert(k, new_value.clone());
                }
            }
        }
    }

    fn transform_path(&self, path: &SyntaxNode) -> SyntaxNode {
        fn find_child_paths(root_path: &SyntaxNode) -> Vec<ast::Path> {
            let mut result = Vec::new();
            for child in root_path.children() {
                if let Some(child_path) = ast::Path::cast(child.clone()) {
                    result.push(child_path);
                } else {
                    result.extend(find_child_paths(&child));
                }
            }
            result
        }
        let root_path = path.clone_subtree();
        let result = find_child_paths(&root_path);
        let mut editor = SyntaxEditor::new(root_path.clone());
        for sub_path in result {
            let new = self.transform_path(sub_path.syntax());
            editor.replace(sub_path.syntax(), new);
        }
        let update_sub_item = editor.finish().new_root().clone().clone_subtree();
        let item = find_child_paths(&update_sub_item);
        let mut editor = SyntaxEditor::new(update_sub_item);
        for sub_path in item {
            self.transform_path_(&mut editor, &sub_path);
        }
        editor.finish().new_root().clone()
    }

    fn transform_path_(&self, editor: &mut SyntaxEditor, path: &ast::Path) -> Option<()> {
        if path.qualifier().is_some() {
            return None;
        }
        if path.segment().is_some_and(|s| {
            s.parenthesized_arg_list().is_some()
                || (s.self_token().is_some() && path.parent_path().is_none())
        }) {
            // don't try to qualify `Fn(Foo) -> Bar` paths, they are in prelude anyway
            // don't try to qualify sole `self` either, they are usually locals, but are returned as modules due to namespace clashing
            return None;
        }
        let resolution = self.source_scope.speculative_resolve(path)?;

        match resolution {
            hir::PathResolution::TypeParam(tp) => {
                if let Some(subst) = self.type_substs.get(&tp) {
                    let parent = path.syntax().parent()?;
                    if let Some(parent) = ast::Path::cast(parent.clone()) {
                        // Path inside path means that there is an associated
                        // type/constant on the type parameter. It is necessary
                        // to fully qualify the type with `as Trait`. Even
                        // though it might be unnecessary if `subst` is generic
                        // type, always fully qualifying the path is safer
                        // because of potential clash of associated types from
                        // multiple traits

                        let trait_ref = find_trait_for_assoc_item(
                            self.source_scope,
                            tp,
                            parent.segment()?.name_ref()?,
                        )
                        .and_then(|trait_ref| {
                            let cfg = ImportPathConfig {
                                prefer_no_std: false,
                                prefer_prelude: true,
                                prefer_absolute: false,
                                allow_unstable: true,
                            };
                            let found_path = self.target_module.find_path(
                                self.source_scope.db,
                                hir::ModuleDef::Trait(trait_ref),
                                cfg,
                            )?;
                            match make::ty_path(mod_path_to_ast(&found_path, self.target_edition)) {
                                ast::Type::PathType(path_ty) => Some(path_ty),
                                _ => None,
                            }
                        });

                        let segment = make::path_segment_ty(subst.clone(), trait_ref);
                        let qualified = make::path_from_segments(std::iter::once(segment), false);
                        editor.replace(path.syntax(), qualified.clone_for_update().syntax());
                    } else if let Some(path_ty) = ast::PathType::cast(parent) {
                        let old = path_ty.syntax();

                        if old.parent().is_some() {
                            editor.replace(old, subst.clone_subtree().clone_for_update().syntax());
                        } else {
                            // Some `path_ty` has no parent, especially ones made for default value
                            // of type parameters.
                            // In this case, `ted` cannot replace `path_ty` with `subst` directly.
                            // So, just replace its children as long as the `subst` is the same type.
                            let new = subst.clone_subtree().clone_for_update();
                            if !matches!(new, ast::Type::PathType(..)) {
                                return None;
                            }
                            let start = path_ty.syntax().first_child().map(NodeOrToken::Node)?;
                            let end = path_ty.syntax().last_child().map(NodeOrToken::Node)?;
                            editor.replace_all(
                                start..=end,
                                new.syntax().children().map(NodeOrToken::Node).collect::<Vec<_>>(),
                            );
                        }
                    } else {
                        editor.replace(
                            path.syntax(),
                            subst.clone_subtree().clone_for_update().syntax(),
                        );
                    }
                }
            }
            hir::PathResolution::Def(def) if def.as_assoc_item(self.source_scope.db).is_none() => {
                if let hir::ModuleDef::Trait(_) = def
                    && matches!(path.segment()?.kind()?, ast::PathSegmentKind::Type { .. })
                {
                    // `speculative_resolve` resolves segments like `<T as
                    // Trait>` into `Trait`, but just the trait name should
                    // not be used as the replacement of the original
                    // segment.
                    return None;
                }

                let cfg = ImportPathConfig {
                    prefer_no_std: false,
                    prefer_prelude: true,
                    prefer_absolute: false,
                    allow_unstable: true,
                };
                let found_path = self.target_module.find_path(self.source_scope.db, def, cfg)?;
                let res = mod_path_to_ast(&found_path, self.target_edition).clone_for_update();
                let mut res_editor = SyntaxEditor::new(res.syntax().clone_subtree());
                if let Some(args) = path.segment().and_then(|it| it.generic_arg_list())
                    && let Some(segment) = res.segment()
                {
                    if let Some(old) = segment.generic_arg_list() {
                        res_editor
                            .replace(old.syntax(), args.clone_subtree().syntax().clone_for_update())
                    } else {
                        res_editor.insert(
                            syntax_editor::Position::last_child_of(segment.syntax()),
                            args.clone_subtree().syntax().clone_for_update(),
                        );
                    }
                }
                let res = res_editor.finish().new_root().clone();
                editor.replace(path.syntax().clone(), res);
            }
            hir::PathResolution::ConstParam(cp) => {
                if let Some(subst) = self.const_substs.get(&cp) {
                    editor.replace(path.syntax(), subst.clone_subtree().clone_for_update());
                }
            }
            hir::PathResolution::SelfType(imp) => {
                // keep Self type if it does not need to be replaced
                if self.same_self_type {
                    return None;
                }

                let ty = imp.self_ty(self.source_scope.db);
                let ty_str = &ty
                    .display_source_code(
                        self.source_scope.db,
                        self.source_scope.module().into(),
                        true,
                    )
                    .ok()?;
                let ast_ty = make::ty(ty_str).clone_for_update();

                if let Some(adt) = ty.as_adt()
                    && let ast::Type::PathType(path_ty) = &ast_ty
                {
                    let cfg = ImportPathConfig {
                        prefer_no_std: false,
                        prefer_prelude: true,
                        prefer_absolute: false,
                        allow_unstable: true,
                    };
                    let found_path = self.target_module.find_path(
                        self.source_scope.db,
                        ModuleDef::from(adt),
                        cfg,
                    )?;

                    if let Some(qual) =
                        mod_path_to_ast(&found_path, self.target_edition).qualifier()
                    {
                        let res = make::path_concat(qual, path_ty.path()?).clone_for_update();
                        editor.replace(path.syntax(), res.syntax());
                        return Some(());
                    }
                }

                editor.replace(path.syntax(), ast_ty.syntax());
            }
            hir::PathResolution::Local(_)
            | hir::PathResolution::Def(_)
            | hir::PathResolution::BuiltinAttr(_)
            | hir::PathResolution::ToolModule(_)
            | hir::PathResolution::DeriveHelper(_) => (),
        }
        Some(())
    }
}

// FIXME: It would probably be nicer if we could get this via HIR (i.e. get the
// trait ref, and then go from the types in the substs back to the syntax).
fn get_syntactic_substs(impl_def: ast::Impl) -> Option<AstSubsts> {
    let target_trait = impl_def.trait_()?;
    let path_type = match target_trait {
        ast::Type::PathType(path) => path,
        _ => return None,
    };
    let generic_arg_list = path_type.path()?.segment()?.generic_arg_list()?;

    get_type_args_from_arg_list(generic_arg_list)
}

fn get_type_args_from_arg_list(generic_arg_list: ast::GenericArgList) -> Option<AstSubsts> {
    let mut result = AstSubsts::default();
    generic_arg_list.generic_args().for_each(|generic_arg| match generic_arg {
        // Const params are marked as consts on definition only,
        // being passed to the trait they are indistguishable from type params;
        // anyway, we don't really need to distinguish them here.
        ast::GenericArg::TypeArg(type_arg) => {
            result.types_and_consts.push(TypeOrConst::Either(type_arg))
        }
        // Some const values are recognized correctly.
        ast::GenericArg::ConstArg(const_arg) => {
            result.types_and_consts.push(TypeOrConst::Const(const_arg));
        }
        ast::GenericArg::LifetimeArg(l_arg) => result.lifetimes.push(l_arg),
        _ => (),
    });

    Some(result)
}

fn find_trait_for_assoc_item(
    scope: &SemanticsScope<'_>,
    type_param: hir::TypeParam,
    assoc_item: ast::NameRef,
) -> Option<hir::Trait> {
    let db = scope.db;
    let trait_bounds = type_param.trait_bounds(db);

    let assoc_item_name = assoc_item.text();

    for trait_ in trait_bounds {
        let names = trait_.items(db).into_iter().filter_map(|item| match item {
            hir::AssocItem::TypeAlias(ta) => Some(ta.name(db)),
            hir::AssocItem::Const(cst) => cst.name(db),
            _ => None,
        });

        for name in names {
            if assoc_item_name.as_str() == name.as_str() {
                // It is fine to return the first match because in case of
                // multiple possibilities, the exact trait must be disambiguated
                // in the definition of trait being implemented, so this search
                // should not be needed.
                return Some(trait_);
            }
        }
    }

    None
}
