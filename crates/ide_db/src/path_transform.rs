//! See [`PathTransform`].

use crate::helpers::mod_path_to_ast;
use hir::{HirDisplay, SemanticsScope};
use rustc_hash::FxHashMap;
use syntax::{
    ast::{self, AstNode},
    ted,
};

/// `PathTransform` substitutes path in SyntaxNodes in bulk.
///
/// This is mostly useful for IDE code generation. If you paste some existing
/// code into a new context (for example, to add method overrides to an `impl`
/// block), you generally want to appropriately qualify the names, and sometimes
/// you might want to substitute generic parameters as well:
///
/// ```
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
    pub subst: (hir::Trait, ast::Impl),
    pub target_scope: &'a SemanticsScope<'a>,
    pub source_scope: &'a SemanticsScope<'a>,
}

impl<'a> PathTransform<'a> {
    pub fn apply(&self, item: ast::AssocItem) {
        if let Some(ctx) = self.build_ctx() {
            ctx.apply(item)
        }
    }
    fn build_ctx(&self) -> Option<Ctx<'a>> {
        let db = self.source_scope.db;
        let target_module = self.target_scope.module()?;
        let source_module = self.source_scope.module()?;

        let substs = get_syntactic_substs(self.subst.1.clone()).unwrap_or_default();
        let generic_def: hir::GenericDef = self.subst.0.into();
        let substs_by_param: FxHashMap<_, _> = generic_def
            .type_params(db)
            .into_iter()
            // this is a trait impl, so we need to skip the first type parameter -- this is a bit hacky
            .skip(1)
            // The actual list of trait type parameters may be longer than the one
            // used in the `impl` block due to trailing default type parameters.
            // For that case we extend the `substs` with an empty iterator so we
            // can still hit those trailing values and check if they actually have
            // a default type. If they do, go for that type from `hir` to `ast` so
            // the resulting change can be applied correctly.
            .zip(substs.into_iter().map(Some).chain(std::iter::repeat(None)))
            .filter_map(|(k, v)| match v {
                Some(v) => Some((k, v)),
                None => {
                    let default = k.default(db)?;
                    Some((
                        k,
                        ast::make::ty(&default.display_source_code(db, source_module.into()).ok()?),
                    ))
                }
            })
            .collect();

        let res = Ctx { substs: substs_by_param, target_module, source_scope: self.source_scope };
        Some(res)
    }
}

struct Ctx<'a> {
    substs: FxHashMap<hir::TypeParam, ast::Type>,
    target_module: hir::Module,
    source_scope: &'a SemanticsScope<'a>,
}

impl<'a> Ctx<'a> {
    fn apply(&self, item: ast::AssocItem) {
        for event in item.syntax().preorder() {
            let node = match event {
                syntax::WalkEvent::Enter(_) => continue,
                syntax::WalkEvent::Leave(it) => it,
            };
            if let Some(path) = ast::Path::cast(node.clone()) {
                self.transform_path(path);
            }
        }
    }
    fn transform_path(&self, path: ast::Path) -> Option<()> {
        if path.qualifier().is_some() {
            return None;
        }
        if path.segment().and_then(|s| s.param_list()).is_some() {
            // don't try to qualify `Fn(Foo) -> Bar` paths, they are in prelude anyway
            return None;
        }

        let resolution = self.source_scope.speculative_resolve(&path)?;

        match resolution {
            hir::PathResolution::TypeParam(tp) => {
                if let Some(subst) = self.substs.get(&tp) {
                    ted::replace(path.syntax(), subst.clone_subtree().clone_for_update().syntax())
                }
            }
            hir::PathResolution::Def(def) => {
                let found_path =
                    self.target_module.find_use_path(self.source_scope.db.upcast(), def)?;
                let res = mod_path_to_ast(&found_path).clone_for_update();
                if let Some(args) = path.segment().and_then(|it| it.generic_arg_list()) {
                    if let Some(segment) = res.segment() {
                        let old = segment.get_or_create_generic_arg_list();
                        ted::replace(old.syntax(), args.clone_subtree().syntax().clone_for_update())
                    }
                }
                ted::replace(path.syntax(), res.syntax())
            }
            hir::PathResolution::Local(_)
            | hir::PathResolution::ConstParam(_)
            | hir::PathResolution::SelfType(_)
            | hir::PathResolution::Macro(_)
            | hir::PathResolution::AssocItem(_) => (),
        }
        Some(())
    }
}

// FIXME: It would probably be nicer if we could get this via HIR (i.e. get the
// trait ref, and then go from the types in the substs back to the syntax).
fn get_syntactic_substs(impl_def: ast::Impl) -> Option<Vec<ast::Type>> {
    let target_trait = impl_def.trait_()?;
    let path_type = match target_trait {
        ast::Type::PathType(path) => path,
        _ => return None,
    };
    let generic_arg_list = path_type.path()?.segment()?.generic_arg_list()?;

    let mut result = Vec::new();
    for generic_arg in generic_arg_list.generic_args() {
        match generic_arg {
            ast::GenericArg::TypeArg(type_arg) => result.push(type_arg.ty()?),
            ast::GenericArg::AssocTypeArg(_)
            | ast::GenericArg::LifetimeArg(_)
            | ast::GenericArg::ConstArg(_) => (),
        }
    }

    Some(result)
}
