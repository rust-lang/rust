//! `AstTransformer`s are functions that replace nodes in an AST and can be easily combined.
use hir::{HirDisplay, PathResolution, SemanticsScope};
use ide_db::helpers::mod_path_to_ast;
use rustc_hash::FxHashMap;
use syntax::{
    ast::{self, AstNode},
    ted, SyntaxNode,
};

pub fn apply<'a, N: AstNode>(transformer: &dyn AstTransform<'a>, node: &N) {
    let mut skip_to = None;
    for event in node.syntax().preorder() {
        match event {
            syntax::WalkEvent::Enter(node) if skip_to.is_none() => {
                skip_to = transformer.get_substitution(&node, transformer).zip(Some(node));
            }
            syntax::WalkEvent::Enter(_) => (),
            syntax::WalkEvent::Leave(node) => match &skip_to {
                Some((replacement, skip_target)) if *skip_target == node => {
                    ted::replace(node, replacement.clone_for_update());
                    skip_to.take();
                }
                _ => (),
            },
        }
    }
}

/// `AstTransform` helps with applying bulk transformations to syntax nodes.
///
/// This is mostly useful for IDE code generation. If you paste some existing
/// code into a new context (for example, to add method overrides to an `impl`
/// block), you generally want to appropriately qualify the names, and sometimes
/// you might want to substitute generic parameters as well:
///
/// ```
/// mod x {
///   pub struct A;
///   pub trait T<U> { fn foo(&self, _: U) -> A; }
/// }
///
/// mod y {
///   use x::T;
///
///   impl T<()> for () {
///      // If we invoke **Add Missing Members** here, we want to copy-paste `foo`.
///      // But we want a slightly-modified version of it:
///      fn foo(&self, _: ()) -> x::A {}
///   }
/// }
/// ```
///
/// So, a single `AstTransform` describes such function from `SyntaxNode` to
/// `SyntaxNode`. Note that the API here is a bit too high-order and high-brow.
/// We'd want to somehow express this concept simpler, but so far nobody got to
/// simplifying this!
pub trait AstTransform<'a> {
    fn get_substitution(
        &self,
        node: &SyntaxNode,
        recur: &dyn AstTransform<'a>,
    ) -> Option<SyntaxNode>;

    fn or<T: AstTransform<'a> + 'a>(self, other: T) -> Box<dyn AstTransform<'a> + 'a>
    where
        Self: Sized + 'a,
    {
        Box::new(Or(Box::new(self), Box::new(other)))
    }
}

struct Or<'a>(Box<dyn AstTransform<'a> + 'a>, Box<dyn AstTransform<'a> + 'a>);

impl<'a> AstTransform<'a> for Or<'a> {
    fn get_substitution(
        &self,
        node: &SyntaxNode,
        recur: &dyn AstTransform<'a>,
    ) -> Option<SyntaxNode> {
        self.0.get_substitution(node, recur).or_else(|| self.1.get_substitution(node, recur))
    }
}

pub struct SubstituteTypeParams<'a> {
    source_scope: &'a SemanticsScope<'a>,
    substs: FxHashMap<hir::TypeParam, ast::Type>,
}

impl<'a> SubstituteTypeParams<'a> {
    pub fn for_trait_impl(
        source_scope: &'a SemanticsScope<'a>,
        // FIXME: there's implicit invariant that `trait_` and  `source_scope` match...
        trait_: hir::Trait,
        impl_def: ast::Impl,
    ) -> SubstituteTypeParams<'a> {
        let substs = get_syntactic_substs(impl_def).unwrap_or_default();
        let generic_def: hir::GenericDef = trait_.into();
        let substs_by_param: FxHashMap<_, _> = generic_def
            .type_params(source_scope.db)
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
                    let default = k.default(source_scope.db)?;
                    Some((
                        k,
                        ast::make::ty(
                            &default
                                .display_source_code(source_scope.db, source_scope.module()?.into())
                                .ok()?,
                        ),
                    ))
                }
            })
            .collect();
        return SubstituteTypeParams { source_scope, substs: substs_by_param };

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
    }
}

impl<'a> AstTransform<'a> for SubstituteTypeParams<'a> {
    fn get_substitution(
        &self,
        node: &SyntaxNode,
        _recur: &dyn AstTransform<'a>,
    ) -> Option<SyntaxNode> {
        let type_ref = ast::Type::cast(node.clone())?;
        let path = match &type_ref {
            ast::Type::PathType(path_type) => path_type.path()?,
            _ => return None,
        };
        let resolution = self.source_scope.speculative_resolve(&path)?;
        match resolution {
            hir::PathResolution::TypeParam(tp) => Some(self.substs.get(&tp)?.syntax().clone()),
            _ => None,
        }
    }
}

pub struct QualifyPaths<'a> {
    target_scope: &'a SemanticsScope<'a>,
    source_scope: &'a SemanticsScope<'a>,
}

impl<'a> QualifyPaths<'a> {
    pub fn new(target_scope: &'a SemanticsScope<'a>, source_scope: &'a SemanticsScope<'a>) -> Self {
        Self { target_scope, source_scope }
    }
}

impl<'a> AstTransform<'a> for QualifyPaths<'a> {
    fn get_substitution(
        &self,
        node: &SyntaxNode,
        recur: &dyn AstTransform<'a>,
    ) -> Option<SyntaxNode> {
        // FIXME handle value ns?
        let from = self.target_scope.module()?;
        let p = ast::Path::cast(node.clone())?;
        if p.segment().and_then(|s| s.param_list()).is_some() {
            // don't try to qualify `Fn(Foo) -> Bar` paths, they are in prelude anyway
            return None;
        }
        let resolution = self.source_scope.speculative_resolve(&p)?;
        match resolution {
            PathResolution::Def(def) => {
                let found_path = from.find_use_path(self.source_scope.db.upcast(), def)?;
                let mut path = mod_path_to_ast(&found_path);

                let type_args = p.segment().and_then(|s| s.generic_arg_list());
                if let Some(type_args) = type_args {
                    apply(recur, &type_args);
                    let last_segment = path.segment().unwrap();
                    path = path.with_segment(last_segment.with_generic_args(type_args))
                }

                Some(path.syntax().clone())
            }
            PathResolution::Local(_)
            | PathResolution::TypeParam(_)
            | PathResolution::SelfType(_)
            | PathResolution::ConstParam(_) => None,
            PathResolution::Macro(_) => None,
            PathResolution::AssocItem(_) => None,
        }
    }
}
