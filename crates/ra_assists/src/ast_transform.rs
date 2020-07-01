//! `AstTransformer`s are functions that replace nodes in an AST and can be easily combined.
use rustc_hash::FxHashMap;

use hir::{HirDisplay, PathResolution, SemanticsScope};
use ra_syntax::{
    algo::SyntaxRewriter,
    ast::{self, AstNode},
};

pub trait AstTransform<'a> {
    fn get_substitution(&self, node: &ra_syntax::SyntaxNode) -> Option<ra_syntax::SyntaxNode>;

    fn chain_before(self, other: Box<dyn AstTransform<'a> + 'a>) -> Box<dyn AstTransform<'a> + 'a>;
    fn or<T: AstTransform<'a> + 'a>(self, other: T) -> Box<dyn AstTransform<'a> + 'a>
    where
        Self: Sized + 'a,
    {
        self.chain_before(Box::new(other))
    }
}

struct NullTransformer;

impl<'a> AstTransform<'a> for NullTransformer {
    fn get_substitution(&self, _node: &ra_syntax::SyntaxNode) -> Option<ra_syntax::SyntaxNode> {
        None
    }
    fn chain_before(self, other: Box<dyn AstTransform<'a> + 'a>) -> Box<dyn AstTransform<'a> + 'a> {
        other
    }
}

pub struct SubstituteTypeParams<'a> {
    source_scope: &'a SemanticsScope<'a>,
    substs: FxHashMap<hir::TypeParam, ast::TypeRef>,
    previous: Box<dyn AstTransform<'a> + 'a>,
}

impl<'a> SubstituteTypeParams<'a> {
    pub fn for_trait_impl(
        source_scope: &'a SemanticsScope<'a>,
        // FIXME: there's implicit invariant that `trait_` and  `source_scope` match...
        trait_: hir::Trait,
        impl_def: ast::ImplDef,
    ) -> SubstituteTypeParams<'a> {
        let substs = get_syntactic_substs(impl_def).unwrap_or_default();
        let generic_def: hir::GenericDef = trait_.into();
        let substs_by_param: FxHashMap<_, _> = generic_def
            .params(source_scope.db)
            .into_iter()
            // this is a trait impl, so we need to skip the first type parameter -- this is a bit hacky
            .skip(1)
            // The actual list of trait type parameters may be longer than the one
            // used in the `impl` block due to trailing default type parametrs.
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
                        ast::make::type_ref(
                            &default
                                .display_source_code(source_scope.db, source_scope.module()?.into())
                                .ok()?,
                        ),
                    ))
                }
            })
            .collect();
        return SubstituteTypeParams {
            source_scope,
            substs: substs_by_param,
            previous: Box::new(NullTransformer),
        };

        // FIXME: It would probably be nicer if we could get this via HIR (i.e. get the
        // trait ref, and then go from the types in the substs back to the syntax)
        fn get_syntactic_substs(impl_def: ast::ImplDef) -> Option<Vec<ast::TypeRef>> {
            let target_trait = impl_def.target_trait()?;
            let path_type = match target_trait {
                ast::TypeRef::PathType(path) => path,
                _ => return None,
            };
            let type_arg_list = path_type.path()?.segment()?.type_arg_list()?;
            let mut result = Vec::new();
            for type_arg in type_arg_list.type_args() {
                let type_arg: ast::TypeArg = type_arg;
                result.push(type_arg.type_ref()?);
            }
            Some(result)
        }
    }
    fn get_substitution_inner(
        &self,
        node: &ra_syntax::SyntaxNode,
    ) -> Option<ra_syntax::SyntaxNode> {
        let type_ref = ast::TypeRef::cast(node.clone())?;
        let path = match &type_ref {
            ast::TypeRef::PathType(path_type) => path_type.path()?,
            _ => return None,
        };
        // FIXME: use `hir::Path::from_src` instead.
        #[allow(deprecated)]
        let path = hir::Path::from_ast(path)?;
        let resolution = self.source_scope.resolve_hir_path(&path)?;
        match resolution {
            hir::PathResolution::TypeParam(tp) => Some(self.substs.get(&tp)?.syntax().clone()),
            _ => None,
        }
    }
}

impl<'a> AstTransform<'a> for SubstituteTypeParams<'a> {
    fn get_substitution(&self, node: &ra_syntax::SyntaxNode) -> Option<ra_syntax::SyntaxNode> {
        self.get_substitution_inner(node).or_else(|| self.previous.get_substitution(node))
    }
    fn chain_before(self, other: Box<dyn AstTransform<'a> + 'a>) -> Box<dyn AstTransform<'a> + 'a> {
        Box::new(SubstituteTypeParams { previous: other, ..self })
    }
}

pub struct QualifyPaths<'a> {
    target_scope: &'a SemanticsScope<'a>,
    source_scope: &'a SemanticsScope<'a>,
    previous: Box<dyn AstTransform<'a> + 'a>,
}

impl<'a> QualifyPaths<'a> {
    pub fn new(target_scope: &'a SemanticsScope<'a>, source_scope: &'a SemanticsScope<'a>) -> Self {
        Self { target_scope, source_scope, previous: Box::new(NullTransformer) }
    }

    fn get_substitution_inner(
        &self,
        node: &ra_syntax::SyntaxNode,
    ) -> Option<ra_syntax::SyntaxNode> {
        // FIXME handle value ns?
        let from = self.target_scope.module()?;
        let p = ast::Path::cast(node.clone())?;
        if p.segment().and_then(|s| s.param_list()).is_some() {
            // don't try to qualify `Fn(Foo) -> Bar` paths, they are in prelude anyway
            return None;
        }
        // FIXME: use `hir::Path::from_src` instead.
        #[allow(deprecated)]
        let hir_path = hir::Path::from_ast(p.clone());
        let resolution = self.source_scope.resolve_hir_path(&hir_path?)?;
        match resolution {
            PathResolution::Def(def) => {
                let found_path = from.find_use_path(self.source_scope.db.upcast(), def)?;
                let mut path = path_to_ast(found_path);

                let type_args = p
                    .segment()
                    .and_then(|s| s.type_arg_list())
                    .map(|arg_list| apply(self, arg_list));
                if let Some(type_args) = type_args {
                    let last_segment = path.segment().unwrap();
                    path = path.with_segment(last_segment.with_type_args(type_args))
                }

                Some(path.syntax().clone())
            }
            PathResolution::Local(_)
            | PathResolution::TypeParam(_)
            | PathResolution::SelfType(_) => None,
            PathResolution::Macro(_) => None,
            PathResolution::AssocItem(_) => None,
        }
    }
}

pub fn apply<'a, N: AstNode>(transformer: &dyn AstTransform<'a>, node: N) -> N {
    SyntaxRewriter::from_fn(|element| match element {
        ra_syntax::SyntaxElement::Node(n) => {
            let replacement = transformer.get_substitution(&n)?;
            Some(replacement.into())
        }
        _ => None,
    })
    .rewrite_ast(&node)
}

impl<'a> AstTransform<'a> for QualifyPaths<'a> {
    fn get_substitution(&self, node: &ra_syntax::SyntaxNode) -> Option<ra_syntax::SyntaxNode> {
        self.get_substitution_inner(node).or_else(|| self.previous.get_substitution(node))
    }
    fn chain_before(self, other: Box<dyn AstTransform<'a> + 'a>) -> Box<dyn AstTransform<'a> + 'a> {
        Box::new(QualifyPaths { previous: other, ..self })
    }
}

pub(crate) fn path_to_ast(path: hir::ModPath) -> ast::Path {
    let parse = ast::SourceFile::parse(&path.to_string());
    parse
        .tree()
        .syntax()
        .descendants()
        .find_map(ast::Path::cast)
        .unwrap_or_else(|| panic!("failed to parse path {:?}, `{}`", path, path))
}
