//! `AstTransformer`s are functions that replace nodes in an AST and can be easily combined.
use rustc_hash::FxHashMap;

use hir::{PathResolution, SemanticsScope};
use ra_ide_db::RootDatabase;
use ra_syntax::ast::{self, AstNode};

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
    source_scope: &'a SemanticsScope<'a, RootDatabase>,
    substs: FxHashMap<hir::TypeParam, ast::TypeRef>,
    previous: Box<dyn AstTransform<'a> + 'a>,
}

impl<'a> SubstituteTypeParams<'a> {
    pub fn for_trait_impl(
        source_scope: &'a SemanticsScope<'a, RootDatabase>,
        db: &'a RootDatabase,
        // FIXME: there's implicit invariant that `trait_` and  `source_scope` match...
        trait_: hir::Trait,
        impl_block: ast::ImplBlock,
    ) -> SubstituteTypeParams<'a> {
        let substs = get_syntactic_substs(impl_block).unwrap_or_default();
        let generic_def: hir::GenericDef = trait_.into();
        let substs_by_param: FxHashMap<_, _> = generic_def
            .params(db)
            .into_iter()
            // this is a trait impl, so we need to skip the first type parameter -- this is a bit hacky
            .skip(1)
            .zip(substs.into_iter())
            .collect();
        return SubstituteTypeParams {
            source_scope,
            substs: substs_by_param,
            previous: Box::new(NullTransformer),
        };

        // FIXME: It would probably be nicer if we could get this via HIR (i.e. get the
        // trait ref, and then go from the types in the substs back to the syntax)
        fn get_syntactic_substs(impl_block: ast::ImplBlock) -> Option<Vec<ast::TypeRef>> {
            let target_trait = impl_block.target_trait()?;
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
    target_scope: &'a SemanticsScope<'a, RootDatabase>,
    source_scope: &'a SemanticsScope<'a, RootDatabase>,
    db: &'a RootDatabase,
    previous: Box<dyn AstTransform<'a> + 'a>,
}

impl<'a> QualifyPaths<'a> {
    pub fn new(
        target_scope: &'a SemanticsScope<'a, RootDatabase>,
        source_scope: &'a SemanticsScope<'a, RootDatabase>,
        db: &'a RootDatabase,
    ) -> Self {
        Self { target_scope, source_scope, db, previous: Box::new(NullTransformer) }
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
        let hir_path = hir::Path::from_ast(p.clone());
        let resolution = self.source_scope.resolve_hir_path(&hir_path?)?;
        match resolution {
            PathResolution::Def(def) => {
                let found_path = from.find_use_path(self.db, def)?;
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
    let syntax = node.syntax();
    let result = ra_syntax::algo::replace_descendants(syntax, &|element| match element {
        ra_syntax::SyntaxElement::Node(n) => {
            let replacement = transformer.get_substitution(&n)?;
            Some(replacement.into())
        }
        _ => None,
    });
    N::cast(result).unwrap()
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
    parse.tree().syntax().descendants().find_map(ast::Path::cast).unwrap()
}
