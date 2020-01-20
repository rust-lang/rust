//! Various extension methods to ast Nodes, which are hard to code-generate.
//! Extensions for various expressions live in a sibling `expr_extensions` module.

use crate::{
    ast::{self, child_opt, children, AstNode, AttrInput, SyntaxNode},
    SmolStr, SyntaxElement,
    SyntaxKind::*,
    SyntaxToken, T,
};
use ra_parser::SyntaxKind;

impl ast::Name {
    pub fn text(&self) -> &SmolStr {
        text_of_first_token(self.syntax())
    }
}

impl ast::NameRef {
    pub fn text(&self) -> &SmolStr {
        text_of_first_token(self.syntax())
    }

    pub fn as_tuple_field(&self) -> Option<usize> {
        self.syntax().children_with_tokens().find_map(|c| {
            if c.kind() == SyntaxKind::INT_NUMBER {
                c.as_token().and_then(|tok| tok.text().as_str().parse().ok())
            } else {
                None
            }
        })
    }
}

fn text_of_first_token(node: &SyntaxNode) -> &SmolStr {
    node.green().children().next().and_then(|it| it.into_token()).unwrap().text()
}

impl ast::Attr {
    pub fn as_simple_atom(&self) -> Option<SmolStr> {
        match self.input() {
            None => self.simple_name(),
            Some(_) => None,
        }
    }

    pub fn as_simple_call(&self) -> Option<(SmolStr, ast::TokenTree)> {
        match self.input() {
            Some(AttrInput::TokenTree(tt)) => Some((self.simple_name()?, tt)),
            _ => None,
        }
    }

    pub fn as_simple_key_value(&self) -> Option<(SmolStr, SmolStr)> {
        match self.input() {
            Some(AttrInput::Literal(lit)) => {
                let key = self.simple_name()?;
                // FIXME: escape? raw string?
                let value = lit.syntax().first_token()?.text().trim_matches('"').into();
                Some((key, value))
            }
            _ => None,
        }
    }

    pub fn simple_name(&self) -> Option<SmolStr> {
        let path = self.path()?;
        match (path.segment(), path.qualifier()) {
            (Some(segment), None) => Some(segment.syntax().first_token()?.text().clone()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSegmentKind {
    Name(ast::NameRef),
    Type { type_ref: Option<ast::TypeRef>, trait_ref: Option<ast::PathType> },
    SelfKw,
    SuperKw,
    CrateKw,
}

impl ast::PathSegment {
    pub fn parent_path(&self) -> ast::Path {
        self.syntax()
            .parent()
            .and_then(ast::Path::cast)
            .expect("segments are always nested in paths")
    }

    pub fn kind(&self) -> Option<PathSegmentKind> {
        let res = if let Some(name_ref) = self.name_ref() {
            PathSegmentKind::Name(name_ref)
        } else {
            match self.syntax().first_child_or_token()?.kind() {
                T![self] => PathSegmentKind::SelfKw,
                T![super] => PathSegmentKind::SuperKw,
                T![crate] => PathSegmentKind::CrateKw,
                T![<] => {
                    // <T> or <T as Trait>
                    // T is any TypeRef, Trait has to be a PathType
                    let mut type_refs =
                        self.syntax().children().filter(|node| ast::TypeRef::can_cast(node.kind()));
                    let type_ref = type_refs.next().and_then(ast::TypeRef::cast);
                    let trait_ref = type_refs.next().and_then(ast::PathType::cast);
                    PathSegmentKind::Type { type_ref, trait_ref }
                }
                _ => return None,
            }
        };
        Some(res)
    }

    pub fn has_colon_colon(&self) -> bool {
        match self.syntax.first_child_or_token().map(|s| s.kind()) {
            Some(T![::]) => true,
            _ => false,
        }
    }
}

impl ast::Path {
    pub fn parent_path(&self) -> Option<ast::Path> {
        self.syntax().parent().and_then(ast::Path::cast)
    }
}

impl ast::Module {
    pub fn has_semi(&self) -> bool {
        match self.syntax().last_child_or_token() {
            None => false,
            Some(node) => node.kind() == T![;],
        }
    }
}

impl ast::UseTree {
    pub fn has_star(&self) -> bool {
        self.syntax().children_with_tokens().any(|it| it.kind() == T![*])
    }
}

impl ast::UseTreeList {
    pub fn parent_use_tree(&self) -> ast::UseTree {
        self.syntax()
            .parent()
            .and_then(ast::UseTree::cast)
            .expect("UseTreeLists are always nested in UseTrees")
    }
}

impl ast::ImplBlock {
    pub fn target_type(&self) -> Option<ast::TypeRef> {
        match self.target() {
            (Some(t), None) | (_, Some(t)) => Some(t),
            _ => None,
        }
    }

    pub fn target_trait(&self) -> Option<ast::TypeRef> {
        match self.target() {
            (Some(t), Some(_)) => Some(t),
            _ => None,
        }
    }

    fn target(&self) -> (Option<ast::TypeRef>, Option<ast::TypeRef>) {
        let mut types = children(self);
        let first = types.next();
        let second = types.next();
        (first, second)
    }

    pub fn is_negative(&self) -> bool {
        self.syntax().children_with_tokens().any(|t| t.kind() == T![!])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructKind {
    Record(ast::RecordFieldDefList),
    Tuple(ast::TupleFieldDefList),
    Unit,
}

impl StructKind {
    fn from_node<N: AstNode>(node: &N) -> StructKind {
        if let Some(nfdl) = child_opt::<_, ast::RecordFieldDefList>(node) {
            StructKind::Record(nfdl)
        } else if let Some(pfl) = child_opt::<_, ast::TupleFieldDefList>(node) {
            StructKind::Tuple(pfl)
        } else {
            StructKind::Unit
        }
    }
}

impl ast::StructDef {
    pub fn kind(&self) -> StructKind {
        StructKind::from_node(self)
    }
}

impl ast::EnumVariant {
    pub fn parent_enum(&self) -> ast::EnumDef {
        self.syntax()
            .parent()
            .and_then(|it| it.parent())
            .and_then(ast::EnumDef::cast)
            .expect("EnumVariants are always nested in Enums")
    }
    pub fn kind(&self) -> StructKind {
        StructKind::from_node(self)
    }
}

impl ast::FnDef {
    pub fn semicolon_token(&self) -> Option<SyntaxToken> {
        self.syntax()
            .last_child_or_token()
            .and_then(|it| it.into_token())
            .filter(|it| it.kind() == T![;])
    }

    pub fn is_async(&self) -> bool {
        self.syntax().children_with_tokens().any(|it| it.kind() == T![async])
    }
}

impl ast::LetStmt {
    pub fn has_semi(&self) -> bool {
        match self.syntax().last_child_or_token() {
            None => false,
            Some(node) => node.kind() == T![;],
        }
    }

    pub fn eq_token(&self) -> Option<SyntaxToken> {
        self.syntax().children_with_tokens().find(|t| t.kind() == EQ).and_then(|it| it.into_token())
    }
}

impl ast::ExprStmt {
    pub fn has_semi(&self) -> bool {
        match self.syntax().last_child_or_token() {
            None => false,
            Some(node) => node.kind() == T![;],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldKind {
    Name(ast::NameRef),
    Index(SyntaxToken),
}

impl ast::FieldExpr {
    pub fn index_token(&self) -> Option<SyntaxToken> {
        self.syntax
            .children_with_tokens()
            // FIXME: Accepting floats here to reject them in validation later
            .find(|c| c.kind() == SyntaxKind::INT_NUMBER || c.kind() == SyntaxKind::FLOAT_NUMBER)
            .as_ref()
            .and_then(SyntaxElement::as_token)
            .cloned()
    }

    pub fn field_access(&self) -> Option<FieldKind> {
        if let Some(nr) = self.name_ref() {
            Some(FieldKind::Name(nr))
        } else if let Some(tok) = self.index_token() {
            Some(FieldKind::Index(tok))
        } else {
            None
        }
    }
}

impl ast::RefPat {
    pub fn is_mut(&self) -> bool {
        self.syntax().children_with_tokens().any(|n| n.kind() == T![mut])
    }
}

impl ast::BindPat {
    pub fn is_mutable(&self) -> bool {
        self.syntax().children_with_tokens().any(|n| n.kind() == T![mut])
    }

    pub fn is_ref(&self) -> bool {
        self.syntax().children_with_tokens().any(|n| n.kind() == T![ref])
    }
}

impl ast::PointerType {
    pub fn is_mut(&self) -> bool {
        self.syntax().children_with_tokens().any(|n| n.kind() == T![mut])
    }
}

impl ast::ReferenceType {
    pub fn is_mut(&self) -> bool {
        self.syntax().children_with_tokens().any(|n| n.kind() == T![mut])
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SelfParamKind {
    /// self
    Owned,
    /// &self
    Ref,
    /// &mut self
    MutRef,
}

impl ast::SelfParam {
    pub fn self_kw_token(&self) -> SyntaxToken {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == T![self])
            .expect("invalid tree: self param must have self")
    }

    pub fn kind(&self) -> SelfParamKind {
        let borrowed = self.syntax().children_with_tokens().any(|n| n.kind() == T![&]);
        if borrowed {
            // check for a `mut` coming after the & -- `mut &self` != `&mut self`
            if self
                .syntax()
                .children_with_tokens()
                .skip_while(|n| n.kind() != T![&])
                .any(|n| n.kind() == T![mut])
            {
                SelfParamKind::MutRef
            } else {
                SelfParamKind::Ref
            }
        } else {
            SelfParamKind::Owned
        }
    }
}

impl ast::LifetimeParam {
    pub fn lifetime_token(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == LIFETIME)
    }
}

impl ast::TypeParam {
    pub fn colon_token(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == T![:])
    }
}

impl ast::WherePred {
    pub fn lifetime_token(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == LIFETIME)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeBoundKind {
    /// Trait
    PathType(ast::PathType),
    /// for<'a> ...
    ForType(ast::ForType),
    /// 'a
    Lifetime(ast::SyntaxToken),
}

impl ast::TypeBound {
    pub fn kind(&self) -> TypeBoundKind {
        if let Some(path_type) = children(self).next() {
            TypeBoundKind::PathType(path_type)
        } else if let Some(for_type) = children(self).next() {
            TypeBoundKind::ForType(for_type)
        } else if let Some(lifetime) = self.lifetime() {
            TypeBoundKind::Lifetime(lifetime)
        } else {
            unreachable!()
        }
    }

    fn lifetime(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == LIFETIME)
    }

    pub fn question_mark_token(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == T![?])
    }
    pub fn has_question_mark(&self) -> bool {
        self.question_mark_token().is_some()
    }
}

impl ast::TraitDef {
    pub fn is_auto(&self) -> bool {
        self.syntax().children_with_tokens().any(|t| t.kind() == T![auto])
    }
}

pub enum VisibilityKind {
    In(ast::Path),
    PubCrate,
    PubSuper,
    Pub,
}

impl ast::Visibility {
    pub fn kind(&self) -> VisibilityKind {
        if let Some(path) = children(self).next() {
            VisibilityKind::In(path)
        } else if self.is_pub_crate() {
            VisibilityKind::PubCrate
        } else if self.is_pub_super() {
            VisibilityKind::PubSuper
        } else {
            VisibilityKind::Pub
        }
    }

    fn is_pub_crate(&self) -> bool {
        self.syntax().children_with_tokens().any(|it| it.kind() == T![crate])
    }

    fn is_pub_super(&self) -> bool {
        self.syntax().children_with_tokens().any(|it| it.kind() == T![super])
    }
}
