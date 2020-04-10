//! Various extension methods to ast Nodes, which are hard to code-generate.
//! Extensions for various expressions live in a sibling `expr_extensions` module.

use itertools::Itertools;
use ra_parser::SyntaxKind;

use crate::{
    ast::{self, support, AstNode, AstToken, AttrInput, NameOwner, SyntaxNode},
    SmolStr, SyntaxElement, SyntaxToken, T,
};

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
        if let Some(ast::NameRefToken::IntNumber(token)) = self.name_ref_token_token() {
            token.text().as_str().parse().ok()
        } else {
            None
        }
    }
}

fn text_of_first_token(node: &SyntaxNode) -> &SmolStr {
    node.green().children().next().and_then(|it| it.into_token()).unwrap().text()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttrKind {
    Inner,
    Outer,
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

    pub fn kind(&self) -> AttrKind {
        let first_token = self.syntax().first_token();
        let first_token_kind = first_token.as_ref().map(SyntaxToken::kind);
        let second_token_kind =
            first_token.and_then(|token| token.next_token()).as_ref().map(SyntaxToken::kind);

        match (first_token_kind, second_token_kind) {
            (Some(SyntaxKind::POUND), Some(SyntaxKind::EXCL)) => AttrKind::Inner,
            _ => AttrKind::Outer,
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
}

impl ast::Path {
    pub fn parent_path(&self) -> Option<ast::Path> {
        self.syntax().parent().and_then(ast::Path::cast)
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

impl ast::ImplDef {
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
        let mut types = support::children(self.syntax());
        let first = types.next();
        let second = types.next();
        (first, second)
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
        if let Some(nfdl) = support::child::<ast::RecordFieldDefList>(node.syntax()) {
            StructKind::Record(nfdl)
        } else if let Some(pfl) = support::child::<ast::TupleFieldDefList>(node.syntax()) {
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

pub struct SlicePatComponents {
    pub prefix: Vec<ast::Pat>,
    pub slice: Option<ast::Pat>,
    pub suffix: Vec<ast::Pat>,
}

impl ast::SlicePat {
    pub fn components(&self) -> SlicePatComponents {
        let mut args = self.args().peekable();
        let prefix = args
            .peeking_take_while(|p| match p {
                ast::Pat::DotDotPat(_) => false,
                ast::Pat::BindPat(bp) => match bp.pat() {
                    Some(ast::Pat::DotDotPat(_)) => false,
                    _ => true,
                },
                ast::Pat::RefPat(rp) => match rp.pat() {
                    Some(ast::Pat::DotDotPat(_)) => false,
                    Some(ast::Pat::BindPat(bp)) => match bp.pat() {
                        Some(ast::Pat::DotDotPat(_)) => false,
                        _ => true,
                    },
                    _ => true,
                },
                _ => true,
            })
            .collect();
        let slice = args.next();
        let suffix = args.collect();

        SlicePatComponents { prefix, slice, suffix }
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
    pub fn kind(&self) -> SelfParamKind {
        if self.amp_token().is_some() {
            if self.amp_mut_token().is_some() {
                SelfParamKind::MutRef
            } else {
                SelfParamKind::Ref
            }
        } else {
            SelfParamKind::Owned
        }
    }

    /// the "mut" in "mut self", not the one in "&mut self"
    pub fn mut_token(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.into_token())
            .take_while(|it| it.kind() != T![&])
            .find(|it| it.kind() == T![mut])
    }

    /// the "mut" in "&mut self", not the one in "mut self"
    pub fn amp_mut_token(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.into_token())
            .skip_while(|it| it.kind() != T![&])
            .find(|it| it.kind() == T![mut])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeBoundKind {
    /// Trait
    PathType(ast::PathType),
    /// for<'a> ...
    ForType(ast::ForType),
    /// 'a
    Lifetime(ast::Lifetime),
}

impl ast::TypeBound {
    pub fn kind(&self) -> TypeBoundKind {
        if let Some(path_type) = support::children(self.syntax()).next() {
            TypeBoundKind::PathType(path_type)
        } else if let Some(for_type) = support::children(self.syntax()).next() {
            TypeBoundKind::ForType(for_type)
        } else if let Some(lifetime) = self.lifetime_token() {
            TypeBoundKind::Lifetime(lifetime)
        } else {
            unreachable!()
        }
    }

    pub fn const_question_token(&self) -> Option<ast::Question> {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.into_token())
            .take_while(|it| it.kind() != T![const])
            .find_map(ast::Question::cast)
    }

    pub fn question_token(&self) -> Option<ast::Question> {
        if self.const_token().is_some() {
            self.syntax()
                .children_with_tokens()
                .filter_map(|it| it.into_token())
                .skip_while(|it| it.kind() != T![const])
                .find_map(ast::Question::cast)
        } else {
            support::token(&self.syntax)
        }
    }
}

pub enum VisibilityKind {
    In(ast::Path),
    PubCrate,
    PubSuper,
    PubSelf,
    Pub,
}

impl ast::Visibility {
    pub fn kind(&self) -> VisibilityKind {
        if let Some(path) = support::children(self.syntax()).next() {
            VisibilityKind::In(path)
        } else if self.crate_token().is_some() {
            VisibilityKind::PubCrate
        } else if self.super_token().is_some() {
            VisibilityKind::PubSuper
        } else if self.self_token().is_some() {
            VisibilityKind::PubSuper
        } else {
            VisibilityKind::Pub
        }
    }
}

impl ast::MacroCall {
    pub fn is_macro_rules(&self) -> Option<ast::Name> {
        let name_ref = self.path()?.segment()?.name_ref()?;
        if name_ref.text() == "macro_rules" {
            self.name()
        } else {
            None
        }
    }
}

impl ast::LifetimeParam {
    pub fn lifetime_bounds(&self) -> impl Iterator<Item = ast::Lifetime> {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.into_token())
            .skip_while(|x| x.kind() != T![:])
            .filter_map(ast::Lifetime::cast)
    }
}

impl ast::RangePat {
    pub fn start(&self) -> Option<ast::Pat> {
        self.syntax()
            .children_with_tokens()
            .take_while(|it| !ast::RangeSeparator::can_cast(it.kind()))
            .filter_map(|it| it.into_node())
            .find_map(ast::Pat::cast)
    }

    pub fn end(&self) -> Option<ast::Pat> {
        self.syntax()
            .children_with_tokens()
            .skip_while(|it| !ast::RangeSeparator::can_cast(it.kind()))
            .filter_map(|it| it.into_node())
            .find_map(ast::Pat::cast)
    }
}

impl ast::TokenTree {
    pub fn left_delimiter_token(&self) -> Option<SyntaxToken> {
        self.syntax().first_child_or_token()?.into_token().filter(|it| match it.kind() {
            T!['{'] | T!['('] | T!['['] => true,
            _ => false,
        })
    }

    pub fn right_delimiter_token(&self) -> Option<SyntaxToken> {
        self.syntax().last_child_or_token()?.into_token().filter(|it| match it.kind() {
            T!['{'] | T!['('] | T!['['] => true,
            _ => false,
        })
    }
}
