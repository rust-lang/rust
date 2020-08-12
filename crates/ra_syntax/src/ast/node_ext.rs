//! Various extension methods to ast Nodes, which are hard to code-generate.
//! Extensions for various expressions live in a sibling `expr_extensions` module.

use std::fmt;

use itertools::Itertools;
use parser::SyntaxKind;

use crate::{
    ast::{self, support, AstNode, NameOwner, SyntaxNode},
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
        self.text().parse().ok()
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
        if self.eq_token().is_some() || self.token_tree().is_some() {
            return None;
        }
        self.simple_name()
    }

    pub fn as_simple_call(&self) -> Option<(SmolStr, ast::TokenTree)> {
        let tt = self.token_tree()?;
        Some((self.simple_name()?, tt))
    }

    pub fn as_simple_key_value(&self) -> Option<(SmolStr, SmolStr)> {
        let lit = self.literal()?;
        let key = self.simple_name()?;
        // FIXME: escape? raw string?
        let value = lit.syntax().first_token()?.text().trim_matches('"').into();
        Some((key, value))
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
            (Some(SyntaxKind::POUND), Some(T![!])) => AttrKind::Inner,
            _ => AttrKind::Outer,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSegmentKind {
    Name(ast::NameRef),
    Type { type_ref: Option<ast::Type>, trait_ref: Option<ast::PathType> },
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
                        self.syntax().children().filter(|node| ast::Type::can_cast(node.kind()));
                    let type_ref = type_refs.next().and_then(ast::Type::cast);
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

impl ast::Impl {
    pub fn self_ty(&self) -> Option<ast::Type> {
        match self.target() {
            (Some(t), None) | (_, Some(t)) => Some(t),
            _ => None,
        }
    }

    pub fn trait_(&self) -> Option<ast::Type> {
        match self.target() {
            (Some(t), Some(_)) => Some(t),
            _ => None,
        }
    }

    fn target(&self) -> (Option<ast::Type>, Option<ast::Type>) {
        let mut types = support::children(self.syntax());
        let first = types.next();
        let second = types.next();
        (first, second)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructKind {
    Record(ast::RecordFieldList),
    Tuple(ast::TupleFieldList),
    Unit,
}

impl StructKind {
    fn from_node<N: AstNode>(node: &N) -> StructKind {
        if let Some(nfdl) = support::child::<ast::RecordFieldList>(node.syntax()) {
            StructKind::Record(nfdl)
        } else if let Some(pfl) = support::child::<ast::TupleFieldList>(node.syntax()) {
            StructKind::Tuple(pfl)
        } else {
            StructKind::Unit
        }
    }
}

impl ast::Struct {
    pub fn kind(&self) -> StructKind {
        StructKind::from_node(self)
    }
}

impl ast::RecordExprField {
    pub fn for_field_name(field_name: &ast::NameRef) -> Option<ast::RecordExprField> {
        let candidate =
            field_name.syntax().parent().and_then(ast::RecordExprField::cast).or_else(|| {
                field_name.syntax().ancestors().nth(4).and_then(ast::RecordExprField::cast)
            })?;
        if candidate.field_name().as_ref() == Some(field_name) {
            Some(candidate)
        } else {
            None
        }
    }

    /// Deals with field init shorthand
    pub fn field_name(&self) -> Option<ast::NameRef> {
        if let Some(name_ref) = self.name_ref() {
            return Some(name_ref);
        }
        if let Some(ast::Expr::PathExpr(expr)) = self.expr() {
            let path = expr.path()?;
            let segment = path.segment()?;
            let name_ref = segment.name_ref()?;
            if path.qualifier().is_none() {
                return Some(name_ref);
            }
        }
        None
    }
}

pub enum NameOrNameRef {
    Name(ast::Name),
    NameRef(ast::NameRef),
}

impl fmt::Display for NameOrNameRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NameOrNameRef::Name(it) => fmt::Display::fmt(it, f),
            NameOrNameRef::NameRef(it) => fmt::Display::fmt(it, f),
        }
    }
}

impl ast::RecordPatField {
    /// Deals with field init shorthand
    pub fn field_name(&self) -> Option<NameOrNameRef> {
        if let Some(name_ref) = self.name_ref() {
            return Some(NameOrNameRef::NameRef(name_ref));
        }
        if let Some(ast::Pat::IdentPat(pat)) = self.pat() {
            let name = pat.name()?;
            return Some(NameOrNameRef::Name(name));
        }
        None
    }
}

impl ast::Variant {
    pub fn parent_enum(&self) -> ast::Enum {
        self.syntax()
            .parent()
            .and_then(|it| it.parent())
            .and_then(ast::Enum::cast)
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
        let mut args = self.pats().peekable();
        let prefix = args
            .peeking_take_while(|p| match p {
                ast::Pat::RestPat(_) => false,
                ast::Pat::IdentPat(bp) => match bp.pat() {
                    Some(ast::Pat::RestPat(_)) => false,
                    _ => true,
                },
                ast::Pat::RefPat(rp) => match rp.pat() {
                    Some(ast::Pat::RestPat(_)) => false,
                    Some(ast::Pat::IdentPat(bp)) => match bp.pat() {
                        Some(ast::Pat::RestPat(_)) => false,
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
            if self.mut_token().is_some() {
                SelfParamKind::MutRef
            } else {
                SelfParamKind::Ref
            }
        } else {
            SelfParamKind::Owned
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeBoundKind {
    /// Trait
    PathType(ast::PathType),
    /// for<'a> ...
    ForType(ast::ForType),
    /// 'a
    Lifetime(SyntaxToken),
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
            VisibilityKind::PubSelf
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

    pub fn is_bang(&self) -> bool {
        self.is_macro_rules().is_none()
    }
}

impl ast::LifetimeParam {
    pub fn lifetime_bounds(&self) -> impl Iterator<Item = SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.into_token())
            .skip_while(|x| x.kind() != T![:])
            .filter(|it| it.kind() == T![lifetime])
    }
}

impl ast::RangePat {
    pub fn start(&self) -> Option<ast::Pat> {
        self.syntax()
            .children_with_tokens()
            .take_while(|it| !(it.kind() == T![..] || it.kind() == T![..=]))
            .filter_map(|it| it.into_node())
            .find_map(ast::Pat::cast)
    }

    pub fn end(&self) -> Option<ast::Pat> {
        self.syntax()
            .children_with_tokens()
            .skip_while(|it| !(it.kind() == T![..] || it.kind() == T![..=]))
            .filter_map(|it| it.into_node())
            .find_map(ast::Pat::cast)
    }
}

impl ast::TokenTree {
    pub fn left_delimiter_token(&self) -> Option<SyntaxToken> {
        self.syntax()
            .first_child_or_token()?
            .into_token()
            .filter(|it| matches!(it.kind(), T!['{'] | T!['('] | T!['[']))
    }

    pub fn right_delimiter_token(&self) -> Option<SyntaxToken> {
        self.syntax()
            .last_child_or_token()?
            .into_token()
            .filter(|it| matches!(it.kind(), T!['}'] | T![')'] | T![']']))
    }
}

impl ast::GenericParamList {
    pub fn lifetime_params(&self) -> impl Iterator<Item = ast::LifetimeParam> {
        self.generic_params().filter_map(|param| match param {
            ast::GenericParam::LifetimeParam(it) => Some(it),
            ast::GenericParam::TypeParam(_) | ast::GenericParam::ConstParam(_) => None,
        })
    }
    pub fn type_params(&self) -> impl Iterator<Item = ast::TypeParam> {
        self.generic_params().filter_map(|param| match param {
            ast::GenericParam::TypeParam(it) => Some(it),
            ast::GenericParam::LifetimeParam(_) | ast::GenericParam::ConstParam(_) => None,
        })
    }
    pub fn const_params(&self) -> impl Iterator<Item = ast::ConstParam> {
        self.generic_params().filter_map(|param| match param {
            ast::GenericParam::ConstParam(it) => Some(it),
            ast::GenericParam::TypeParam(_) | ast::GenericParam::LifetimeParam(_) => None,
        })
    }
}

impl ast::DocCommentsOwner for ast::SourceFile {}
impl ast::DocCommentsOwner for ast::Fn {}
impl ast::DocCommentsOwner for ast::Struct {}
impl ast::DocCommentsOwner for ast::Union {}
impl ast::DocCommentsOwner for ast::RecordField {}
impl ast::DocCommentsOwner for ast::TupleField {}
impl ast::DocCommentsOwner for ast::Enum {}
impl ast::DocCommentsOwner for ast::Variant {}
impl ast::DocCommentsOwner for ast::Trait {}
impl ast::DocCommentsOwner for ast::Module {}
impl ast::DocCommentsOwner for ast::Static {}
impl ast::DocCommentsOwner for ast::Const {}
impl ast::DocCommentsOwner for ast::TypeAlias {}
impl ast::DocCommentsOwner for ast::Impl {}
impl ast::DocCommentsOwner for ast::MacroCall {}
