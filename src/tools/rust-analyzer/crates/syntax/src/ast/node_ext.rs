//! Various extension methods to ast Nodes, which are hard to code-generate.
//! Extensions for various expressions live in a sibling `expr_extensions` module.
//!
//! These methods should only do simple, shallow tasks related to the syntax of the node itself.

use std::{borrow::Cow, fmt, iter::successors};

use itertools::Itertools;
use parser::SyntaxKind;
use rowan::{GreenNodeData, GreenTokenData};

use crate::{
    NodeOrToken, SmolStr, SyntaxElement, SyntaxToken, T, TokenText,
    ast::{
        self, AstNode, AstToken, HasAttrs, HasGenericArgs, HasGenericParams, HasName, SyntaxNode,
        support,
    },
    ted,
};

use super::{GenericParam, RangeItem, RangeOp};

impl ast::Lifetime {
    pub fn text(&self) -> TokenText<'_> {
        text_of_first_token(self.syntax())
    }
}

impl ast::Name {
    pub fn text(&self) -> TokenText<'_> {
        text_of_first_token(self.syntax())
    }
    pub fn text_non_mutable(&self) -> &str {
        fn first_token(green_ref: &GreenNodeData) -> &GreenTokenData {
            green_ref.children().next().and_then(NodeOrToken::into_token).unwrap()
        }

        match self.syntax().green() {
            Cow::Borrowed(green_ref) => first_token(green_ref).text(),
            Cow::Owned(_) => unreachable!(),
        }
    }
}

impl ast::NameRef {
    pub fn text(&self) -> TokenText<'_> {
        text_of_first_token(self.syntax())
    }
    pub fn text_non_mutable(&self) -> &str {
        fn first_token(green_ref: &GreenNodeData) -> &GreenTokenData {
            green_ref.children().next().and_then(NodeOrToken::into_token).unwrap()
        }

        match self.syntax().green() {
            Cow::Borrowed(green_ref) => first_token(green_ref).text(),
            Cow::Owned(_) => unreachable!(),
        }
    }

    pub fn as_tuple_field(&self) -> Option<usize> {
        self.text().parse().ok()
    }

    pub fn token_kind(&self) -> SyntaxKind {
        self.syntax().first_token().map_or(SyntaxKind::ERROR, |it| it.kind())
    }
}

fn text_of_first_token(node: &SyntaxNode) -> TokenText<'_> {
    fn first_token(green_ref: &GreenNodeData) -> &GreenTokenData {
        green_ref.children().next().and_then(NodeOrToken::into_token).unwrap()
    }

    match node.green() {
        Cow::Borrowed(green_ref) => TokenText::borrowed(first_token(green_ref).text()),
        Cow::Owned(green) => TokenText::owned(first_token(&green).to_owned()),
    }
}

impl ast::Abi {
    pub fn abi_string(&self) -> Option<ast::String> {
        support::token(&self.syntax, SyntaxKind::STRING).and_then(ast::String::cast)
    }
}

impl ast::HasModuleItem for ast::StmtList {}

impl ast::BlockExpr {
    // FIXME: remove all these methods, they belong to ast::StmtList
    pub fn statements(&self) -> impl Iterator<Item = ast::Stmt> {
        self.stmt_list().into_iter().flat_map(|it| it.statements())
    }
    pub fn tail_expr(&self) -> Option<ast::Expr> {
        self.stmt_list()?.tail_expr()
    }
    /// Block expressions accept outer and inner attributes, but only when they are the outer
    /// expression of an expression statement or the final expression of another block expression.
    pub fn may_carry_attributes(&self) -> bool {
        matches!(
            self.syntax().parent().map(|it| it.kind()),
            Some(SyntaxKind::BLOCK_EXPR | SyntaxKind::EXPR_STMT)
        )
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Macro {
    MacroRules(ast::MacroRules),
    MacroDef(ast::MacroDef),
}

impl From<ast::MacroRules> for Macro {
    fn from(it: ast::MacroRules) -> Self {
        Macro::MacroRules(it)
    }
}

impl From<ast::MacroDef> for Macro {
    fn from(it: ast::MacroDef) -> Self {
        Macro::MacroDef(it)
    }
}

impl AstNode for Macro {
    fn can_cast(kind: SyntaxKind) -> bool {
        matches!(kind, SyntaxKind::MACRO_RULES | SyntaxKind::MACRO_DEF)
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            SyntaxKind::MACRO_RULES => Macro::MacroRules(ast::MacroRules { syntax }),
            SyntaxKind::MACRO_DEF => Macro::MacroDef(ast::MacroDef { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            Macro::MacroRules(it) => it.syntax(),
            Macro::MacroDef(it) => it.syntax(),
        }
    }
}

impl HasName for Macro {
    fn name(&self) -> Option<ast::Name> {
        match self {
            Macro::MacroRules(mac) => mac.name(),
            Macro::MacroDef(mac) => mac.name(),
        }
    }
}

impl HasAttrs for Macro {}

impl From<ast::AssocItem> for ast::Item {
    fn from(assoc: ast::AssocItem) -> Self {
        match assoc {
            ast::AssocItem::Const(it) => ast::Item::Const(it),
            ast::AssocItem::Fn(it) => ast::Item::Fn(it),
            ast::AssocItem::MacroCall(it) => ast::Item::MacroCall(it),
            ast::AssocItem::TypeAlias(it) => ast::Item::TypeAlias(it),
        }
    }
}

impl From<ast::ExternItem> for ast::Item {
    fn from(extern_item: ast::ExternItem) -> Self {
        match extern_item {
            ast::ExternItem::Static(it) => ast::Item::Static(it),
            ast::ExternItem::Fn(it) => ast::Item::Fn(it),
            ast::ExternItem::MacroCall(it) => ast::Item::MacroCall(it),
            ast::ExternItem::TypeAlias(it) => ast::Item::TypeAlias(it),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AttrKind {
    Inner,
    Outer,
}

impl AttrKind {
    /// Returns `true` if the attr_kind is [`Inner`](Self::Inner).
    pub fn is_inner(&self) -> bool {
        matches!(self, Self::Inner)
    }

    /// Returns `true` if the attr_kind is [`Outer`](Self::Outer).
    pub fn is_outer(&self) -> bool {
        matches!(self, Self::Outer)
    }
}

impl ast::Attr {
    pub fn as_simple_atom(&self) -> Option<SmolStr> {
        let meta = self.meta()?;
        if meta.eq_token().is_some() || meta.token_tree().is_some() {
            return None;
        }
        self.simple_name()
    }

    pub fn as_simple_call(&self) -> Option<(SmolStr, ast::TokenTree)> {
        let tt = self.meta()?.token_tree()?;
        Some((self.simple_name()?, tt))
    }

    pub fn as_simple_path(&self) -> Option<ast::Path> {
        let meta = self.meta()?;
        if meta.eq_token().is_some() || meta.token_tree().is_some() {
            return None;
        }
        self.path()
    }

    pub fn simple_name(&self) -> Option<SmolStr> {
        let path = self.meta()?.path()?;
        match (path.segment(), path.qualifier()) {
            (Some(segment), None) => Some(segment.syntax().first_token()?.text().into()),
            _ => None,
        }
    }

    pub fn kind(&self) -> AttrKind {
        match self.excl_token() {
            Some(_) => AttrKind::Inner,
            None => AttrKind::Outer,
        }
    }

    pub fn path(&self) -> Option<ast::Path> {
        self.meta()?.path()
    }

    pub fn expr(&self) -> Option<ast::Expr> {
        self.meta()?.expr()
    }

    pub fn token_tree(&self) -> Option<ast::TokenTree> {
        self.meta()?.token_tree()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSegmentKind {
    Name(ast::NameRef),
    Type { type_ref: Option<ast::Type>, trait_ref: Option<ast::PathType> },
    SelfTypeKw,
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

    pub fn crate_token(&self) -> Option<SyntaxToken> {
        self.name_ref().and_then(|it| it.crate_token())
    }

    pub fn self_token(&self) -> Option<SyntaxToken> {
        self.name_ref().and_then(|it| it.self_token())
    }

    pub fn self_type_token(&self) -> Option<SyntaxToken> {
        self.name_ref().and_then(|it| it.Self_token())
    }

    pub fn super_token(&self) -> Option<SyntaxToken> {
        self.name_ref().and_then(|it| it.super_token())
    }

    pub fn kind(&self) -> Option<PathSegmentKind> {
        let res = if let Some(name_ref) = self.name_ref() {
            match name_ref.token_kind() {
                T![Self] => PathSegmentKind::SelfTypeKw,
                T![self] => PathSegmentKind::SelfKw,
                T![super] => PathSegmentKind::SuperKw,
                T![crate] => PathSegmentKind::CrateKw,
                _ => PathSegmentKind::Name(name_ref),
            }
        } else {
            let anchor = self.type_anchor()?;
            // FIXME: Move this over to `ast::TypeAnchor`
            // <T> or <T as Trait>
            // T is any TypeRef, Trait has to be a PathType
            let mut type_refs =
                anchor.syntax().children().filter(|node| ast::Type::can_cast(node.kind()));
            let type_ref = type_refs.next().and_then(ast::Type::cast);
            let trait_ref = type_refs.next().and_then(ast::PathType::cast);
            PathSegmentKind::Type { type_ref, trait_ref }
        };
        Some(res)
    }
}

impl ast::Path {
    pub fn parent_path(&self) -> Option<ast::Path> {
        self.syntax().parent().and_then(ast::Path::cast)
    }

    pub fn as_single_segment(&self) -> Option<ast::PathSegment> {
        match self.qualifier() {
            Some(_) => None,
            None => self.segment(),
        }
    }

    pub fn as_single_name_ref(&self) -> Option<ast::NameRef> {
        match self.qualifier() {
            Some(_) => None,
            None => self.segment()?.name_ref(),
        }
    }

    pub fn first_qualifier_or_self(&self) -> ast::Path {
        successors(Some(self.clone()), ast::Path::qualifier).last().unwrap()
    }

    pub fn first_qualifier(&self) -> Option<ast::Path> {
        successors(self.qualifier(), ast::Path::qualifier).last()
    }

    pub fn first_segment(&self) -> Option<ast::PathSegment> {
        self.first_qualifier_or_self().segment()
    }

    pub fn segments(&self) -> impl Iterator<Item = ast::PathSegment> + Clone {
        let path_range = self.syntax().text_range();
        successors(self.first_segment(), move |p| {
            p.parent_path().parent_path().and_then(|p| {
                if path_range.contains_range(p.syntax().text_range()) { p.segment() } else { None }
            })
        })
    }

    pub fn qualifiers(&self) -> impl Iterator<Item = ast::Path> + Clone {
        successors(self.qualifier(), |p| p.qualifier())
    }

    pub fn top_path(&self) -> ast::Path {
        let mut this = self.clone();
        while let Some(path) = this.parent_path() {
            this = path;
        }
        this
    }
}

impl ast::Use {
    pub fn is_simple_glob(&self) -> bool {
        self.use_tree().is_some_and(|use_tree| {
            use_tree.use_tree_list().is_none() && use_tree.star_token().is_some()
        })
    }
}

impl ast::UseTree {
    pub fn is_simple_path(&self) -> bool {
        self.use_tree_list().is_none() && self.star_token().is_none()
    }

    pub fn parent_use_tree_list(&self) -> Option<ast::UseTreeList> {
        self.syntax().parent().and_then(ast::UseTreeList::cast)
    }

    pub fn top_use_tree(&self) -> ast::UseTree {
        let mut this = self.clone();
        while let Some(use_tree_list) = this.parent_use_tree_list() {
            this = use_tree_list.parent_use_tree();
        }
        this
    }
}

impl ast::UseTreeList {
    pub fn parent_use_tree(&self) -> ast::UseTree {
        self.syntax()
            .parent()
            .and_then(ast::UseTree::cast)
            .expect("UseTreeLists are always nested in UseTrees")
    }

    pub fn has_inner_comment(&self) -> bool {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.into_token())
            .find_map(ast::Comment::cast)
            .is_some()
    }

    pub fn comma(&self) -> impl Iterator<Item = SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.into_token().filter(|it| it.kind() == T![,]))
    }

    /// Remove the unnecessary braces in current `UseTreeList`
    pub fn remove_unnecessary_braces(mut self) {
        // Returns true iff there is a single subtree and it is not the self keyword. The braces in
        // `use x::{self};` are necessary and so we should not remove them.
        let has_single_subtree_that_is_not_self = |u: &ast::UseTreeList| {
            if let Some((single_subtree,)) = u.use_trees().collect_tuple() {
                // We have a single subtree, check whether it is self.

                let is_self = single_subtree.path().as_ref().is_some_and(|path| {
                    path.segment().and_then(|seg| seg.self_token()).is_some()
                        && path.qualifier().is_none()
                });

                !is_self
            } else {
                // Not a single subtree
                false
            }
        };

        let remove_brace_in_use_tree_list = |u: &ast::UseTreeList| {
            if has_single_subtree_that_is_not_self(u) {
                if let Some(a) = u.l_curly_token() {
                    ted::remove(a)
                }
                if let Some(a) = u.r_curly_token() {
                    ted::remove(a)
                }
                u.comma().for_each(ted::remove);
            }
        };

        // take `use crate::{{{{A}}}}` for example
        // the below remove the innermost {}, got `use crate::{{{A}}}`
        remove_brace_in_use_tree_list(&self);

        // the below remove other unnecessary {}, got `use crate::A`
        while let Some(parent_use_tree_list) = self.parent_use_tree().parent_use_tree_list() {
            remove_brace_in_use_tree_list(&parent_use_tree_list);
            self = parent_use_tree_list;
        }
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

    pub fn for_trait_name_ref(name_ref: &ast::NameRef) -> Option<ast::Impl> {
        let this = name_ref.syntax().ancestors().find_map(ast::Impl::cast)?;
        if this.trait_()?.syntax().text_range().start() == name_ref.syntax().text_range().start() {
            Some(this)
        } else {
            None
        }
    }
}

// [#15778](https://github.com/rust-lang/rust-analyzer/issues/15778)
impl ast::PathSegment {
    pub fn qualifying_trait(&self) -> Option<ast::PathType> {
        let mut path_types = support::children(self.type_anchor()?.syntax());
        let first = path_types.next()?;
        path_types.next().or(Some(first))
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

impl ast::Union {
    pub fn kind(&self) -> StructKind {
        StructKind::from_node(self)
    }
}

impl ast::RecordExprField {
    pub fn for_field_name(field_name: &ast::NameRef) -> Option<ast::RecordExprField> {
        let candidate = Self::for_name_ref(field_name)?;
        if candidate.field_name().as_ref() == Some(field_name) { Some(candidate) } else { None }
    }

    pub fn for_name_ref(name_ref: &ast::NameRef) -> Option<ast::RecordExprField> {
        let syn = name_ref.syntax();
        syn.parent()
            .and_then(ast::RecordExprField::cast)
            .or_else(|| syn.ancestors().nth(4).and_then(ast::RecordExprField::cast))
    }

    /// Deals with field init shorthand
    pub fn field_name(&self) -> Option<ast::NameRef> {
        if let Some(name_ref) = self.name_ref() {
            return Some(name_ref);
        }
        if let ast::Expr::PathExpr(expr) = self.expr()? {
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

#[derive(Debug, Clone)]
pub enum NameLike {
    NameRef(ast::NameRef),
    Name(ast::Name),
    Lifetime(ast::Lifetime),
}

impl NameLike {
    pub fn as_name_ref(&self) -> Option<&ast::NameRef> {
        match self {
            NameLike::NameRef(name_ref) => Some(name_ref),
            _ => None,
        }
    }
    pub fn as_lifetime(&self) -> Option<&ast::Lifetime> {
        match self {
            NameLike::Lifetime(lifetime) => Some(lifetime),
            _ => None,
        }
    }
    pub fn text(&self) -> TokenText<'_> {
        match self {
            NameLike::NameRef(name_ref) => name_ref.text(),
            NameLike::Name(name) => name.text(),
            NameLike::Lifetime(lifetime) => lifetime.text(),
        }
    }
}

impl ast::AstNode for NameLike {
    fn can_cast(kind: SyntaxKind) -> bool {
        matches!(kind, SyntaxKind::NAME | SyntaxKind::NAME_REF | SyntaxKind::LIFETIME)
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            SyntaxKind::NAME => NameLike::Name(ast::Name { syntax }),
            SyntaxKind::NAME_REF => NameLike::NameRef(ast::NameRef { syntax }),
            SyntaxKind::LIFETIME => NameLike::Lifetime(ast::Lifetime { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            NameLike::NameRef(it) => it.syntax(),
            NameLike::Name(it) => it.syntax(),
            NameLike::Lifetime(it) => it.syntax(),
        }
    }
}

const _: () = {
    use ast::{Lifetime, Name, NameRef};
    stdx::impl_from!(NameRef, Name, Lifetime for NameLike);
};

#[derive(Debug, Clone, PartialEq)]
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

impl ast::AstNode for NameOrNameRef {
    fn can_cast(kind: SyntaxKind) -> bool {
        matches!(kind, SyntaxKind::NAME | SyntaxKind::NAME_REF)
    }
    fn cast(syntax: SyntaxNode) -> Option<Self> {
        let res = match syntax.kind() {
            SyntaxKind::NAME => NameOrNameRef::Name(ast::Name { syntax }),
            SyntaxKind::NAME_REF => NameOrNameRef::NameRef(ast::NameRef { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxNode {
        match self {
            NameOrNameRef::NameRef(it) => it.syntax(),
            NameOrNameRef::Name(it) => it.syntax(),
        }
    }
}

impl NameOrNameRef {
    pub fn text(&self) -> TokenText<'_> {
        match self {
            NameOrNameRef::Name(name) => name.text(),
            NameOrNameRef::NameRef(name_ref) => name_ref.text(),
        }
    }
}

impl ast::RecordPatField {
    pub fn for_field_name_ref(field_name: &ast::NameRef) -> Option<ast::RecordPatField> {
        let candidate = field_name.syntax().parent().and_then(ast::RecordPatField::cast)?;
        match candidate.field_name()? {
            NameOrNameRef::NameRef(name_ref) if name_ref == *field_name => Some(candidate),
            _ => None,
        }
    }

    pub fn for_field_name(field_name: &ast::Name) -> Option<ast::RecordPatField> {
        let candidate =
            field_name.syntax().ancestors().nth(2).and_then(ast::RecordPatField::cast)?;
        match candidate.field_name()? {
            NameOrNameRef::Name(name) if name == *field_name => Some(candidate),
            _ => None,
        }
    }

    pub fn parent_record_pat(&self) -> ast::RecordPat {
        self.syntax().ancestors().find_map(ast::RecordPat::cast).unwrap()
    }

    /// Deals with field init shorthand
    pub fn field_name(&self) -> Option<NameOrNameRef> {
        if let Some(name_ref) = self.name_ref() {
            return Some(NameOrNameRef::NameRef(name_ref));
        }
        match self.pat() {
            Some(ast::Pat::IdentPat(pat)) => {
                let name = pat.name()?;
                Some(NameOrNameRef::Name(name))
            }
            Some(ast::Pat::BoxPat(pat)) => match pat.pat() {
                Some(ast::Pat::IdentPat(pat)) => {
                    let name = pat.name()?;
                    Some(NameOrNameRef::Name(name))
                }
                _ => None,
            },
            _ => None,
        }
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

impl ast::Item {
    pub fn generic_param_list(&self) -> Option<ast::GenericParamList> {
        ast::AnyHasGenericParams::cast(self.syntax().clone())?.generic_param_list()
    }
}

impl ast::Type {
    pub fn generic_arg_list(&self) -> Option<ast::GenericArgList> {
        if let ast::Type::PathType(path_type) = self {
            path_type.path()?.segment()?.generic_arg_list()
        } else {
            None
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
        match self.name_ref() {
            Some(nr) => Some(FieldKind::Name(nr)),
            None => self.index_token().map(FieldKind::Index),
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
                ast::Pat::IdentPat(bp) => !matches!(bp.pat(), Some(ast::Pat::RestPat(_))),
                ast::Pat::RefPat(rp) => match rp.pat() {
                    Some(ast::Pat::RestPat(_)) => false,
                    Some(ast::Pat::IdentPat(bp)) => !matches!(bp.pat(), Some(ast::Pat::RestPat(_))),
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

impl ast::IdentPat {
    pub fn is_simple_ident(&self) -> bool {
        self.at_token().is_none()
            && self.mut_token().is_none()
            && self.ref_token().is_none()
            && self.pat().is_none()
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
            if self.mut_token().is_some() { SelfParamKind::MutRef } else { SelfParamKind::Ref }
        } else {
            SelfParamKind::Owned
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeBoundKind {
    /// Trait
    PathType(Option<ast::ForBinder>, ast::PathType),
    /// use
    Use(ast::UseBoundGenericArgs),
    /// 'a
    Lifetime(ast::Lifetime),
}

impl ast::TypeBound {
    pub fn kind(&self) -> TypeBoundKind {
        if let Some(path_type) = support::children(self.syntax()).next() {
            TypeBoundKind::PathType(self.for_binder(), path_type)
        } else if let Some(args) = self.use_bound_generic_args() {
            TypeBoundKind::Use(args)
        } else if let Some(lifetime) = self.lifetime() {
            TypeBoundKind::Lifetime(lifetime)
        } else {
            unreachable!()
        }
    }
}

#[derive(Debug, Clone)]
pub enum TypeOrConstParam {
    Type(ast::TypeParam),
    Const(ast::ConstParam),
}

impl From<TypeOrConstParam> for GenericParam {
    fn from(value: TypeOrConstParam) -> Self {
        match value {
            TypeOrConstParam::Type(it) => GenericParam::TypeParam(it),
            TypeOrConstParam::Const(it) => GenericParam::ConstParam(it),
        }
    }
}

impl TypeOrConstParam {
    pub fn name(&self) -> Option<ast::Name> {
        match self {
            TypeOrConstParam::Type(x) => x.name(),
            TypeOrConstParam::Const(x) => x.name(),
        }
    }
}

impl AstNode for TypeOrConstParam {
    fn can_cast(kind: SyntaxKind) -> bool
    where
        Self: Sized,
    {
        matches!(kind, SyntaxKind::TYPE_PARAM | SyntaxKind::CONST_PARAM)
    }

    fn cast(syntax: SyntaxNode) -> Option<Self>
    where
        Self: Sized,
    {
        let res = match syntax.kind() {
            SyntaxKind::TYPE_PARAM => TypeOrConstParam::Type(ast::TypeParam { syntax }),
            SyntaxKind::CONST_PARAM => TypeOrConstParam::Const(ast::ConstParam { syntax }),
            _ => return None,
        };
        Some(res)
    }

    fn syntax(&self) -> &SyntaxNode {
        match self {
            TypeOrConstParam::Type(it) => it.syntax(),
            TypeOrConstParam::Const(it) => it.syntax(),
        }
    }
}

impl HasAttrs for TypeOrConstParam {}

#[derive(Debug, Clone)]
pub enum TraitOrAlias {
    Trait(ast::Trait),
    TraitAlias(ast::TraitAlias),
}

impl TraitOrAlias {
    pub fn name(&self) -> Option<ast::Name> {
        match self {
            TraitOrAlias::Trait(x) => x.name(),
            TraitOrAlias::TraitAlias(x) => x.name(),
        }
    }
}

impl AstNode for TraitOrAlias {
    fn can_cast(kind: SyntaxKind) -> bool
    where
        Self: Sized,
    {
        matches!(kind, SyntaxKind::TRAIT | SyntaxKind::TRAIT_ALIAS)
    }

    fn cast(syntax: SyntaxNode) -> Option<Self>
    where
        Self: Sized,
    {
        let res = match syntax.kind() {
            SyntaxKind::TRAIT => TraitOrAlias::Trait(ast::Trait { syntax }),
            SyntaxKind::TRAIT_ALIAS => TraitOrAlias::TraitAlias(ast::TraitAlias { syntax }),
            _ => return None,
        };
        Some(res)
    }

    fn syntax(&self) -> &SyntaxNode {
        match self {
            TraitOrAlias::Trait(it) => it.syntax(),
            TraitOrAlias::TraitAlias(it) => it.syntax(),
        }
    }
}

impl HasAttrs for TraitOrAlias {}

pub enum VisibilityKind {
    In(ast::Path),
    PubCrate,
    PubSuper,
    PubSelf,
    Pub,
}

impl ast::Visibility {
    pub fn kind(&self) -> VisibilityKind {
        match self.path() {
            Some(path) => {
                if let Some(segment) =
                    path.as_single_segment().filter(|it| it.coloncolon_token().is_none())
                {
                    if segment.crate_token().is_some() {
                        return VisibilityKind::PubCrate;
                    } else if segment.super_token().is_some() {
                        return VisibilityKind::PubSuper;
                    } else if segment.self_token().is_some() {
                        return VisibilityKind::PubSelf;
                    }
                }
                VisibilityKind::In(path)
            }
            None => VisibilityKind::Pub,
        }
    }
}

impl ast::LifetimeParam {
    pub fn lifetime_bounds(&self) -> impl Iterator<Item = SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.into_token())
            .skip_while(|x| x.kind() != T![:])
            .filter(|it| it.kind() == T![lifetime_ident])
    }
}

impl ast::Module {
    /// Returns the parent ast::Module, this is different than the semantic parent in that this only
    /// considers parent declarations in the AST
    pub fn parent(&self) -> Option<ast::Module> {
        self.syntax().ancestors().nth(2).and_then(ast::Module::cast)
    }
}

impl RangeItem for ast::RangePat {
    type Bound = ast::Pat;

    fn start(&self) -> Option<ast::Pat> {
        self.syntax()
            .children_with_tokens()
            .take_while(|it| !(it.kind() == T![..] || it.kind() == T![..=]))
            .filter_map(|it| it.into_node())
            .find_map(ast::Pat::cast)
    }

    fn end(&self) -> Option<ast::Pat> {
        self.syntax()
            .children_with_tokens()
            .skip_while(|it| !(it.kind() == T![..] || it.kind() == T![..=]))
            .filter_map(|it| it.into_node())
            .find_map(ast::Pat::cast)
    }

    fn op_token(&self) -> Option<SyntaxToken> {
        self.syntax().children_with_tokens().find_map(|it| {
            let token = it.into_token()?;

            match token.kind() {
                T![..] => Some(token),
                T![..=] => Some(token),
                _ => None,
            }
        })
    }

    fn op_kind(&self) -> Option<RangeOp> {
        self.syntax().children_with_tokens().find_map(|it| {
            let token = it.into_token()?;

            match token.kind() {
                T![..] => Some(RangeOp::Exclusive),
                T![..=] => Some(RangeOp::Inclusive),
                _ => None,
            }
        })
    }
}

impl ast::TokenTree {
    pub fn token_trees_and_tokens(
        &self,
    ) -> impl Iterator<Item = NodeOrToken<ast::TokenTree, SyntaxToken>> {
        self.syntax().children_with_tokens().filter_map(|not| match not {
            NodeOrToken::Node(node) => ast::TokenTree::cast(node).map(NodeOrToken::Node),
            NodeOrToken::Token(t) => Some(NodeOrToken::Token(t)),
        })
    }

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

    pub fn parent_meta(&self) -> Option<ast::Meta> {
        self.syntax().parent().and_then(ast::Meta::cast)
    }
}

impl ast::Meta {
    pub fn parent_attr(&self) -> Option<ast::Attr> {
        self.syntax().parent().and_then(ast::Attr::cast)
    }
}

impl ast::GenericArgList {
    pub fn lifetime_args(&self) -> impl Iterator<Item = ast::LifetimeArg> {
        self.generic_args().filter_map(|arg| match arg {
            ast::GenericArg::LifetimeArg(it) => Some(it),
            _ => None,
        })
    }
}

impl ast::GenericParamList {
    pub fn lifetime_params(&self) -> impl Iterator<Item = ast::LifetimeParam> {
        self.generic_params().filter_map(|param| match param {
            ast::GenericParam::LifetimeParam(it) => Some(it),
            ast::GenericParam::TypeParam(_) | ast::GenericParam::ConstParam(_) => None,
        })
    }
    pub fn type_or_const_params(&self) -> impl Iterator<Item = ast::TypeOrConstParam> + use<> {
        self.generic_params().filter_map(|param| match param {
            ast::GenericParam::TypeParam(it) => Some(ast::TypeOrConstParam::Type(it)),
            ast::GenericParam::LifetimeParam(_) => None,
            ast::GenericParam::ConstParam(it) => Some(ast::TypeOrConstParam::Const(it)),
        })
    }
}

impl ast::ForExpr {
    pub fn iterable(&self) -> Option<ast::Expr> {
        // If the iterable is a BlockExpr, check if the body is missing.
        // If it is assume the iterable is the expression that is missing instead.
        let mut exprs = support::children(self.syntax());
        let first = exprs.next();
        match first {
            Some(ast::Expr::BlockExpr(_)) => exprs.next().and(first),
            first => first,
        }
    }
}

impl ast::HasLoopBody for ast::ForExpr {
    fn loop_body(&self) -> Option<ast::BlockExpr> {
        let mut exprs = support::children(self.syntax());
        let first = exprs.next();
        let second = exprs.next();
        second.or(first)
    }
}

impl ast::WhileExpr {
    pub fn condition(&self) -> Option<ast::Expr> {
        // If the condition is a BlockExpr, check if the body is missing.
        // If it is assume the condition is the expression that is missing instead.
        let mut exprs = support::children(self.syntax());
        let first = exprs.next();
        match first {
            Some(ast::Expr::BlockExpr(_)) => exprs.next().and(first),
            first => first,
        }
    }
}

impl ast::HasLoopBody for ast::WhileExpr {
    fn loop_body(&self) -> Option<ast::BlockExpr> {
        let mut exprs = support::children(self.syntax());
        let first = exprs.next();
        let second = exprs.next();
        second.or(first)
    }
}

impl ast::HasAttrs for ast::AnyHasDocComments {}

impl From<ast::Adt> for ast::Item {
    fn from(it: ast::Adt) -> Self {
        match it {
            ast::Adt::Enum(it) => ast::Item::Enum(it),
            ast::Adt::Struct(it) => ast::Item::Struct(it),
            ast::Adt::Union(it) => ast::Item::Union(it),
        }
    }
}

impl ast::MatchGuard {
    pub fn condition(&self) -> Option<ast::Expr> {
        support::child(&self.syntax)
    }
}

impl From<ast::Item> for ast::AnyHasAttrs {
    fn from(node: ast::Item) -> Self {
        Self::new(node)
    }
}

impl From<ast::AssocItem> for ast::AnyHasAttrs {
    fn from(node: ast::AssocItem) -> Self {
        Self::new(node)
    }
}

impl ast::OrPat {
    pub fn leading_pipe(&self) -> Option<SyntaxToken> {
        self.syntax
            .children_with_tokens()
            .find(|it| !it.kind().is_trivia())
            .and_then(NodeOrToken::into_token)
            .filter(|it| it.kind() == T![|])
    }
}
