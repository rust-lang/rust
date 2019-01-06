mod generated;

use std::marker::PhantomData;
use std::string::String as RustString;

use itertools::Itertools;

pub use self::generated::*;
use crate::{
    yellow::{RefRoot, SyntaxNodeChildren},
    SmolStr,
    SyntaxKind::*,
    SyntaxNodeRef,
};

/// The main trait to go from untyped `SyntaxNode`  to a typed ast. The
/// conversion itself has zero runtime cost: ast and syntax nodes have exactly
/// the same representation: a pointer to the tree root and a pointer to the
/// node itself.
pub trait AstNode<'a>: Clone + Copy + 'a {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self>
    where
        Self: Sized;
    fn syntax(self) -> SyntaxNodeRef<'a>;
}

pub trait NameOwner<'a>: AstNode<'a> {
    fn name(self) -> Option<Name<'a>> {
        child_opt(self)
    }
}

pub trait VisibilityOwner<'a>: AstNode<'a> {
    fn visibility(self) -> Option<Visibility<'a>> {
        child_opt(self)
    }
}

pub trait LoopBodyOwner<'a>: AstNode<'a> {
    fn loop_body(self) -> Option<Block<'a>> {
        child_opt(self)
    }
}

pub trait ArgListOwner<'a>: AstNode<'a> {
    fn arg_list(self) -> Option<ArgList<'a>> {
        child_opt(self)
    }
}

pub trait FnDefOwner<'a>: AstNode<'a> {
    fn functions(self) -> AstChildren<'a, FnDef<'a>> {
        children(self)
    }
}

// ModuleItem
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemOrMacro<'a> {
    Item(ModuleItem<'a>),
    Macro(MacroCall<'a>),
}

impl<'a> AstNode<'a> for ItemOrMacro<'a> {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self> {
        let res = if let Some(item) = ModuleItem::cast(syntax) {
            ItemOrMacro::Item(item)
        } else if let Some(macro_call) = MacroCall::cast(syntax) {
            ItemOrMacro::Macro(macro_call)
        } else {
            return None;
        };
        Some(res)
    }
    fn syntax(self) -> SyntaxNodeRef<'a> {
        match self {
            ItemOrMacro::Item(it) => it.syntax(),
            ItemOrMacro::Macro(it) => it.syntax(),
        }
    }
}

pub trait ModuleItemOwner<'a>: AstNode<'a> {
    fn items(self) -> AstChildren<'a, ModuleItem<'a>> {
        children(self)
    }

    fn items_with_macros(self) -> AstChildren<'a, ItemOrMacro<'a>> {
        children(self)
    }
}

pub trait TypeParamsOwner<'a>: AstNode<'a> {
    fn type_param_list(self) -> Option<TypeParamList<'a>> {
        child_opt(self)
    }

    fn where_clause(self) -> Option<WhereClause<'a>> {
        child_opt(self)
    }
}

pub trait AttrsOwner<'a>: AstNode<'a> {
    fn attrs(self) -> AstChildren<'a, Attr<'a>> {
        children(self)
    }
}

pub trait DocCommentsOwner<'a>: AstNode<'a> {
    fn doc_comments(self) -> AstChildren<'a, Comment<'a>> {
        children(self)
    }

    /// Returns the textual content of a doc comment block as a single string.
    /// That is, strips leading `///` and joins lines
    fn doc_comment_text(self) -> RustString {
        self.doc_comments()
            .filter(|comment| comment.is_doc_comment())
            .map(|comment| {
                let prefix = comment.prefix();
                let trimmed = comment
                    .text()
                    .as_str()
                    .trim()
                    .trim_start_matches(prefix)
                    .trim_start();
                trimmed.to_owned()
            })
            .join("\n")
    }
}

impl<'a> FnDef<'a> {
    pub fn has_atom_attr(&self, atom: &str) -> bool {
        self.attrs().filter_map(|x| x.as_atom()).any(|x| x == atom)
    }
}

impl<'a> Attr<'a> {
    pub fn as_atom(&self) -> Option<SmolStr> {
        let tt = self.value()?;
        let (_bra, attr, _ket) = tt.syntax().children().collect_tuple()?;
        if attr.kind() == IDENT {
            Some(attr.leaf_text().unwrap().clone())
        } else {
            None
        }
    }

    pub fn as_call(&self) -> Option<(SmolStr, TokenTree<'a>)> {
        let tt = self.value()?;
        let (_bra, attr, args, _ket) = tt.syntax().children().collect_tuple()?;
        let args = TokenTree::cast(args)?;
        if attr.kind() == IDENT {
            Some((attr.leaf_text().unwrap().clone(), args))
        } else {
            None
        }
    }
}

impl<'a> Lifetime<'a> {
    pub fn text(&self) -> SmolStr {
        self.syntax().leaf_text().unwrap().clone()
    }
}

impl<'a> Char<'a> {
    pub fn text(&self) -> &SmolStr {
        &self.syntax().leaf_text().unwrap()
    }
}

impl<'a> Byte<'a> {
    pub fn text(&self) -> &SmolStr {
        &self.syntax().leaf_text().unwrap()
    }
}

impl<'a> ByteString<'a> {
    pub fn text(&self) -> &SmolStr {
        &self.syntax().leaf_text().unwrap()
    }
}

impl<'a> String<'a> {
    pub fn text(&self) -> &SmolStr {
        &self.syntax().leaf_text().unwrap()
    }
}

impl<'a> Comment<'a> {
    pub fn text(&self) -> &SmolStr {
        self.syntax().leaf_text().unwrap()
    }

    pub fn flavor(&self) -> CommentFlavor {
        let text = self.text();
        if text.starts_with("///") {
            CommentFlavor::Doc
        } else if text.starts_with("//!") {
            CommentFlavor::ModuleDoc
        } else if text.starts_with("//") {
            CommentFlavor::Line
        } else {
            CommentFlavor::Multiline
        }
    }

    pub fn is_doc_comment(&self) -> bool {
        self.flavor().is_doc_comment()
    }

    pub fn prefix(&self) -> &'static str {
        self.flavor().prefix()
    }

    pub fn count_newlines_lazy(&self) -> impl Iterator<Item = &()> {
        self.text().chars().filter(|&c| c == '\n').map(|_| &())
    }

    pub fn has_newlines(&self) -> bool {
        self.count_newlines_lazy().count() > 0
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum CommentFlavor {
    Line,
    Doc,
    ModuleDoc,
    Multiline,
}

impl CommentFlavor {
    pub fn prefix(&self) -> &'static str {
        use self::CommentFlavor::*;
        match *self {
            Line => "//",
            Doc => "///",
            ModuleDoc => "//!",
            Multiline => "/*",
        }
    }

    pub fn is_doc_comment(&self) -> bool {
        match self {
            CommentFlavor::Doc | CommentFlavor::ModuleDoc => true,
            _ => false,
        }
    }
}

impl<'a> Whitespace<'a> {
    pub fn text(&self) -> &SmolStr {
        &self.syntax().leaf_text().unwrap()
    }

    pub fn count_newlines_lazy(&self) -> impl Iterator<Item = &()> {
        self.text().chars().filter(|&c| c == '\n').map(|_| &())
    }

    pub fn has_newlines(&self) -> bool {
        self.count_newlines_lazy().count() > 0
    }
}

impl<'a> Name<'a> {
    pub fn text(&self) -> SmolStr {
        let ident = self.syntax().first_child().unwrap();
        ident.leaf_text().unwrap().clone()
    }
}

impl<'a> NameRef<'a> {
    pub fn text(&self) -> SmolStr {
        let ident = self.syntax().first_child().unwrap();
        ident.leaf_text().unwrap().clone()
    }
}

impl<'a> ImplBlock<'a> {
    pub fn target_type(self) -> Option<TypeRef<'a>> {
        match self.target() {
            (Some(t), None) | (_, Some(t)) => Some(t),
            _ => None,
        }
    }

    pub fn target_trait(self) -> Option<TypeRef<'a>> {
        match self.target() {
            (Some(t), Some(_)) => Some(t),
            _ => None,
        }
    }

    fn target(self) -> (Option<TypeRef<'a>>, Option<TypeRef<'a>>) {
        let mut types = children(self);
        let first = types.next();
        let second = types.next();
        (first, second)
    }
}

impl<'a> Module<'a> {
    pub fn has_semi(self) -> bool {
        match self.syntax().last_child() {
            None => false,
            Some(node) => node.kind() == SEMI,
        }
    }
}

impl<'a> LetStmt<'a> {
    pub fn has_semi(self) -> bool {
        match self.syntax().last_child() {
            None => false,
            Some(node) => node.kind() == SEMI,
        }
    }
}

impl<'a> IfExpr<'a> {
    pub fn then_branch(self) -> Option<Block<'a>> {
        self.blocks().nth(0)
    }
    pub fn else_branch(self) -> Option<Block<'a>> {
        self.blocks().nth(1)
    }
    fn blocks(self) -> AstChildren<'a, Block<'a>> {
        children(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathSegmentKind<'a> {
    Name(NameRef<'a>),
    SelfKw,
    SuperKw,
    CrateKw,
}

impl<'a> PathSegment<'a> {
    pub fn parent_path(self) -> Path<'a> {
        self.syntax()
            .parent()
            .and_then(Path::cast)
            .expect("segments are always nested in paths")
    }

    pub fn kind(self) -> Option<PathSegmentKind<'a>> {
        let res = if let Some(name_ref) = self.name_ref() {
            PathSegmentKind::Name(name_ref)
        } else {
            match self.syntax().first_child()?.kind() {
                SELF_KW => PathSegmentKind::SelfKw,
                SUPER_KW => PathSegmentKind::SuperKw,
                CRATE_KW => PathSegmentKind::CrateKw,
                _ => return None,
            }
        };
        Some(res)
    }
}

impl<'a> Path<'a> {
    pub fn parent_path(self) -> Option<Path<'a>> {
        self.syntax().parent().and_then(Path::cast)
    }
}

impl<'a> UseTree<'a> {
    pub fn has_star(self) -> bool {
        self.syntax().children().any(|it| it.kind() == STAR)
    }
}

impl<'a> UseTreeList<'a> {
    pub fn parent_use_tree(self) -> UseTree<'a> {
        self.syntax()
            .parent()
            .and_then(UseTree::cast)
            .expect("UseTreeLists are always nested in UseTrees")
    }
}

fn child_opt<'a, P: AstNode<'a>, C: AstNode<'a>>(parent: P) -> Option<C> {
    children(parent).next()
}

fn children<'a, P: AstNode<'a>, C: AstNode<'a>>(parent: P) -> AstChildren<'a, C> {
    AstChildren::new(parent.syntax())
}

#[derive(Debug)]
pub struct AstChildren<'a, N> {
    inner: SyntaxNodeChildren<RefRoot<'a>>,
    ph: PhantomData<N>,
}

impl<'a, N> AstChildren<'a, N> {
    fn new(parent: SyntaxNodeRef<'a>) -> Self {
        AstChildren {
            inner: parent.children(),
            ph: PhantomData,
        }
    }
}

impl<'a, N: AstNode<'a>> Iterator for AstChildren<'a, N> {
    type Item = N;
    fn next(&mut self) -> Option<N> {
        loop {
            if let Some(n) = N::cast(self.inner.next()?) {
                return Some(n);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructFlavor<'a> {
    Tuple(PosFieldList<'a>),
    Named(NamedFieldDefList<'a>),
    Unit,
}

impl<'a> StructFlavor<'a> {
    fn from_node<N: AstNode<'a>>(node: N) -> StructFlavor<'a> {
        if let Some(nfdl) = child_opt::<_, NamedFieldDefList>(node) {
            StructFlavor::Named(nfdl)
        } else if let Some(pfl) = child_opt::<_, PosFieldList>(node) {
            StructFlavor::Tuple(pfl)
        } else {
            StructFlavor::Unit
        }
    }
}

impl<'a> StructDef<'a> {
    pub fn flavor(self) -> StructFlavor<'a> {
        StructFlavor::from_node(self)
    }
}

impl<'a> EnumVariant<'a> {
    pub fn flavor(self) -> StructFlavor<'a> {
        StructFlavor::from_node(self)
    }
}

impl<'a> PointerType<'a> {
    pub fn is_mut(&self) -> bool {
        self.syntax().children().any(|n| n.kind() == MUT_KW)
    }
}

impl<'a> ReferenceType<'a> {
    pub fn is_mut(&self) -> bool {
        self.syntax().children().any(|n| n.kind() == MUT_KW)
    }
}

impl<'a> RefExpr<'a> {
    pub fn is_mut(&self) -> bool {
        self.syntax().children().any(|n| n.kind() == MUT_KW)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PrefixOp {
    /// The `*` operator for dereferencing
    Deref,
    /// The `!` operator for logical inversion
    Not,
    /// The `-` operator for negation
    Neg,
}

impl<'a> PrefixExpr<'a> {
    pub fn op(&self) -> Option<PrefixOp> {
        match self.syntax().first_child()?.kind() {
            STAR => Some(PrefixOp::Deref),
            EXCL => Some(PrefixOp::Not),
            MINUS => Some(PrefixOp::Neg),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    /// The `||` operator for boolean OR
    BooleanOr,
    /// The `&&` operator for boolean AND
    BooleanAnd,
    /// The `==` operator for equality testing
    EqualityTest,
    /// The `<=` operator for lesser-equal testing
    LesserEqualTest,
    /// The `>=` operator for greater-equal testing
    GreaterEqualTest,
    /// The `<` operator for comparison
    LesserTest,
    /// The `>` operator for comparison
    GreaterTest,
    // TODO: lots of others
}

impl<'a> BinExpr<'a> {
    pub fn op(&self) -> Option<BinOp> {
        self.syntax()
            .children()
            .filter_map(|c| match c.kind() {
                PIPEPIPE => Some(BinOp::BooleanOr),
                AMPAMP => Some(BinOp::BooleanAnd),
                EQEQ => Some(BinOp::EqualityTest),
                LTEQ => Some(BinOp::LesserEqualTest),
                GTEQ => Some(BinOp::GreaterEqualTest),
                L_ANGLE => Some(BinOp::LesserTest),
                R_ANGLE => Some(BinOp::GreaterTest),
                _ => None,
            })
            .next()
    }

    pub fn lhs(self) -> Option<Expr<'a>> {
        children(self).nth(0)
    }

    pub fn rhs(self) -> Option<Expr<'a>> {
        children(self).nth(1)
    }

    pub fn sub_exprs(self) -> (Option<Expr<'a>>, Option<Expr<'a>>) {
        let mut children = children(self);
        let first = children.next();
        let second = children.next();
        (first, second)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SelfParamFlavor {
    /// self
    Owned,
    /// &self
    Ref,
    /// &mut self
    MutRef,
}

impl<'a> SelfParam<'a> {
    pub fn flavor(&self) -> SelfParamFlavor {
        let borrowed = self.syntax().children().any(|n| n.kind() == AMP);
        if borrowed {
            // check for a `mut` coming after the & -- `mut &self` != `&mut self`
            if self
                .syntax()
                .children()
                .skip_while(|n| n.kind() != AMP)
                .any(|n| n.kind() == MUT_KW)
            {
                SelfParamFlavor::MutRef
            } else {
                SelfParamFlavor::Ref
            }
        } else {
            SelfParamFlavor::Owned
        }
    }
}

#[test]
fn test_doc_comment_of_items() {
    let file = SourceFileNode::parse(
        r#"
        //! doc
        // non-doc
        mod foo {}
        "#,
    );
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!("doc", module.doc_comment_text());
}
