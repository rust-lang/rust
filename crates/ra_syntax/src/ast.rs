//! Abstract Syntax Tree, layered on top of untyped `SyntaxNode`s
mod generated;

use std::marker::PhantomData;

use itertools::Itertools;

pub use self::generated::*;
use crate::{
    syntax_node::{SyntaxNode, SyntaxNodeChildren, TreeArc, RaTypes},
    SmolStr,
    SyntaxKind::*,
};

/// The main trait to go from untyped `SyntaxNode`  to a typed ast. The
/// conversion itself has zero runtime cost: ast and syntax nodes have exactly
/// the same representation: a pointer to the tree root and a pointer to the
/// node itself.
pub trait AstNode:
    rowan::TransparentNewType<Repr = rowan::SyntaxNode<RaTypes>> + ToOwned<Owned = TreeArc<Self>>
{
    fn cast(syntax: &SyntaxNode) -> Option<&Self>
    where
        Self: Sized;
    fn syntax(&self) -> &SyntaxNode;
}

pub trait AstToken: AstNode {
    fn text(&self) -> &SmolStr {
        self.syntax().leaf_text().unwrap()
    }
}

pub trait TypeAscriptionOwner: AstNode {
    fn ascribed_type(&self) -> Option<&TypeRef> {
        child_opt(self)
    }
}

pub trait NameOwner: AstNode {
    fn name(&self) -> Option<&Name> {
        child_opt(self)
    }
}

pub trait VisibilityOwner: AstNode {
    fn visibility(&self) -> Option<&Visibility> {
        child_opt(self)
    }
}

pub trait LoopBodyOwner: AstNode {
    fn loop_body(&self) -> Option<&Block> {
        child_opt(self)
    }
}

pub trait ArgListOwner: AstNode {
    fn arg_list(&self) -> Option<&ArgList> {
        child_opt(self)
    }
}

pub trait FnDefOwner: AstNode {
    fn functions(&self) -> AstChildren<FnDef> {
        children(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemOrMacro<'a> {
    Item(&'a ModuleItem),
    Macro(&'a MacroCall),
}

pub trait ModuleItemOwner: AstNode {
    fn items(&self) -> AstChildren<ModuleItem> {
        children(self)
    }
    fn items_with_macros(&self) -> ItemOrMacroIter {
        ItemOrMacroIter(self.syntax().children())
    }
}

#[derive(Debug)]
pub struct ItemOrMacroIter<'a>(SyntaxNodeChildren<'a>);

impl<'a> Iterator for ItemOrMacroIter<'a> {
    type Item = ItemOrMacro<'a>;
    fn next(&mut self) -> Option<ItemOrMacro<'a>> {
        loop {
            let n = self.0.next()?;
            if let Some(item) = ModuleItem::cast(n) {
                return Some(ItemOrMacro::Item(item));
            }
            if let Some(call) = MacroCall::cast(n) {
                return Some(ItemOrMacro::Macro(call));
            }
        }
    }
}

pub trait TypeParamsOwner: AstNode {
    fn type_param_list(&self) -> Option<&TypeParamList> {
        child_opt(self)
    }

    fn where_clause(&self) -> Option<&WhereClause> {
        child_opt(self)
    }
}

pub trait AttrsOwner: AstNode {
    fn attrs(&self) -> AstChildren<Attr> {
        children(self)
    }
    fn has_atom_attr(&self, atom: &str) -> bool {
        self.attrs().filter_map(|x| x.as_atom()).any(|x| x == atom)
    }
}

pub trait DocCommentsOwner: AstNode {
    fn doc_comments(&self) -> AstChildren<Comment> {
        children(self)
    }

    /// Returns the textual content of a doc comment block as a single string.
    /// That is, strips leading `///` (+ optional 1 character of whitespace)
    /// and joins lines.
    fn doc_comment_text(&self) -> Option<std::string::String> {
        let docs = self
            .doc_comments()
            .filter(|comment| comment.is_doc_comment())
            .map(|comment| {
                let prefix_len = comment.prefix().len();

                let line = comment.text().as_str();

                // Determine if the prefix or prefix + 1 char is stripped
                let pos =
                    if line.chars().nth(prefix_len).map(|c| c.is_whitespace()).unwrap_or(false) {
                        prefix_len + 1
                    } else {
                        prefix_len
                    };

                line[pos..].to_owned()
            })
            .join("\n");

        if docs.is_empty() {
            None
        } else {
            Some(docs)
        }
    }
}

impl Attr {
    pub fn is_inner(&self) -> bool {
        let tt = match self.value() {
            None => return false,
            Some(tt) => tt,
        };

        let prev = match tt.syntax().prev_sibling() {
            None => return false,
            Some(prev) => prev,
        };

        prev.kind() == EXCL
    }

    pub fn as_atom(&self) -> Option<SmolStr> {
        let tt = self.value()?;
        let (_bra, attr, _ket) = tt.syntax().children().collect_tuple()?;
        if attr.kind() == IDENT {
            Some(attr.leaf_text().unwrap().clone())
        } else {
            None
        }
    }

    pub fn as_call(&self) -> Option<(SmolStr, &TokenTree)> {
        let tt = self.value()?;
        let (_bra, attr, args, _ket) = tt.syntax().children().collect_tuple()?;
        let args = TokenTree::cast(args)?;
        if attr.kind() == IDENT {
            Some((attr.leaf_text().unwrap().clone(), args))
        } else {
            None
        }
    }

    pub fn as_named(&self) -> Option<SmolStr> {
        let tt = self.value()?;
        let attr = tt.syntax().children().nth(1)?;
        if attr.kind() == IDENT {
            Some(attr.leaf_text().unwrap().clone())
        } else {
            None
        }
    }
}

impl Comment {
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

impl Whitespace {
    pub fn count_newlines_lazy(&self) -> impl Iterator<Item = &()> {
        self.text().chars().filter(|&c| c == '\n').map(|_| &())
    }

    pub fn has_newlines(&self) -> bool {
        self.text().contains('\n')
    }
}

impl Name {
    pub fn text(&self) -> &SmolStr {
        let ident = self.syntax().first_child().unwrap();
        ident.leaf_text().unwrap()
    }
}

impl NameRef {
    pub fn text(&self) -> &SmolStr {
        let ident = self.syntax().first_child().unwrap();
        ident.leaf_text().unwrap()
    }
}

impl ImplBlock {
    pub fn target_type(&self) -> Option<&TypeRef> {
        match self.target() {
            (Some(t), None) | (_, Some(t)) => Some(t),
            _ => None,
        }
    }

    pub fn target_trait(&self) -> Option<&TypeRef> {
        match self.target() {
            (Some(t), Some(_)) => Some(t),
            _ => None,
        }
    }

    fn target(&self) -> (Option<&TypeRef>, Option<&TypeRef>) {
        let mut types = children(self);
        let first = types.next();
        let second = types.next();
        (first, second)
    }
}

impl Module {
    pub fn has_semi(&self) -> bool {
        match self.syntax().last_child() {
            None => false,
            Some(node) => node.kind() == SEMI,
        }
    }
}

impl LetStmt {
    pub fn has_semi(&self) -> bool {
        match self.syntax().last_child() {
            None => false,
            Some(node) => node.kind() == SEMI,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElseBranchFlavor<'a> {
    Block(&'a Block),
    IfExpr(&'a IfExpr),
}

impl IfExpr {
    pub fn then_branch(&self) -> Option<&Block> {
        self.blocks().nth(0)
    }
    pub fn else_branch(&self) -> Option<ElseBranchFlavor> {
        let res = match self.blocks().nth(1) {
            Some(block) => ElseBranchFlavor::Block(block),
            None => {
                let elif: &IfExpr = child_opt(self)?;
                ElseBranchFlavor::IfExpr(elif)
            }
        };
        Some(res)
    }

    fn blocks(&self) -> AstChildren<Block> {
        children(self)
    }
}

impl ExprStmt {
    pub fn has_semi(&self) -> bool {
        match self.syntax().last_child() {
            None => false,
            Some(node) => node.kind() == SEMI,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathSegmentKind<'a> {
    Name(&'a NameRef),
    SelfKw,
    SuperKw,
    CrateKw,
}

impl PathSegment {
    pub fn parent_path(&self) -> &Path {
        self.syntax().parent().and_then(Path::cast).expect("segments are always nested in paths")
    }

    pub fn kind(&self) -> Option<PathSegmentKind> {
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

    pub fn has_colon_colon(&self) -> bool {
        match self.syntax.first_child().map(|s| s.kind()) {
            Some(COLONCOLON) => true,
            _ => false,
        }
    }
}

impl Path {
    pub fn parent_path(&self) -> Option<&Path> {
        self.syntax().parent().and_then(Path::cast)
    }
}

impl UseTree {
    pub fn has_star(&self) -> bool {
        self.syntax().children().any(|it| it.kind() == STAR)
    }
}

impl UseTreeList {
    pub fn parent_use_tree(&self) -> &UseTree {
        self.syntax()
            .parent()
            .and_then(UseTree::cast)
            .expect("UseTreeLists are always nested in UseTrees")
    }
}

impl RefPat {
    pub fn is_mut(&self) -> bool {
        self.syntax().children().any(|n| n.kind() == MUT_KW)
    }
}

fn child_opt<P: AstNode, C: AstNode>(parent: &P) -> Option<&C> {
    children(parent).next()
}

fn children<P: AstNode, C: AstNode>(parent: &P) -> AstChildren<C> {
    AstChildren::new(parent.syntax())
}

#[derive(Debug)]
pub struct AstChildren<'a, N> {
    inner: SyntaxNodeChildren<'a>,
    ph: PhantomData<N>,
}

impl<'a, N> AstChildren<'a, N> {
    fn new(parent: &'a SyntaxNode) -> Self {
        AstChildren { inner: parent.children(), ph: PhantomData }
    }
}

impl<'a, N: AstNode + 'a> Iterator for AstChildren<'a, N> {
    type Item = &'a N;
    fn next(&mut self) -> Option<&'a N> {
        loop {
            if let Some(n) = N::cast(self.inner.next()?) {
                return Some(n);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructFlavor<'a> {
    Tuple(&'a PosFieldDefList),
    Named(&'a NamedFieldDefList),
    Unit,
}

impl StructFlavor<'_> {
    fn from_node<N: AstNode>(node: &N) -> StructFlavor {
        if let Some(nfdl) = child_opt::<_, NamedFieldDefList>(node) {
            StructFlavor::Named(nfdl)
        } else if let Some(pfl) = child_opt::<_, PosFieldDefList>(node) {
            StructFlavor::Tuple(pfl)
        } else {
            StructFlavor::Unit
        }
    }
}

impl StructDef {
    pub fn flavor(&self) -> StructFlavor {
        StructFlavor::from_node(self)
    }
}

impl EnumVariant {
    pub fn parent_enum(&self) -> &EnumDef {
        self.syntax()
            .parent()
            .and_then(|it| it.parent())
            .and_then(EnumDef::cast)
            .expect("EnumVariants are always nested in Enums")
    }
    pub fn flavor(&self) -> StructFlavor {
        StructFlavor::from_node(self)
    }
}

impl PointerType {
    pub fn is_mut(&self) -> bool {
        self.syntax().children().any(|n| n.kind() == MUT_KW)
    }
}

impl ReferenceType {
    pub fn is_mut(&self) -> bool {
        self.syntax().children().any(|n| n.kind() == MUT_KW)
    }
}

impl RefExpr {
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

impl PrefixExpr {
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
    /// The `!=` operator for equality testing
    NegatedEqualityTest,
    /// The `<=` operator for lesser-equal testing
    LesserEqualTest,
    /// The `>=` operator for greater-equal testing
    GreaterEqualTest,
    /// The `<` operator for comparison
    LesserTest,
    /// The `>` operator for comparison
    GreaterTest,
    /// The `+` operator for addition
    Addition,
    /// The `*` operator for multiplication
    Multiplication,
    /// The `-` operator for subtraction
    Subtraction,
    /// The `/` operator for division
    Division,
    /// The `%` operator for remainder after division
    Remainder,
    /// The `<<` operator for left shift
    LeftShift,
    /// The `>>` operator for right shift
    RightShift,
    /// The `^` operator for bitwise XOR
    BitwiseXor,
    /// The `|` operator for bitwise OR
    BitwiseOr,
    /// The `&` operator for bitwise AND
    BitwiseAnd,
    /// The `..` operator for right-open ranges
    RangeRightOpen,
    /// The `..=` operator for right-closed ranges
    RangeRightClosed,
    /// The `=` operator for assignment
    Assignment,
    /// The `+=` operator for assignment after addition
    AddAssign,
    /// The `/=` operator for assignment after division
    DivAssign,
    /// The `*=` operator for assignment after multiplication
    MulAssign,
    /// The `%=` operator for assignment after remainders
    RemAssign,
    /// The `>>=` operator for assignment after shifting right
    ShrAssign,
    /// The `<<=` operator for assignment after shifting left
    ShlAssign,
    /// The `-=` operator for assignment after subtraction
    SubAssign,
    /// The `|=` operator for assignment after bitwise OR
    BitOrAssign,
    /// The `&=` operator for assignment after bitwise AND
    BitAndAssign,
    /// The `^=` operator for assignment after bitwise XOR
    BitXorAssign,
}

impl BinExpr {
    pub fn op(&self) -> Option<BinOp> {
        self.syntax()
            .children()
            .filter_map(|c| match c.kind() {
                PIPEPIPE => Some(BinOp::BooleanOr),
                AMPAMP => Some(BinOp::BooleanAnd),
                EQEQ => Some(BinOp::EqualityTest),
                NEQ => Some(BinOp::NegatedEqualityTest),
                LTEQ => Some(BinOp::LesserEqualTest),
                GTEQ => Some(BinOp::GreaterEqualTest),
                L_ANGLE => Some(BinOp::LesserTest),
                R_ANGLE => Some(BinOp::GreaterTest),
                PLUS => Some(BinOp::Addition),
                STAR => Some(BinOp::Multiplication),
                MINUS => Some(BinOp::Subtraction),
                SLASH => Some(BinOp::Division),
                PERCENT => Some(BinOp::Remainder),
                SHL => Some(BinOp::LeftShift),
                SHR => Some(BinOp::RightShift),
                CARET => Some(BinOp::BitwiseXor),
                PIPE => Some(BinOp::BitwiseOr),
                AMP => Some(BinOp::BitwiseAnd),
                DOTDOT => Some(BinOp::RangeRightOpen),
                DOTDOTEQ => Some(BinOp::RangeRightClosed),
                EQ => Some(BinOp::Assignment),
                PLUSEQ => Some(BinOp::AddAssign),
                SLASHEQ => Some(BinOp::DivAssign),
                STAREQ => Some(BinOp::MulAssign),
                PERCENTEQ => Some(BinOp::RemAssign),
                SHREQ => Some(BinOp::ShrAssign),
                SHLEQ => Some(BinOp::ShlAssign),
                MINUSEQ => Some(BinOp::SubAssign),
                PIPEEQ => Some(BinOp::BitOrAssign),
                AMPEQ => Some(BinOp::BitAndAssign),
                CARETEQ => Some(BinOp::BitXorAssign),
                _ => None,
            })
            .next()
    }

    pub fn lhs(&self) -> Option<&Expr> {
        children(self).nth(0)
    }

    pub fn rhs(&self) -> Option<&Expr> {
        children(self).nth(1)
    }

    pub fn sub_exprs(&self) -> (Option<&Expr>, Option<&Expr>) {
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

impl SelfParam {
    pub fn flavor(&self) -> SelfParamFlavor {
        let borrowed = self.syntax().children().any(|n| n.kind() == AMP);
        if borrowed {
            // check for a `mut` coming after the & -- `mut &self` != `&mut self`
            if self.syntax().children().skip_while(|n| n.kind() != AMP).any(|n| n.kind() == MUT_KW)
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LiteralFlavor {
    String,
    ByteString,
    Char,
    Byte,
    IntNumber { suffix: Option<SmolStr> },
    FloatNumber { suffix: Option<SmolStr> },
    Bool,
}

impl LiteralExpr {
    pub fn flavor(&self) -> LiteralFlavor {
        let syntax = self.syntax();
        match syntax.kind() {
            INT_NUMBER => {
                let allowed_suffix_list = [
                    "isize", "i128", "i64", "i32", "i16", "i8", "usize", "u128", "u64", "u32",
                    "u16", "u8",
                ];
                let text = syntax.text().to_string();
                let suffix = allowed_suffix_list
                    .iter()
                    .find(|&s| text.ends_with(s))
                    .map(|&suf| SmolStr::new(suf));
                LiteralFlavor::IntNumber { suffix: suffix }
            }
            FLOAT_NUMBER => {
                let allowed_suffix_list = ["f64", "f32"];
                let text = syntax.text().to_string();
                let suffix = allowed_suffix_list
                    .iter()
                    .find(|&s| text.ends_with(s))
                    .map(|&suf| SmolStr::new(suf));
                LiteralFlavor::FloatNumber { suffix: suffix }
            }
            STRING | RAW_STRING => LiteralFlavor::String,
            TRUE_KW | FALSE_KW => LiteralFlavor::Bool,
            BYTE_STRING | RAW_BYTE_STRING => LiteralFlavor::ByteString,
            CHAR => LiteralFlavor::Char,
            BYTE => LiteralFlavor::Byte,
            _ => unreachable!(),
        }
    }
}

impl NamedField {
    pub fn parent_struct_lit(&self) -> &StructLit {
        self.syntax().ancestors().find_map(StructLit::cast).unwrap()
    }
}

impl BindPat {
    pub fn is_mutable(&self) -> bool {
        self.syntax().children().any(|n| n.kind() == MUT_KW)
    }

    pub fn is_ref(&self) -> bool {
        self.syntax().children().any(|n| n.kind() == REF_KW)
    }
}

#[test]
fn test_doc_comment_none() {
    let file = SourceFile::parse(
        r#"
        // non-doc
        mod foo {}
        "#,
    );
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert!(module.doc_comment_text().is_none());
}

#[test]
fn test_doc_comment_of_items() {
    let file = SourceFile::parse(
        r#"
        //! doc
        // non-doc
        mod foo {}
        "#,
    );
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!("doc", module.doc_comment_text().unwrap());
}

#[test]
fn test_doc_comment_preserves_indents() {
    let file = SourceFile::parse(
        r#"
        /// doc1
        /// ```
        /// fn foo() {
        ///     // ...
        /// }
        /// ```
        mod foo {}
        "#,
    );
    let module = file.syntax().descendants().find_map(Module::cast).unwrap();
    assert_eq!("doc1\n```\nfn foo() {\n    // ...\n}\n```", module.doc_comment_text().unwrap());
}
