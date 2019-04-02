//! Abstract Syntax Tree, layered on top of untyped `SyntaxNode`s
mod generated;
mod traits;

use std::marker::PhantomData;

use itertools::Itertools;

use crate::{
    syntax_node::{SyntaxNode, SyntaxNodeChildren, TreeArc, RaTypes, SyntaxToken, SyntaxElement},
    SmolStr,
    SyntaxKind::*,
};

pub use self::{
    generated::*,
    traits::*,
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
        self.inner.by_ref().find_map(N::cast)
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
        let (_bra, attr, _ket) = tt.syntax().children_with_tokens().collect_tuple()?;
        if attr.kind() == IDENT {
            Some(attr.as_token()?.text().clone())
        } else {
            None
        }
    }

    pub fn as_call(&self) -> Option<(SmolStr, &TokenTree)> {
        let tt = self.value()?;
        let (_bra, attr, args, _ket) = tt.syntax().children_with_tokens().collect_tuple()?;
        let args = TokenTree::cast(args.as_node()?)?;
        if attr.kind() == IDENT {
            Some((attr.as_token()?.text().clone(), args))
        } else {
            None
        }
    }

    pub fn as_named(&self) -> Option<SmolStr> {
        let tt = self.value()?;
        let attr = tt.syntax().children_with_tokens().nth(1)?;
        if attr.kind() == IDENT {
            Some(attr.as_token()?.text().clone())
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Comment<'a>(SyntaxToken<'a>);

impl<'a> Comment<'a> {
    pub fn cast(token: SyntaxToken<'a>) -> Option<Self> {
        if token.kind() == COMMENT {
            Some(Comment(token))
        } else {
            None
        }
    }

    pub fn syntax(&self) -> SyntaxToken<'a> {
        self.0
    }

    pub fn text(&self) -> &'a SmolStr {
        self.0.text()
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

pub struct Whitespace<'a>(SyntaxToken<'a>);

impl<'a> Whitespace<'a> {
    pub fn cast(token: SyntaxToken<'a>) -> Option<Self> {
        if token.kind() == WHITESPACE {
            Some(Whitespace(token))
        } else {
            None
        }
    }

    pub fn syntax(&self) -> SyntaxToken<'a> {
        self.0
    }

    pub fn text(&self) -> &'a SmolStr {
        self.0.text()
    }

    pub fn spans_multiple_lines(&self) -> bool {
        let text = self.text();
        text.find('\n').map_or(false, |idx| text[idx + 1..].contains('\n'))
    }
}

impl Name {
    pub fn text(&self) -> &SmolStr {
        let ident = self.syntax().first_child_or_token().unwrap().as_token().unwrap();
        ident.text()
    }
}

impl NameRef {
    pub fn text(&self) -> &SmolStr {
        let ident = self.syntax().first_child_or_token().unwrap().as_token().unwrap();
        ident.text()
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
        match self.syntax().last_child_or_token() {
            None => false,
            Some(node) => node.kind() == SEMI,
        }
    }
}

impl LetStmt {
    pub fn has_semi(&self) -> bool {
        match self.syntax().last_child_or_token() {
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
        match self.syntax().last_child_or_token() {
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
            match self.syntax().first_child_or_token()?.kind() {
                SELF_KW => PathSegmentKind::SelfKw,
                SUPER_KW => PathSegmentKind::SuperKw,
                CRATE_KW => PathSegmentKind::CrateKw,
                _ => return None,
            }
        };
        Some(res)
    }

    pub fn has_colon_colon(&self) -> bool {
        match self.syntax.first_child_or_token().map(|s| s.kind()) {
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
        self.syntax().children_with_tokens().any(|it| it.kind() == STAR)
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
        self.syntax().children_with_tokens().any(|n| n.kind() == MUT_KW)
    }
}

fn child_opt<P: AstNode, C: AstNode>(parent: &P) -> Option<&C> {
    children(parent).next()
}

fn children<P: AstNode, C: AstNode>(parent: &P) -> AstChildren<C> {
    AstChildren::new(parent.syntax())
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
        self.syntax().children_with_tokens().any(|n| n.kind() == MUT_KW)
    }
}

impl ReferenceType {
    pub fn is_mut(&self) -> bool {
        self.syntax().children_with_tokens().any(|n| n.kind() == MUT_KW)
    }
}

impl RefExpr {
    pub fn is_mut(&self) -> bool {
        self.syntax().children_with_tokens().any(|n| n.kind() == MUT_KW)
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
    pub fn op_kind(&self) -> Option<PrefixOp> {
        match self.op_token()?.kind() {
            STAR => Some(PrefixOp::Deref),
            EXCL => Some(PrefixOp::Not),
            MINUS => Some(PrefixOp::Neg),
            _ => None,
        }
    }

    pub fn op_token(&self) -> Option<SyntaxToken> {
        self.syntax().first_child_or_token()?.as_token()
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
    fn op_details(&self) -> Option<(SyntaxToken, BinOp)> {
        self.syntax().children_with_tokens().filter_map(|it| it.as_token()).find_map(|c| {
            match c.kind() {
                PIPEPIPE => Some((c, BinOp::BooleanOr)),
                AMPAMP => Some((c, BinOp::BooleanAnd)),
                EQEQ => Some((c, BinOp::EqualityTest)),
                NEQ => Some((c, BinOp::NegatedEqualityTest)),
                LTEQ => Some((c, BinOp::LesserEqualTest)),
                GTEQ => Some((c, BinOp::GreaterEqualTest)),
                L_ANGLE => Some((c, BinOp::LesserTest)),
                R_ANGLE => Some((c, BinOp::GreaterTest)),
                PLUS => Some((c, BinOp::Addition)),
                STAR => Some((c, BinOp::Multiplication)),
                MINUS => Some((c, BinOp::Subtraction)),
                SLASH => Some((c, BinOp::Division)),
                PERCENT => Some((c, BinOp::Remainder)),
                SHL => Some((c, BinOp::LeftShift)),
                SHR => Some((c, BinOp::RightShift)),
                CARET => Some((c, BinOp::BitwiseXor)),
                PIPE => Some((c, BinOp::BitwiseOr)),
                AMP => Some((c, BinOp::BitwiseAnd)),
                DOTDOT => Some((c, BinOp::RangeRightOpen)),
                DOTDOTEQ => Some((c, BinOp::RangeRightClosed)),
                EQ => Some((c, BinOp::Assignment)),
                PLUSEQ => Some((c, BinOp::AddAssign)),
                SLASHEQ => Some((c, BinOp::DivAssign)),
                STAREQ => Some((c, BinOp::MulAssign)),
                PERCENTEQ => Some((c, BinOp::RemAssign)),
                SHREQ => Some((c, BinOp::ShrAssign)),
                SHLEQ => Some((c, BinOp::ShlAssign)),
                MINUSEQ => Some((c, BinOp::SubAssign)),
                PIPEEQ => Some((c, BinOp::BitOrAssign)),
                AMPEQ => Some((c, BinOp::BitAndAssign)),
                CARETEQ => Some((c, BinOp::BitXorAssign)),
                _ => None,
            }
        })
    }

    pub fn op_kind(&self) -> Option<BinOp> {
        self.op_details().map(|t| t.1)
    }

    pub fn op_token(&self) -> Option<SyntaxToken> {
        self.op_details().map(|t| t.0)
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
    pub fn self_kw_token(&self) -> SyntaxToken {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.as_token())
            .find(|it| it.kind() == SELF_KW)
            .expect("invalid tree: self param must have self")
    }

    pub fn flavor(&self) -> SelfParamFlavor {
        let borrowed = self.syntax().children_with_tokens().any(|n| n.kind() == AMP);
        if borrowed {
            // check for a `mut` coming after the & -- `mut &self` != `&mut self`
            if self
                .syntax()
                .children_with_tokens()
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

impl Literal {
    pub fn token(&self) -> SyntaxToken {
        match self.syntax().first_child_or_token().unwrap() {
            SyntaxElement::Token(token) => token,
            _ => unreachable!(),
        }
    }

    pub fn flavor(&self) -> LiteralFlavor {
        match self.token().kind() {
            INT_NUMBER => {
                let allowed_suffix_list = [
                    "isize", "i128", "i64", "i32", "i16", "i8", "usize", "u128", "u64", "u32",
                    "u16", "u8",
                ];
                let text = self.token().text().to_string();
                let suffix = allowed_suffix_list
                    .iter()
                    .find(|&s| text.ends_with(s))
                    .map(|&suf| SmolStr::new(suf));
                LiteralFlavor::IntNumber { suffix }
            }
            FLOAT_NUMBER => {
                let allowed_suffix_list = ["f64", "f32"];
                let text = self.token().text().to_string();
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
        self.syntax().children_with_tokens().any(|n| n.kind() == MUT_KW)
    }

    pub fn is_ref(&self) -> bool {
        self.syntax().children_with_tokens().any(|n| n.kind() == REF_KW)
    }
}

impl LifetimeParam {
    pub fn lifetime_token(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.as_token())
            .find(|it| it.kind() == LIFETIME)
    }
}

impl WherePred {
    pub fn lifetime_token(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(|it| it.as_token())
            .find(|it| it.kind() == LIFETIME)
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

#[test]
fn test_where_predicates() {
    fn assert_bound(text: &str, bound: Option<&TypeBound>) {
        assert_eq!(text, bound.unwrap().syntax().text().to_string());
    }

    let file = SourceFile::parse(
        r#"
fn foo()
where
   T: Clone + Copy + Debug + 'static,
   'a: 'b + 'c,
   Iterator::Item: 'a + Debug,
   Iterator::Item: Debug + 'a,
   <T as Iterator>::Item: Debug + 'a,
   for<'a> F: Fn(&'a str)
{}
        "#,
    );
    let where_clause = file.syntax().descendants().find_map(WhereClause::cast).unwrap();

    let mut predicates = where_clause.predicates();

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert_eq!("T", pred.type_ref().unwrap().syntax().text().to_string());
    assert_bound("Clone", bounds.next());
    assert_bound("Copy", bounds.next());
    assert_bound("Debug", bounds.next());
    assert_bound("'static", bounds.next());

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert_eq!("'a", pred.lifetime_token().unwrap().text());

    assert_bound("'b", bounds.next());
    assert_bound("'c", bounds.next());

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert_eq!("Iterator::Item", pred.type_ref().unwrap().syntax().text().to_string());
    assert_bound("'a", bounds.next());

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert_eq!("Iterator::Item", pred.type_ref().unwrap().syntax().text().to_string());
    assert_bound("Debug", bounds.next());
    assert_bound("'a", bounds.next());

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert_eq!("<T as Iterator>::Item", pred.type_ref().unwrap().syntax().text().to_string());
    assert_bound("Debug", bounds.next());
    assert_bound("'a", bounds.next());

    let pred = predicates.next().unwrap();
    let mut bounds = pred.type_bound_list().unwrap().bounds();

    assert_eq!("for<'a> F", pred.type_ref().unwrap().syntax().text().to_string());
    assert_bound("Fn(&'a str)", bounds.next());
}
