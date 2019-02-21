//! Syntax Tree library used throughout the rust analyzer.
//!
//! Properties:
//!   - easy and fast incremental re-parsing
//!   - graceful handling of errors
//!   - full-fidelity representation (*any* text can be precisely represented as
//!     a syntax tree)
//!
//! For more information, see the [RFC]. Current implementation is inspired by
//! the [Swift] one.
//!
//! The most interesting modules here are `syntax_node` (which defines concrete
//! syntax tree) and `ast` (which defines abstract syntax tree on top of the
//! CST). The actual parser live in a separate `ra_parser` crate, thought the
//! lexer lives in this crate.
//!
//! See `api_walkthrough` test in this file for a quick API tour!
//!
//! [RFC]: <https://github.com/rust-lang/rfcs/pull/2256>
//! [Swift]: <https://github.com/apple/swift/blob/13d593df6f359d0cb2fc81cfaac273297c539455/lib/Syntax/README.md>

mod syntax_node;
mod syntax_text;
mod syntax_error;
mod parsing;
mod string_lexing;
mod validation;
mod ptr;

pub mod algo;
pub mod ast;

pub use rowan::{SmolStr, TextRange, TextUnit};
pub use ra_parser::SyntaxKind;
pub use crate::{
    ast::AstNode,
    syntax_error::{SyntaxError, SyntaxErrorKind, Location},
    syntax_text::SyntaxText,
    syntax_node::{Direction,  SyntaxNode, WalkEvent, TreeArc},
    ptr::{SyntaxNodePtr, AstPtr},
    parsing::{tokenize, Token},
};

use ra_text_edit::AtomTextEdit;
use crate::syntax_node::GreenNode;

/// `SourceFile` represents a parse tree for a single Rust file.
pub use crate::ast::SourceFile;

impl SourceFile {
    fn new(green: GreenNode, errors: Vec<SyntaxError>) -> TreeArc<SourceFile> {
        let root = SyntaxNode::new(green, errors);
        if cfg!(debug_assertions) {
            validation::validate_block_structure(&root);
        }
        assert_eq!(root.kind(), SyntaxKind::SOURCE_FILE);
        TreeArc::cast(root)
    }

    pub fn parse(text: &str) -> TreeArc<SourceFile> {
        let (green, errors) = parsing::parse_text(text);
        SourceFile::new(green, errors)
    }

    pub fn reparse(&self, edit: &AtomTextEdit) -> TreeArc<SourceFile> {
        self.incremental_reparse(edit).unwrap_or_else(|| self.full_reparse(edit))
    }

    pub fn incremental_reparse(&self, edit: &AtomTextEdit) -> Option<TreeArc<SourceFile>> {
        parsing::incremental_reparse(self.syntax(), edit, self.errors())
            .map(|(green_node, errors)| SourceFile::new(green_node, errors))
    }

    fn full_reparse(&self, edit: &AtomTextEdit) -> TreeArc<SourceFile> {
        let text = edit.apply(self.syntax().text().to_string());
        SourceFile::parse(&text)
    }

    pub fn errors(&self) -> Vec<SyntaxError> {
        let mut errors = self.syntax.root_data().clone();
        errors.extend(validation::validate(self));
        errors
    }
}

pub fn check_fuzz_invariants(text: &str) {
    let file = SourceFile::parse(text);
    let root = file.syntax();
    validation::validate_block_structure(root);
    let _ = file.errors();
}

/// This test does not assert anything and instead just shows off the crate's
/// API.
#[test]
fn api_walkthrough() {
    use ast::{ModuleItemOwner, NameOwner};

    let source_code = "
        fn foo() {
            1 + 1
        }
    ";
    // `SourceFile` is the main entry point.
    //
    // Note how `parse` does not return a `Result`: even completely invalid
    // source code might be parsed.
    let file = SourceFile::parse(source_code);

    // Due to the way ownership is set up, owned syntax Nodes always live behind
    // a `TreeArc` smart pointer. `TreeArc` is roughly an `std::sync::Arc` which
    // points to the whole file instead of an individual node.
    let file: TreeArc<SourceFile> = file;

    // `SourceFile` is the root of the syntax tree. We can iterate file's items:
    let mut func = None;
    for item in file.items() {
        match item.kind() {
            ast::ModuleItemKind::FnDef(f) => func = Some(f),
            _ => unreachable!(),
        }
    }
    // The returned items are always references.
    let func: &ast::FnDef = func.unwrap();

    // All nodes implement `ToOwned` trait, with `Owned = TreeArc<Self>`.
    // `to_owned` is a cheap operation: atomic increment.
    let _owned_func: TreeArc<ast::FnDef> = func.to_owned();

    // Each AST node has a bunch of getters for children. All getters return
    // `Option`s though, to account for incomplete code. Some getters are common
    // for several kinds of node. In this case, a trait like `ast::NameOwner`
    // usually exists. By convention, all ast types should be used with `ast::`
    // qualifier.
    let name: Option<&ast::Name> = func.name();
    let name = name.unwrap();
    assert_eq!(name.text(), "foo");

    // Let's get the `1 + 1` expression!
    let block: &ast::Block = func.body().unwrap();
    let expr: &ast::Expr = block.expr().unwrap();

    // "Enum"-like nodes are represented using the "kind" pattern. It allows us
    // to match exhaustively against all flavors of nodes, while maintaining
    // internal representation flexibility. The drawback is that one can't write
    // nested matches as one pattern.
    let bin_expr: &ast::BinExpr = match expr.kind() {
        ast::ExprKind::BinExpr(e) => e,
        _ => unreachable!(),
    };

    // Besides the "typed" AST API, there's an untyped CST one as well.
    // To switch from AST to CST, call `.syntax()` method:
    let expr_syntax: &SyntaxNode = expr.syntax();

    // Note how `expr` and `bin_expr` are in fact the same node underneath:
    assert!(std::ptr::eq(expr_syntax, bin_expr.syntax()));

    // To go from CST to AST, `AstNode::cast` function is used:
    let expr = match ast::Expr::cast(expr_syntax) {
        Some(e) => e,
        None => unreachable!(),
    };

    // Note how expr is also a reference!
    let expr: &ast::Expr = expr;

    // This is possible because the underlying representation is the same:
    assert_eq!(
        expr as *const ast::Expr as *const u8,
        expr_syntax as *const SyntaxNode as *const u8
    );

    // The two properties each syntax node has is a `SyntaxKind`:
    assert_eq!(expr_syntax.kind(), SyntaxKind::BIN_EXPR);

    // And text range:
    assert_eq!(expr_syntax.range(), TextRange::from_to(32.into(), 37.into()));

    // You can get node's text as a `SyntaxText` object, which will traverse the
    // tree collecting token's text:
    let text: SyntaxText<'_> = expr_syntax.text();
    assert_eq!(text.to_string(), "1 + 1");

    // There's a bunch of traversal methods on `SyntaxNode`:
    assert_eq!(expr_syntax.parent(), Some(block.syntax()));
    assert_eq!(block.syntax().first_child().map(|it| it.kind()), Some(SyntaxKind::L_CURLY));
    assert_eq!(expr_syntax.next_sibling().map(|it| it.kind()), Some(SyntaxKind::WHITESPACE));

    // As well as some iterator helpers:
    let f = expr_syntax.ancestors().find_map(ast::FnDef::cast);
    assert_eq!(f, Some(&*func));
    assert!(expr_syntax.siblings(Direction::Next).any(|it| it.kind() == SyntaxKind::R_CURLY));
    assert_eq!(
        expr_syntax.descendants().count(),
        8, // 5 tokens `1`, ` `, `+`, ` `, `!`
           // 2 child literal expressions: `1`, `1`
           // 1 the node itself: `1 + 1`
    );

    // There's also a `preorder` method with a more fine-grained iteration control:
    let mut buf = String::new();
    let mut indent = 0;
    for event in expr_syntax.preorder() {
        match event {
            WalkEvent::Enter(node) => {
                buf += &format!(
                    "{:indent$}{:?} {:?}\n",
                    " ",
                    node.text(),
                    node.kind(),
                    indent = indent
                );
                indent += 2;
            }
            WalkEvent::Leave(_) => indent -= 2,
        }
    }
    assert_eq!(indent, 0);
    assert_eq!(
        buf.trim(),
        r#"
"1 + 1" BIN_EXPR
  "1" LITERAL
    "1" INT_NUMBER
  " " WHITESPACE
  "+" PLUS
  " " WHITESPACE
  "1" LITERAL
    "1" INT_NUMBER
"#
        .trim()
    );

    // To recursively process the tree, there are three approaches:
    // 1. explicitly call getter methods on AST nodes.
    // 2. use descendants and `AstNode::cast`.
    // 3. use descendants and the visitor.
    //
    // Here's how the first one looks like:
    let exprs_cast: Vec<String> = file
        .syntax()
        .descendants()
        .filter_map(ast::Expr::cast)
        .map(|expr| expr.syntax().text().to_string())
        .collect();

    // An alternative is to use a visitor. The visitor does not do traversal
    // automatically (so it's more akin to a generic lambda) and is constructed
    // from closures. This seems more flexible than a single generated visitor
    // trait.
    use algo::visit::{visitor, Visitor};
    let mut exprs_visit = Vec::new();
    for node in file.syntax().descendants() {
        if let Some(result) =
            visitor().visit::<ast::Expr, _>(|expr| expr.syntax().text().to_string()).accept(node)
        {
            exprs_visit.push(result);
        }
    }
    assert_eq!(exprs_cast, exprs_visit);
}
