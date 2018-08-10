extern crate libsyntax2;
extern crate superslice;

mod extend_selection;
mod line_index;

use libsyntax2::{
    SyntaxNodeRef, AstNode,
    algo::walk,
    SyntaxKind::*,
};
pub use libsyntax2::{TextRange, TextUnit, ast};
pub use self::line_index::{LineIndex, LineCol};

#[derive(Debug)]
pub struct HighlightedRange {
    pub range: TextRange,
    pub tag: &'static str,
}

#[derive(Debug)]
pub struct Diagnostic {
    pub range: TextRange,
    pub msg: String,
}

#[derive(Debug)]
pub struct Symbol {
    // pub parent: ???,
    pub name: String,
    pub range: TextRange,
}

#[derive(Debug)]
pub struct Runnable {
    pub range: TextRange,
    pub kind: RunnableKind,
}

#[derive(Debug)]
pub enum RunnableKind {
    Test { name: String },
    Bin,
}

pub fn highlight(file: &ast::File) -> Vec<HighlightedRange> {
    let syntax = file.syntax();
    let mut res = Vec::new();
    for node in walk::preorder(syntax.as_ref()) {
        let tag = match node.kind() {
            ERROR => "error",
            COMMENT | DOC_COMMENT => "comment",
            STRING | RAW_STRING | RAW_BYTE_STRING | BYTE_STRING => "string",
            ATTR => "attribute",
            NAME_REF => "text",
            NAME => "function",
            INT_NUMBER | FLOAT_NUMBER | CHAR | BYTE => "literal",
            LIFETIME => "parameter",
            k if k.is_keyword() => "keyword",
            _ => continue,
        };
        res.push(HighlightedRange {
            range: node.range(),
            tag,
        })
    }
    res
}

pub fn diagnostics(file: &ast::File) -> Vec<Diagnostic> {
    let syntax = file.syntax();
    let mut res = Vec::new();

    for node in walk::preorder(syntax.as_ref()) {
        if node.kind() == ERROR {
            res.push(Diagnostic {
                range: node.range(),
                msg: "Syntax Error".to_string(),
            });
        }
    }
    res.extend(file.errors().into_iter().map(|err| Diagnostic {
        range: TextRange::offset_len(err.offset, 1.into()),
        msg: err.msg,
    }));
    res
}

pub fn syntax_tree(file: &ast::File) -> String {
    ::libsyntax2::utils::dump_tree(&file.syntax())
}

pub fn symbols(file: &ast::File) -> Vec<Symbol> {
    let syntax = file.syntax();
    let res: Vec<Symbol> = walk::preorder(syntax.as_ref())
        .filter_map(Declaration::cast)
        .filter_map(|decl| {
            let name = decl.name()?;
            let range = decl.range();
            Some(Symbol { name, range })
        })
        .collect();
    res // NLL :-(
}

pub fn extend_selection(file: &ast::File, range: TextRange) -> Option<TextRange> {
    let syntax = file.syntax();
    extend_selection::extend_selection(syntax.as_ref(), range)
}

pub fn runnables(file: &ast::File) -> Vec<Runnable> {
    file
        .functions()
        .filter_map(|f| {
            let name = f.name()?.text();
            let kind = if name == "main" {
                RunnableKind::Bin
            } else if f.has_atom_attr("test") {
                RunnableKind::Test {
                    name: name.to_string()
                }
            } else {
                return None;
            };
            Some(Runnable {
                range: f.syntax().range(),
                kind,
            })
        })
        .collect()
}


struct Declaration<'f> (SyntaxNodeRef<'f>);

impl<'f> Declaration<'f> {
    fn cast(node: SyntaxNodeRef<'f>) -> Option<Declaration<'f>> {
        match node.kind() {
            | STRUCT_ITEM | ENUM_ITEM | FUNCTION | TRAIT_ITEM
            | CONST_ITEM | STATIC_ITEM | MOD_ITEM | NAMED_FIELD
            | TYPE_ITEM => Some(Declaration(node)),
            _ => None
        }
    }

    fn name(&self) -> Option<String> {
        let name = self.0.children()
            .find(|child| child.kind() == NAME)?;
        Some(name.text())
    }

    fn range(&self) -> TextRange {
        self.0.range()
    }
}
