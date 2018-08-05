extern crate libsyntax2;
extern crate text_unit;

use libsyntax2::{
    SyntaxNodeRef,
    algo::walk,
    SyntaxKind::*,
};
use text_unit::TextRange;

pub struct File {
    inner: libsyntax2::File
}

pub struct HighlightedRange {
    pub range: TextRange,
    pub tag: &'static str,
}

#[derive(Debug)]
pub struct Symbol {
    // pub parent: ???,
    pub name: String,
    pub range: TextRange,
}

impl File {
    pub fn new(text: &str) -> File {
        File {
            inner: libsyntax2::File::parse(text)
        }
    }

    pub fn highlight(&self) -> Vec<HighlightedRange> {
        let syntax = self.inner.syntax();
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

    pub fn syntax_tree(&self) -> String {
        ::libsyntax2::utils::dump_tree(&self.inner.syntax())
    }

    pub fn symbols(&self) -> Vec<Symbol> {
        let syntax = self.inner.syntax();
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
}


struct Declaration<'f>(SyntaxNodeRef<'f>);

impl<'f> Declaration<'f> {
    fn cast(node: SyntaxNodeRef<'f>) -> Option<Declaration<'f>> {
        match node.kind() {
            | STRUCT_ITEM | ENUM_ITEM | FN_ITEM | TRAIT_ITEM
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
