extern crate libsyntax2;
extern crate text_unit;

use libsyntax2::{
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
                tag
            })
        }
        res
    }

    pub fn syntax_tree(&self) -> String {
        ::libsyntax2::utils::dump_tree(&self.inner.syntax())
    }
}
