// Generated from grammar.ron
use tree::{SyntaxKind, SyntaxInfo};

pub const IDENT: SyntaxKind = SyntaxKind(0);
pub const WHITESPACE: SyntaxKind = SyntaxKind(1);

static IDENT_INFO: SyntaxInfo = SyntaxInfo {
   name: "IDENT",
};
static WHITESPACE_INFO: SyntaxInfo = SyntaxInfo {
   name: "WHITESPACE",
};

pub(crate) fn syntax_info(kind: SyntaxKind) -> &'static SyntaxInfo {
    match kind {
        IDENT => &IDENT_INFO,
        WHITESPACE => &WHITESPACE_INFO,
        _ => unreachable!()
    }
}
