use tree::{SyntaxKind, SyntaxInfo};

pub const IDENT: SyntaxKind = SyntaxKind(1);
pub const WHITESPACE: SyntaxKind = SyntaxKind(2);


static IDENT_INFO: SyntaxInfo = SyntaxInfo {
    name: "IDENT",
};

pub(crate) fn syntax_info(kind: SyntaxKind) -> &'static SyntaxInfo {
    match kind {
        IDENT => &IDENT_INFO,
        _ => unreachable!(),
    }
}