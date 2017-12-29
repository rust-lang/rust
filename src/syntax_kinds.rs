// Generated from grammar.ron
use tree::{SyntaxKind, SyntaxInfo};

pub const ERROR: SyntaxKind = SyntaxKind(0);
pub const IDENT: SyntaxKind = SyntaxKind(1);
pub const UNDERSCORE: SyntaxKind = SyntaxKind(2);
pub const WHITESPACE: SyntaxKind = SyntaxKind(3);

static INFOS: [SyntaxInfo; 4] = [
    SyntaxInfo { name: "ERROR" },
    SyntaxInfo { name: "IDENT" },
    SyntaxInfo { name: "UNDERSCORE" },
    SyntaxInfo { name: "WHITESPACE" },
];

pub(crate) fn syntax_info(kind: SyntaxKind) -> &'static SyntaxInfo {
    &INFOS[kind.0 as usize]
}
