// Generated from grammar.ron
use tree::{SyntaxKind, SyntaxInfo};

pub const ERROR: SyntaxKind = SyntaxKind(0);
pub const IDENT: SyntaxKind = SyntaxKind(1);
pub const UNDERSCORE: SyntaxKind = SyntaxKind(2);
pub const WHITESPACE: SyntaxKind = SyntaxKind(3);
pub const INT_NUMBER: SyntaxKind = SyntaxKind(4);
pub const FLOAT_NUMBER: SyntaxKind = SyntaxKind(5);

static INFOS: [SyntaxInfo; 6] = [
    SyntaxInfo { name: "ERROR" },
    SyntaxInfo { name: "IDENT" },
    SyntaxInfo { name: "UNDERSCORE" },
    SyntaxInfo { name: "WHITESPACE" },
    SyntaxInfo { name: "INT_NUMBER" },
    SyntaxInfo { name: "FLOAT_NUMBER" },
];

pub(crate) fn syntax_info(kind: SyntaxKind) -> &'static SyntaxInfo {
    &INFOS[kind.0 as usize]
}
