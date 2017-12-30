// Generated from grammar.ron
use tree::{SyntaxKind, SyntaxInfo};

pub const ERROR: SyntaxKind = SyntaxKind(0);
pub const IDENT: SyntaxKind = SyntaxKind(1);
pub const UNDERSCORE: SyntaxKind = SyntaxKind(2);
pub const WHITESPACE: SyntaxKind = SyntaxKind(3);
pub const INT_NUMBER: SyntaxKind = SyntaxKind(4);
pub const FLOAT_NUMBER: SyntaxKind = SyntaxKind(5);
pub const SEMI: SyntaxKind = SyntaxKind(6);
pub const COMMA: SyntaxKind = SyntaxKind(7);
pub const DOT: SyntaxKind = SyntaxKind(8);
pub const DOTDOT: SyntaxKind = SyntaxKind(9);
pub const DOTDOTDOT: SyntaxKind = SyntaxKind(10);
pub const DOTDOTEQ: SyntaxKind = SyntaxKind(11);
pub const L_PAREN: SyntaxKind = SyntaxKind(12);
pub const R_PAREN: SyntaxKind = SyntaxKind(13);
pub const L_CURLY: SyntaxKind = SyntaxKind(14);
pub const R_CURLY: SyntaxKind = SyntaxKind(15);
pub const L_BRACK: SyntaxKind = SyntaxKind(16);
pub const R_BRACK: SyntaxKind = SyntaxKind(17);
pub const AT: SyntaxKind = SyntaxKind(18);
pub const POUND: SyntaxKind = SyntaxKind(19);
pub const TILDE: SyntaxKind = SyntaxKind(20);
pub const QUESTION: SyntaxKind = SyntaxKind(21);
pub const COLON: SyntaxKind = SyntaxKind(22);
pub const COLONCOLON: SyntaxKind = SyntaxKind(23);
pub const DOLLAR: SyntaxKind = SyntaxKind(24);
pub const EQ: SyntaxKind = SyntaxKind(25);
pub const EQEQ: SyntaxKind = SyntaxKind(26);
pub const FAT_ARROW: SyntaxKind = SyntaxKind(27);
pub const NEQ: SyntaxKind = SyntaxKind(28);
pub const NOT: SyntaxKind = SyntaxKind(29);

static INFOS: [SyntaxInfo; 30] = [
    SyntaxInfo { name: "ERROR" },
    SyntaxInfo { name: "IDENT" },
    SyntaxInfo { name: "UNDERSCORE" },
    SyntaxInfo { name: "WHITESPACE" },
    SyntaxInfo { name: "INT_NUMBER" },
    SyntaxInfo { name: "FLOAT_NUMBER" },
    SyntaxInfo { name: "SEMI" },
    SyntaxInfo { name: "COMMA" },
    SyntaxInfo { name: "DOT" },
    SyntaxInfo { name: "DOTDOT" },
    SyntaxInfo { name: "DOTDOTDOT" },
    SyntaxInfo { name: "DOTDOTEQ" },
    SyntaxInfo { name: "L_PAREN" },
    SyntaxInfo { name: "R_PAREN" },
    SyntaxInfo { name: "L_CURLY" },
    SyntaxInfo { name: "R_CURLY" },
    SyntaxInfo { name: "L_BRACK" },
    SyntaxInfo { name: "R_BRACK" },
    SyntaxInfo { name: "AT" },
    SyntaxInfo { name: "POUND" },
    SyntaxInfo { name: "TILDE" },
    SyntaxInfo { name: "QUESTION" },
    SyntaxInfo { name: "COLON" },
    SyntaxInfo { name: "COLONCOLON" },
    SyntaxInfo { name: "DOLLAR" },
    SyntaxInfo { name: "EQ" },
    SyntaxInfo { name: "EQEQ" },
    SyntaxInfo { name: "FAT_ARROW" },
    SyntaxInfo { name: "NEQ" },
    SyntaxInfo { name: "NOT" },
];

pub(crate) fn syntax_info(kind: SyntaxKind) -> &'static SyntaxInfo {
    &INFOS[kind.0 as usize]
}
