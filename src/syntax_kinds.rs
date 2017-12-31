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
pub const L_ANGLE: SyntaxKind = SyntaxKind(18);
pub const R_ANGLE: SyntaxKind = SyntaxKind(19);
pub const AT: SyntaxKind = SyntaxKind(20);
pub const POUND: SyntaxKind = SyntaxKind(21);
pub const TILDE: SyntaxKind = SyntaxKind(22);
pub const QUESTION: SyntaxKind = SyntaxKind(23);
pub const COLON: SyntaxKind = SyntaxKind(24);
pub const COLONCOLON: SyntaxKind = SyntaxKind(25);
pub const DOLLAR: SyntaxKind = SyntaxKind(26);
pub const EQ: SyntaxKind = SyntaxKind(27);
pub const EQEQ: SyntaxKind = SyntaxKind(28);
pub const FAT_ARROW: SyntaxKind = SyntaxKind(29);
pub const NEQ: SyntaxKind = SyntaxKind(30);
pub const NOT: SyntaxKind = SyntaxKind(31);
pub const LIFETIME: SyntaxKind = SyntaxKind(32);
pub const CHAR: SyntaxKind = SyntaxKind(33);
pub const BYTE: SyntaxKind = SyntaxKind(34);
pub const STRING: SyntaxKind = SyntaxKind(35);
pub const RAW_STRING: SyntaxKind = SyntaxKind(36);
pub const BYTE_STRING: SyntaxKind = SyntaxKind(37);
pub const RAW_BYTE_STRING: SyntaxKind = SyntaxKind(38);
pub const PLUS: SyntaxKind = SyntaxKind(39);
pub const MINUS: SyntaxKind = SyntaxKind(40);
pub const STAR: SyntaxKind = SyntaxKind(41);
pub const SLASH: SyntaxKind = SyntaxKind(42);
pub const CARET: SyntaxKind = SyntaxKind(43);
pub const PERCENT: SyntaxKind = SyntaxKind(44);
pub const AMPERSAND: SyntaxKind = SyntaxKind(45);
pub const PIPE: SyntaxKind = SyntaxKind(46);
pub const THIN_ARROW: SyntaxKind = SyntaxKind(47);

static INFOS: [SyntaxInfo; 48] = [
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
    SyntaxInfo { name: "L_ANGLE" },
    SyntaxInfo { name: "R_ANGLE" },
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
    SyntaxInfo { name: "LIFETIME" },
    SyntaxInfo { name: "CHAR" },
    SyntaxInfo { name: "BYTE" },
    SyntaxInfo { name: "STRING" },
    SyntaxInfo { name: "RAW_STRING" },
    SyntaxInfo { name: "BYTE_STRING" },
    SyntaxInfo { name: "RAW_BYTE_STRING" },
    SyntaxInfo { name: "PLUS" },
    SyntaxInfo { name: "MINUS" },
    SyntaxInfo { name: "STAR" },
    SyntaxInfo { name: "SLASH" },
    SyntaxInfo { name: "CARET" },
    SyntaxInfo { name: "PERCENT" },
    SyntaxInfo { name: "AMPERSAND" },
    SyntaxInfo { name: "PIPE" },
    SyntaxInfo { name: "THIN_ARROW" },
];

pub(crate) fn syntax_info(kind: SyntaxKind) -> &'static SyntaxInfo {
    &INFOS[kind.0 as usize]
}
