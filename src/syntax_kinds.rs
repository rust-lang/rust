// Generated from grammar.ron
use tree::{SyntaxKind, SyntaxInfo};

pub const USE_KW: SyntaxKind = SyntaxKind(0);
pub const FN_KW: SyntaxKind = SyntaxKind(1);
pub const STRUCT_KW: SyntaxKind = SyntaxKind(2);
pub const ENUM_KW: SyntaxKind = SyntaxKind(3);
pub const TRAIT_KW: SyntaxKind = SyntaxKind(4);
pub const IMPL_KW: SyntaxKind = SyntaxKind(5);
pub const ERROR: SyntaxKind = SyntaxKind(6);
pub const IDENT: SyntaxKind = SyntaxKind(7);
pub const UNDERSCORE: SyntaxKind = SyntaxKind(8);
pub const WHITESPACE: SyntaxKind = SyntaxKind(9);
pub const INT_NUMBER: SyntaxKind = SyntaxKind(10);
pub const FLOAT_NUMBER: SyntaxKind = SyntaxKind(11);
pub const SEMI: SyntaxKind = SyntaxKind(12);
pub const COMMA: SyntaxKind = SyntaxKind(13);
pub const DOT: SyntaxKind = SyntaxKind(14);
pub const DOTDOT: SyntaxKind = SyntaxKind(15);
pub const DOTDOTDOT: SyntaxKind = SyntaxKind(16);
pub const DOTDOTEQ: SyntaxKind = SyntaxKind(17);
pub const L_PAREN: SyntaxKind = SyntaxKind(18);
pub const R_PAREN: SyntaxKind = SyntaxKind(19);
pub const L_CURLY: SyntaxKind = SyntaxKind(20);
pub const R_CURLY: SyntaxKind = SyntaxKind(21);
pub const L_BRACK: SyntaxKind = SyntaxKind(22);
pub const R_BRACK: SyntaxKind = SyntaxKind(23);
pub const L_ANGLE: SyntaxKind = SyntaxKind(24);
pub const R_ANGLE: SyntaxKind = SyntaxKind(25);
pub const AT: SyntaxKind = SyntaxKind(26);
pub const POUND: SyntaxKind = SyntaxKind(27);
pub const TILDE: SyntaxKind = SyntaxKind(28);
pub const QUESTION: SyntaxKind = SyntaxKind(29);
pub const COLON: SyntaxKind = SyntaxKind(30);
pub const COLONCOLON: SyntaxKind = SyntaxKind(31);
pub const DOLLAR: SyntaxKind = SyntaxKind(32);
pub const EQ: SyntaxKind = SyntaxKind(33);
pub const EQEQ: SyntaxKind = SyntaxKind(34);
pub const FAT_ARROW: SyntaxKind = SyntaxKind(35);
pub const NEQ: SyntaxKind = SyntaxKind(36);
pub const NOT: SyntaxKind = SyntaxKind(37);
pub const LIFETIME: SyntaxKind = SyntaxKind(38);
pub const CHAR: SyntaxKind = SyntaxKind(39);
pub const BYTE: SyntaxKind = SyntaxKind(40);
pub const STRING: SyntaxKind = SyntaxKind(41);
pub const RAW_STRING: SyntaxKind = SyntaxKind(42);
pub const BYTE_STRING: SyntaxKind = SyntaxKind(43);
pub const RAW_BYTE_STRING: SyntaxKind = SyntaxKind(44);
pub const PLUS: SyntaxKind = SyntaxKind(45);
pub const MINUS: SyntaxKind = SyntaxKind(46);
pub const STAR: SyntaxKind = SyntaxKind(47);
pub const SLASH: SyntaxKind = SyntaxKind(48);
pub const CARET: SyntaxKind = SyntaxKind(49);
pub const PERCENT: SyntaxKind = SyntaxKind(50);
pub const AMPERSAND: SyntaxKind = SyntaxKind(51);
pub const PIPE: SyntaxKind = SyntaxKind(52);
pub const THIN_ARROW: SyntaxKind = SyntaxKind(53);
pub const COMMENT: SyntaxKind = SyntaxKind(54);
pub const DOC_COMMENT: SyntaxKind = SyntaxKind(55);
pub const SHEBANG: SyntaxKind = SyntaxKind(56);
pub const FILE: SyntaxKind = SyntaxKind(57);

static INFOS: [SyntaxInfo; 58] = [
    SyntaxInfo { name: "USE_KW" },
    SyntaxInfo { name: "FN_KW" },
    SyntaxInfo { name: "STRUCT_KW" },
    SyntaxInfo { name: "ENUM_KW" },
    SyntaxInfo { name: "TRAIT_KW" },
    SyntaxInfo { name: "IMPL_KW" },
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
    SyntaxInfo { name: "COMMENT" },
    SyntaxInfo { name: "DOC_COMMENT" },
    SyntaxInfo { name: "SHEBANG" },
    SyntaxInfo { name: "FILE" },
];

pub(crate) fn syntax_info(kind: SyntaxKind) -> &'static SyntaxInfo {
    &INFOS[kind.0 as usize]
}

pub(crate) fn ident_to_keyword(ident: &str) -> Option<SyntaxKind> {
   match ident {
       "use" => Some(USE_KW),
       "fn" => Some(FN_KW),
       "struct" => Some(STRUCT_KW),
       "enum" => Some(ENUM_KW),
       "trait" => Some(TRAIT_KW),
       "impl" => Some(IMPL_KW),
       _ => None,
   }
}
