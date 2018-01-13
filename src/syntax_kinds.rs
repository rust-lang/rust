// Generated from grammar.ron
use tree::{SyntaxKind, SyntaxInfo};

pub const USE_KW: SyntaxKind = SyntaxKind(0);
pub const FN_KW: SyntaxKind = SyntaxKind(1);
pub const STRUCT_KW: SyntaxKind = SyntaxKind(2);
pub const ENUM_KW: SyntaxKind = SyntaxKind(3);
pub const TRAIT_KW: SyntaxKind = SyntaxKind(4);
pub const IMPL_KW: SyntaxKind = SyntaxKind(5);
pub const TRUE_KW: SyntaxKind = SyntaxKind(6);
pub const FALSE_KW: SyntaxKind = SyntaxKind(7);
pub const AS_KW: SyntaxKind = SyntaxKind(8);
pub const EXTERN_KW: SyntaxKind = SyntaxKind(9);
pub const CRATE_KW: SyntaxKind = SyntaxKind(10);
pub const MOD_KW: SyntaxKind = SyntaxKind(11);
pub const PUB_KW: SyntaxKind = SyntaxKind(12);
pub const SELF_KW: SyntaxKind = SyntaxKind(13);
pub const SUPER_KW: SyntaxKind = SyntaxKind(14);
pub const IN_KW: SyntaxKind = SyntaxKind(15);
pub const WHERE_KW: SyntaxKind = SyntaxKind(16);
pub const ERROR: SyntaxKind = SyntaxKind(17);
pub const IDENT: SyntaxKind = SyntaxKind(18);
pub const UNDERSCORE: SyntaxKind = SyntaxKind(19);
pub const WHITESPACE: SyntaxKind = SyntaxKind(20);
pub const INT_NUMBER: SyntaxKind = SyntaxKind(21);
pub const FLOAT_NUMBER: SyntaxKind = SyntaxKind(22);
pub const SEMI: SyntaxKind = SyntaxKind(23);
pub const COMMA: SyntaxKind = SyntaxKind(24);
pub const DOT: SyntaxKind = SyntaxKind(25);
pub const DOTDOT: SyntaxKind = SyntaxKind(26);
pub const DOTDOTDOT: SyntaxKind = SyntaxKind(27);
pub const DOTDOTEQ: SyntaxKind = SyntaxKind(28);
pub const L_PAREN: SyntaxKind = SyntaxKind(29);
pub const R_PAREN: SyntaxKind = SyntaxKind(30);
pub const L_CURLY: SyntaxKind = SyntaxKind(31);
pub const R_CURLY: SyntaxKind = SyntaxKind(32);
pub const L_BRACK: SyntaxKind = SyntaxKind(33);
pub const R_BRACK: SyntaxKind = SyntaxKind(34);
pub const L_ANGLE: SyntaxKind = SyntaxKind(35);
pub const R_ANGLE: SyntaxKind = SyntaxKind(36);
pub const AT: SyntaxKind = SyntaxKind(37);
pub const POUND: SyntaxKind = SyntaxKind(38);
pub const TILDE: SyntaxKind = SyntaxKind(39);
pub const QUESTION: SyntaxKind = SyntaxKind(40);
pub const COLON: SyntaxKind = SyntaxKind(41);
pub const COLONCOLON: SyntaxKind = SyntaxKind(42);
pub const DOLLAR: SyntaxKind = SyntaxKind(43);
pub const EQ: SyntaxKind = SyntaxKind(44);
pub const EQEQ: SyntaxKind = SyntaxKind(45);
pub const FAT_ARROW: SyntaxKind = SyntaxKind(46);
pub const NEQ: SyntaxKind = SyntaxKind(47);
pub const EXCL: SyntaxKind = SyntaxKind(48);
pub const LIFETIME: SyntaxKind = SyntaxKind(49);
pub const CHAR: SyntaxKind = SyntaxKind(50);
pub const BYTE: SyntaxKind = SyntaxKind(51);
pub const STRING: SyntaxKind = SyntaxKind(52);
pub const RAW_STRING: SyntaxKind = SyntaxKind(53);
pub const BYTE_STRING: SyntaxKind = SyntaxKind(54);
pub const RAW_BYTE_STRING: SyntaxKind = SyntaxKind(55);
pub const PLUS: SyntaxKind = SyntaxKind(56);
pub const MINUS: SyntaxKind = SyntaxKind(57);
pub const STAR: SyntaxKind = SyntaxKind(58);
pub const SLASH: SyntaxKind = SyntaxKind(59);
pub const CARET: SyntaxKind = SyntaxKind(60);
pub const PERCENT: SyntaxKind = SyntaxKind(61);
pub const AMPERSAND: SyntaxKind = SyntaxKind(62);
pub const PIPE: SyntaxKind = SyntaxKind(63);
pub const THIN_ARROW: SyntaxKind = SyntaxKind(64);
pub const COMMENT: SyntaxKind = SyntaxKind(65);
pub const DOC_COMMENT: SyntaxKind = SyntaxKind(66);
pub const SHEBANG: SyntaxKind = SyntaxKind(67);
pub const FILE: SyntaxKind = SyntaxKind(68);
pub const STRUCT_ITEM: SyntaxKind = SyntaxKind(69);
pub const NAMED_FIELD: SyntaxKind = SyntaxKind(70);
pub const POS_FIELD: SyntaxKind = SyntaxKind(71);
pub const FN_ITEM: SyntaxKind = SyntaxKind(72);
pub const EXTERN_CRATE_ITEM: SyntaxKind = SyntaxKind(73);
pub const ATTR: SyntaxKind = SyntaxKind(74);
pub const META_ITEM: SyntaxKind = SyntaxKind(75);
pub const MOD_ITEM: SyntaxKind = SyntaxKind(76);
pub const USE_ITEM: SyntaxKind = SyntaxKind(77);
pub const USE_TREE: SyntaxKind = SyntaxKind(78);
pub const PATH: SyntaxKind = SyntaxKind(79);
pub const PATH_SEGMENT: SyntaxKind = SyntaxKind(80);
pub const LITERAL: SyntaxKind = SyntaxKind(81);
pub const ALIAS: SyntaxKind = SyntaxKind(82);
pub const VISIBILITY: SyntaxKind = SyntaxKind(83);

static INFOS: [SyntaxInfo; 84] = [
    SyntaxInfo { name: "USE_KW" },
    SyntaxInfo { name: "FN_KW" },
    SyntaxInfo { name: "STRUCT_KW" },
    SyntaxInfo { name: "ENUM_KW" },
    SyntaxInfo { name: "TRAIT_KW" },
    SyntaxInfo { name: "IMPL_KW" },
    SyntaxInfo { name: "TRUE_KW" },
    SyntaxInfo { name: "FALSE_KW" },
    SyntaxInfo { name: "AS_KW" },
    SyntaxInfo { name: "EXTERN_KW" },
    SyntaxInfo { name: "CRATE_KW" },
    SyntaxInfo { name: "MOD_KW" },
    SyntaxInfo { name: "PUB_KW" },
    SyntaxInfo { name: "SELF_KW" },
    SyntaxInfo { name: "SUPER_KW" },
    SyntaxInfo { name: "IN_KW" },
    SyntaxInfo { name: "WHERE_KW" },
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
    SyntaxInfo { name: "EXCL" },
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
    SyntaxInfo { name: "STRUCT_ITEM" },
    SyntaxInfo { name: "NAMED_FIELD" },
    SyntaxInfo { name: "POS_FIELD" },
    SyntaxInfo { name: "FN_ITEM" },
    SyntaxInfo { name: "EXTERN_CRATE_ITEM" },
    SyntaxInfo { name: "ATTR" },
    SyntaxInfo { name: "META_ITEM" },
    SyntaxInfo { name: "MOD_ITEM" },
    SyntaxInfo { name: "USE_ITEM" },
    SyntaxInfo { name: "USE_TREE" },
    SyntaxInfo { name: "PATH" },
    SyntaxInfo { name: "PATH_SEGMENT" },
    SyntaxInfo { name: "LITERAL" },
    SyntaxInfo { name: "ALIAS" },
    SyntaxInfo { name: "VISIBILITY" },
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
       "true" => Some(TRUE_KW),
       "false" => Some(FALSE_KW),
       "as" => Some(AS_KW),
       "extern" => Some(EXTERN_KW),
       "crate" => Some(CRATE_KW),
       "mod" => Some(MOD_KW),
       "pub" => Some(PUB_KW),
       "self" => Some(SELF_KW),
       "super" => Some(SUPER_KW),
       "in" => Some(IN_KW),
       "where" => Some(WHERE_KW),
       _ => None,
   }
}
