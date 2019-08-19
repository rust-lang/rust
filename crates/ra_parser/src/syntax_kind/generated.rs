// Generated file, do not edit by hand, see `crate/ra_tools/src/codegen`

#![allow(bad_style, missing_docs, unreachable_pub)]
use super::SyntaxInfo;
#[doc = r" The kind of syntax node, e.g. `IDENT`, `USE_KW`, or `STRUCT_DEF`."]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u16)]
pub enum SyntaxKind {
    #[doc(hidden)]
    TOMBSTONE,
    #[doc(hidden)]
    EOF,
    SEMI,
    COMMA,
    L_PAREN,
    R_PAREN,
    L_CURLY,
    R_CURLY,
    L_BRACK,
    R_BRACK,
    L_ANGLE,
    R_ANGLE,
    AT,
    POUND,
    TILDE,
    QUESTION,
    DOLLAR,
    AMP,
    PIPE,
    PLUS,
    STAR,
    SLASH,
    CARET,
    PERCENT,
    UNDERSCORE,
    DOT,
    DOTDOT,
    DOTDOTDOT,
    DOTDOTEQ,
    COLON,
    COLONCOLON,
    EQ,
    EQEQ,
    FAT_ARROW,
    EXCL,
    NEQ,
    MINUS,
    THIN_ARROW,
    LTEQ,
    GTEQ,
    PLUSEQ,
    MINUSEQ,
    PIPEEQ,
    AMPEQ,
    CARETEQ,
    SLASHEQ,
    STAREQ,
    PERCENTEQ,
    AMPAMP,
    PIPEPIPE,
    SHL,
    SHR,
    SHLEQ,
    SHREQ,
    ASYNC_KW,
    USE_KW,
    FN_KW,
    STRUCT_KW,
    ENUM_KW,
    TRAIT_KW,
    IMPL_KW,
    DYN_KW,
    TRUE_KW,
    FALSE_KW,
    AS_KW,
    EXTERN_KW,
    CRATE_KW,
    MOD_KW,
    PUB_KW,
    SELF_KW,
    SUPER_KW,
    IN_KW,
    WHERE_KW,
    FOR_KW,
    LOOP_KW,
    WHILE_KW,
    CONTINUE_KW,
    BREAK_KW,
    IF_KW,
    ELSE_KW,
    MATCH_KW,
    CONST_KW,
    STATIC_KW,
    MUT_KW,
    UNSAFE_KW,
    TYPE_KW,
    REF_KW,
    LET_KW,
    MOVE_KW,
    RETURN_KW,
    TRY_KW,
    BOX_KW,
    AWAIT_KW,
    AUTO_KW,
    DEFAULT_KW,
    EXISTENTIAL_KW,
    UNION_KW,
    INT_NUMBER,
    FLOAT_NUMBER,
    CHAR,
    BYTE,
    STRING,
    RAW_STRING,
    BYTE_STRING,
    RAW_BYTE_STRING,
    ERROR,
    IDENT,
    WHITESPACE,
    LIFETIME,
    COMMENT,
    SHEBANG,
    L_DOLLAR,
    R_DOLLAR,
    SOURCE_FILE,
    STRUCT_DEF,
    ENUM_DEF,
    FN_DEF,
    RET_TYPE,
    EXTERN_CRATE_ITEM,
    MODULE,
    USE_ITEM,
    STATIC_DEF,
    CONST_DEF,
    TRAIT_DEF,
    IMPL_BLOCK,
    TYPE_ALIAS_DEF,
    MACRO_CALL,
    TOKEN_TREE,
    PAREN_TYPE,
    TUPLE_TYPE,
    NEVER_TYPE,
    PATH_TYPE,
    POINTER_TYPE,
    ARRAY_TYPE,
    SLICE_TYPE,
    REFERENCE_TYPE,
    PLACEHOLDER_TYPE,
    FN_POINTER_TYPE,
    FOR_TYPE,
    IMPL_TRAIT_TYPE,
    DYN_TRAIT_TYPE,
    REF_PAT,
    BIND_PAT,
    PLACEHOLDER_PAT,
    PATH_PAT,
    STRUCT_PAT,
    FIELD_PAT_LIST,
    FIELD_PAT,
    TUPLE_STRUCT_PAT,
    TUPLE_PAT,
    SLICE_PAT,
    RANGE_PAT,
    LITERAL_PAT,
    TUPLE_EXPR,
    ARRAY_EXPR,
    PAREN_EXPR,
    PATH_EXPR,
    LAMBDA_EXPR,
    IF_EXPR,
    WHILE_EXPR,
    CONDITION,
    LOOP_EXPR,
    FOR_EXPR,
    CONTINUE_EXPR,
    BREAK_EXPR,
    LABEL,
    BLOCK_EXPR,
    RETURN_EXPR,
    MATCH_EXPR,
    MATCH_ARM_LIST,
    MATCH_ARM,
    MATCH_GUARD,
    STRUCT_LIT,
    NAMED_FIELD_LIST,
    NAMED_FIELD,
    TRY_BLOCK_EXPR,
    BOX_EXPR,
    CALL_EXPR,
    INDEX_EXPR,
    METHOD_CALL_EXPR,
    FIELD_EXPR,
    AWAIT_EXPR,
    TRY_EXPR,
    CAST_EXPR,
    REF_EXPR,
    PREFIX_EXPR,
    RANGE_EXPR,
    BIN_EXPR,
    BLOCK,
    EXTERN_BLOCK,
    EXTERN_ITEM_LIST,
    ENUM_VARIANT,
    NAMED_FIELD_DEF_LIST,
    NAMED_FIELD_DEF,
    POS_FIELD_DEF_LIST,
    POS_FIELD_DEF,
    ENUM_VARIANT_LIST,
    ITEM_LIST,
    ATTR,
    META_ITEM,
    USE_TREE,
    USE_TREE_LIST,
    PATH,
    PATH_SEGMENT,
    LITERAL,
    ALIAS,
    VISIBILITY,
    WHERE_CLAUSE,
    WHERE_PRED,
    ABI,
    NAME,
    NAME_REF,
    LET_STMT,
    EXPR_STMT,
    TYPE_PARAM_LIST,
    LIFETIME_PARAM,
    TYPE_PARAM,
    TYPE_ARG_LIST,
    LIFETIME_ARG,
    TYPE_ARG,
    ASSOC_TYPE_ARG,
    PARAM_LIST,
    PARAM,
    SELF_PARAM,
    ARG_LIST,
    TYPE_BOUND,
    TYPE_BOUND_LIST,
    MACRO_ITEMS,
    MACRO_STMTS,
    #[doc(hidden)]
    __LAST,
}
use self::SyntaxKind::*;
impl From<u16> for SyntaxKind {
    fn from(d: u16) -> SyntaxKind {
        assert!(d <= (__LAST as u16));
        unsafe { std::mem::transmute::<u16, SyntaxKind>(d) }
    }
}
impl From<SyntaxKind> for u16 {
    fn from(k: SyntaxKind) -> u16 {
        k as u16
    }
}
impl SyntaxKind {
    pub fn is_keyword(self) -> bool {
        match self {
            ASYNC_KW | USE_KW | FN_KW | STRUCT_KW | ENUM_KW | TRAIT_KW | IMPL_KW | DYN_KW
            | TRUE_KW | FALSE_KW | AS_KW | EXTERN_KW | CRATE_KW | MOD_KW | PUB_KW | SELF_KW
            | SUPER_KW | IN_KW | WHERE_KW | FOR_KW | LOOP_KW | WHILE_KW | CONTINUE_KW
            | BREAK_KW | IF_KW | ELSE_KW | MATCH_KW | CONST_KW | STATIC_KW | MUT_KW | UNSAFE_KW
            | TYPE_KW | REF_KW | LET_KW | MOVE_KW | RETURN_KW | TRY_KW | BOX_KW | AWAIT_KW
            | AUTO_KW | DEFAULT_KW | EXISTENTIAL_KW | UNION_KW => true,
            _ => false,
        }
    }
    pub fn is_punct(self) -> bool {
        match self {
            SEMI | COMMA | L_PAREN | R_PAREN | L_CURLY | R_CURLY | L_BRACK | R_BRACK | L_ANGLE
            | R_ANGLE | AT | POUND | TILDE | QUESTION | DOLLAR | AMP | PIPE | PLUS | STAR
            | SLASH | CARET | PERCENT | UNDERSCORE | DOT | DOTDOT | DOTDOTDOT | DOTDOTEQ
            | COLON | COLONCOLON | EQ | EQEQ | FAT_ARROW | EXCL | NEQ | MINUS | THIN_ARROW
            | LTEQ | GTEQ | PLUSEQ | MINUSEQ | PIPEEQ | AMPEQ | CARETEQ | SLASHEQ | STAREQ
            | PERCENTEQ | AMPAMP | PIPEPIPE | SHL | SHR | SHLEQ | SHREQ => true,
            _ => false,
        }
    }
    pub fn is_literal(self) -> bool {
        match self {
            INT_NUMBER | FLOAT_NUMBER | CHAR | BYTE | STRING | RAW_STRING | BYTE_STRING
            | RAW_BYTE_STRING => true,
            _ => false,
        }
    }
    pub(crate) fn info(self) -> &'static SyntaxInfo {
        match self {
            SEMI => &SyntaxInfo { name: stringify!(SEMI) },
            COMMA => &SyntaxInfo { name: stringify!(COMMA) },
            L_PAREN => &SyntaxInfo { name: stringify!(L_PAREN) },
            R_PAREN => &SyntaxInfo { name: stringify!(R_PAREN) },
            L_CURLY => &SyntaxInfo { name: stringify!(L_CURLY) },
            R_CURLY => &SyntaxInfo { name: stringify!(R_CURLY) },
            L_BRACK => &SyntaxInfo { name: stringify!(L_BRACK) },
            R_BRACK => &SyntaxInfo { name: stringify!(R_BRACK) },
            L_ANGLE => &SyntaxInfo { name: stringify!(L_ANGLE) },
            R_ANGLE => &SyntaxInfo { name: stringify!(R_ANGLE) },
            AT => &SyntaxInfo { name: stringify!(AT) },
            POUND => &SyntaxInfo { name: stringify!(POUND) },
            TILDE => &SyntaxInfo { name: stringify!(TILDE) },
            QUESTION => &SyntaxInfo { name: stringify!(QUESTION) },
            DOLLAR => &SyntaxInfo { name: stringify!(DOLLAR) },
            AMP => &SyntaxInfo { name: stringify!(AMP) },
            PIPE => &SyntaxInfo { name: stringify!(PIPE) },
            PLUS => &SyntaxInfo { name: stringify!(PLUS) },
            STAR => &SyntaxInfo { name: stringify!(STAR) },
            SLASH => &SyntaxInfo { name: stringify!(SLASH) },
            CARET => &SyntaxInfo { name: stringify!(CARET) },
            PERCENT => &SyntaxInfo { name: stringify!(PERCENT) },
            UNDERSCORE => &SyntaxInfo { name: stringify!(UNDERSCORE) },
            DOT => &SyntaxInfo { name: stringify!(DOT) },
            DOTDOT => &SyntaxInfo { name: stringify!(DOTDOT) },
            DOTDOTDOT => &SyntaxInfo { name: stringify!(DOTDOTDOT) },
            DOTDOTEQ => &SyntaxInfo { name: stringify!(DOTDOTEQ) },
            COLON => &SyntaxInfo { name: stringify!(COLON) },
            COLONCOLON => &SyntaxInfo { name: stringify!(COLONCOLON) },
            EQ => &SyntaxInfo { name: stringify!(EQ) },
            EQEQ => &SyntaxInfo { name: stringify!(EQEQ) },
            FAT_ARROW => &SyntaxInfo { name: stringify!(FAT_ARROW) },
            EXCL => &SyntaxInfo { name: stringify!(EXCL) },
            NEQ => &SyntaxInfo { name: stringify!(NEQ) },
            MINUS => &SyntaxInfo { name: stringify!(MINUS) },
            THIN_ARROW => &SyntaxInfo { name: stringify!(THIN_ARROW) },
            LTEQ => &SyntaxInfo { name: stringify!(LTEQ) },
            GTEQ => &SyntaxInfo { name: stringify!(GTEQ) },
            PLUSEQ => &SyntaxInfo { name: stringify!(PLUSEQ) },
            MINUSEQ => &SyntaxInfo { name: stringify!(MINUSEQ) },
            PIPEEQ => &SyntaxInfo { name: stringify!(PIPEEQ) },
            AMPEQ => &SyntaxInfo { name: stringify!(AMPEQ) },
            CARETEQ => &SyntaxInfo { name: stringify!(CARETEQ) },
            SLASHEQ => &SyntaxInfo { name: stringify!(SLASHEQ) },
            STAREQ => &SyntaxInfo { name: stringify!(STAREQ) },
            PERCENTEQ => &SyntaxInfo { name: stringify!(PERCENTEQ) },
            AMPAMP => &SyntaxInfo { name: stringify!(AMPAMP) },
            PIPEPIPE => &SyntaxInfo { name: stringify!(PIPEPIPE) },
            SHL => &SyntaxInfo { name: stringify!(SHL) },
            SHR => &SyntaxInfo { name: stringify!(SHR) },
            SHLEQ => &SyntaxInfo { name: stringify!(SHLEQ) },
            SHREQ => &SyntaxInfo { name: stringify!(SHREQ) },
            ASYNC_KW => &SyntaxInfo { name: stringify!(ASYNC_KW) },
            USE_KW => &SyntaxInfo { name: stringify!(USE_KW) },
            FN_KW => &SyntaxInfo { name: stringify!(FN_KW) },
            STRUCT_KW => &SyntaxInfo { name: stringify!(STRUCT_KW) },
            ENUM_KW => &SyntaxInfo { name: stringify!(ENUM_KW) },
            TRAIT_KW => &SyntaxInfo { name: stringify!(TRAIT_KW) },
            IMPL_KW => &SyntaxInfo { name: stringify!(IMPL_KW) },
            DYN_KW => &SyntaxInfo { name: stringify!(DYN_KW) },
            TRUE_KW => &SyntaxInfo { name: stringify!(TRUE_KW) },
            FALSE_KW => &SyntaxInfo { name: stringify!(FALSE_KW) },
            AS_KW => &SyntaxInfo { name: stringify!(AS_KW) },
            EXTERN_KW => &SyntaxInfo { name: stringify!(EXTERN_KW) },
            CRATE_KW => &SyntaxInfo { name: stringify!(CRATE_KW) },
            MOD_KW => &SyntaxInfo { name: stringify!(MOD_KW) },
            PUB_KW => &SyntaxInfo { name: stringify!(PUB_KW) },
            SELF_KW => &SyntaxInfo { name: stringify!(SELF_KW) },
            SUPER_KW => &SyntaxInfo { name: stringify!(SUPER_KW) },
            IN_KW => &SyntaxInfo { name: stringify!(IN_KW) },
            WHERE_KW => &SyntaxInfo { name: stringify!(WHERE_KW) },
            FOR_KW => &SyntaxInfo { name: stringify!(FOR_KW) },
            LOOP_KW => &SyntaxInfo { name: stringify!(LOOP_KW) },
            WHILE_KW => &SyntaxInfo { name: stringify!(WHILE_KW) },
            CONTINUE_KW => &SyntaxInfo { name: stringify!(CONTINUE_KW) },
            BREAK_KW => &SyntaxInfo { name: stringify!(BREAK_KW) },
            IF_KW => &SyntaxInfo { name: stringify!(IF_KW) },
            ELSE_KW => &SyntaxInfo { name: stringify!(ELSE_KW) },
            MATCH_KW => &SyntaxInfo { name: stringify!(MATCH_KW) },
            CONST_KW => &SyntaxInfo { name: stringify!(CONST_KW) },
            STATIC_KW => &SyntaxInfo { name: stringify!(STATIC_KW) },
            MUT_KW => &SyntaxInfo { name: stringify!(MUT_KW) },
            UNSAFE_KW => &SyntaxInfo { name: stringify!(UNSAFE_KW) },
            TYPE_KW => &SyntaxInfo { name: stringify!(TYPE_KW) },
            REF_KW => &SyntaxInfo { name: stringify!(REF_KW) },
            LET_KW => &SyntaxInfo { name: stringify!(LET_KW) },
            MOVE_KW => &SyntaxInfo { name: stringify!(MOVE_KW) },
            RETURN_KW => &SyntaxInfo { name: stringify!(RETURN_KW) },
            TRY_KW => &SyntaxInfo { name: stringify!(TRY_KW) },
            BOX_KW => &SyntaxInfo { name: stringify!(BOX_KW) },
            AWAIT_KW => &SyntaxInfo { name: stringify!(AWAIT_KW) },
            AUTO_KW => &SyntaxInfo { name: stringify!(AUTO_KW) },
            DEFAULT_KW => &SyntaxInfo { name: stringify!(DEFAULT_KW) },
            EXISTENTIAL_KW => &SyntaxInfo { name: stringify!(EXISTENTIAL_KW) },
            UNION_KW => &SyntaxInfo { name: stringify!(UNION_KW) },
            INT_NUMBER => &SyntaxInfo { name: stringify!(INT_NUMBER) },
            FLOAT_NUMBER => &SyntaxInfo { name: stringify!(FLOAT_NUMBER) },
            CHAR => &SyntaxInfo { name: stringify!(CHAR) },
            BYTE => &SyntaxInfo { name: stringify!(BYTE) },
            STRING => &SyntaxInfo { name: stringify!(STRING) },
            RAW_STRING => &SyntaxInfo { name: stringify!(RAW_STRING) },
            BYTE_STRING => &SyntaxInfo { name: stringify!(BYTE_STRING) },
            RAW_BYTE_STRING => &SyntaxInfo { name: stringify!(RAW_BYTE_STRING) },
            ERROR => &SyntaxInfo { name: stringify!(ERROR) },
            IDENT => &SyntaxInfo { name: stringify!(IDENT) },
            WHITESPACE => &SyntaxInfo { name: stringify!(WHITESPACE) },
            LIFETIME => &SyntaxInfo { name: stringify!(LIFETIME) },
            COMMENT => &SyntaxInfo { name: stringify!(COMMENT) },
            SHEBANG => &SyntaxInfo { name: stringify!(SHEBANG) },
            L_DOLLAR => &SyntaxInfo { name: stringify!(L_DOLLAR) },
            R_DOLLAR => &SyntaxInfo { name: stringify!(R_DOLLAR) },
            SOURCE_FILE => &SyntaxInfo { name: stringify!(SOURCE_FILE) },
            STRUCT_DEF => &SyntaxInfo { name: stringify!(STRUCT_DEF) },
            ENUM_DEF => &SyntaxInfo { name: stringify!(ENUM_DEF) },
            FN_DEF => &SyntaxInfo { name: stringify!(FN_DEF) },
            RET_TYPE => &SyntaxInfo { name: stringify!(RET_TYPE) },
            EXTERN_CRATE_ITEM => &SyntaxInfo { name: stringify!(EXTERN_CRATE_ITEM) },
            MODULE => &SyntaxInfo { name: stringify!(MODULE) },
            USE_ITEM => &SyntaxInfo { name: stringify!(USE_ITEM) },
            STATIC_DEF => &SyntaxInfo { name: stringify!(STATIC_DEF) },
            CONST_DEF => &SyntaxInfo { name: stringify!(CONST_DEF) },
            TRAIT_DEF => &SyntaxInfo { name: stringify!(TRAIT_DEF) },
            IMPL_BLOCK => &SyntaxInfo { name: stringify!(IMPL_BLOCK) },
            TYPE_ALIAS_DEF => &SyntaxInfo { name: stringify!(TYPE_ALIAS_DEF) },
            MACRO_CALL => &SyntaxInfo { name: stringify!(MACRO_CALL) },
            TOKEN_TREE => &SyntaxInfo { name: stringify!(TOKEN_TREE) },
            PAREN_TYPE => &SyntaxInfo { name: stringify!(PAREN_TYPE) },
            TUPLE_TYPE => &SyntaxInfo { name: stringify!(TUPLE_TYPE) },
            NEVER_TYPE => &SyntaxInfo { name: stringify!(NEVER_TYPE) },
            PATH_TYPE => &SyntaxInfo { name: stringify!(PATH_TYPE) },
            POINTER_TYPE => &SyntaxInfo { name: stringify!(POINTER_TYPE) },
            ARRAY_TYPE => &SyntaxInfo { name: stringify!(ARRAY_TYPE) },
            SLICE_TYPE => &SyntaxInfo { name: stringify!(SLICE_TYPE) },
            REFERENCE_TYPE => &SyntaxInfo { name: stringify!(REFERENCE_TYPE) },
            PLACEHOLDER_TYPE => &SyntaxInfo { name: stringify!(PLACEHOLDER_TYPE) },
            FN_POINTER_TYPE => &SyntaxInfo { name: stringify!(FN_POINTER_TYPE) },
            FOR_TYPE => &SyntaxInfo { name: stringify!(FOR_TYPE) },
            IMPL_TRAIT_TYPE => &SyntaxInfo { name: stringify!(IMPL_TRAIT_TYPE) },
            DYN_TRAIT_TYPE => &SyntaxInfo { name: stringify!(DYN_TRAIT_TYPE) },
            REF_PAT => &SyntaxInfo { name: stringify!(REF_PAT) },
            BIND_PAT => &SyntaxInfo { name: stringify!(BIND_PAT) },
            PLACEHOLDER_PAT => &SyntaxInfo { name: stringify!(PLACEHOLDER_PAT) },
            PATH_PAT => &SyntaxInfo { name: stringify!(PATH_PAT) },
            STRUCT_PAT => &SyntaxInfo { name: stringify!(STRUCT_PAT) },
            FIELD_PAT_LIST => &SyntaxInfo { name: stringify!(FIELD_PAT_LIST) },
            FIELD_PAT => &SyntaxInfo { name: stringify!(FIELD_PAT) },
            TUPLE_STRUCT_PAT => &SyntaxInfo { name: stringify!(TUPLE_STRUCT_PAT) },
            TUPLE_PAT => &SyntaxInfo { name: stringify!(TUPLE_PAT) },
            SLICE_PAT => &SyntaxInfo { name: stringify!(SLICE_PAT) },
            RANGE_PAT => &SyntaxInfo { name: stringify!(RANGE_PAT) },
            LITERAL_PAT => &SyntaxInfo { name: stringify!(LITERAL_PAT) },
            TUPLE_EXPR => &SyntaxInfo { name: stringify!(TUPLE_EXPR) },
            ARRAY_EXPR => &SyntaxInfo { name: stringify!(ARRAY_EXPR) },
            PAREN_EXPR => &SyntaxInfo { name: stringify!(PAREN_EXPR) },
            PATH_EXPR => &SyntaxInfo { name: stringify!(PATH_EXPR) },
            LAMBDA_EXPR => &SyntaxInfo { name: stringify!(LAMBDA_EXPR) },
            IF_EXPR => &SyntaxInfo { name: stringify!(IF_EXPR) },
            WHILE_EXPR => &SyntaxInfo { name: stringify!(WHILE_EXPR) },
            CONDITION => &SyntaxInfo { name: stringify!(CONDITION) },
            LOOP_EXPR => &SyntaxInfo { name: stringify!(LOOP_EXPR) },
            FOR_EXPR => &SyntaxInfo { name: stringify!(FOR_EXPR) },
            CONTINUE_EXPR => &SyntaxInfo { name: stringify!(CONTINUE_EXPR) },
            BREAK_EXPR => &SyntaxInfo { name: stringify!(BREAK_EXPR) },
            LABEL => &SyntaxInfo { name: stringify!(LABEL) },
            BLOCK_EXPR => &SyntaxInfo { name: stringify!(BLOCK_EXPR) },
            RETURN_EXPR => &SyntaxInfo { name: stringify!(RETURN_EXPR) },
            MATCH_EXPR => &SyntaxInfo { name: stringify!(MATCH_EXPR) },
            MATCH_ARM_LIST => &SyntaxInfo { name: stringify!(MATCH_ARM_LIST) },
            MATCH_ARM => &SyntaxInfo { name: stringify!(MATCH_ARM) },
            MATCH_GUARD => &SyntaxInfo { name: stringify!(MATCH_GUARD) },
            STRUCT_LIT => &SyntaxInfo { name: stringify!(STRUCT_LIT) },
            NAMED_FIELD_LIST => &SyntaxInfo { name: stringify!(NAMED_FIELD_LIST) },
            NAMED_FIELD => &SyntaxInfo { name: stringify!(NAMED_FIELD) },
            TRY_BLOCK_EXPR => &SyntaxInfo { name: stringify!(TRY_BLOCK_EXPR) },
            BOX_EXPR => &SyntaxInfo { name: stringify!(BOX_EXPR) },
            CALL_EXPR => &SyntaxInfo { name: stringify!(CALL_EXPR) },
            INDEX_EXPR => &SyntaxInfo { name: stringify!(INDEX_EXPR) },
            METHOD_CALL_EXPR => &SyntaxInfo { name: stringify!(METHOD_CALL_EXPR) },
            FIELD_EXPR => &SyntaxInfo { name: stringify!(FIELD_EXPR) },
            AWAIT_EXPR => &SyntaxInfo { name: stringify!(AWAIT_EXPR) },
            TRY_EXPR => &SyntaxInfo { name: stringify!(TRY_EXPR) },
            CAST_EXPR => &SyntaxInfo { name: stringify!(CAST_EXPR) },
            REF_EXPR => &SyntaxInfo { name: stringify!(REF_EXPR) },
            PREFIX_EXPR => &SyntaxInfo { name: stringify!(PREFIX_EXPR) },
            RANGE_EXPR => &SyntaxInfo { name: stringify!(RANGE_EXPR) },
            BIN_EXPR => &SyntaxInfo { name: stringify!(BIN_EXPR) },
            BLOCK => &SyntaxInfo { name: stringify!(BLOCK) },
            EXTERN_BLOCK => &SyntaxInfo { name: stringify!(EXTERN_BLOCK) },
            EXTERN_ITEM_LIST => &SyntaxInfo { name: stringify!(EXTERN_ITEM_LIST) },
            ENUM_VARIANT => &SyntaxInfo { name: stringify!(ENUM_VARIANT) },
            NAMED_FIELD_DEF_LIST => &SyntaxInfo { name: stringify!(NAMED_FIELD_DEF_LIST) },
            NAMED_FIELD_DEF => &SyntaxInfo { name: stringify!(NAMED_FIELD_DEF) },
            POS_FIELD_DEF_LIST => &SyntaxInfo { name: stringify!(POS_FIELD_DEF_LIST) },
            POS_FIELD_DEF => &SyntaxInfo { name: stringify!(POS_FIELD_DEF) },
            ENUM_VARIANT_LIST => &SyntaxInfo { name: stringify!(ENUM_VARIANT_LIST) },
            ITEM_LIST => &SyntaxInfo { name: stringify!(ITEM_LIST) },
            ATTR => &SyntaxInfo { name: stringify!(ATTR) },
            META_ITEM => &SyntaxInfo { name: stringify!(META_ITEM) },
            USE_TREE => &SyntaxInfo { name: stringify!(USE_TREE) },
            USE_TREE_LIST => &SyntaxInfo { name: stringify!(USE_TREE_LIST) },
            PATH => &SyntaxInfo { name: stringify!(PATH) },
            PATH_SEGMENT => &SyntaxInfo { name: stringify!(PATH_SEGMENT) },
            LITERAL => &SyntaxInfo { name: stringify!(LITERAL) },
            ALIAS => &SyntaxInfo { name: stringify!(ALIAS) },
            VISIBILITY => &SyntaxInfo { name: stringify!(VISIBILITY) },
            WHERE_CLAUSE => &SyntaxInfo { name: stringify!(WHERE_CLAUSE) },
            WHERE_PRED => &SyntaxInfo { name: stringify!(WHERE_PRED) },
            ABI => &SyntaxInfo { name: stringify!(ABI) },
            NAME => &SyntaxInfo { name: stringify!(NAME) },
            NAME_REF => &SyntaxInfo { name: stringify!(NAME_REF) },
            LET_STMT => &SyntaxInfo { name: stringify!(LET_STMT) },
            EXPR_STMT => &SyntaxInfo { name: stringify!(EXPR_STMT) },
            TYPE_PARAM_LIST => &SyntaxInfo { name: stringify!(TYPE_PARAM_LIST) },
            LIFETIME_PARAM => &SyntaxInfo { name: stringify!(LIFETIME_PARAM) },
            TYPE_PARAM => &SyntaxInfo { name: stringify!(TYPE_PARAM) },
            TYPE_ARG_LIST => &SyntaxInfo { name: stringify!(TYPE_ARG_LIST) },
            LIFETIME_ARG => &SyntaxInfo { name: stringify!(LIFETIME_ARG) },
            TYPE_ARG => &SyntaxInfo { name: stringify!(TYPE_ARG) },
            ASSOC_TYPE_ARG => &SyntaxInfo { name: stringify!(ASSOC_TYPE_ARG) },
            PARAM_LIST => &SyntaxInfo { name: stringify!(PARAM_LIST) },
            PARAM => &SyntaxInfo { name: stringify!(PARAM) },
            SELF_PARAM => &SyntaxInfo { name: stringify!(SELF_PARAM) },
            ARG_LIST => &SyntaxInfo { name: stringify!(ARG_LIST) },
            TYPE_BOUND => &SyntaxInfo { name: stringify!(TYPE_BOUND) },
            TYPE_BOUND_LIST => &SyntaxInfo { name: stringify!(TYPE_BOUND_LIST) },
            MACRO_ITEMS => &SyntaxInfo { name: stringify!(MACRO_ITEMS) },
            MACRO_STMTS => &SyntaxInfo { name: stringify!(MACRO_STMTS) },
            TOMBSTONE => &SyntaxInfo { name: "TOMBSTONE" },
            EOF => &SyntaxInfo { name: "EOF" },
            __LAST => &SyntaxInfo { name: "__LAST" },
        }
    }
    pub fn from_keyword(ident: &str) -> Option<SyntaxKind> {
        let kw = match ident {
            "async" => ASYNC_KW,
            "use" => USE_KW,
            "fn" => FN_KW,
            "struct" => STRUCT_KW,
            "enum" => ENUM_KW,
            "trait" => TRAIT_KW,
            "impl" => IMPL_KW,
            "dyn" => DYN_KW,
            "true" => TRUE_KW,
            "false" => FALSE_KW,
            "as" => AS_KW,
            "extern" => EXTERN_KW,
            "crate" => CRATE_KW,
            "mod" => MOD_KW,
            "pub" => PUB_KW,
            "self" => SELF_KW,
            "super" => SUPER_KW,
            "in" => IN_KW,
            "where" => WHERE_KW,
            "for" => FOR_KW,
            "loop" => LOOP_KW,
            "while" => WHILE_KW,
            "continue" => CONTINUE_KW,
            "break" => BREAK_KW,
            "if" => IF_KW,
            "else" => ELSE_KW,
            "match" => MATCH_KW,
            "const" => CONST_KW,
            "static" => STATIC_KW,
            "mut" => MUT_KW,
            "unsafe" => UNSAFE_KW,
            "type" => TYPE_KW,
            "ref" => REF_KW,
            "let" => LET_KW,
            "move" => MOVE_KW,
            "return" => RETURN_KW,
            "try" => TRY_KW,
            "box" => BOX_KW,
            "await" => AWAIT_KW,
            _ => return None,
        };
        Some(kw)
    }
    pub fn from_char(c: char) -> Option<SyntaxKind> {
        let tok = match c {
            ';' => SEMI,
            ',' => COMMA,
            '(' => L_PAREN,
            ')' => R_PAREN,
            '{' => L_CURLY,
            '}' => R_CURLY,
            '[' => L_BRACK,
            ']' => R_BRACK,
            '<' => L_ANGLE,
            '>' => R_ANGLE,
            '@' => AT,
            '#' => POUND,
            '~' => TILDE,
            '?' => QUESTION,
            '$' => DOLLAR,
            '&' => AMP,
            '|' => PIPE,
            '+' => PLUS,
            '*' => STAR,
            '/' => SLASH,
            '^' => CARET,
            '%' => PERCENT,
            '_' => UNDERSCORE,
            _ => return None,
        };
        Some(tok)
    }
}
#[macro_export]
macro_rules! T {
    ( ; ) => {
        $crate::SyntaxKind::SEMI
    };
    ( , ) => {
        $crate::SyntaxKind::COMMA
    };
    ( '(' ) => {
        $crate::SyntaxKind::L_PAREN
    };
    ( ')' ) => {
        $crate::SyntaxKind::R_PAREN
    };
    ( '{' ) => {
        $crate::SyntaxKind::L_CURLY
    };
    ( '}' ) => {
        $crate::SyntaxKind::R_CURLY
    };
    ( '[' ) => {
        $crate::SyntaxKind::L_BRACK
    };
    ( ']' ) => {
        $crate::SyntaxKind::R_BRACK
    };
    ( < ) => {
        $crate::SyntaxKind::L_ANGLE
    };
    ( > ) => {
        $crate::SyntaxKind::R_ANGLE
    };
    ( @ ) => {
        $crate::SyntaxKind::AT
    };
    ( # ) => {
        $crate::SyntaxKind::POUND
    };
    ( ~ ) => {
        $crate::SyntaxKind::TILDE
    };
    ( ? ) => {
        $crate::SyntaxKind::QUESTION
    };
    ( $ ) => {
        $crate::SyntaxKind::DOLLAR
    };
    ( & ) => {
        $crate::SyntaxKind::AMP
    };
    ( | ) => {
        $crate::SyntaxKind::PIPE
    };
    ( + ) => {
        $crate::SyntaxKind::PLUS
    };
    ( * ) => {
        $crate::SyntaxKind::STAR
    };
    ( / ) => {
        $crate::SyntaxKind::SLASH
    };
    ( ^ ) => {
        $crate::SyntaxKind::CARET
    };
    ( % ) => {
        $crate::SyntaxKind::PERCENT
    };
    ( _ ) => {
        $crate::SyntaxKind::UNDERSCORE
    };
    ( . ) => {
        $crate::SyntaxKind::DOT
    };
    ( .. ) => {
        $crate::SyntaxKind::DOTDOT
    };
    ( ... ) => {
        $crate::SyntaxKind::DOTDOTDOT
    };
    ( ..= ) => {
        $crate::SyntaxKind::DOTDOTEQ
    };
    ( : ) => {
        $crate::SyntaxKind::COLON
    };
    ( :: ) => {
        $crate::SyntaxKind::COLONCOLON
    };
    ( = ) => {
        $crate::SyntaxKind::EQ
    };
    ( == ) => {
        $crate::SyntaxKind::EQEQ
    };
    ( => ) => {
        $crate::SyntaxKind::FAT_ARROW
    };
    ( ! ) => {
        $crate::SyntaxKind::EXCL
    };
    ( != ) => {
        $crate::SyntaxKind::NEQ
    };
    ( - ) => {
        $crate::SyntaxKind::MINUS
    };
    ( -> ) => {
        $crate::SyntaxKind::THIN_ARROW
    };
    ( <= ) => {
        $crate::SyntaxKind::LTEQ
    };
    ( >= ) => {
        $crate::SyntaxKind::GTEQ
    };
    ( += ) => {
        $crate::SyntaxKind::PLUSEQ
    };
    ( -= ) => {
        $crate::SyntaxKind::MINUSEQ
    };
    ( |= ) => {
        $crate::SyntaxKind::PIPEEQ
    };
    ( &= ) => {
        $crate::SyntaxKind::AMPEQ
    };
    ( ^= ) => {
        $crate::SyntaxKind::CARETEQ
    };
    ( /= ) => {
        $crate::SyntaxKind::SLASHEQ
    };
    ( *= ) => {
        $crate::SyntaxKind::STAREQ
    };
    ( %= ) => {
        $crate::SyntaxKind::PERCENTEQ
    };
    ( && ) => {
        $crate::SyntaxKind::AMPAMP
    };
    ( || ) => {
        $crate::SyntaxKind::PIPEPIPE
    };
    ( << ) => {
        $crate::SyntaxKind::SHL
    };
    ( >> ) => {
        $crate::SyntaxKind::SHR
    };
    ( <<= ) => {
        $crate::SyntaxKind::SHLEQ
    };
    ( >>= ) => {
        $crate::SyntaxKind::SHREQ
    };
    ( async ) => {
        $crate::SyntaxKind::ASYNC_KW
    };
    ( use ) => {
        $crate::SyntaxKind::USE_KW
    };
    ( fn ) => {
        $crate::SyntaxKind::FN_KW
    };
    ( struct ) => {
        $crate::SyntaxKind::STRUCT_KW
    };
    ( enum ) => {
        $crate::SyntaxKind::ENUM_KW
    };
    ( trait ) => {
        $crate::SyntaxKind::TRAIT_KW
    };
    ( impl ) => {
        $crate::SyntaxKind::IMPL_KW
    };
    ( dyn ) => {
        $crate::SyntaxKind::DYN_KW
    };
    ( true ) => {
        $crate::SyntaxKind::TRUE_KW
    };
    ( false ) => {
        $crate::SyntaxKind::FALSE_KW
    };
    ( as ) => {
        $crate::SyntaxKind::AS_KW
    };
    ( extern ) => {
        $crate::SyntaxKind::EXTERN_KW
    };
    ( crate ) => {
        $crate::SyntaxKind::CRATE_KW
    };
    ( mod ) => {
        $crate::SyntaxKind::MOD_KW
    };
    ( pub ) => {
        $crate::SyntaxKind::PUB_KW
    };
    ( self ) => {
        $crate::SyntaxKind::SELF_KW
    };
    ( super ) => {
        $crate::SyntaxKind::SUPER_KW
    };
    ( in ) => {
        $crate::SyntaxKind::IN_KW
    };
    ( where ) => {
        $crate::SyntaxKind::WHERE_KW
    };
    ( for ) => {
        $crate::SyntaxKind::FOR_KW
    };
    ( loop ) => {
        $crate::SyntaxKind::LOOP_KW
    };
    ( while ) => {
        $crate::SyntaxKind::WHILE_KW
    };
    ( continue ) => {
        $crate::SyntaxKind::CONTINUE_KW
    };
    ( break ) => {
        $crate::SyntaxKind::BREAK_KW
    };
    ( if ) => {
        $crate::SyntaxKind::IF_KW
    };
    ( else ) => {
        $crate::SyntaxKind::ELSE_KW
    };
    ( match ) => {
        $crate::SyntaxKind::MATCH_KW
    };
    ( const ) => {
        $crate::SyntaxKind::CONST_KW
    };
    ( static ) => {
        $crate::SyntaxKind::STATIC_KW
    };
    ( mut ) => {
        $crate::SyntaxKind::MUT_KW
    };
    ( unsafe ) => {
        $crate::SyntaxKind::UNSAFE_KW
    };
    ( type ) => {
        $crate::SyntaxKind::TYPE_KW
    };
    ( ref ) => {
        $crate::SyntaxKind::REF_KW
    };
    ( let ) => {
        $crate::SyntaxKind::LET_KW
    };
    ( move ) => {
        $crate::SyntaxKind::MOVE_KW
    };
    ( return ) => {
        $crate::SyntaxKind::RETURN_KW
    };
    ( try ) => {
        $crate::SyntaxKind::TRY_KW
    };
    ( box ) => {
        $crate::SyntaxKind::BOX_KW
    };
    ( await ) => {
        $crate::SyntaxKind::AWAIT_KW
    };
    ( auto ) => {
        $crate::SyntaxKind::AUTO_KW
    };
    ( default ) => {
        $crate::SyntaxKind::DEFAULT_KW
    };
    ( existential ) => {
        $crate::SyntaxKind::EXISTENTIAL_KW
    };
    ( union ) => {
        $crate::SyntaxKind::UNION_KW
    };
}
