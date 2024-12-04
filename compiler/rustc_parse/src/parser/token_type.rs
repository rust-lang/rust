use rustc_ast::token::TokenKind;
use rustc_span::symbol::{Symbol, kw, sym};

/// Used in "expected"/"expected one of" error messages. Tokens are added here
/// as necessary. Tokens with values (e.g. literals, identifiers) are
/// represented by a single variant (e.g. `Literal`, `Ident`).
///
/// It's an awkward representation, but it's important for performance. It's a
/// C-style parameterless enum so that `TokenTypeSet` can be a bitset. This is
/// important because `Parser::expected_token_types` is very hot. `TokenType`
/// used to have variants with parameters (e.g. all the keywords were in a
/// single `Keyword` variant with a `Symbol` parameter) and
/// `Parser::expected_token_types` was a `Vec<TokenType>` which was much slower
/// to manipulate.
///
/// We really want to keep the number of variants to 128 or fewer, so that
/// `TokenTypeSet` can be implemented with a `u128`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenType {
    // Expression-operator symbols
    Eq,
    Lt,
    Le,
    EqEq,
    Gt,
    AndAnd,
    OrOr,
    Not,
    Tilde,

    // BinOps
    Plus,
    Minus,
    Star,
    And,
    Or,

    // Structural symbols
    At,
    Dot,
    DotDot,
    DotDotDot,
    DotDotEq,
    Comma,
    Semi,
    Colon,
    PathSep,
    RArrow,
    FatArrow,
    Pound,
    Question,
    OpenParen,
    CloseParen,
    OpenBrace,
    CloseBrace,
    OpenBracket,
    CloseBracket,
    Eof,

    // Token types with some details elided.
    /// Any operator.
    Operator,
    /// Any identifier token.
    Ident,
    /// Any lifetime token.
    Lifetime,
    /// Any token that can start a path.
    Path,
    /// Any token that can start a type.
    Type,
    /// Any token that can start a const expression.
    Const,

    // Keywords
    // tidy-alphabetical-start
    KwAs,
    KwAsync,
    KwAuto,
    KwAwait,
    KwBecome,
    KwBox,
    KwBreak,
    KwCatch,
    KwConst,
    KwContinue,
    KwCrate,
    KwDefault,
    KwDyn,
    KwElse,
    KwEnum,
    KwExtern,
    KwFn,
    KwFor,
    KwGen,
    KwIf,
    KwImpl,
    KwIn,
    KwLet,
    KwLoop,
    KwMacro,
    KwMacroRules,
    KwMatch,
    KwMod,
    KwMove,
    KwMut,
    KwPub,
    KwRaw,
    KwRef,
    KwReturn,
    KwReuse,
    KwSafe,
    KwSelfUpper,
    KwStatic,
    KwStruct,
    KwTrait,
    KwTry,
    KwType,
    KwUnderscore,
    KwUnsafe,
    KwUse,
    KwWhere,
    KwWhile,
    KwYield,
    // tidy-alphabetical-end

    // Keyword-like symbols.
    // tidy-alphabetical-start
    SymAttSyntax,
    SymClobberAbi,
    SymInlateout,
    SymInout,
    SymIs,
    SymLabel,
    SymLateout,
    SymMayUnwind,
    SymNomem,
    SymNoreturn,
    SymNostack,
    SymOptions,
    SymOut,
    SymPreservesFlags,
    SymPure,
    SymReadonly,
    SymSym,
    // tidy-alphabetical-end
}

impl TokenType {
    fn from_u32(val: u32) -> TokenType {
        let token_type = match val {
            0 => TokenType::Eq,
            1 => TokenType::Lt,
            2 => TokenType::Le,
            3 => TokenType::EqEq,
            4 => TokenType::Gt,
            5 => TokenType::AndAnd,
            6 => TokenType::OrOr,
            7 => TokenType::Not,
            8 => TokenType::Tilde,

            9 => TokenType::Plus,
            10 => TokenType::Minus,
            11 => TokenType::Star,
            12 => TokenType::And,
            13 => TokenType::Or,

            14 => TokenType::At,
            15 => TokenType::Dot,
            16 => TokenType::DotDot,
            17 => TokenType::DotDotDot,
            18 => TokenType::DotDotEq,
            19 => TokenType::Comma,
            20 => TokenType::Semi,
            21 => TokenType::Colon,
            22 => TokenType::PathSep,
            23 => TokenType::RArrow,
            24 => TokenType::FatArrow,
            25 => TokenType::Pound,
            26 => TokenType::Question,
            27 => TokenType::OpenParen,
            28 => TokenType::CloseParen,
            29 => TokenType::OpenBrace,
            30 => TokenType::CloseBrace,
            31 => TokenType::OpenBracket,
            32 => TokenType::CloseBracket,
            33 => TokenType::Eof,

            34 => TokenType::Operator,
            35 => TokenType::Ident,
            36 => TokenType::Lifetime,
            37 => TokenType::Path,
            38 => TokenType::Type,
            39 => TokenType::Const,

            40 => TokenType::KwAs,
            41 => TokenType::KwAsync,
            42 => TokenType::KwAuto,
            43 => TokenType::KwAwait,
            44 => TokenType::KwBecome,
            45 => TokenType::KwBox,
            46 => TokenType::KwBreak,
            47 => TokenType::KwCatch,
            48 => TokenType::KwConst,
            49 => TokenType::KwContinue,
            50 => TokenType::KwCrate,
            51 => TokenType::KwDefault,
            52 => TokenType::KwDyn,
            53 => TokenType::KwElse,
            54 => TokenType::KwEnum,
            55 => TokenType::KwExtern,
            56 => TokenType::KwFn,
            57 => TokenType::KwFor,
            58 => TokenType::KwGen,
            59 => TokenType::KwIf,
            60 => TokenType::KwImpl,
            61 => TokenType::KwIn,
            62 => TokenType::KwLet,
            63 => TokenType::KwLoop,
            64 => TokenType::KwMacro,
            65 => TokenType::KwMacroRules,
            66 => TokenType::KwMatch,
            67 => TokenType::KwMod,
            68 => TokenType::KwMove,
            69 => TokenType::KwMut,
            70 => TokenType::KwPub,
            71 => TokenType::KwRaw,
            72 => TokenType::KwRef,
            73 => TokenType::KwReturn,
            74 => TokenType::KwReuse,
            75 => TokenType::KwSafe,
            76 => TokenType::KwSelfUpper,
            77 => TokenType::KwStatic,
            78 => TokenType::KwStruct,
            79 => TokenType::KwTrait,
            80 => TokenType::KwTry,
            81 => TokenType::KwType,
            82 => TokenType::KwUnderscore,
            83 => TokenType::KwUnsafe,
            84 => TokenType::KwUse,
            85 => TokenType::KwWhere,
            86 => TokenType::KwWhile,
            87 => TokenType::KwYield,

            88 => TokenType::SymAttSyntax,
            89 => TokenType::SymClobberAbi,
            90 => TokenType::SymInlateout,
            91 => TokenType::SymInout,
            92 => TokenType::SymIs,
            93 => TokenType::SymLabel,
            94 => TokenType::SymLateout,
            95 => TokenType::SymMayUnwind,
            96 => TokenType::SymNomem,
            97 => TokenType::SymNoreturn,
            98 => TokenType::SymNostack,
            99 => TokenType::SymOptions,
            100 => TokenType::SymOut,
            101 => TokenType::SymPreservesFlags,
            102 => TokenType::SymPure,
            103 => TokenType::SymReadonly,
            104 => TokenType::SymSym,

            _ => panic!("unhandled value: {val}"),
        };
        // This assertion will detect if this method and the type definition get out of sync.
        assert_eq!(token_type as u32, val);
        token_type
    }

    pub(super) fn is_keyword(&self) -> Option<Symbol> {
        match self {
            TokenType::KwAs => Some(kw::As),
            TokenType::KwAsync => Some(kw::Async),
            TokenType::KwAuto => Some(kw::Auto),
            TokenType::KwAwait => Some(kw::Await),
            TokenType::KwBecome => Some(kw::Become),
            TokenType::KwBox => Some(kw::Box),
            TokenType::KwBreak => Some(kw::Break),
            TokenType::KwCatch => Some(kw::Catch),
            TokenType::KwConst => Some(kw::Const),
            TokenType::KwContinue => Some(kw::Continue),
            TokenType::KwCrate => Some(kw::Crate),
            TokenType::KwDefault => Some(kw::Default),
            TokenType::KwDyn => Some(kw::Dyn),
            TokenType::KwElse => Some(kw::Else),
            TokenType::KwEnum => Some(kw::Enum),
            TokenType::KwExtern => Some(kw::Extern),
            TokenType::KwFn => Some(kw::Fn),
            TokenType::KwFor => Some(kw::For),
            TokenType::KwGen => Some(kw::Gen),
            TokenType::KwIf => Some(kw::If),
            TokenType::KwImpl => Some(kw::Impl),
            TokenType::KwIn => Some(kw::In),
            TokenType::KwLet => Some(kw::Let),
            TokenType::KwLoop => Some(kw::Loop),
            TokenType::KwMacroRules => Some(kw::MacroRules),
            TokenType::KwMacro => Some(kw::Macro),
            TokenType::KwMatch => Some(kw::Match),
            TokenType::KwMod => Some(kw::Mod),
            TokenType::KwMove => Some(kw::Move),
            TokenType::KwMut => Some(kw::Mut),
            TokenType::KwPub => Some(kw::Pub),
            TokenType::KwRaw => Some(kw::Raw),
            TokenType::KwRef => Some(kw::Ref),
            TokenType::KwReturn => Some(kw::Return),
            TokenType::KwReuse => Some(kw::Reuse),
            TokenType::KwSafe => Some(kw::Safe),
            TokenType::KwSelfUpper => Some(kw::SelfUpper),
            TokenType::KwStatic => Some(kw::Static),
            TokenType::KwStruct => Some(kw::Struct),
            TokenType::KwTrait => Some(kw::Trait),
            TokenType::KwTry => Some(kw::Try),
            TokenType::KwType => Some(kw::Type),
            TokenType::KwUnderscore => Some(kw::Underscore),
            TokenType::KwUnsafe => Some(kw::Unsafe),
            TokenType::KwUse => Some(kw::Use),
            TokenType::KwWhere => Some(kw::Where),
            TokenType::KwWhile => Some(kw::While),
            TokenType::KwYield => Some(kw::Yield),

            TokenType::SymAttSyntax => Some(sym::att_syntax),
            TokenType::SymClobberAbi => Some(sym::clobber_abi),
            TokenType::SymInlateout => Some(sym::inlateout),
            TokenType::SymInout => Some(sym::inout),
            TokenType::SymIs => Some(sym::is),
            TokenType::SymLabel => Some(sym::label),
            TokenType::SymLateout => Some(sym::lateout),
            TokenType::SymMayUnwind => Some(sym::may_unwind),
            TokenType::SymNomem => Some(sym::nomem),
            TokenType::SymNoreturn => Some(sym::noreturn),
            TokenType::SymNostack => Some(sym::nostack),
            TokenType::SymOptions => Some(sym::options),
            TokenType::SymOut => Some(sym::out),
            TokenType::SymPreservesFlags => Some(sym::preserves_flags),
            TokenType::SymPure => Some(sym::pure),
            TokenType::SymReadonly => Some(sym::readonly),
            TokenType::SymSym => Some(sym::sym),
            _ => None,
        }
    }

    // The output should be the same as that produced by
    // `rustc_ast_pretty::pprust::token_to_string`.
    pub(super) fn to_string(&self) -> String {
        match self {
            TokenType::Eq => "`=`",
            TokenType::Lt => "`<`",
            TokenType::Le => "`<=`",
            TokenType::EqEq => "`==`",
            TokenType::Gt => "`>`",
            TokenType::AndAnd => "`&&`",
            TokenType::OrOr => "`||`",
            TokenType::Not => "`!`",
            TokenType::Tilde => "`~`",

            TokenType::Plus => "`+`",
            TokenType::Minus => "`-`",
            TokenType::Star => "`*`",
            TokenType::And => "`&`",
            TokenType::Or => "`|`",

            TokenType::At => "`@`",
            TokenType::Dot => "`.`",
            TokenType::DotDot => "`..`",
            TokenType::DotDotDot => "`...`",
            TokenType::DotDotEq => "`..=`",
            TokenType::Comma => "`,`",
            TokenType::Semi => "`;`",
            TokenType::Colon => "`:`",
            TokenType::PathSep => "`::`",
            TokenType::RArrow => "`->`",
            TokenType::FatArrow => "`=>`",
            TokenType::Pound => "`#`",
            TokenType::Question => "`?`",
            TokenType::OpenParen => "`(`",
            TokenType::CloseParen => "`)`",
            TokenType::OpenBrace => "`{`",
            TokenType::CloseBrace => "`}`",
            TokenType::OpenBracket => "`[`",
            TokenType::CloseBracket => "`]`",
            TokenType::Eof => "<eof>",

            TokenType::Operator => "an operator",
            TokenType::Ident => "identifier",
            TokenType::Lifetime => "lifetime",
            TokenType::Path => "path",
            TokenType::Type => "type",
            TokenType::Const => "a const expression",

            _ => return format!("`{}`", self.is_keyword().unwrap()),
        }
        .to_string()
    }
}

/// Used by various `Parser` methods such as `check` and `eat`. The first field
/// is always by used those methods. The second field is only used when the
/// first field doesn't match.
#[derive(Clone, Copy, Debug)]
pub struct ExpTokenPair<'a> {
    pub tok: &'a TokenKind,
    pub token_type: TokenType,
}

/// Used by various `Parser` methods such as `check_keyword` and `eat_keyword`.
/// The first field is always used by those methods. The second field is only
/// used when the first field doesn't match.
#[derive(Clone, Copy)]
pub struct ExpKeywordPair {
    pub kw: Symbol,
    pub token_type: TokenType,
}

// Gets a statically-known `ExpTokenPair` pair (for non-keywords) or
// `ExpKeywordPair` (for keywords), as used with various `check`/`expect`
// methods in `Parser`.
//
// The name is short because it's used a lot.
#[macro_export]
// We don't use the normal `#[rustfmt::skip]` here because that triggers a
// bogus "macro-expanded `macro_export` macros from the current crate cannot be
// referred to by absolute paths" error, ugh. See #52234.
#[cfg_attr(rustfmt, rustfmt::skip)]
macro_rules! exp {
    // `ExpTokenPair` helper rules.
    (@tok, $tok:ident) => {
        $crate::parser::token_type::ExpTokenPair {
            tok: &rustc_ast::token::$tok,
            token_type: $crate::parser::token_type::TokenType::$tok
        }
    };
    (@binop, $op:ident) => {
        $crate::parser::token_type::ExpTokenPair {
            tok: &rustc_ast::token::BinOp(rustc_ast::token::BinOpToken::$op),
            token_type: $crate::parser::token_type::TokenType::$op,
        }
    };
    (@open, $delim:ident, $token_type:ident) => {
        $crate::parser::token_type::ExpTokenPair {
            tok: &rustc_ast::token::OpenDelim(rustc_ast::token::Delimiter::$delim),
            token_type: $crate::parser::token_type::TokenType::$token_type,
        }
    };
    (@close, $delim:ident, $token_type:ident) => {
        $crate::parser::token_type::ExpTokenPair {
            tok: &rustc_ast::token::CloseDelim(rustc_ast::token::Delimiter::$delim),
            token_type: $crate::parser::token_type::TokenType::$token_type,
        }
    };

    // `ExpKeywordPair` helper rules.
    (@kw, $kw:ident, $token_type:ident) => {
        $crate::parser::token_type::ExpKeywordPair {
            kw: rustc_span::symbol::kw::$kw,
            token_type: $crate::parser::token_type::TokenType::$token_type,
        }
    };
    (@sym, $kw:ident, $token_type:ident) => {
        $crate::parser::token_type::ExpKeywordPair {
            kw: rustc_span::symbol::sym::$kw,
            token_type: $crate::parser::token_type::TokenType::$token_type,
        }
    };

    (Eq)             => { exp!(@tok, Eq) };
    (Lt)             => { exp!(@tok, Lt) };
    (Le)             => { exp!(@tok, Le) };
    (EqEq)           => { exp!(@tok, EqEq) };
    (Gt)             => { exp!(@tok, Gt) };
    (AndAnd)         => { exp!(@tok, AndAnd) };
    (OrOr)           => { exp!(@tok, OrOr) };
    (Not)            => { exp!(@tok, Not) };
    (Tilde)          => { exp!(@tok, Tilde) };
    (At)             => { exp!(@tok, At) };
    (Dot)            => { exp!(@tok, Dot) };
    (DotDot)         => { exp!(@tok, DotDot) };
    (DotDotDot)      => { exp!(@tok, DotDotDot) };
    (DotDotEq)       => { exp!(@tok, DotDotEq) };
    (Comma)          => { exp!(@tok, Comma) };
    (Semi)           => { exp!(@tok, Semi) };
    (Colon)          => { exp!(@tok, Colon) };
    (PathSep)        => { exp!(@tok, PathSep) };
    (RArrow)         => { exp!(@tok, RArrow) };
    (FatArrow)       => { exp!(@tok, FatArrow) };
    (Pound)          => { exp!(@tok, Pound) };
    (Question)       => { exp!(@tok, Question) };
    (Eof)            => { exp!(@tok, Eof) };

    (Plus)           => { exp!(@binop, Plus) };
    (Minus)          => { exp!(@binop, Minus) };
    (Star)           => { exp!(@binop, Star) };
    (And)            => { exp!(@binop, And) };
    (Or)             => { exp!(@binop, Or) };

    (OpenParen)      => { exp!(@open,  Parenthesis, OpenParen) };
    (OpenBrace)      => { exp!(@open,  Brace,       OpenBrace) };
    (OpenBracket)    => { exp!(@open,  Bracket,     OpenBracket) };
    (CloseParen)     => { exp!(@close, Parenthesis, CloseParen) };
    (CloseBrace)     => { exp!(@close, Brace,       CloseBrace) };
    (CloseBracket)   => { exp!(@close, Bracket,     CloseBracket) };

    (As)             => { exp!(@kw, As,         KwAs) };
    (Async)          => { exp!(@kw, Async,      KwAsync) };
    (Auto)           => { exp!(@kw, Auto,       KwAuto) };
    (Await)          => { exp!(@kw, Await,      KwAwait) };
    (Become)         => { exp!(@kw, Become,     KwBecome) };
    (Box)            => { exp!(@kw, Box,        KwBox) };
    (Break)          => { exp!(@kw, Break,      KwBreak) };
    (Catch)          => { exp!(@kw, Catch,      KwCatch) };
    (Const)          => { exp!(@kw, Const,      KwConst) };
    (Continue)       => { exp!(@kw, Continue,   KwContinue) };
    (Crate)          => { exp!(@kw, Crate,      KwCrate) };
    (Default)        => { exp!(@kw, Default,    KwDefault) };
    (Dyn)            => { exp!(@kw, Dyn,        KwDyn) };
    (Else)           => { exp!(@kw, Else,       KwElse) };
    (Enum)           => { exp!(@kw, Enum,       KwEnum) };
    (Extern)         => { exp!(@kw, Extern,     KwExtern) };
    (Fn)             => { exp!(@kw, Fn,         KwFn) };
    (For)            => { exp!(@kw, For,        KwFor) };
    (Gen)            => { exp!(@kw, Gen,        KwGen) };
    (If)             => { exp!(@kw, If,         KwIf) };
    (Impl)           => { exp!(@kw, Impl,       KwImpl) };
    (In)             => { exp!(@kw, In,         KwIn) };
    (Let)            => { exp!(@kw, Let,        KwLet) };
    (Loop)           => { exp!(@kw, Loop,       KwLoop) };
    (Macro)          => { exp!(@kw, Macro,      KwMacro) };
    (MacroRules)     => { exp!(@kw, MacroRules, KwMacroRules) };
    (Match)          => { exp!(@kw, Match,      KwMatch) };
    (Mod)            => { exp!(@kw, Mod,        KwMod) };
    (Move)           => { exp!(@kw, Move,       KwMove) };
    (Mut)            => { exp!(@kw, Mut,        KwMut) };
    (Pub)            => { exp!(@kw, Pub,        KwPub) };
    (Raw)            => { exp!(@kw, Raw,        KwRaw) };
    (Ref)            => { exp!(@kw, Ref,        KwRef) };
    (Return)         => { exp!(@kw, Return,     KwReturn) };
    (Reuse)          => { exp!(@kw, Reuse,      KwReuse) };
    (Safe)           => { exp!(@kw, Safe,       KwSafe) };
    (SelfUpper)      => { exp!(@kw, SelfUpper,  KwSelfUpper) };
    (Static)         => { exp!(@kw, Static,     KwStatic) };
    (Struct)         => { exp!(@kw, Struct,     KwStruct) };
    (Trait)          => { exp!(@kw, Trait,      KwTrait) };
    (Try)            => { exp!(@kw, Try,        KwTry) };
    (Type)           => { exp!(@kw, Type,       KwType) };
    (Underscore)     => { exp!(@kw, Underscore, KwUnderscore) };
    (Unsafe)         => { exp!(@kw, Unsafe,     KwUnsafe) };
    (Use)            => { exp!(@kw, Use,        KwUse) };
    (Where)          => { exp!(@kw, Where,      KwWhere) };
    (While)          => { exp!(@kw, While,      KwWhile) };
    (Yield)          => { exp!(@kw, Yield,      KwYield) };

    (AttSyntax)      => { exp!(@sym, att_syntax,      SymAttSyntax) };
    (ClobberAbi)     => { exp!(@sym, clobber_abi,     SymClobberAbi) };
    (Inlateout)      => { exp!(@sym, inlateout,       SymInlateout) };
    (Inout)          => { exp!(@sym, inout,           SymInout) };
    (Is)             => { exp!(@sym, is,              SymIs) };
    (Label)          => { exp!(@sym, label,           SymLabel) };
    (Lateout)        => { exp!(@sym, lateout,         SymLateout) };
    (MayUnwind)      => { exp!(@sym, may_unwind,      SymMayUnwind) };
    (Nomem)          => { exp!(@sym, nomem,           SymNomem) };
    (Noreturn)       => { exp!(@sym, noreturn,        SymNoreturn) };
    (Nostack)        => { exp!(@sym, nostack,         SymNostack) };
    (Options)        => { exp!(@sym, options,         SymOptions) };
    (Out)            => { exp!(@sym, out,             SymOut) };
    (PreservesFlags) => { exp!(@sym, preserves_flags, SymPreservesFlags) };
    (Pure)           => { exp!(@sym, pure,            SymPure) };
    (Readonly)       => { exp!(@sym, readonly,        SymReadonly) };
    (Sym)            => { exp!(@sym, sym,             SymSym) };
}

/// A bitset type designed specifically for `Parser::expected_token_types`,
/// which is very hot. `u128` is the smallest integer that will fit every
/// `TokenType` value.
#[derive(Clone, Copy)]
pub(super) struct TokenTypeSet(u128);

impl TokenTypeSet {
    pub(super) fn new() -> TokenTypeSet {
        TokenTypeSet(0)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub(super) fn insert(&mut self, token_type: TokenType) {
        self.0 = self.0 | (1u128 << token_type as u32)
    }

    pub(super) fn clear(&mut self) {
        self.0 = 0
    }

    pub(super) fn contains(&self, token_type: TokenType) -> bool {
        self.0 & (1u128 << token_type as u32) != 0
    }

    pub(super) fn iter(&self) -> TokenTypeSetIter {
        TokenTypeSetIter(*self)
    }
}

// The `TokenTypeSet` is a copy of the set being iterated. It initially holds
// the entire set. Each bit is cleared as it is returned. We have finished once
// it is all zeroes.
pub(super) struct TokenTypeSetIter(TokenTypeSet);

impl Iterator for TokenTypeSetIter {
    type Item = TokenType;

    fn next(&mut self) -> Option<TokenType> {
        let num_bits: u32 = (std::mem::size_of_val(&self.0.0) * 8) as u32;
        assert_eq!(num_bits, 128);
        let z = self.0.0.trailing_zeros();
        if z == num_bits {
            None
        } else {
            self.0.0 &= !(1 << z); // clear the trailing 1 bit
            Some(TokenType::from_u32(z))
        }
    }
}
