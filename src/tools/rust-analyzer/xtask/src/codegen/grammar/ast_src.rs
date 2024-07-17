//! Defines input for code generation process.

use crate::codegen::grammar::to_upper_snake_case;

#[derive(Copy, Clone, Debug)]
pub(crate) struct KindsSrc {
    pub(crate) punct: &'static [(&'static str, &'static str)],
    pub(crate) keywords: &'static [&'static str],
    pub(crate) contextual_keywords: &'static [&'static str],
    pub(crate) literals: &'static [&'static str],
    pub(crate) tokens: &'static [&'static str],
    pub(crate) nodes: &'static [&'static str],
}

/// The punctuations of the language.
const PUNCT: &[(&str, &str)] = &[
    // KEEP THE DOLLAR AT THE TOP ITS SPECIAL
    ("$", "DOLLAR"),
    (";", "SEMICOLON"),
    (",", "COMMA"),
    ("(", "L_PAREN"),
    (")", "R_PAREN"),
    ("{", "L_CURLY"),
    ("}", "R_CURLY"),
    ("[", "L_BRACK"),
    ("]", "R_BRACK"),
    ("<", "L_ANGLE"),
    (">", "R_ANGLE"),
    ("@", "AT"),
    ("#", "POUND"),
    ("~", "TILDE"),
    ("?", "QUESTION"),
    ("&", "AMP"),
    ("|", "PIPE"),
    ("+", "PLUS"),
    ("*", "STAR"),
    ("/", "SLASH"),
    ("^", "CARET"),
    ("%", "PERCENT"),
    ("_", "UNDERSCORE"),
    (".", "DOT"),
    ("..", "DOT2"),
    ("...", "DOT3"),
    ("..=", "DOT2EQ"),
    (":", "COLON"),
    ("::", "COLON2"),
    ("=", "EQ"),
    ("==", "EQ2"),
    ("=>", "FAT_ARROW"),
    ("!", "BANG"),
    ("!=", "NEQ"),
    ("-", "MINUS"),
    ("->", "THIN_ARROW"),
    ("<=", "LTEQ"),
    (">=", "GTEQ"),
    ("+=", "PLUSEQ"),
    ("-=", "MINUSEQ"),
    ("|=", "PIPEEQ"),
    ("&=", "AMPEQ"),
    ("^=", "CARETEQ"),
    ("/=", "SLASHEQ"),
    ("*=", "STAREQ"),
    ("%=", "PERCENTEQ"),
    ("&&", "AMP2"),
    ("||", "PIPE2"),
    ("<<", "SHL"),
    (">>", "SHR"),
    ("<<=", "SHLEQ"),
    (">>=", "SHREQ"),
];
const TOKENS: &[&str] = &["ERROR", "WHITESPACE", "NEWLINE", "COMMENT"];
// &["ERROR", "IDENT", "WHITESPACE", "LIFETIME_IDENT", "COMMENT", "SHEBANG"],;

const EOF: &str = "EOF";

const RESERVED: &[&str] = &[
    "abstract", "become", "box", "do", "final", "macro", "override", "priv", "typeof", "unsized",
    "virtual", "yield", "try",
];
const CONTEXTUAL_RESERVED: &[&str] = &[];

pub(crate) fn generate_kind_src(
    nodes: &[AstNodeSrc],
    enums: &[AstEnumSrc],
    grammar: &ungrammar::Grammar,
) -> KindsSrc {
    let mut keywords: Vec<&_> = Vec::new();
    let mut contextual_keywords: Vec<&_> = Vec::new();
    let mut tokens: Vec<&_> = TOKENS.to_vec();
    let mut literals: Vec<&_> = Vec::new();
    let mut used_puncts = vec![false; PUNCT.len()];
    // Mark $ as used
    used_puncts[0] = true;
    grammar.tokens().for_each(|token| {
        let name = &*grammar[token].name;
        if name == EOF {
            return;
        }
        match name.split_at(1) {
            ("@", lit) if !lit.is_empty() => {
                literals.push(String::leak(to_upper_snake_case(lit)));
            }
            ("#", token) if !token.is_empty() => {
                tokens.push(String::leak(to_upper_snake_case(token)));
            }
            ("?", kw) if !kw.is_empty() => {
                contextual_keywords.push(String::leak(kw.to_owned()));
            }
            _ if name.chars().all(char::is_alphabetic) => {
                keywords.push(String::leak(name.to_owned()));
            }
            _ => {
                let idx = PUNCT
                    .iter()
                    .position(|(punct, _)| punct == &name)
                    .unwrap_or_else(|| panic!("Grammar references unknown punctuation {name:?}"));
                used_puncts[idx] = true;
            }
        }
    });
    PUNCT.iter().zip(used_puncts).filter(|(_, used)| !used).for_each(|((punct, _), _)| {
        panic!("Punctuation {punct:?} is not used in grammar");
    });
    keywords.extend(RESERVED.iter().copied());
    keywords.sort();
    keywords.dedup();
    contextual_keywords.extend(CONTEXTUAL_RESERVED.iter().copied());
    contextual_keywords.sort();
    contextual_keywords.dedup();

    // we leak things here for simplicity, that way we don't have to deal with lifetimes
    // The execution is a one shot job so thats fine
    let nodes = nodes
        .iter()
        .map(|it| &it.name)
        .chain(enums.iter().map(|it| &it.name))
        .map(|it| to_upper_snake_case(it))
        .map(String::leak)
        .map(|it| &*it)
        .collect();
    let nodes = Vec::leak(nodes);
    nodes.sort();
    let keywords = Vec::leak(keywords);
    let contextual_keywords = Vec::leak(contextual_keywords);
    let literals = Vec::leak(literals);
    literals.sort();
    let tokens = Vec::leak(tokens);
    tokens.sort();

    KindsSrc { punct: PUNCT, nodes, keywords, contextual_keywords, literals, tokens }
}

#[derive(Default, Debug)]
pub(crate) struct AstSrc {
    pub(crate) tokens: Vec<String>,
    pub(crate) nodes: Vec<AstNodeSrc>,
    pub(crate) enums: Vec<AstEnumSrc>,
}

#[derive(Debug)]
pub(crate) struct AstNodeSrc {
    pub(crate) doc: Vec<String>,
    pub(crate) name: String,
    pub(crate) traits: Vec<String>,
    pub(crate) fields: Vec<Field>,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum Field {
    Token(String),
    Node { name: String, ty: String, cardinality: Cardinality },
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum Cardinality {
    Optional,
    Many,
}

#[derive(Debug)]
pub(crate) struct AstEnumSrc {
    pub(crate) doc: Vec<String>,
    pub(crate) name: String,
    pub(crate) traits: Vec<String>,
    pub(crate) variants: Vec<String>,
}
