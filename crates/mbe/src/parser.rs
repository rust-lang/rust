//! Parser recognizes special macro syntax, `$var` and `$(repeat)*`, in token
//! trees.

use smallvec::SmallVec;
use syntax::SmolStr;
use tt::Delimiter;

use crate::{tt_iter::TtIter, MetaTemplate, ParseError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    Var { name: SmolStr, kind: Option<SmolStr>, id: tt::TokenId },
    Repeat { tokens: MetaTemplate, kind: RepeatKind, separator: Option<Separator> },
    Leaf(tt::Leaf),
    Subtree { tokens: MetaTemplate, delimiter: Option<Delimiter> },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum RepeatKind {
    ZeroOrMore,
    OneOrMore,
    ZeroOrOne,
}

#[derive(Clone, Debug, Eq)]
pub(crate) enum Separator {
    Literal(tt::Literal),
    Ident(tt::Ident),
    Puncts(SmallVec<[tt::Punct; 3]>),
}

// Note that when we compare a Separator, we just care about its textual value.
impl PartialEq for Separator {
    fn eq(&self, other: &Separator) -> bool {
        use Separator::*;

        match (self, other) {
            (Ident(ref a), Ident(ref b)) => a.text == b.text,
            (Literal(ref a), Literal(ref b)) => a.text == b.text,
            (Puncts(ref a), Puncts(ref b)) if a.len() == b.len() => {
                let a_iter = a.iter().map(|a| a.char);
                let b_iter = b.iter().map(|b| b.char);
                a_iter.eq(b_iter)
            }
            _ => false,
        }
    }
}

pub(crate) fn parse_template(template: &tt::Subtree) -> Result<Vec<Op>, ParseError> {
    parse_inner(&template, Mode::Template).into_iter().collect()
}

pub(crate) fn parse_pattern(pattern: &tt::Subtree) -> Result<Vec<Op>, ParseError> {
    parse_inner(&pattern, Mode::Pattern).into_iter().collect()
}

#[derive(Clone, Copy)]
enum Mode {
    Pattern,
    Template,
}

fn parse_inner(tt: &tt::Subtree, mode: Mode) -> Vec<Result<Op, ParseError>> {
    let mut src = TtIter::new(&tt);
    std::iter::from_fn(move || {
        let first = src.next()?;
        Some(next_op(first, &mut src, mode))
    })
    .collect()
}

macro_rules! err {
    ($($tt:tt)*) => {
        ParseError::UnexpectedToken(($($tt)*).to_string())
    };
}

macro_rules! bail {
    ($($tt:tt)*) => {
        return Err(err!($($tt)*))
    };
}

fn next_op<'a>(first: &tt::TokenTree, src: &mut TtIter<'a>, mode: Mode) -> Result<Op, ParseError> {
    let res = match first {
        tt::TokenTree::Leaf(leaf @ tt::Leaf::Punct(tt::Punct { char: '$', .. })) => {
            // Note that the '$' itself is a valid token inside macro_rules.
            let second = match src.next() {
                None => return Ok(Op::Leaf(leaf.clone())),
                Some(it) => it,
            };
            match second {
                tt::TokenTree::Subtree(subtree) => {
                    let (separator, kind) = parse_repeat(src)?;
                    let tokens = parse_inner(&subtree, mode)
                        .into_iter()
                        .collect::<Result<Vec<Op>, ParseError>>()?;
                    Op::Repeat { tokens: MetaTemplate(tokens), separator, kind }
                }
                tt::TokenTree::Leaf(leaf) => match leaf {
                    tt::Leaf::Punct(punct) => {
                        static UNDERSCORE: SmolStr = SmolStr::new_inline("_");

                        if punct.char != '_' {
                            return Err(ParseError::Expected("_".to_string()));
                        }
                        let name = UNDERSCORE.clone();
                        let kind = eat_fragment_kind(src, mode)?;
                        let id = punct.id;
                        Op::Var { name, kind, id }
                    }
                    tt::Leaf::Ident(ident) if ident.text == "crate" => {
                        // We simply produce identifier `$crate` here. And it will be resolved when lowering ast to Path.
                        Op::Leaf(tt::Leaf::from(tt::Ident { text: "$crate".into(), id: ident.id }))
                    }
                    tt::Leaf::Ident(ident) => {
                        let name = ident.text.clone();
                        let kind = eat_fragment_kind(src, mode)?;
                        let id = ident.id;
                        Op::Var { name, kind, id }
                    }
                    tt::Leaf::Literal(lit) => {
                        if is_boolean_literal(&lit) {
                            let name = lit.text.clone();
                            let kind = eat_fragment_kind(src, mode)?;
                            let id = lit.id;
                            Op::Var { name, kind, id }
                        } else {
                            bail!("bad var 2");
                        }
                    }
                },
            }
        }
        tt::TokenTree::Leaf(tt) => Op::Leaf(tt.clone()),
        tt::TokenTree::Subtree(subtree) => {
            let tokens =
                parse_inner(&subtree, mode).into_iter().collect::<Result<Vec<Op>, ParseError>>()?;
            Op::Subtree { tokens: MetaTemplate(tokens), delimiter: subtree.delimiter }
        }
    };
    Ok(res)
}

fn eat_fragment_kind<'a>(src: &mut TtIter<'a>, mode: Mode) -> Result<Option<SmolStr>, ParseError> {
    if let Mode::Pattern = mode {
        src.expect_char(':').map_err(|()| err!("bad fragment specifier 1"))?;
        let ident = src.expect_ident().map_err(|()| err!("bad fragment specifier 1"))?;
        return Ok(Some(ident.text.clone()));
    };
    Ok(None)
}

fn is_boolean_literal(lit: &tt::Literal) -> bool {
    matches!(lit.text.as_str(), "true" | "false")
}

fn parse_repeat(src: &mut TtIter) -> Result<(Option<Separator>, RepeatKind), ParseError> {
    let mut separator = Separator::Puncts(SmallVec::new());
    for tt in src {
        let tt = match tt {
            tt::TokenTree::Leaf(leaf) => leaf,
            tt::TokenTree::Subtree(_) => return Err(ParseError::InvalidRepeat),
        };
        let has_sep = match &separator {
            Separator::Puncts(puncts) => !puncts.is_empty(),
            _ => true,
        };
        match tt {
            tt::Leaf::Ident(_) | tt::Leaf::Literal(_) if has_sep => {
                return Err(ParseError::InvalidRepeat)
            }
            tt::Leaf::Ident(ident) => separator = Separator::Ident(ident.clone()),
            tt::Leaf::Literal(lit) => separator = Separator::Literal(lit.clone()),
            tt::Leaf::Punct(punct) => {
                let repeat_kind = match punct.char {
                    '*' => RepeatKind::ZeroOrMore,
                    '+' => RepeatKind::OneOrMore,
                    '?' => RepeatKind::ZeroOrOne,
                    _ => {
                        match &mut separator {
                            Separator::Puncts(puncts) => {
                                if puncts.len() == 3 {
                                    return Err(ParseError::InvalidRepeat);
                                }
                                puncts.push(punct.clone())
                            }
                            _ => return Err(ParseError::InvalidRepeat),
                        }
                        continue;
                    }
                };
                let separator = if has_sep { Some(separator) } else { None };
                return Ok((separator, repeat_kind));
            }
        }
    }
    Err(ParseError::InvalidRepeat)
}
