//! Parser recognizes special macro syntax, `$var` and `$(repeat)*`, in token
//! trees.

use ra_syntax::SmolStr;
use smallvec::SmallVec;

use crate::{tt_iter::TtIter, ExpandError};

#[derive(Debug)]
pub(crate) enum Op<'a> {
    Var { name: &'a SmolStr, kind: Option<&'a SmolStr> },
    Repeat { subtree: &'a tt::Subtree, kind: RepeatKind, separator: Option<Separator> },
    TokenTree(&'a tt::TokenTree),
}

#[derive(Clone, Debug, PartialEq, Eq)]
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

pub(crate) fn parse_template(
    template: &tt::Subtree,
) -> impl Iterator<Item = Result<Op<'_>, ExpandError>> {
    parse_inner(template, Mode::Template)
}

pub(crate) fn parse_pattern(
    pattern: &tt::Subtree,
) -> impl Iterator<Item = Result<Op<'_>, ExpandError>> {
    parse_inner(pattern, Mode::Pattern)
}

#[derive(Clone, Copy)]
enum Mode {
    Pattern,
    Template,
}

fn parse_inner(src: &tt::Subtree, mode: Mode) -> impl Iterator<Item = Result<Op<'_>, ExpandError>> {
    let mut src = TtIter::new(src);
    std::iter::from_fn(move || {
        let first = src.next()?;
        Some(next_op(first, &mut src, mode))
    })
}

macro_rules! err {
    ($($tt:tt)*) => {
        ExpandError::UnexpectedToken
    };
}

macro_rules! bail {
    ($($tt:tt)*) => {
        return Err(err!($($tt)*))
    };
}

fn next_op<'a>(
    first: &'a tt::TokenTree,
    src: &mut TtIter<'a>,
    mode: Mode,
) -> Result<Op<'a>, ExpandError> {
    let res = match first {
        tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct { char: '$', .. })) => {
            // Note that the '$' itself is a valid token inside macro_rules.
            let second = match src.next() {
                None => return Ok(Op::TokenTree(first)),
                Some(it) => it,
            };
            match second {
                tt::TokenTree::Subtree(subtree) => {
                    let (separator, kind) = parse_repeat(src)?;
                    Op::Repeat { subtree, separator, kind }
                }
                tt::TokenTree::Leaf(leaf) => match leaf {
                    tt::Leaf::Punct(..) => return Err(ExpandError::UnexpectedToken),
                    tt::Leaf::Ident(ident) => {
                        let name = &ident.text;
                        let kind = eat_fragment_kind(src, mode)?;
                        Op::Var { name, kind }
                    }
                    tt::Leaf::Literal(lit) => {
                        if is_boolean_literal(lit) {
                            let name = &lit.text;
                            let kind = eat_fragment_kind(src, mode)?;
                            Op::Var { name, kind }
                        } else {
                            bail!("bad var 2");
                        }
                    }
                },
            }
        }
        tt => Op::TokenTree(tt),
    };
    Ok(res)
}

fn eat_fragment_kind<'a>(
    src: &mut TtIter<'a>,
    mode: Mode,
) -> Result<Option<&'a SmolStr>, ExpandError> {
    if let Mode::Pattern = mode {
        src.expect_char(':').map_err(|()| err!("bad fragment specifier 1"))?;
        let ident = src.expect_ident().map_err(|()| err!("bad fragment specifier 1"))?;
        return Ok(Some(&ident.text));
    };
    Ok(None)
}

fn is_boolean_literal(lit: &tt::Literal) -> bool {
    matches!(lit.text.as_str(), "true" | "false")
}

fn parse_repeat(src: &mut TtIter) -> Result<(Option<Separator>, RepeatKind), ExpandError> {
    let mut separator = Separator::Puncts(SmallVec::new());
    for tt in src {
        let tt = match tt {
            tt::TokenTree::Leaf(leaf) => leaf,
            tt::TokenTree::Subtree(_) => return Err(ExpandError::InvalidRepeat),
        };
        let has_sep = match &separator {
            Separator::Puncts(puncts) => !puncts.is_empty(),
            _ => true,
        };
        match tt {
            tt::Leaf::Ident(_) | tt::Leaf::Literal(_) if has_sep => {
                return Err(ExpandError::InvalidRepeat)
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
                                    return Err(ExpandError::InvalidRepeat);
                                }
                                puncts.push(punct.clone())
                            }
                            _ => return Err(ExpandError::InvalidRepeat),
                        }
                        continue;
                    }
                };
                let separator = if has_sep { Some(separator) } else { None };
                return Ok((separator, repeat_kind));
            }
        }
    }
    Err(ExpandError::InvalidRepeat)
}
