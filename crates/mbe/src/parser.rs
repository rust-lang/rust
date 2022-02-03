//! Parser recognizes special macro syntax, `$var` and `$(repeat)*`, in token
//! trees.

use smallvec::SmallVec;
use syntax::SmolStr;

use crate::{tt_iter::TtIter, ParseError};

/// Consider
///
/// ```
/// macro_rules! an_macro {
///     ($x:expr + $y:expr) => ($y * $x)
/// }
/// ```
///
/// Stuff to the left of `=>` is a [`MetaTemplate`] pattern (which is matched
/// with input).
///
/// Stuff to the right is a [`MetaTemplate`] template which is used to produce
/// output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MetaTemplate(pub(crate) Vec<Op>);

impl MetaTemplate {
    pub(crate) fn parse_pattern(pattern: &tt::Subtree) -> Result<MetaTemplate, ParseError> {
        MetaTemplate::parse(pattern, Mode::Pattern)
    }

    pub(crate) fn parse_template(template: &tt::Subtree) -> Result<MetaTemplate, ParseError> {
        MetaTemplate::parse(template, Mode::Template)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &Op> {
        self.0.iter()
    }

    fn parse(tt: &tt::Subtree, mode: Mode) -> Result<MetaTemplate, ParseError> {
        let mut src = TtIter::new(tt);

        let mut res = Vec::new();
        while let Some(first) = src.next() {
            let op = next_op(first, &mut src, mode)?;
            res.push(op);
        }

        Ok(MetaTemplate(res))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    Var { name: SmolStr, kind: Option<SmolStr>, id: tt::TokenId },
    Repeat { tokens: MetaTemplate, kind: RepeatKind, separator: Option<Separator> },
    Leaf(tt::Leaf),
    Subtree { tokens: MetaTemplate, delimiter: Option<tt::Delimiter> },
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
            (Ident(a), Ident(b)) => a.text == b.text,
            (Literal(a), Literal(b)) => a.text == b.text,
            (Puncts(a), Puncts(b)) if a.len() == b.len() => {
                let a_iter = a.iter().map(|a| a.char);
                let b_iter = b.iter().map(|b| b.char);
                a_iter.eq(b_iter)
            }
            _ => false,
        }
    }
}

impl Separator {
    pub(crate) fn tt_count(&self) -> usize {
        match self {
            Separator::Literal(_) => 1,
            Separator::Ident(_) => 1,
            Separator::Puncts(it) => it.len(),
        }
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Pattern,
    Template,
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
                    let tokens = MetaTemplate::parse(subtree, mode)?;
                    Op::Repeat { tokens, separator, kind }
                }
                tt::TokenTree::Leaf(leaf) => match leaf {
                    tt::Leaf::Ident(ident) if ident.text == "crate" => {
                        // We simply produce identifier `$crate` here. And it will be resolved when lowering ast to Path.
                        Op::Leaf(tt::Leaf::from(tt::Ident { text: "$crate".into(), id: ident.id }))
                    }
                    tt::Leaf::Ident(ident) => {
                        let kind = eat_fragment_kind(src, mode)?;
                        let name = ident.text.clone();
                        let id = ident.id;
                        Op::Var { name, kind, id }
                    }
                    tt::Leaf::Literal(lit) if is_boolean_literal(lit) => {
                        let kind = eat_fragment_kind(src, mode)?;
                        let name = lit.text.clone();
                        let id = lit.id;
                        Op::Var { name, kind, id }
                    }
                    tt::Leaf::Punct(_) | tt::Leaf::Literal(_) => {
                        return Err(ParseError::expected("expected ident"))
                    }
                },
            }
        }
        tt::TokenTree::Leaf(tt) => Op::Leaf(tt.clone()),
        tt::TokenTree::Subtree(subtree) => {
            let tokens = MetaTemplate::parse(subtree, mode)?;
            Op::Subtree { tokens, delimiter: subtree.delimiter }
        }
    };
    Ok(res)
}

fn eat_fragment_kind(src: &mut TtIter<'_>, mode: Mode) -> Result<Option<SmolStr>, ParseError> {
    if let Mode::Pattern = mode {
        src.expect_char(':').map_err(|()| ParseError::unexpected("missing fragment specifier"))?;
        let ident = src
            .expect_ident()
            .map_err(|()| ParseError::unexpected("missing fragment specifier"))?;
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
                    _ => match &mut separator {
                        Separator::Puncts(puncts) if puncts.len() != 3 => {
                            puncts.push(*punct);
                            continue;
                        }
                        _ => return Err(ParseError::InvalidRepeat),
                    },
                };
                return Ok((has_sep.then(|| separator), repeat_kind));
            }
        }
    }
    Err(ParseError::InvalidRepeat)
}
