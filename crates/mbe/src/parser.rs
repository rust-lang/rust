//! Parser recognizes special macro syntax, `$var` and `$(repeat)*`, in token
//! trees.

use smallvec::{smallvec, SmallVec};
use syntax::SmolStr;
use tt::Span;

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
pub(crate) struct MetaTemplate<S>(pub(crate) Box<[Op<S>]>);

impl<S: Span> MetaTemplate<S> {
    pub(crate) fn parse_pattern(pattern: &tt::Subtree<S>) -> Result<Self, ParseError> {
        MetaTemplate::parse(pattern, Mode::Pattern, false)
    }

    pub(crate) fn parse_template(
        template: &tt::Subtree<S>,
        new_meta_vars: bool,
    ) -> Result<Self, ParseError> {
        MetaTemplate::parse(template, Mode::Template, new_meta_vars)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &Op<S>> {
        self.0.iter()
    }

    fn parse(tt: &tt::Subtree<S>, mode: Mode, new_meta_vars: bool) -> Result<Self, ParseError> {
        let mut src = TtIter::new(tt);

        let mut res = Vec::new();
        while let Some(first) = src.peek_n(0) {
            let op = next_op(first, &mut src, mode, new_meta_vars)?;
            res.push(op);
        }

        Ok(MetaTemplate(res.into_boxed_slice()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Op<S> {
    Var {
        name: SmolStr,
        kind: Option<MetaVarKind>,
        id: S,
    },
    Ignore {
        name: SmolStr,
        id: S,
    },
    Index {
        depth: usize,
    },
    Length {
        depth: usize,
    },
    Count {
        name: SmolStr,
        // FIXME: `usize`` once we drop support for 1.76
        depth: Option<usize>,
    },
    Repeat {
        tokens: MetaTemplate<S>,
        kind: RepeatKind,
        separator: Option<Separator<S>>,
    },
    Subtree {
        tokens: MetaTemplate<S>,
        delimiter: tt::Delimiter<S>,
    },
    Literal(tt::Literal<S>),
    Punct(SmallVec<[tt::Punct<S>; 3]>),
    Ident(tt::Ident<S>),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum RepeatKind {
    ZeroOrMore,
    OneOrMore,
    ZeroOrOne,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum MetaVarKind {
    Path,
    Ty,
    Pat,
    PatParam,
    Stmt,
    Block,
    Meta,
    Item,
    Vis,
    Expr,
    Ident,
    Tt,
    Lifetime,
    Literal,
}

#[derive(Clone, Debug, Eq)]
pub(crate) enum Separator<S> {
    Literal(tt::Literal<S>),
    Ident(tt::Ident<S>),
    Puncts(SmallVec<[tt::Punct<S>; 3]>),
}

// Note that when we compare a Separator, we just care about its textual value.
impl<S> PartialEq for Separator<S> {
    fn eq(&self, other: &Separator<S>) -> bool {
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

#[derive(Clone, Copy)]
enum Mode {
    Pattern,
    Template,
}

fn next_op<S: Span>(
    first_peeked: &tt::TokenTree<S>,
    src: &mut TtIter<'_, S>,
    mode: Mode,
    new_meta_vars: bool,
) -> Result<Op<S>, ParseError> {
    let res = match first_peeked {
        tt::TokenTree::Leaf(tt::Leaf::Punct(p @ tt::Punct { char: '$', .. })) => {
            src.next().expect("first token already peeked");
            // Note that the '$' itself is a valid token inside macro_rules.
            let second = match src.next() {
                None => return Ok(Op::Punct(smallvec![*p])),
                Some(it) => it,
            };
            match second {
                tt::TokenTree::Subtree(subtree) => match subtree.delimiter.kind {
                    tt::DelimiterKind::Parenthesis => {
                        let (separator, kind) = parse_repeat(src)?;
                        let tokens = MetaTemplate::parse(subtree, mode, new_meta_vars)?;
                        Op::Repeat { tokens, separator, kind }
                    }
                    tt::DelimiterKind::Brace => match mode {
                        Mode::Template => {
                            parse_metavar_expr(new_meta_vars, &mut TtIter::new(subtree)).map_err(
                                |()| ParseError::unexpected("invalid metavariable expression"),
                            )?
                        }
                        Mode::Pattern => {
                            return Err(ParseError::unexpected(
                                "`${}` metavariable expressions are not allowed in matchers",
                            ))
                        }
                    },
                    _ => {
                        return Err(ParseError::expected(
                            "expected `$()` repetition or `${}` expression",
                        ))
                    }
                },
                tt::TokenTree::Leaf(leaf) => match leaf {
                    tt::Leaf::Ident(ident) if ident.text == "crate" => {
                        // We simply produce identifier `$crate` here. And it will be resolved when lowering ast to Path.
                        Op::Ident(tt::Ident { text: "$crate".into(), span: ident.span })
                    }
                    tt::Leaf::Ident(ident) => {
                        let kind = eat_fragment_kind(src, mode)?;
                        let name = ident.text.clone();
                        let id = ident.span;
                        Op::Var { name, kind, id }
                    }
                    tt::Leaf::Literal(lit) if is_boolean_literal(lit) => {
                        let kind = eat_fragment_kind(src, mode)?;
                        let name = lit.text.clone();
                        let id = lit.span;
                        Op::Var { name, kind, id }
                    }
                    tt::Leaf::Punct(punct @ tt::Punct { char: '$', .. }) => match mode {
                        Mode::Pattern => {
                            return Err(ParseError::unexpected(
                                "`$$` is not allowed on the pattern side",
                            ))
                        }
                        Mode::Template => Op::Punct(smallvec![*punct]),
                    },
                    tt::Leaf::Punct(_) | tt::Leaf::Literal(_) => {
                        return Err(ParseError::expected("expected ident"))
                    }
                },
            }
        }

        tt::TokenTree::Leaf(tt::Leaf::Literal(it)) => {
            src.next().expect("first token already peeked");
            Op::Literal(it.clone())
        }

        tt::TokenTree::Leaf(tt::Leaf::Ident(it)) => {
            src.next().expect("first token already peeked");
            Op::Ident(it.clone())
        }

        tt::TokenTree::Leaf(tt::Leaf::Punct(_)) => {
            // There's at least one punct so this shouldn't fail.
            let puncts = src.expect_glued_punct().unwrap();
            Op::Punct(puncts)
        }

        tt::TokenTree::Subtree(subtree) => {
            src.next().expect("first token already peeked");
            let tokens = MetaTemplate::parse(subtree, mode, new_meta_vars)?;
            Op::Subtree { tokens, delimiter: subtree.delimiter }
        }
    };
    Ok(res)
}

fn eat_fragment_kind<S: Span>(
    src: &mut TtIter<'_, S>,
    mode: Mode,
) -> Result<Option<MetaVarKind>, ParseError> {
    if let Mode::Pattern = mode {
        src.expect_char(':').map_err(|()| ParseError::unexpected("missing fragment specifier"))?;
        let ident = src
            .expect_ident()
            .map_err(|()| ParseError::unexpected("missing fragment specifier"))?;
        let kind = match ident.text.as_str() {
            "path" => MetaVarKind::Path,
            "ty" => MetaVarKind::Ty,
            "pat" => MetaVarKind::Pat,
            "pat_param" => MetaVarKind::PatParam,
            "stmt" => MetaVarKind::Stmt,
            "block" => MetaVarKind::Block,
            "meta" => MetaVarKind::Meta,
            "item" => MetaVarKind::Item,
            "vis" => MetaVarKind::Vis,
            "expr" => MetaVarKind::Expr,
            "ident" => MetaVarKind::Ident,
            "tt" => MetaVarKind::Tt,
            "lifetime" => MetaVarKind::Lifetime,
            "literal" => MetaVarKind::Literal,
            _ => return Ok(None),
        };
        return Ok(Some(kind));
    };
    Ok(None)
}

fn is_boolean_literal<S>(lit: &tt::Literal<S>) -> bool {
    matches!(lit.text.as_str(), "true" | "false")
}

fn parse_repeat<S: Span>(
    src: &mut TtIter<'_, S>,
) -> Result<(Option<Separator<S>>, RepeatKind), ParseError> {
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
                return Ok((has_sep.then_some(separator), repeat_kind));
            }
        }
    }
    Err(ParseError::InvalidRepeat)
}

fn parse_metavar_expr<S: Span>(new_meta_vars: bool, src: &mut TtIter<'_, S>) -> Result<Op<S>, ()> {
    let func = src.expect_ident()?;
    let args = src.expect_subtree()?;

    if args.delimiter.kind != tt::DelimiterKind::Parenthesis {
        return Err(());
    }

    let mut args = TtIter::new(args);

    let op = match &*func.text {
        "ignore" => {
            if new_meta_vars {
                args.expect_dollar()?;
            }
            let ident = args.expect_ident()?;
            Op::Ignore { name: ident.text.clone(), id: ident.span }
        }
        "index" => Op::Index { depth: parse_depth(&mut args)? },
        "length" => Op::Length { depth: parse_depth(&mut args)? },
        "count" => {
            if new_meta_vars {
                args.expect_dollar()?;
            }
            let ident = args.expect_ident()?;
            let depth = if try_eat_comma(&mut args) { Some(parse_depth(&mut args)?) } else { None };
            Op::Count { name: ident.text.clone(), depth }
        }
        _ => return Err(()),
    };

    if args.next().is_some() {
        return Err(());
    }

    Ok(op)
}

fn parse_depth<S: Span>(src: &mut TtIter<'_, S>) -> Result<usize, ()> {
    if src.len() == 0 {
        Ok(0)
    } else if let tt::Leaf::Literal(lit) = src.expect_literal()? {
        // Suffixes are not allowed.
        lit.text.parse().map_err(|_| ())
    } else {
        Err(())
    }
}

fn try_eat_comma<S: Span>(src: &mut TtIter<'_, S>) -> bool {
    if let Some(tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct { char: ',', .. }))) = src.peek_n(0) {
        let _ = src.next();
        return true;
    }
    false
}
