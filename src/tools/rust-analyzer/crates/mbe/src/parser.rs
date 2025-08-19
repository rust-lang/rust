//! Parser recognizes special macro syntax, `$var` and `$(repeat)*`, in token
//! trees.

use std::sync::Arc;

use arrayvec::ArrayVec;
use intern::{Symbol, sym};
use span::{Edition, Span, SyntaxContext};
use tt::{
    MAX_GLUED_PUNCT_LEN,
    iter::{TtElement, TtIter},
};

use crate::ParseError;

/// Consider
///
/// ```
/// macro_rules! a_macro {
///     ($x:expr, $y:expr) => ($y * $x)
/// }
/// ```
///
/// Stuff to the left of `=>` is a [`MetaTemplate`] pattern (which is matched
/// with input).
///
/// Stuff to the right is a [`MetaTemplate`] template which is used to produce
/// output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MetaTemplate(pub(crate) Box<[Op]>);

impl MetaTemplate {
    pub(crate) fn parse_pattern(
        edition: impl Copy + Fn(SyntaxContext) -> Edition,
        pattern: TtIter<'_, Span>,
    ) -> Result<Self, ParseError> {
        MetaTemplate::parse(edition, pattern, Mode::Pattern)
    }

    pub(crate) fn parse_template(
        edition: impl Copy + Fn(SyntaxContext) -> Edition,
        template: TtIter<'_, Span>,
    ) -> Result<Self, ParseError> {
        MetaTemplate::parse(edition, template, Mode::Template)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &Op> {
        self.0.iter()
    }

    fn parse(
        edition: impl Copy + Fn(SyntaxContext) -> Edition,
        mut src: TtIter<'_, Span>,
        mode: Mode,
    ) -> Result<Self, ParseError> {
        let mut res = Vec::new();
        while let Some(first) = src.peek() {
            let op = next_op(edition, first, &mut src, mode)?;
            res.push(op);
        }

        Ok(MetaTemplate(res.into_boxed_slice()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    Var {
        name: Symbol,
        kind: Option<MetaVarKind>,
        id: Span,
    },
    Ignore {
        name: Symbol,
        id: Span,
    },
    Index {
        depth: usize,
    },
    Len {
        depth: usize,
    },
    Count {
        name: Symbol,
        // FIXME: `usize`` once we drop support for 1.76
        depth: Option<usize>,
    },
    Concat {
        elements: Box<[ConcatMetaVarExprElem]>,
        span: Span,
    },
    Repeat {
        tokens: MetaTemplate,
        kind: RepeatKind,
        separator: Option<Arc<Separator>>,
    },
    Subtree {
        tokens: MetaTemplate,
        delimiter: tt::Delimiter<Span>,
    },
    Literal(tt::Literal<Span>),
    Punct(Box<ArrayVec<tt::Punct<Span>, MAX_GLUED_PUNCT_LEN>>),
    Ident(tt::Ident<Span>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ConcatMetaVarExprElem {
    /// There is NO preceding dollar sign, which means that this identifier should be interpreted
    /// as a literal.
    Ident(tt::Ident<Span>),
    /// There is a preceding dollar sign, which means that this identifier should be expanded
    /// and interpreted as a variable.
    Var(tt::Ident<Span>),
    /// For example, a number or a string.
    Literal(tt::Literal<Span>),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum RepeatKind {
    ZeroOrMore,
    OneOrMore,
    ZeroOrOne,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ExprKind {
    // Matches expressions using the post-edition 2024. Was written using
    // `expr` in edition 2024 or later.
    Expr,
    // Matches expressions using the pre-edition 2024 rules.
    // Either written using `expr` in edition 2021 or earlier or.was written using `expr_2021`.
    Expr2021,
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
    Expr(ExprKind),
    Ident,
    Tt,
    Lifetime,
    Literal,
}

#[derive(Clone, Debug, Eq)]
pub(crate) enum Separator {
    Literal(tt::Literal<Span>),
    Ident(tt::Ident<Span>),
    Puncts(ArrayVec<tt::Punct<Span>, MAX_GLUED_PUNCT_LEN>),
    Lifetime(tt::Punct<Span>, tt::Ident<Span>),
}

// Note that when we compare a Separator, we just care about its textual value.
impl PartialEq for Separator {
    fn eq(&self, other: &Separator) -> bool {
        use Separator::*;

        match (self, other) {
            (Ident(a), Ident(b)) => a.sym == b.sym,
            (Literal(a), Literal(b)) => a.symbol == b.symbol,
            (Puncts(a), Puncts(b)) if a.len() == b.len() => {
                let a_iter = a.iter().map(|a| a.char);
                let b_iter = b.iter().map(|b| b.char);
                a_iter.eq(b_iter)
            }
            (Lifetime(_, a), Lifetime(_, b)) => a.sym == b.sym,
            _ => false,
        }
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Pattern,
    Template,
}

fn next_op(
    edition: impl Copy + Fn(SyntaxContext) -> Edition,
    first_peeked: TtElement<'_, Span>,
    src: &mut TtIter<'_, Span>,
    mode: Mode,
) -> Result<Op, ParseError> {
    let res = match first_peeked {
        TtElement::Leaf(tt::Leaf::Punct(p @ tt::Punct { char: '$', .. })) => {
            src.next().expect("first token already peeked");
            // Note that the '$' itself is a valid token inside macro_rules.
            let second = match src.next() {
                None => {
                    return Ok(Op::Punct({
                        let mut res = ArrayVec::new();
                        res.push(*p);
                        Box::new(res)
                    }));
                }
                Some(it) => it,
            };
            match second {
                TtElement::Subtree(subtree, mut subtree_iter) => match subtree.delimiter.kind {
                    tt::DelimiterKind::Parenthesis => {
                        let (separator, kind) = parse_repeat(src)?;
                        let tokens = MetaTemplate::parse(edition, subtree_iter, mode)?;
                        Op::Repeat { tokens, separator: separator.map(Arc::new), kind }
                    }
                    tt::DelimiterKind::Brace => match mode {
                        Mode::Template => parse_metavar_expr(&mut subtree_iter).map_err(|()| {
                            ParseError::unexpected("invalid metavariable expression")
                        })?,
                        Mode::Pattern => {
                            return Err(ParseError::unexpected(
                                "`${}` metavariable expressions are not allowed in matchers",
                            ));
                        }
                    },
                    _ => {
                        return Err(ParseError::expected(
                            "expected `$()` repetition or `${}` expression",
                        ));
                    }
                },
                TtElement::Leaf(leaf) => match leaf {
                    tt::Leaf::Ident(ident) if ident.sym == sym::crate_ => {
                        // We simply produce identifier `$crate` here. And it will be resolved when lowering ast to Path.
                        Op::Ident(tt::Ident {
                            sym: sym::dollar_crate,
                            span: ident.span,
                            is_raw: tt::IdentIsRaw::No,
                        })
                    }
                    tt::Leaf::Ident(ident) => {
                        let kind = eat_fragment_kind(edition, src, mode)?;
                        let name = ident.sym.clone();
                        let id = ident.span;
                        Op::Var { name, kind, id }
                    }
                    tt::Leaf::Literal(lit) if is_boolean_literal(lit) => {
                        let kind = eat_fragment_kind(edition, src, mode)?;
                        let name = lit.symbol.clone();
                        let id = lit.span;
                        Op::Var { name, kind, id }
                    }
                    tt::Leaf::Punct(punct @ tt::Punct { char: '$', .. }) => match mode {
                        Mode::Pattern => {
                            return Err(ParseError::unexpected(
                                "`$$` is not allowed on the pattern side",
                            ));
                        }
                        Mode::Template => Op::Punct({
                            let mut res = ArrayVec::new();
                            res.push(*punct);
                            Box::new(res)
                        }),
                    },
                    tt::Leaf::Punct(_) | tt::Leaf::Literal(_) => {
                        return Err(ParseError::expected("expected ident"));
                    }
                },
            }
        }

        TtElement::Leaf(tt::Leaf::Literal(it)) => {
            src.next().expect("first token already peeked");
            Op::Literal(it.clone())
        }

        TtElement::Leaf(tt::Leaf::Ident(it)) => {
            src.next().expect("first token already peeked");
            Op::Ident(it.clone())
        }

        TtElement::Leaf(tt::Leaf::Punct(_)) => {
            // There's at least one punct so this shouldn't fail.
            let puncts = src.expect_glued_punct().unwrap();
            Op::Punct(Box::new(puncts))
        }

        TtElement::Subtree(subtree, subtree_iter) => {
            src.next().expect("first token already peeked");
            let tokens = MetaTemplate::parse(edition, subtree_iter, mode)?;
            Op::Subtree { tokens, delimiter: subtree.delimiter }
        }
    };
    Ok(res)
}

fn eat_fragment_kind(
    edition: impl Copy + Fn(SyntaxContext) -> Edition,
    src: &mut TtIter<'_, Span>,
    mode: Mode,
) -> Result<Option<MetaVarKind>, ParseError> {
    if let Mode::Pattern = mode {
        src.expect_char(':').map_err(|()| ParseError::unexpected("missing fragment specifier"))?;
        let ident = src
            .expect_ident()
            .map_err(|()| ParseError::unexpected("missing fragment specifier"))?;
        let kind = match ident.sym.as_str() {
            "path" => MetaVarKind::Path,
            "ty" => MetaVarKind::Ty,
            "pat" => {
                if edition(ident.span.ctx).at_least_2021() {
                    MetaVarKind::Pat
                } else {
                    MetaVarKind::PatParam
                }
            }
            "pat_param" => MetaVarKind::PatParam,
            "stmt" => MetaVarKind::Stmt,
            "block" => MetaVarKind::Block,
            "meta" => MetaVarKind::Meta,
            "item" => MetaVarKind::Item,
            "vis" => MetaVarKind::Vis,
            "expr" => {
                if edition(ident.span.ctx).at_least_2024() {
                    MetaVarKind::Expr(ExprKind::Expr)
                } else {
                    MetaVarKind::Expr(ExprKind::Expr2021)
                }
            }
            "expr_2021" => MetaVarKind::Expr(ExprKind::Expr2021),
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

fn is_boolean_literal(lit: &tt::Literal<Span>) -> bool {
    matches!(lit.symbol.as_str(), "true" | "false")
}

fn parse_repeat(src: &mut TtIter<'_, Span>) -> Result<(Option<Separator>, RepeatKind), ParseError> {
    let mut separator = Separator::Puncts(ArrayVec::new());
    for tt in src {
        let tt = match tt {
            TtElement::Leaf(leaf) => leaf,
            TtElement::Subtree(..) => return Err(ParseError::InvalidRepeat),
        };
        let has_sep = match &separator {
            Separator::Puncts(puncts) => !puncts.is_empty(),
            _ => true,
        };
        match tt {
            tt::Leaf::Ident(ident) => match separator {
                Separator::Puncts(puncts) if puncts.is_empty() => {
                    separator = Separator::Ident(ident.clone());
                }
                Separator::Puncts(puncts) => match puncts.as_slice() {
                    [tt::Punct { char: '\'', .. }] => {
                        separator = Separator::Lifetime(puncts[0], ident.clone());
                    }
                    _ => return Err(ParseError::InvalidRepeat),
                },
                _ => return Err(ParseError::InvalidRepeat),
            },
            tt::Leaf::Literal(_) if has_sep => return Err(ParseError::InvalidRepeat),
            tt::Leaf::Literal(lit) => separator = Separator::Literal(lit.clone()),
            tt::Leaf::Punct(punct) => {
                let repeat_kind = match punct.char {
                    '*' => RepeatKind::ZeroOrMore,
                    '+' => RepeatKind::OneOrMore,
                    '?' => RepeatKind::ZeroOrOne,
                    _ => match &mut separator {
                        Separator::Puncts(puncts) if puncts.len() < 3 => {
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

fn parse_metavar_expr(src: &mut TtIter<'_, Span>) -> Result<Op, ()> {
    let func = src.expect_ident()?;
    let (args, mut args_iter) = src.expect_subtree()?;

    if args.delimiter.kind != tt::DelimiterKind::Parenthesis {
        return Err(());
    }

    let op = match &func.sym {
        s if sym::ignore == *s => {
            args_iter.expect_dollar()?;
            let ident = args_iter.expect_ident()?;
            Op::Ignore { name: ident.sym.clone(), id: ident.span }
        }
        s if sym::index == *s => Op::Index { depth: parse_depth(&mut args_iter)? },
        s if sym::len == *s => Op::Len { depth: parse_depth(&mut args_iter)? },
        s if sym::count == *s => {
            args_iter.expect_dollar()?;
            let ident = args_iter.expect_ident()?;
            let depth = if try_eat_comma(&mut args_iter) {
                Some(parse_depth(&mut args_iter)?)
            } else {
                None
            };
            Op::Count { name: ident.sym.clone(), depth }
        }
        s if sym::concat == *s => {
            let mut elements = Vec::new();
            while let Some(next) = args_iter.peek() {
                let element = if let TtElement::Leaf(tt::Leaf::Literal(lit)) = next {
                    args_iter.next().expect("already peeked");
                    ConcatMetaVarExprElem::Literal(lit.clone())
                } else {
                    let is_var = try_eat_dollar(&mut args_iter);
                    let ident = args_iter.expect_ident_or_underscore()?.clone();

                    if is_var {
                        ConcatMetaVarExprElem::Var(ident)
                    } else {
                        ConcatMetaVarExprElem::Ident(ident)
                    }
                };
                elements.push(element);
                if !args_iter.is_empty() {
                    args_iter.expect_comma()?;
                }
            }
            if elements.len() < 2 {
                return Err(());
            }
            Op::Concat { elements: elements.into_boxed_slice(), span: func.span }
        }
        _ => return Err(()),
    };

    if args_iter.next().is_some() {
        return Err(());
    }

    Ok(op)
}

fn parse_depth(src: &mut TtIter<'_, Span>) -> Result<usize, ()> {
    if src.is_empty() {
        Ok(0)
    } else if let tt::Leaf::Literal(tt::Literal { symbol: text, suffix: None, .. }) =
        src.expect_literal()?
    {
        // Suffixes are not allowed.
        text.as_str().parse().map_err(|_| ())
    } else {
        Err(())
    }
}

fn try_eat_comma(src: &mut TtIter<'_, Span>) -> bool {
    if let Some(TtElement::Leaf(tt::Leaf::Punct(tt::Punct { char: ',', .. }))) = src.peek() {
        let _ = src.next();
        return true;
    }
    false
}

fn try_eat_dollar(src: &mut TtIter<'_, Span>) -> bool {
    if let Some(TtElement::Leaf(tt::Leaf::Punct(tt::Punct { char: '$', .. }))) = src.peek() {
        let _ = src.next();
        return true;
    }
    false
}
