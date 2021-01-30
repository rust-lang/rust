//! FIXME: write short doc here

use crate::{
    expander::{Binding, Bindings, Fragment},
    parser::{Op, RepeatKind, Separator},
    subtree_source::SubtreeTokenSource,
    tt_iter::TtIter,
    ExpandError, MetaTemplate,
};

use super::ExpandResult;
use parser::{FragmentKind::*, TreeSink};
use syntax::{SmolStr, SyntaxKind};
use tt::buffer::{Cursor, TokenBuffer};

impl Bindings {
    fn push_optional(&mut self, name: &SmolStr) {
        // FIXME: Do we have a better way to represent an empty token ?
        // Insert an empty subtree for empty token
        let tt = tt::Subtree::default().into();
        self.inner.insert(name.clone(), Binding::Fragment(Fragment::Tokens(tt)));
    }

    fn push_empty(&mut self, name: &SmolStr) {
        self.inner.insert(name.clone(), Binding::Empty);
    }

    fn push_nested(&mut self, idx: usize, nested: Bindings) -> Result<(), ExpandError> {
        for (key, value) in nested.inner {
            if !self.inner.contains_key(&key) {
                self.inner.insert(key.clone(), Binding::Nested(Vec::new()));
            }
            match self.inner.get_mut(&key) {
                Some(Binding::Nested(it)) => {
                    // insert empty nested bindings before this one
                    while it.len() < idx {
                        it.push(Binding::Nested(vec![]));
                    }
                    it.push(value);
                }
                _ => {
                    return Err(ExpandError::BindingError(format!(
                        "could not find binding `{}`",
                        key
                    )));
                }
            }
        }
        Ok(())
    }
}

macro_rules! err {
    () => {
        ExpandError::BindingError(format!(""))
    };
    ($($tt:tt)*) => {
        ExpandError::BindingError(format!($($tt)*))
    };
}

#[derive(Debug, Default)]
pub(super) struct Match {
    pub(super) bindings: Bindings,
    /// We currently just keep the first error and count the rest to compare matches.
    pub(super) err: Option<ExpandError>,
    pub(super) err_count: usize,
    /// How many top-level token trees were left to match.
    pub(super) unmatched_tts: usize,
}

impl Match {
    fn add_err(&mut self, err: ExpandError) {
        let prev_err = self.err.take();
        self.err = prev_err.or(Some(err));
        self.err_count += 1;
    }
}

/// Matching errors are added to the `Match`.
pub(super) fn match_(pattern: &MetaTemplate, src: &tt::Subtree) -> Match {
    let mut res = Match::default();
    let mut src = TtIter::new(src);

    match_tokens(&mut res, pattern, &mut src);

    if src.len() > 0 {
        res.unmatched_tts += src.len();
        res.add_err(err!("leftover tokens"));
    }

    res
}

fn match_tokens(res: &mut Match, pattern: &MetaTemplate, src: &mut TtIter) {
    for op in pattern.iter() {
        match op {
            Op::Leaf(lhs) => {
                if let Err(err) = match_leaf(lhs, src) {
                    res.add_err(err);
                    continue;
                }
            }
            Op::Subtree { tokens, delimiter: delim } => {
                let rhs = match src.expect_subtree() {
                    Ok(s) => s,
                    Err(()) => {
                        res.add_err(err!("expected subtree"));
                        continue;
                    }
                };
                if delim.map(|it| it.kind) != rhs.delimiter_kind() {
                    res.add_err(err!("mismatched delimiter"));
                    continue;
                }
                let mut src = TtIter::new(rhs);
                match_tokens(res, tokens, &mut src);
                if src.len() > 0 {
                    res.add_err(err!("leftover tokens"));
                }
            }
            Op::Var { name, kind, .. } => {
                let kind = match kind {
                    Some(k) => k,
                    None => {
                        res.add_err(ExpandError::UnexpectedToken);
                        continue;
                    }
                };
                let ExpandResult { value: matched, err: match_err } =
                    match_meta_var(kind.as_str(), src);
                match matched {
                    Some(fragment) => {
                        res.bindings.inner.insert(name.clone(), Binding::Fragment(fragment));
                    }
                    None if match_err.is_none() => res.bindings.push_optional(name),
                    _ => {}
                }
                if let Some(err) = match_err {
                    res.add_err(err);
                }
            }
            Op::Repeat { tokens: subtree, kind, separator } => {
                match_repeat(res, subtree, *kind, separator, src);
            }
        }
    }
}

fn match_leaf(lhs: &tt::Leaf, src: &mut TtIter) -> Result<(), ExpandError> {
    let rhs = match src.expect_leaf() {
        Ok(l) => l,
        Err(()) => {
            return Err(err!("expected leaf: `{}`", lhs));
        }
    };
    match (lhs, rhs) {
        (
            tt::Leaf::Punct(tt::Punct { char: lhs, .. }),
            tt::Leaf::Punct(tt::Punct { char: rhs, .. }),
        ) if lhs == rhs => (),
        (
            tt::Leaf::Ident(tt::Ident { text: lhs, .. }),
            tt::Leaf::Ident(tt::Ident { text: rhs, .. }),
        ) if lhs == rhs => (),
        (
            tt::Leaf::Literal(tt::Literal { text: lhs, .. }),
            tt::Leaf::Literal(tt::Literal { text: rhs, .. }),
        ) if lhs == rhs => (),
        _ => {
            return Err(ExpandError::UnexpectedToken);
        }
    }

    Ok(())
}

fn match_repeat(
    res: &mut Match,
    pattern: &MetaTemplate,
    kind: RepeatKind,
    separator: &Option<Separator>,
    src: &mut TtIter,
) {
    // Dirty hack to make macro-expansion terminate.
    // This should be replaced by a proper macro-by-example implementation
    let mut limit = 65536;
    let mut counter = 0;

    for i in 0.. {
        let mut fork = src.clone();

        if let Some(separator) = &separator {
            if i != 0 && !fork.eat_separator(separator) {
                break;
            }
        }

        let mut nested = Match::default();
        match_tokens(&mut nested, pattern, &mut fork);
        if nested.err.is_none() {
            limit -= 1;
            if limit == 0 {
                log::warn!(
                    "match_lhs exceeded repeat pattern limit => {:#?}\n{:#?}\n{:#?}\n{:#?}",
                    pattern,
                    src,
                    kind,
                    separator
                );
                break;
            }
            *src = fork;

            if let Err(err) = res.bindings.push_nested(counter, nested.bindings) {
                res.add_err(err);
            }
            counter += 1;
            if counter == 1 {
                if let RepeatKind::ZeroOrOne = kind {
                    break;
                }
            }
        } else {
            break;
        }
    }

    match (kind, counter) {
        (RepeatKind::OneOrMore, 0) => {
            res.add_err(ExpandError::UnexpectedToken);
        }
        (_, 0) => {
            // Collect all empty variables in subtrees
            let mut vars = Vec::new();
            collect_vars(&mut vars, pattern);
            for var in vars {
                res.bindings.push_empty(&var)
            }
        }
        _ => (),
    }
}

fn match_meta_var(kind: &str, input: &mut TtIter) -> ExpandResult<Option<Fragment>> {
    let fragment = match kind {
        "path" => Path,
        "expr" => Expr,
        "ty" => Type,
        "pat" => Pattern,
        "stmt" => Statement,
        "block" => Block,
        "meta" => MetaItem,
        "item" => Item,
        _ => {
            let tt_result = match kind {
                "ident" => input
                    .expect_ident()
                    .map(|ident| Some(tt::Leaf::from(ident.clone()).into()))
                    .map_err(|()| err!("expected ident")),
                "tt" => input.expect_tt().map(Some).map_err(|()| err!()),
                "lifetime" => input
                    .expect_lifetime()
                    .map(|tt| Some(tt))
                    .map_err(|()| err!("expected lifetime")),
                "literal" => {
                    let neg = input.eat_char('-');
                    input
                        .expect_literal()
                        .map(|literal| {
                            let lit = tt::Leaf::from(literal.clone());
                            match neg {
                                None => Some(lit.into()),
                                Some(neg) => Some(tt::TokenTree::Subtree(tt::Subtree {
                                    delimiter: None,
                                    token_trees: vec![neg, lit.into()],
                                })),
                            }
                        })
                        .map_err(|()| err!())
                }
                // `vis` is optional
                "vis" => match input.eat_vis() {
                    Some(vis) => Ok(Some(vis)),
                    None => Ok(None),
                },
                _ => Err(ExpandError::UnexpectedToken),
            };
            return tt_result.map(|it| it.map(Fragment::Tokens)).into();
        }
    };
    let result = input.expect_fragment(fragment);
    result.map(|tt| if kind == "expr" { tt.map(Fragment::Ast) } else { tt.map(Fragment::Tokens) })
}

fn collect_vars(buf: &mut Vec<SmolStr>, pattern: &MetaTemplate) {
    for op in pattern.iter() {
        match op {
            Op::Var { name, .. } => buf.push(name.clone()),
            Op::Leaf(_) => (),
            Op::Subtree { tokens, .. } => collect_vars(buf, tokens),
            Op::Repeat { tokens, .. } => collect_vars(buf, tokens),
        }
    }
}

impl<'a> TtIter<'a> {
    fn eat_separator(&mut self, separator: &Separator) -> bool {
        let mut fork = self.clone();
        let ok = match separator {
            Separator::Ident(lhs) => match fork.expect_ident() {
                Ok(rhs) => rhs.text == lhs.text,
                _ => false,
            },
            Separator::Literal(lhs) => match fork.expect_literal() {
                Ok(rhs) => match rhs {
                    tt::Leaf::Literal(rhs) => rhs.text == lhs.text,
                    tt::Leaf::Ident(rhs) => rhs.text == lhs.text,
                    tt::Leaf::Punct(_) => false,
                },
                _ => false,
            },
            Separator::Puncts(lhss) => lhss.iter().all(|lhs| match fork.expect_punct() {
                Ok(rhs) => rhs.char == lhs.char,
                _ => false,
            }),
        };
        if ok {
            *self = fork;
        }
        ok
    }

    fn expect_tt(&mut self) -> Result<tt::TokenTree, ()> {
        match self.peek_n(0) {
            Some(tt::TokenTree::Leaf(tt::Leaf::Punct(punct))) if punct.char == '\'' => {
                return self.expect_lifetime();
            }
            _ => (),
        }

        let tt = self.next().ok_or_else(|| ())?.clone();
        let punct = match tt {
            tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) if punct.spacing == tt::Spacing::Joint => {
                punct
            }
            _ => return Ok(tt),
        };

        let (second, third) = match (self.peek_n(0), self.peek_n(1)) {
            (
                Some(tt::TokenTree::Leaf(tt::Leaf::Punct(p2))),
                Some(tt::TokenTree::Leaf(tt::Leaf::Punct(p3))),
            ) if p2.spacing == tt::Spacing::Joint => (p2.char, Some(p3.char)),
            (Some(tt::TokenTree::Leaf(tt::Leaf::Punct(p2))), _) => (p2.char, None),
            _ => return Ok(tt),
        };

        match (punct.char, second, third) {
            ('.', '.', Some('.'))
            | ('.', '.', Some('='))
            | ('<', '<', Some('='))
            | ('>', '>', Some('=')) => {
                let tt2 = self.next().unwrap().clone();
                let tt3 = self.next().unwrap().clone();
                Ok(tt::Subtree { delimiter: None, token_trees: vec![tt, tt2, tt3] }.into())
            }
            ('-', '=', _)
            | ('-', '>', _)
            | (':', ':', _)
            | ('!', '=', _)
            | ('.', '.', _)
            | ('*', '=', _)
            | ('/', '=', _)
            | ('&', '&', _)
            | ('&', '=', _)
            | ('%', '=', _)
            | ('^', '=', _)
            | ('+', '=', _)
            | ('<', '<', _)
            | ('<', '=', _)
            | ('=', '=', _)
            | ('=', '>', _)
            | ('>', '=', _)
            | ('>', '>', _)
            | ('|', '=', _)
            | ('|', '|', _) => {
                let tt2 = self.next().unwrap().clone();
                Ok(tt::Subtree { delimiter: None, token_trees: vec![tt, tt2] }.into())
            }
            _ => Ok(tt),
        }
    }

    fn expect_lifetime(&mut self) -> Result<tt::TokenTree, ()> {
        let punct = self.expect_punct()?;
        if punct.char != '\'' {
            return Err(());
        }
        let ident = self.expect_ident()?;

        Ok(tt::Subtree {
            delimiter: None,
            token_trees: vec![
                tt::Leaf::Punct(*punct).into(),
                tt::Leaf::Ident(ident.clone()).into(),
            ],
        }
        .into())
    }

    fn expect_fragment(
        &mut self,
        fragment_kind: parser::FragmentKind,
    ) -> ExpandResult<Option<tt::TokenTree>> {
        struct OffsetTokenSink<'a> {
            cursor: Cursor<'a>,
            error: bool,
        }

        impl<'a> TreeSink for OffsetTokenSink<'a> {
            fn token(&mut self, kind: SyntaxKind, mut n_tokens: u8) {
                if kind == SyntaxKind::LIFETIME_IDENT {
                    n_tokens = 2;
                }
                for _ in 0..n_tokens {
                    self.cursor = self.cursor.bump_subtree();
                }
            }
            fn start_node(&mut self, _kind: SyntaxKind) {}
            fn finish_node(&mut self) {}
            fn error(&mut self, _error: parser::ParseError) {
                self.error = true;
            }
        }

        let buffer = TokenBuffer::from_tokens(&self.inner.as_slice());
        let mut src = SubtreeTokenSource::new(&buffer);
        let mut sink = OffsetTokenSink { cursor: buffer.begin(), error: false };

        parser::parse_fragment(&mut src, &mut sink, fragment_kind);

        let mut err = None;
        if !sink.cursor.is_root() || sink.error {
            err = Some(err!("expected {:?}", fragment_kind));
        }

        let mut curr = buffer.begin();
        let mut res = vec![];

        if sink.cursor.is_root() {
            while curr != sink.cursor {
                if let Some(token) = curr.token_tree() {
                    res.push(token);
                }
                curr = curr.bump();
            }
        }
        self.inner = self.inner.as_slice()[res.len()..].iter();
        if res.len() == 0 && err.is_none() {
            err = Some(err!("no tokens consumed"));
        }
        let res = match res.len() {
            1 => Some(res[0].cloned()),
            0 => None,
            _ => Some(tt::TokenTree::Subtree(tt::Subtree {
                delimiter: None,
                token_trees: res.into_iter().map(|it| it.cloned()).collect(),
            })),
        };
        ExpandResult { value: res, err }
    }

    fn eat_vis(&mut self) -> Option<tt::TokenTree> {
        let mut fork = self.clone();
        match fork.expect_fragment(Visibility) {
            ExpandResult { value: tt, err: None } => {
                *self = fork;
                tt
            }
            ExpandResult { value: _, err: Some(_) } => None,
        }
    }

    fn eat_char(&mut self, c: char) -> Option<tt::TokenTree> {
        let mut fork = self.clone();
        match fork.expect_char(c) {
            Ok(_) => {
                let tt = self.next().cloned();
                *self = fork;
                tt
            }
            Err(_) => None,
        }
    }
}
