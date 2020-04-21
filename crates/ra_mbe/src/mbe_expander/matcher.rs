//! FIXME: write short doc here

use crate::{
    mbe_expander::{Binding, Bindings, Fragment},
    parser::{parse_pattern, Op, RepeatKind, Separator},
    subtree_source::SubtreeTokenSource,
    tt_iter::TtIter,
    ExpandError,
};

use super::ExpandResult;
use ra_parser::{FragmentKind::*, TreeSink};
use ra_syntax::{SmolStr, SyntaxKind};
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
    pub bindings: Bindings,
    /// We currently just keep the first error and count the rest to compare matches.
    pub err: Option<ExpandError>,
    pub err_count: usize,
    /// How many top-level token trees were left to match.
    pub unmatched_tts: usize,
}

impl Match {
    pub fn add_err(&mut self, err: ExpandError) {
        let prev_err = self.err.take();
        self.err = prev_err.or(Some(err));
        self.err_count += 1;
    }
}

// General note: These functions have two channels to return errors, a `Result`
// return value and the `&mut Match`. The returned Result is for pattern parsing
// errors; if a branch of the macro definition doesn't parse, it doesn't make
// sense to try using it. Matching errors are added to the `Match`. It might
// make sense to make pattern parsing a separate step?

pub(super) fn match_(pattern: &tt::Subtree, src: &tt::Subtree) -> Result<Match, ExpandError> {
    assert!(pattern.delimiter == None);

    let mut res = Match::default();
    let mut src = TtIter::new(src);

    match_subtree(&mut res, pattern, &mut src)?;

    if src.len() > 0 {
        res.unmatched_tts += src.len();
        res.add_err(err!("leftover tokens"));
    }

    Ok(res)
}

fn match_subtree(
    res: &mut Match,
    pattern: &tt::Subtree,
    src: &mut TtIter,
) -> Result<(), ExpandError> {
    for op in parse_pattern(pattern) {
        match op? {
            Op::TokenTree(tt::TokenTree::Leaf(lhs)) => {
                let rhs = match src.expect_leaf() {
                    Ok(l) => l,
                    Err(()) => {
                        res.add_err(err!("expected leaf: `{}`", lhs));
                        continue;
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
                        res.add_err(ExpandError::UnexpectedToken);
                    }
                }
            }
            Op::TokenTree(tt::TokenTree::Subtree(lhs)) => {
                let rhs = match src.expect_subtree() {
                    Ok(s) => s,
                    Err(()) => {
                        res.add_err(err!("expected subtree"));
                        continue;
                    }
                };
                if lhs.delimiter_kind() != rhs.delimiter_kind() {
                    res.add_err(err!("mismatched delimiter"));
                    continue;
                }
                let mut src = TtIter::new(rhs);
                match_subtree(res, lhs, &mut src)?;
                if src.len() > 0 {
                    res.add_err(err!("leftover tokens"));
                }
            }
            Op::Var { name, kind } => {
                let kind = match kind {
                    Some(k) => k,
                    None => {
                        res.add_err(ExpandError::UnexpectedToken);
                        continue;
                    }
                };
                let ExpandResult(matched, match_err) = match_meta_var(kind.as_str(), src);
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
            Op::Repeat { subtree, kind, separator } => {
                match_repeat(res, subtree, kind, separator, src)?;
            }
        }
    }
    Ok(())
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

    pub(crate) fn expect_tt(&mut self) -> Result<tt::TokenTree, ()> {
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
            ('-', '=', None)
            | ('-', '>', None)
            | (':', ':', None)
            | ('!', '=', None)
            | ('.', '.', None)
            | ('*', '=', None)
            | ('/', '=', None)
            | ('&', '&', None)
            | ('&', '=', None)
            | ('%', '=', None)
            | ('^', '=', None)
            | ('+', '=', None)
            | ('<', '<', None)
            | ('<', '=', None)
            | ('=', '=', None)
            | ('=', '>', None)
            | ('>', '=', None)
            | ('>', '>', None)
            | ('|', '=', None)
            | ('|', '|', None) => {
                let tt2 = self.next().unwrap().clone();
                Ok(tt::Subtree { delimiter: None, token_trees: vec![tt.clone(), tt2] }.into())
            }
            _ => Ok(tt),
        }
    }

    pub(crate) fn expect_lifetime(&mut self) -> Result<tt::TokenTree, ()> {
        let punct = self.expect_punct()?;
        if punct.char != '\'' {
            return Err(());
        }
        let ident = self.expect_ident()?;

        Ok(tt::Subtree {
            delimiter: None,
            token_trees: vec![
                tt::Leaf::Punct(punct.clone()).into(),
                tt::Leaf::Ident(ident.clone()).into(),
            ],
        }
        .into())
    }

    pub(crate) fn expect_fragment(
        &mut self,
        fragment_kind: ra_parser::FragmentKind,
    ) -> ExpandResult<Option<tt::TokenTree>> {
        pub(crate) struct OffsetTokenSink<'a> {
            pub(crate) cursor: Cursor<'a>,
            pub(crate) error: bool,
        }

        impl<'a> TreeSink for OffsetTokenSink<'a> {
            fn token(&mut self, kind: SyntaxKind, mut n_tokens: u8) {
                if kind == SyntaxKind::LIFETIME {
                    n_tokens = 2;
                }
                for _ in 0..n_tokens {
                    self.cursor = self.cursor.bump_subtree();
                }
            }
            fn start_node(&mut self, _kind: SyntaxKind) {}
            fn finish_node(&mut self) {}
            fn error(&mut self, _error: ra_parser::ParseError) {
                self.error = true;
            }
        }

        let buffer = TokenBuffer::new(&self.inner.as_slice());
        let mut src = SubtreeTokenSource::new(&buffer);
        let mut sink = OffsetTokenSink { cursor: buffer.begin(), error: false };

        ra_parser::parse_fragment(&mut src, &mut sink, fragment_kind);

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
            1 => Some(res[0].clone()),
            0 => None,
            _ => Some(tt::TokenTree::Subtree(tt::Subtree {
                delimiter: None,
                token_trees: res.into_iter().cloned().collect(),
            })),
        };
        ExpandResult(res, err)
    }

    pub(crate) fn eat_vis(&mut self) -> Option<tt::TokenTree> {
        let mut fork = self.clone();
        match fork.expect_fragment(Visibility) {
            ExpandResult(tt, None) => {
                *self = fork;
                tt
            }
            ExpandResult(_, Some(_)) => None,
        }
    }
}

pub(super) fn match_repeat(
    res: &mut Match,
    pattern: &tt::Subtree,
    kind: RepeatKind,
    separator: Option<Separator>,
    src: &mut TtIter,
) -> Result<(), ExpandError> {
    // Dirty hack to make macro-expansion terminate.
    // This should be replaced by a propper macro-by-example implementation
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
        match_subtree(&mut nested, pattern, &mut fork)?;
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
            collect_vars(&mut vars, pattern)?;
            for var in vars {
                res.bindings.push_empty(&var)
            }
        }
        _ => (),
    }
    Ok(())
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
                "literal" => input
                    .expect_literal()
                    .map(|literal| Some(tt::Leaf::from(literal.clone()).into()))
                    .map_err(|()| err!()),
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

fn collect_vars(buf: &mut Vec<SmolStr>, pattern: &tt::Subtree) -> Result<(), ExpandError> {
    for op in parse_pattern(pattern) {
        match op? {
            Op::Var { name, .. } => buf.push(name.clone()),
            Op::TokenTree(tt::TokenTree::Leaf(_)) => (),
            Op::TokenTree(tt::TokenTree::Subtree(subtree)) => collect_vars(buf, subtree)?,
            Op::Repeat { subtree, .. } => collect_vars(buf, subtree)?,
        }
    }
    Ok(())
}
