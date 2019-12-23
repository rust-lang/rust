//! FIXME: write short doc here

use crate::{
    mbe_expander::{Binding, Bindings, Fragment},
    parser::{parse_pattern, Op, RepeatKind, Separator},
    subtree_source::SubtreeTokenSource,
    tt_iter::TtIter,
    ExpandError,
};

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

macro_rules! bail {
    ($($tt:tt)*) => {
        return Err(err!($($tt)*))
    };
}

pub(super) fn match_(pattern: &tt::Subtree, src: &tt::Subtree) -> Result<Bindings, ExpandError> {
    assert!(pattern.delimiter == None);

    let mut res = Bindings::default();
    let mut src = TtIter::new(src);

    match_subtree(&mut res, pattern, &mut src)?;

    if src.len() > 0 {
        bail!("leftover tokens");
    }

    Ok(res)
}

fn match_subtree(
    bindings: &mut Bindings,
    pattern: &tt::Subtree,
    src: &mut TtIter,
) -> Result<(), ExpandError> {
    for op in parse_pattern(pattern) {
        match op? {
            Op::TokenTree(tt::TokenTree::Leaf(lhs)) => {
                let rhs = src.expect_leaf().map_err(|()| err!("expected leaf: `{}`", lhs))?;
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
                    _ => Err(ExpandError::UnexpectedToken)?,
                }
            }
            Op::TokenTree(tt::TokenTree::Subtree(lhs)) => {
                let rhs = src.expect_subtree().map_err(|()| err!("expected subtree"))?;
                if lhs.delimiter_kind() != rhs.delimiter_kind() {
                    bail!("mismatched delimiter")
                }
                let mut src = TtIter::new(rhs);
                match_subtree(bindings, lhs, &mut src)?;
                if src.len() > 0 {
                    bail!("leftover tokens");
                }
            }
            Op::Var { name, kind } => {
                let kind = kind.as_ref().ok_or(ExpandError::UnexpectedToken)?;
                match match_meta_var(kind.as_str(), src)? {
                    Some(fragment) => {
                        bindings.inner.insert(name.clone(), Binding::Fragment(fragment));
                    }
                    None => bindings.push_optional(name),
                }
            }
            Op::Repeat { subtree, kind, separator } => {
                match_repeat(bindings, subtree, kind, separator, src)?
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
                Ok(rhs) => rhs.text == lhs.text,
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

    pub(crate) fn expect_lifetime(&mut self) -> Result<&tt::Ident, ()> {
        let ident = self.expect_ident()?;
        // check if it start from "`"
        if !ident.text.starts_with('\'') {
            return Err(());
        }
        Ok(ident)
    }

    pub(crate) fn expect_fragment(
        &mut self,
        fragment_kind: ra_parser::FragmentKind,
    ) -> Result<tt::TokenTree, ()> {
        pub(crate) struct OffsetTokenSink<'a> {
            pub(crate) cursor: Cursor<'a>,
            pub(crate) error: bool,
        }

        impl<'a> TreeSink for OffsetTokenSink<'a> {
            fn token(&mut self, _kind: SyntaxKind, n_tokens: u8) {
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

        let buffer = TokenBuffer::new(self.inner.as_slice());
        let mut src = SubtreeTokenSource::new(&buffer);
        let mut sink = OffsetTokenSink { cursor: buffer.begin(), error: false };

        ra_parser::parse_fragment(&mut src, &mut sink, fragment_kind);

        if !sink.cursor.is_root() || sink.error {
            return Err(());
        }

        let mut curr = buffer.begin();
        let mut res = vec![];

        while curr != sink.cursor {
            if let Some(token) = curr.token_tree() {
                res.push(token);
            }
            curr = curr.bump();
        }
        self.inner = self.inner.as_slice()[res.len()..].iter();
        match res.len() {
            0 => Err(()),
            1 => Ok(res[0].clone()),
            _ => Ok(tt::TokenTree::Subtree(tt::Subtree {
                delimiter: None,
                token_trees: res.into_iter().cloned().collect(),
            })),
        }
    }

    pub(crate) fn eat_vis(&mut self) -> Option<tt::TokenTree> {
        let mut fork = self.clone();
        match fork.expect_fragment(Visibility) {
            Ok(tt) => {
                *self = fork;
                Some(tt)
            }
            Err(()) => None,
        }
    }
}

pub(super) fn match_repeat(
    bindings: &mut Bindings,
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

        let mut nested = Bindings::default();
        match match_subtree(&mut nested, pattern, &mut fork) {
            Ok(()) => {
                limit -= 1;
                if limit == 0 {
                    log::warn!("match_lhs excced in repeat pattern exceed limit => {:#?}\n{:#?}\n{:#?}\n{:#?}", pattern, src, kind, separator);
                    break;
                }
                *src = fork;

                bindings.push_nested(counter, nested)?;
                counter += 1;
                if counter == 1 {
                    if let RepeatKind::ZeroOrOne = kind {
                        break;
                    }
                }
            }
            Err(_) => break,
        }
    }

    match (kind, counter) {
        (RepeatKind::OneOrMore, 0) => return Err(ExpandError::UnexpectedToken),
        (_, 0) => {
            // Collect all empty variables in subtrees
            let mut vars = Vec::new();
            collect_vars(&mut vars, pattern)?;
            for var in vars {
                bindings.push_empty(&var)
            }
        }
        _ => (),
    }
    Ok(())
}

fn match_meta_var(kind: &str, input: &mut TtIter) -> Result<Option<Fragment>, ExpandError> {
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
            let tt = match kind {
                "ident" => {
                    let ident = input.expect_ident().map_err(|()| err!("expected ident"))?.clone();
                    tt::Leaf::from(ident).into()
                }
                "tt" => input.next().ok_or_else(|| err!())?.clone(),
                "lifetime" => {
                    let ident = input.expect_lifetime().map_err(|()| err!())?;
                    tt::Leaf::Ident(ident.clone()).into()
                }
                "literal" => {
                    let literal = input.expect_literal().map_err(|()| err!())?.clone();
                    tt::Leaf::from(literal).into()
                }
                // `vis` is optional
                "vis" => match input.eat_vis() {
                    Some(vis) => vis,
                    None => return Ok(None),
                },
                _ => return Err(ExpandError::UnexpectedToken),
            };
            return Ok(Some(Fragment::Tokens(tt)));
        }
    };
    let tt = input.expect_fragment(fragment).map_err(|()| err!())?;
    let fragment = if kind == "expr" { Fragment::Ast(tt) } else { Fragment::Tokens(tt) };
    Ok(Some(fragment))
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
