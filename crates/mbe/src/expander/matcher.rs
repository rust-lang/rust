//! An NFA-based parser, which is porting from rustc mbe parsing code
//!
//! See https://github.com/rust-lang/rust/blob/70b18bc2cbac4712020019f5bf57c00905373205/compiler/rustc_expand/src/mbe/macro_parser.rs
//! Here is a quick intro to how the parser works, copied from rustc:
//!
//! A 'position' is a dot in the middle of a matcher, usually represented as a
//! dot. For example `· a $( a )* a b` is a position, as is `a $( · a )* a b`.
//!
//! The parser walks through the input a character at a time, maintaining a list
//! of threads consistent with the current position in the input string: `cur_items`.
//!
//! As it processes them, it fills up `eof_items` with threads that would be valid if
//! the macro invocation is now over, `bb_items` with threads that are waiting on
//! a Rust non-terminal like `$e:expr`, and `next_items` with threads that are waiting
//! on a particular token. Most of the logic concerns moving the · through the
//! repetitions indicated by Kleene stars. The rules for moving the · without
//! consuming any input are called epsilon transitions. It only advances or calls
//! out to the real Rust parser when no `cur_items` threads remain.
//!
//! Example:
//!
//! ```text, ignore
//! Start parsing a a a a b against [· a $( a )* a b].
//!
//! Remaining input: a a a a b
//! next: [· a $( a )* a b]
//!
//! - - - Advance over an a. - - -
//!
//! Remaining input: a a a b
//! cur: [a · $( a )* a b]
//! Descend/Skip (first item).
//! next: [a $( · a )* a b]  [a $( a )* · a b].
//!
//! - - - Advance over an a. - - -
//!
//! Remaining input: a a b
//! cur: [a $( a · )* a b]  [a $( a )* a · b]
//! Follow epsilon transition: Finish/Repeat (first item)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over an a. - - - (this looks exactly like the last step)
//!
//! Remaining input: a b
//! cur: [a $( a · )* a b]  [a $( a )* a · b]
//! Follow epsilon transition: Finish/Repeat (first item)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over an a. - - - (this looks exactly like the last step)
//!
//! Remaining input: b
//! cur: [a $( a · )* a b]  [a $( a )* a · b]
//! Follow epsilon transition: Finish/Repeat (first item)
//! next: [a $( a )* · a b]  [a $( · a )* a b]  [a $( a )* a · b]
//!
//! - - - Advance over a b. - - -
//!
//! Remaining input: ''
//! eof: [a $( a )* a b ·]
//! ```

use crate::{
    expander::{Binding, Bindings, Fragment},
    parser::{Op, OpDelimited, OpDelimitedIter, RepeatKind, Separator},
    tt_iter::TtIter,
    ExpandError, MetaTemplate,
};

use super::ExpandResult;
use parser::FragmentKind::*;
use smallvec::{smallvec, SmallVec};
use syntax::SmolStr;

impl Bindings {
    fn push_optional(&mut self, name: &SmolStr) {
        // FIXME: Do we have a better way to represent an empty token ?
        // Insert an empty subtree for empty token
        let tt = tt::Subtree::default().into();
        self.inner.push((name.clone(), Binding::Fragment(Fragment::Tokens(tt))));
    }

    fn push_empty(&mut self, name: &SmolStr) {
        self.inner.push((name.clone(), Binding::Empty));
    }

    fn push_nested(&mut self, idx: usize, nested: Bindings) -> Result<(), ExpandError> {
        for (key, value) in nested.inner {
            if self.get_mut(&key).is_none() {
                self.inner.push((key.clone(), Binding::Nested(Vec::new())));
            }
            match self.get_mut(&key) {
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

    fn get_mut(&mut self, name: &str) -> Option<&mut Binding> {
        self.inner.iter_mut().find_map(|(n, b)| if n == name { Some(b) } else { None })
    }

    fn bindings(&self) -> impl Iterator<Item = &Binding> {
        self.inner.iter().map(|(_, b)| b)
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Match {
    pub(super) bindings: Bindings,
    /// We currently just keep the first error and count the rest to compare matches.
    pub(super) err: Option<ExpandError>,
    pub(super) err_count: usize,
    /// How many top-level token trees were left to match.
    pub(super) unmatched_tts: usize,
    /// The number of bound variables
    pub(super) bound_count: usize,
}

impl Match {
    fn add_err(&mut self, err: ExpandError) {
        let prev_err = self.err.take();
        self.err = prev_err.or(Some(err));
        self.err_count += 1;
    }
}

/// Matching errors are added to the `Match`.
pub(super) fn match_(pattern: &MetaTemplate, input: &tt::Subtree) -> Match {
    let mut res = match_loop(pattern, &input);
    res.bound_count = count(res.bindings.bindings());
    return res;

    fn count<'a>(bindings: impl Iterator<Item = &'a Binding>) -> usize {
        bindings
            .map(|it| match it {
                Binding::Fragment(_) => 1,
                Binding::Empty => 1,
                Binding::Nested(it) => count(it.iter()),
            })
            .sum()
    }
}

#[derive(Debug, Clone)]
struct MatchState<'t> {
    /// The position of the "dot" in this matcher
    dot: OpDelimitedIter<'t>,

    /// Token subtree stack
    /// When matching against matchers with nested delimited submatchers (e.g., `pat ( pat ( .. )
    /// pat ) pat`), we need to keep track of the matchers we are descending into. This stack does
    /// that where the bottom of the stack is the outermost matcher.
    stack: SmallVec<[OpDelimitedIter<'t>; 4]>,

    /// The "parent" matcher position if we are in a repetition. That is, the matcher position just
    /// before we enter the repetition.
    up: Option<Box<MatchState<'t>>>,

    /// The separator if we are in a repetition.
    sep: Option<Separator>,

    /// The KleeneOp of this sequence if we are in a repetition.
    sep_kind: Option<RepeatKind>,

    /// Number of tokens of seperator parsed
    sep_parsed: Option<usize>,

    /// Matched meta variables bindings
    bindings: SmallVec<[Bindings; 4]>,

    /// Cached result of meta variable parsing
    meta_result: Option<(TtIter<'t>, ExpandResult<Option<Fragment>>)>,

    /// Is error occuried in this state, will `poised` to "parent"
    is_error: bool,
}

/// Process the matcher positions of `cur_items` until it is empty. In the process, this will
/// produce more items in `next_items`, `eof_items`, and `bb_items`.
///
/// For more info about the how this happens, see the module-level doc comments and the inline
/// comments of this function.
///
/// # Parameters
///
/// - `src`: the current token of the parser.
/// - `stack`: the "parent" frames of the token tree
/// - `res`: the match result to store errors
/// - `cur_items`: the set of current items to be processed. This should be empty by the end of a
///   successful execution of this function.
/// - `next_items`: the set of newly generated items. These are used to replenish `cur_items` in
///   the function `parse`.
/// - `eof_items`: the set of items that would be valid if this was the EOF.
/// - `bb_items`: the set of items that are waiting for the black-box parser.
/// - `error_items`: the set of items in errors, used for error-resilient parsing
fn match_loop_inner<'t>(
    src: TtIter<'t>,
    stack: &[TtIter<'t>],
    res: &mut Match,
    cur_items: &mut SmallVec<[MatchState<'t>; 1]>,
    bb_items: &mut SmallVec<[MatchState<'t>; 1]>,
    next_items: &mut Vec<MatchState<'t>>,
    eof_items: &mut SmallVec<[MatchState<'t>; 1]>,
    error_items: &mut SmallVec<[MatchState<'t>; 1]>,
) {
    macro_rules! try_push {
        ($items: expr, $it:expr) => {
            if $it.is_error {
                error_items.push($it);
            } else {
                $items.push($it);
            }
        };
    }

    while let Some(mut item) = cur_items.pop() {
        while item.dot.is_eof() {
            match item.stack.pop() {
                Some(frame) => {
                    item.dot = frame;
                    item.dot.next();
                }
                None => break,
            }
        }
        let op = match item.dot.peek() {
            None => {
                // We are at or past the end of the matcher of `item`.
                if item.up.is_some() {
                    if item.sep_parsed.is_none() {
                        // Get the `up` matcher
                        let mut new_pos = *item.up.clone().unwrap();
                        // Add matches from this repetition to the `matches` of `up`
                        if let Some(bindings) = new_pos.bindings.last_mut() {
                            for (i, b) in item.bindings.iter_mut().enumerate() {
                                bindings.push_nested(i, b.clone()).unwrap();
                            }
                        }
                        // Move the "dot" past the repetition in `up`
                        new_pos.dot.next();
                        new_pos.is_error = new_pos.is_error || item.is_error;
                        cur_items.push(new_pos);
                    }

                    // Check if we need a separator.
                    // We check the separator one by one
                    let sep_idx = *item.sep_parsed.as_ref().unwrap_or(&0);
                    let sep_len = item.sep.as_ref().map_or(0, Separator::tt_count);
                    if item.sep.is_some() && sep_idx != sep_len {
                        let sep = item.sep.as_ref().unwrap();
                        if src.clone().expect_separator(&sep, sep_idx) {
                            item.dot.next();
                            item.sep_parsed = Some(sep_idx + 1);
                            try_push!(next_items, item);
                        }
                    }
                    // We don't need a separator. Move the "dot" back to the beginning of the matcher
                    // and try to match again UNLESS we are only allowed to have _one_ repetition.
                    else if item.sep_kind != Some(RepeatKind::ZeroOrOne) {
                        item.dot = item.dot.reset();
                        item.sep_parsed = None;
                        item.bindings.push(Bindings::default());
                        cur_items.push(item);
                    }
                } else {
                    // If we are not in a repetition, then being at the end of a matcher means that we have
                    // reached the potential end of the input.
                    try_push!(eof_items, item);
                }
                continue;
            }
            Some(it) => it,
        };

        // We are in the middle of a matcher.
        match op {
            OpDelimited::Op(Op::Repeat { tokens, kind, separator }) => {
                if matches!(kind, RepeatKind::ZeroOrMore | RepeatKind::ZeroOrOne) {
                    let mut new_item = item.clone();
                    new_item.dot.next();
                    let mut vars = Vec::new();
                    let bindings = new_item.bindings.last_mut().unwrap();
                    collect_vars(&mut vars, tokens);
                    for var in vars {
                        bindings.push_empty(&var);
                    }
                    cur_items.push(new_item);
                }
                cur_items.push(MatchState {
                    dot: tokens.iter_delimited(None),
                    stack: Default::default(),
                    up: Some(Box::new(item)),
                    sep: separator.clone(),
                    sep_kind: Some(*kind),
                    sep_parsed: None,
                    bindings: smallvec![Bindings::default()],
                    meta_result: None,
                    is_error: false,
                })
            }
            OpDelimited::Op(Op::Subtree { tokens, delimiter }) => {
                if let Ok(subtree) = src.clone().expect_subtree() {
                    if subtree.delimiter_kind() == delimiter.map(|it| it.kind) {
                        item.stack.push(item.dot);
                        item.dot = tokens.iter_delimited(delimiter.as_ref());
                        cur_items.push(item);
                    }
                }
            }
            OpDelimited::Op(Op::Var { kind, name, .. }) => {
                if let Some(kind) = kind {
                    let mut fork = src.clone();
                    let match_res = match_meta_var(kind.as_str(), &mut fork);
                    match match_res.err {
                        None => {
                            // Some meta variables are optional (e.g. vis)
                            if match_res.value.is_some() {
                                item.meta_result = Some((fork, match_res));
                                try_push!(bb_items, item);
                            } else {
                                item.bindings.last_mut().unwrap().push_optional(name);
                                item.dot.next();
                                cur_items.push(item);
                            }
                        }
                        Some(err) => {
                            res.add_err(err);
                            match match_res.value {
                                Some(fragment) => {
                                    item.bindings
                                        .last_mut()
                                        .unwrap()
                                        .inner
                                        .push((name.clone(), Binding::Fragment(fragment)));
                                }
                                _ => {}
                            }
                            item.is_error = true;
                            error_items.push(item);
                        }
                    }
                }
            }
            OpDelimited::Op(Op::Leaf(leaf)) => {
                if let Err(err) = match_leaf(&leaf, &mut src.clone()) {
                    res.add_err(err);
                    item.is_error = true;
                } else {
                    item.dot.next();
                }
                try_push!(next_items, item);
            }
            OpDelimited::Open => {
                if matches!(src.clone().next(), Some(tt::TokenTree::Subtree(..))) {
                    item.dot.next();
                    try_push!(next_items, item);
                }
            }
            OpDelimited::Close => {
                let is_delim_closed = src.peek_n(0).is_none() && !stack.is_empty();
                if is_delim_closed {
                    item.dot.next();
                    try_push!(next_items, item);
                }
            }
        }
    }
}

fn match_loop(pattern: &MetaTemplate, src: &tt::Subtree) -> Match {
    let mut src = TtIter::new(src);
    let mut stack: SmallVec<[TtIter; 1]> = SmallVec::new();
    let mut res = Match::default();
    let mut error_reover_item = None;

    let mut cur_items = smallvec![MatchState {
        dot: pattern.iter_delimited(None),
        stack: Default::default(),
        up: None,
        sep: None,
        sep_kind: None,
        sep_parsed: None,
        bindings: smallvec![Bindings::default()],
        is_error: false,
        meta_result: None,
    }];

    let mut next_items = vec![];

    loop {
        let mut bb_items = SmallVec::new();
        let mut eof_items = SmallVec::new();
        let mut error_items = SmallVec::new();

        stdx::always!(next_items.is_empty());

        match_loop_inner(
            src.clone(),
            &stack,
            &mut res,
            &mut cur_items,
            &mut bb_items,
            &mut next_items,
            &mut eof_items,
            &mut error_items,
        );
        stdx::always!(cur_items.is_empty());

        if error_items.len() > 0 {
            error_reover_item = error_items.pop();
        } else if eof_items.len() > 0 {
            error_reover_item = Some(eof_items[0].clone());
        }

        // We need to do some post processing after the `match_loop_inner`.
        // If we reached the EOF, check that there is EXACTLY ONE possible matcher. Otherwise,
        // either the parse is ambiguous (which should never happen) or there is a syntax error.
        if src.peek_n(0).is_none() && stack.is_empty() {
            if eof_items.len() == 1 {
                // remove all errors, because it is the correct answer !
                res = Match::default();
                res.bindings = eof_items[0].bindings[0].clone();
            } else {
                // Error recovery
                if error_reover_item.is_some() {
                    res.bindings = error_reover_item.unwrap().bindings[0].clone();
                }
                res.add_err(ExpandError::UnexpectedToken);
            }
            return res;
        }

        // If there are no possible next positions AND we aren't waiting for the black-box parser,
        // then there is a syntax error.
        //
        // Another possibility is that we need to call out to parse some rust nonterminal
        // (black-box) parser. However, if there is not EXACTLY ONE of these, something is wrong.
        if (bb_items.is_empty() && next_items.is_empty())
            || (!bb_items.is_empty() && !next_items.is_empty())
            || bb_items.len() > 1
        {
            res.unmatched_tts += src.len();
            while let Some(it) = stack.pop() {
                src = it;
                res.unmatched_tts += src.len();
            }
            res.add_err(err!("leftover tokens"));

            if let Some(mut error_reover_item) = error_reover_item {
                res.bindings = error_reover_item.bindings.remove(0);
            }
            return res;
        }
        // Dump all possible `next_items` into `cur_items` for the next iteration.
        else if !next_items.is_empty() {
            // Now process the next token
            cur_items.extend(next_items.drain(..));

            match src.next() {
                Some(tt::TokenTree::Subtree(subtree)) => {
                    stack.push(src.clone());
                    src = TtIter::new(subtree);
                }
                None if !stack.is_empty() => src = stack.pop().unwrap(),
                _ => (),
            }
        }
        // Finally, we have the case where we need to call the black-box parser to get some
        // nonterminal.
        else {
            stdx::always!(bb_items.len() == 1);
            let mut item = bb_items.pop().unwrap();

            if let Some(OpDelimited::Op(Op::Var { name, .. })) = item.dot.peek() {
                let (iter, match_res) = item.meta_result.take().unwrap();
                let bindings = item.bindings.last_mut().unwrap();
                match match_res.value {
                    Some(fragment) => {
                        bindings.inner.push((name.clone(), Binding::Fragment(fragment)));
                    }
                    None if match_res.err.is_none() => bindings.push_optional(name),
                    _ => {}
                }
                if let Some(err) = match_res.err {
                    res.add_err(err);
                }
                src = iter.clone();
                item.dot.next();
            } else {
                unreachable!()
            }
            cur_items.push(item);
        }
        stdx::always!(!cur_items.is_empty());
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
    fn expect_separator(&mut self, separator: &Separator, idx: usize) -> bool {
        let mut fork = self.clone();
        let ok = match separator {
            Separator::Ident(lhs) if idx == 0 => match fork.expect_ident() {
                Ok(rhs) => rhs.text == lhs.text,
                _ => false,
            },
            Separator::Literal(lhs) if idx == 0 => match fork.expect_literal() {
                Ok(rhs) => match rhs {
                    tt::Leaf::Literal(rhs) => rhs.text == lhs.text,
                    tt::Leaf::Ident(rhs) => rhs.text == lhs.text,
                    tt::Leaf::Punct(_) => false,
                },
                _ => false,
            },
            Separator::Puncts(lhss) if idx < lhss.len() => match fork.expect_punct() {
                Ok(rhs) => rhs.char == lhss[idx].char,
                _ => false,
            },
            _ => false,
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
