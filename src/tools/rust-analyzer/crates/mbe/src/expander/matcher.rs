//! An NFA-based parser, which is porting from rustc mbe parsing code
//!
//! See <https://github.com/rust-lang/rust/blob/70b18bc2cbac4712020019f5bf57c00905373205/compiler/rustc_expand/src/mbe/macro_parser.rs>
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

use std::rc::Rc;

use smallvec::{smallvec, SmallVec};
use syntax::SmolStr;

use crate::{
    expander::{Binding, Bindings, ExpandResult, Fragment},
    parser::{MetaVarKind, Op, RepeatKind, Separator},
    tt,
    tt_iter::TtIter,
    ExpandError, MetaTemplate, ValueResult,
};

impl Bindings {
    fn push_optional(&mut self, name: &SmolStr) {
        // FIXME: Do we have a better way to represent an empty token ?
        // Insert an empty subtree for empty token
        let tt =
            tt::Subtree { delimiter: tt::Delimiter::unspecified(), token_trees: vec![] }.into();
        self.inner.insert(name.clone(), Binding::Fragment(Fragment::Tokens(tt)));
    }

    fn push_empty(&mut self, name: &SmolStr) {
        self.inner.insert(name.clone(), Binding::Empty);
    }

    fn bindings(&self) -> impl Iterator<Item = &Binding> {
        self.inner.values()
    }
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
pub(super) fn match_(pattern: &MetaTemplate, input: &tt::Subtree, is_2021: bool) -> Match {
    let mut res = match_loop(pattern, input, is_2021);
    res.bound_count = count(res.bindings.bindings());
    return res;

    fn count<'a>(bindings: impl Iterator<Item = &'a Binding>) -> usize {
        bindings
            .map(|it| match it {
                Binding::Fragment(_) => 1,
                Binding::Empty => 1,
                Binding::Missing(_) => 1,
                Binding::Nested(it) => count(it.iter()),
            })
            .sum()
    }
}

#[derive(Debug, Clone)]
enum BindingKind {
    Empty(SmolStr),
    Optional(SmolStr),
    Fragment(SmolStr, Fragment),
    Missing(SmolStr, MetaVarKind),
    Nested(usize, usize),
}

#[derive(Debug, Clone)]
struct BindingsIdx(usize, usize);

#[derive(Debug, Clone)]
enum LinkNode<T> {
    Node(T),
    Parent { idx: usize, len: usize },
}

#[derive(Default)]
struct BindingsBuilder {
    nodes: Vec<Vec<LinkNode<Rc<BindingKind>>>>,
    nested: Vec<Vec<LinkNode<usize>>>,
}

impl BindingsBuilder {
    fn alloc(&mut self) -> BindingsIdx {
        let idx = self.nodes.len();
        self.nodes.push(Vec::new());
        let nidx = self.nested.len();
        self.nested.push(Vec::new());
        BindingsIdx(idx, nidx)
    }

    fn copy(&mut self, bindings: &BindingsIdx) -> BindingsIdx {
        let idx = copy_parent(bindings.0, &mut self.nodes);
        let nidx = copy_parent(bindings.1, &mut self.nested);
        return BindingsIdx(idx, nidx);

        fn copy_parent<T>(idx: usize, target: &mut Vec<Vec<LinkNode<T>>>) -> usize
        where
            T: Clone,
        {
            let new_idx = target.len();
            let len = target[idx].len();
            if len < 4 {
                target.push(target[idx].clone())
            } else {
                target.push(vec![LinkNode::Parent { idx, len }]);
            }
            new_idx
        }
    }

    fn push_empty(&mut self, idx: &mut BindingsIdx, var: &SmolStr) {
        self.nodes[idx.0].push(LinkNode::Node(Rc::new(BindingKind::Empty(var.clone()))));
    }

    fn push_optional(&mut self, idx: &mut BindingsIdx, var: &SmolStr) {
        self.nodes[idx.0].push(LinkNode::Node(Rc::new(BindingKind::Optional(var.clone()))));
    }

    fn push_fragment(&mut self, idx: &mut BindingsIdx, var: &SmolStr, fragment: Fragment) {
        self.nodes[idx.0]
            .push(LinkNode::Node(Rc::new(BindingKind::Fragment(var.clone(), fragment))));
    }

    fn push_missing(&mut self, idx: &mut BindingsIdx, var: &SmolStr, kind: MetaVarKind) {
        self.nodes[idx.0].push(LinkNode::Node(Rc::new(BindingKind::Missing(var.clone(), kind))));
    }

    fn push_nested(&mut self, parent: &mut BindingsIdx, child: &BindingsIdx) {
        let BindingsIdx(idx, nidx) = self.copy(child);
        self.nodes[parent.0].push(LinkNode::Node(Rc::new(BindingKind::Nested(idx, nidx))));
    }

    fn push_default(&mut self, idx: &mut BindingsIdx) {
        self.nested[idx.1].push(LinkNode::Node(idx.0));
        let new_idx = self.nodes.len();
        self.nodes.push(Vec::new());
        idx.0 = new_idx;
    }

    fn build(self, idx: &BindingsIdx) -> Bindings {
        self.build_inner(&self.nodes[idx.0])
    }

    fn build_inner(&self, link_nodes: &[LinkNode<Rc<BindingKind>>]) -> Bindings {
        let mut bindings = Bindings::default();
        let mut nodes = Vec::new();
        self.collect_nodes(link_nodes, &mut nodes);

        for cmd in nodes {
            match cmd {
                BindingKind::Empty(name) => {
                    bindings.push_empty(name);
                }
                BindingKind::Optional(name) => {
                    bindings.push_optional(name);
                }
                BindingKind::Fragment(name, fragment) => {
                    bindings.inner.insert(name.clone(), Binding::Fragment(fragment.clone()));
                }
                BindingKind::Missing(name, kind) => {
                    bindings.inner.insert(name.clone(), Binding::Missing(*kind));
                }
                BindingKind::Nested(idx, nested_idx) => {
                    let mut nested_nodes = Vec::new();
                    self.collect_nested(*idx, *nested_idx, &mut nested_nodes);

                    for (idx, iter) in nested_nodes.into_iter().enumerate() {
                        for (key, value) in &iter.inner {
                            let bindings = bindings
                                .inner
                                .entry(key.clone())
                                .or_insert_with(|| Binding::Nested(Vec::new()));

                            if let Binding::Nested(it) = bindings {
                                // insert empty nested bindings before this one
                                while it.len() < idx {
                                    it.push(Binding::Nested(Vec::new()));
                                }
                                it.push(value.clone());
                            }
                        }
                    }
                }
            }
        }

        bindings
    }

    fn collect_nested_ref<'a>(
        &'a self,
        id: usize,
        len: usize,
        nested_refs: &mut Vec<&'a [LinkNode<Rc<BindingKind>>]>,
    ) {
        self.nested[id].iter().take(len).for_each(|it| match it {
            LinkNode::Node(id) => nested_refs.push(&self.nodes[*id]),
            LinkNode::Parent { idx, len } => self.collect_nested_ref(*idx, *len, nested_refs),
        });
    }

    fn collect_nested(&self, idx: usize, nested_idx: usize, nested: &mut Vec<Bindings>) {
        let last = &self.nodes[idx];
        let mut nested_refs: Vec<&[_]> = Vec::new();
        self.nested[nested_idx].iter().for_each(|it| match *it {
            LinkNode::Node(idx) => nested_refs.push(&self.nodes[idx]),
            LinkNode::Parent { idx, len } => self.collect_nested_ref(idx, len, &mut nested_refs),
        });
        nested_refs.push(last);
        nested.extend(nested_refs.into_iter().map(|iter| self.build_inner(iter)));
    }

    fn collect_nodes_ref<'a>(&'a self, id: usize, len: usize, nodes: &mut Vec<&'a BindingKind>) {
        self.nodes[id].iter().take(len).for_each(|it| match it {
            LinkNode::Node(it) => nodes.push(it),
            LinkNode::Parent { idx, len } => self.collect_nodes_ref(*idx, *len, nodes),
        });
    }

    fn collect_nodes<'a>(
        &'a self,
        link_nodes: &'a [LinkNode<Rc<BindingKind>>],
        nodes: &mut Vec<&'a BindingKind>,
    ) {
        link_nodes.iter().for_each(|it| match it {
            LinkNode::Node(it) => nodes.push(it),
            LinkNode::Parent { idx, len } => self.collect_nodes_ref(*idx, *len, nodes),
        });
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

    /// Whether we already matched separator token.
    sep_matched: bool,

    /// Matched meta variables bindings
    bindings: BindingsIdx,

    /// Cached result of meta variable parsing
    meta_result: Option<(TtIter<'t>, ExpandResult<Option<Fragment>>)>,

    /// Is error occurred in this state, will `poised` to "parent"
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
#[inline]
fn match_loop_inner<'t>(
    src: TtIter<'t>,
    stack: &[TtIter<'t>],
    res: &mut Match,
    bindings_builder: &mut BindingsBuilder,
    cur_items: &mut SmallVec<[MatchState<'t>; 1]>,
    bb_items: &mut SmallVec<[MatchState<'t>; 1]>,
    next_items: &mut Vec<MatchState<'t>>,
    eof_items: &mut SmallVec<[MatchState<'t>; 1]>,
    error_items: &mut SmallVec<[MatchState<'t>; 1]>,
    is_2021: bool,
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
                if let Some(up) = &item.up {
                    if !item.sep_matched {
                        // Get the `up` matcher
                        let mut new_pos = (**up).clone();
                        new_pos.bindings = bindings_builder.copy(&new_pos.bindings);
                        // Add matches from this repetition to the `matches` of `up`
                        bindings_builder.push_nested(&mut new_pos.bindings, &item.bindings);

                        // Move the "dot" past the repetition in `up`
                        new_pos.dot.next();
                        new_pos.is_error = new_pos.is_error || item.is_error;
                        cur_items.push(new_pos);
                    }

                    // Check if we need a separator.
                    if item.sep.is_some() && !item.sep_matched {
                        let sep = item.sep.as_ref().unwrap();
                        let mut fork = src.clone();
                        if fork.expect_separator(sep) {
                            // HACK: here we use `meta_result` to pass `TtIter` back to caller because
                            // it might have been advanced multiple times. `ValueResult` is
                            // insignificant.
                            item.meta_result = Some((fork, ValueResult::ok(None)));
                            item.dot.next();
                            // item.sep_parsed = Some(sep_len);
                            item.sep_matched = true;
                            try_push!(next_items, item);
                        }
                    }
                    // We don't need a separator. Move the "dot" back to the beginning of the matcher
                    // and try to match again UNLESS we are only allowed to have _one_ repetition.
                    else if item.sep_kind != Some(RepeatKind::ZeroOrOne) {
                        item.dot = item.dot.reset();
                        item.sep_matched = false;
                        bindings_builder.push_default(&mut item.bindings);
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
                    new_item.bindings = bindings_builder.copy(&new_item.bindings);
                    new_item.dot.next();
                    collect_vars(
                        &mut |s| {
                            bindings_builder.push_empty(&mut new_item.bindings, &s);
                        },
                        tokens,
                    );
                    cur_items.push(new_item);
                }
                cur_items.push(MatchState {
                    dot: tokens.iter_delimited(None),
                    stack: Default::default(),
                    up: Some(Box::new(item)),
                    sep: separator.clone(),
                    sep_kind: Some(*kind),
                    sep_matched: false,
                    bindings: bindings_builder.alloc(),
                    meta_result: None,
                    is_error: false,
                })
            }
            OpDelimited::Op(Op::Subtree { tokens, delimiter }) => {
                if let Ok(subtree) = src.clone().expect_subtree() {
                    if subtree.delimiter.kind == delimiter.kind {
                        item.stack.push(item.dot);
                        item.dot = tokens.iter_delimited(Some(delimiter));
                        cur_items.push(item);
                    }
                }
            }
            OpDelimited::Op(Op::Var { kind, name, .. }) => {
                if let &Some(kind) = kind {
                    let mut fork = src.clone();
                    let match_res = match_meta_var(kind, &mut fork, is_2021);
                    match match_res.err {
                        None => {
                            // Some meta variables are optional (e.g. vis)
                            if match_res.value.is_some() {
                                item.meta_result = Some((fork, match_res));
                                try_push!(bb_items, item);
                            } else {
                                bindings_builder.push_optional(&mut item.bindings, name);
                                item.dot.next();
                                cur_items.push(item);
                            }
                        }
                        Some(err) => {
                            res.add_err(err);
                            match match_res.value {
                                Some(fragment) => bindings_builder.push_fragment(
                                    &mut item.bindings,
                                    name,
                                    fragment,
                                ),
                                None => {
                                    bindings_builder.push_missing(&mut item.bindings, name, kind)
                                }
                            }
                            item.is_error = true;
                            error_items.push(item);
                        }
                    }
                }
            }
            OpDelimited::Op(Op::Literal(lhs)) => {
                if let Ok(rhs) = src.clone().expect_leaf() {
                    if matches!(rhs, tt::Leaf::Literal(it) if it.text == lhs.text) {
                        item.dot.next();
                    } else {
                        res.add_err(ExpandError::UnexpectedToken);
                        item.is_error = true;
                    }
                } else {
                    res.add_err(ExpandError::binding_error(format!("expected literal: `{lhs}`")));
                    item.is_error = true;
                }
                try_push!(next_items, item);
            }
            OpDelimited::Op(Op::Ident(lhs)) => {
                if let Ok(rhs) = src.clone().expect_leaf() {
                    if matches!(rhs, tt::Leaf::Ident(it) if it.text == lhs.text) {
                        item.dot.next();
                    } else {
                        res.add_err(ExpandError::UnexpectedToken);
                        item.is_error = true;
                    }
                } else {
                    res.add_err(ExpandError::binding_error(format!("expected ident: `{lhs}`")));
                    item.is_error = true;
                }
                try_push!(next_items, item);
            }
            OpDelimited::Op(Op::Punct(lhs)) => {
                let mut fork = src.clone();
                let error = if let Ok(rhs) = fork.expect_glued_punct() {
                    let first_is_single_quote = rhs[0].char == '\'';
                    let lhs = lhs.iter().map(|it| it.char);
                    let rhs = rhs.iter().map(|it| it.char);
                    if lhs.clone().eq(rhs) {
                        // HACK: here we use `meta_result` to pass `TtIter` back to caller because
                        // it might have been advanced multiple times. `ValueResult` is
                        // insignificant.
                        item.meta_result = Some((fork, ValueResult::ok(None)));
                        item.dot.next();
                        next_items.push(item);
                        continue;
                    }

                    if first_is_single_quote {
                        // If the first punct token is a single quote, that's a part of a lifetime
                        // ident, not a punct.
                        ExpandError::UnexpectedToken
                    } else {
                        let lhs: SmolStr = lhs.collect();
                        ExpandError::binding_error(format!("expected punct: `{lhs}`"))
                    }
                } else {
                    ExpandError::UnexpectedToken
                };

                res.add_err(error);
                item.is_error = true;
                error_items.push(item);
            }
            OpDelimited::Op(Op::Ignore { .. } | Op::Index { .. } | Op::Count { .. }) => {
                stdx::never!("metavariable expression in lhs found");
            }
            OpDelimited::Open => {
                if matches!(src.peek_n(0), Some(tt::TokenTree::Subtree(..))) {
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

fn match_loop(pattern: &MetaTemplate, src: &tt::Subtree, is_2021: bool) -> Match {
    let mut src = TtIter::new(src);
    let mut stack: SmallVec<[TtIter<'_>; 1]> = SmallVec::new();
    let mut res = Match::default();
    let mut error_recover_item = None;

    let mut bindings_builder = BindingsBuilder::default();

    let mut cur_items = smallvec![MatchState {
        dot: pattern.iter_delimited(None),
        stack: Default::default(),
        up: None,
        sep: None,
        sep_kind: None,
        sep_matched: false,
        bindings: bindings_builder.alloc(),
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
            &mut bindings_builder,
            &mut cur_items,
            &mut bb_items,
            &mut next_items,
            &mut eof_items,
            &mut error_items,
            is_2021,
        );
        stdx::always!(cur_items.is_empty());

        if !error_items.is_empty() {
            error_recover_item = error_items.pop().map(|it| it.bindings);
        } else if let [state, ..] = &*eof_items {
            error_recover_item = Some(state.bindings.clone());
        }

        // We need to do some post processing after the `match_loop_inner`.
        // If we reached the EOF, check that there is EXACTLY ONE possible matcher. Otherwise,
        // either the parse is ambiguous (which should never happen) or there is a syntax error.
        if src.peek_n(0).is_none() && stack.is_empty() {
            if let [state] = &*eof_items {
                // remove all errors, because it is the correct answer !
                res = Match::default();
                res.bindings = bindings_builder.build(&state.bindings);
            } else {
                // Error recovery
                if let Some(item) = error_recover_item {
                    res.bindings = bindings_builder.build(&item);
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
        let has_leftover_tokens = (bb_items.is_empty() && next_items.is_empty())
            || !(bb_items.is_empty() || next_items.is_empty())
            || bb_items.len() > 1;
        if has_leftover_tokens {
            res.unmatched_tts += src.len();
            while let Some(it) = stack.pop() {
                src = it;
                res.unmatched_tts += src.len();
            }
            res.add_err(ExpandError::LeftoverTokens);

            if let Some(error_recover_item) = error_recover_item {
                res.bindings = bindings_builder.build(&error_recover_item);
            }
            return res;
        }
        // Dump all possible `next_items` into `cur_items` for the next iteration.
        else if !next_items.is_empty() {
            if let Some((iter, _)) = next_items[0].meta_result.take() {
                // We've matched a possibly "glued" punct. The matched punct (hence
                // `meta_result` also) must be the same for all items.
                // FIXME: If there are multiple items, it's definitely redundant (and it's hacky!
                // `meta_result` isn't supposed to be used this way).

                // We already bumped, so no need to call `.next()` like in the other branch.
                src = iter;
                for item in next_items.iter_mut() {
                    item.meta_result = None;
                }
            } else {
                match src.next() {
                    Some(tt::TokenTree::Subtree(subtree)) => {
                        stack.push(src.clone());
                        src = TtIter::new(subtree);
                    }
                    None => {
                        if let Some(iter) = stack.pop() {
                            src = iter;
                        }
                    }
                    _ => (),
                }
            }
            // Now process the next token
            cur_items.extend(next_items.drain(..));
        }
        // Finally, we have the case where we need to call the black-box parser to get some
        // nonterminal.
        else {
            stdx::always!(bb_items.len() == 1);
            let mut item = bb_items.pop().unwrap();

            if let Some(OpDelimited::Op(Op::Var { name, .. })) = item.dot.peek() {
                let (iter, match_res) = item.meta_result.take().unwrap();
                match match_res.value {
                    Some(fragment) => {
                        bindings_builder.push_fragment(&mut item.bindings, name, fragment);
                    }
                    None if match_res.err.is_none() => {
                        bindings_builder.push_optional(&mut item.bindings, name);
                    }
                    None => {}
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

fn match_meta_var(
    kind: MetaVarKind,
    input: &mut TtIter<'_>,
    is_2021: bool,
) -> ExpandResult<Option<Fragment>> {
    let fragment = match kind {
        MetaVarKind::Path => parser::PrefixEntryPoint::Path,
        MetaVarKind::Ty => parser::PrefixEntryPoint::Ty,
        MetaVarKind::Pat if is_2021 => parser::PrefixEntryPoint::PatTop,
        MetaVarKind::Pat => parser::PrefixEntryPoint::Pat,
        MetaVarKind::PatParam => parser::PrefixEntryPoint::Pat,
        MetaVarKind::Stmt => parser::PrefixEntryPoint::Stmt,
        MetaVarKind::Block => parser::PrefixEntryPoint::Block,
        MetaVarKind::Meta => parser::PrefixEntryPoint::MetaItem,
        MetaVarKind::Item => parser::PrefixEntryPoint::Item,
        MetaVarKind::Vis => parser::PrefixEntryPoint::Vis,
        MetaVarKind::Expr => {
            // `expr` should not match underscores, let expressions, or inline const. The latter
            // two are for [backwards compatibility][0].
            // HACK: Macro expansion should not be done using "rollback and try another alternative".
            // rustc [explicitly checks the next token][1].
            // [0]: https://github.com/rust-lang/rust/issues/86730
            // [1]: https://github.com/rust-lang/rust/blob/f0c4da499/compiler/rustc_expand/src/mbe/macro_parser.rs#L576
            match input.peek_n(0) {
                Some(tt::TokenTree::Leaf(tt::Leaf::Ident(it)))
                    if it.text == "_" || it.text == "let" || it.text == "const" =>
                {
                    return ExpandResult::only_err(ExpandError::NoMatchingRule)
                }
                _ => {}
            };
            return input
                .expect_fragment(parser::PrefixEntryPoint::Expr)
                .map(|tt| tt.map(Fragment::Expr));
        }
        _ => {
            let tt_result = match kind {
                MetaVarKind::Ident => input
                    .expect_ident()
                    .map(|ident| tt::Leaf::from(ident.clone()).into())
                    .map_err(|()| ExpandError::binding_error("expected ident")),
                MetaVarKind::Tt => input
                    .expect_tt()
                    .map_err(|()| ExpandError::binding_error("expected token tree")),
                MetaVarKind::Lifetime => input
                    .expect_lifetime()
                    .map_err(|()| ExpandError::binding_error("expected lifetime")),
                MetaVarKind::Literal => {
                    let neg = input.eat_char('-');
                    input
                        .expect_literal()
                        .map(|literal| {
                            let lit = literal.clone();
                            match neg {
                                None => lit.into(),
                                Some(neg) => tt::TokenTree::Subtree(tt::Subtree {
                                    delimiter: tt::Delimiter::unspecified(),
                                    token_trees: vec![neg, lit.into()],
                                }),
                            }
                        })
                        .map_err(|()| ExpandError::binding_error("expected literal"))
                }
                _ => Err(ExpandError::UnexpectedToken),
            };
            return tt_result.map(|it| Some(Fragment::Tokens(it))).into();
        }
    };
    input.expect_fragment(fragment).map(|it| it.map(Fragment::Tokens))
}

fn collect_vars(collector_fun: &mut impl FnMut(SmolStr), pattern: &MetaTemplate) {
    for op in pattern.iter() {
        match op {
            Op::Var { name, .. } => collector_fun(name.clone()),
            Op::Subtree { tokens, .. } => collect_vars(collector_fun, tokens),
            Op::Repeat { tokens, .. } => collect_vars(collector_fun, tokens),
            Op::Literal(_) | Op::Ident(_) | Op::Punct(_) => {}
            Op::Ignore { .. } | Op::Index { .. } | Op::Count { .. } => {
                stdx::never!("metavariable expression in lhs found");
            }
        }
    }
}
impl MetaTemplate {
    fn iter_delimited<'a>(&'a self, delimited: Option<&'a tt::Delimiter>) -> OpDelimitedIter<'a> {
        OpDelimitedIter {
            inner: &self.0,
            idx: 0,
            delimited: delimited.unwrap_or(&tt::Delimiter::UNSPECIFIED),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum OpDelimited<'a> {
    Op(&'a Op),
    Open,
    Close,
}

#[derive(Debug, Clone, Copy)]
struct OpDelimitedIter<'a> {
    inner: &'a [Op],
    delimited: &'a tt::Delimiter,
    idx: usize,
}

impl<'a> OpDelimitedIter<'a> {
    fn is_eof(&self) -> bool {
        let len = self.inner.len()
            + if self.delimited.kind != tt::DelimiterKind::Invisible { 2 } else { 0 };
        self.idx >= len
    }

    fn peek(&self) -> Option<OpDelimited<'a>> {
        match self.delimited.kind {
            tt::DelimiterKind::Invisible => self.inner.get(self.idx).map(OpDelimited::Op),
            _ => match self.idx {
                0 => Some(OpDelimited::Open),
                i if i == self.inner.len() + 1 => Some(OpDelimited::Close),
                i => self.inner.get(i - 1).map(OpDelimited::Op),
            },
        }
    }

    fn reset(&self) -> Self {
        Self { inner: self.inner, idx: 0, delimited: self.delimited }
    }
}

impl<'a> Iterator for OpDelimitedIter<'a> {
    type Item = OpDelimited<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let res = self.peek();
        self.idx += 1;
        res
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.len()
            + if self.delimited.kind != tt::DelimiterKind::Invisible { 2 } else { 0 };
        let remain = len.saturating_sub(self.idx);
        (remain, Some(remain))
    }
}

impl<'a> TtIter<'a> {
    fn expect_separator(&mut self, separator: &Separator) -> bool {
        let mut fork = self.clone();
        let ok = match separator {
            Separator::Ident(lhs) => match fork.expect_ident_or_underscore() {
                Ok(rhs) => rhs.text == lhs.text,
                Err(_) => false,
            },
            Separator::Literal(lhs) => match fork.expect_literal() {
                Ok(rhs) => match rhs {
                    tt::Leaf::Literal(rhs) => rhs.text == lhs.text,
                    tt::Leaf::Ident(rhs) => rhs.text == lhs.text,
                    tt::Leaf::Punct(_) => false,
                },
                Err(_) => false,
            },
            Separator::Puncts(lhs) => match fork.expect_glued_punct() {
                Ok(rhs) => {
                    let lhs = lhs.iter().map(|it| it.char);
                    let rhs = rhs.iter().map(|it| it.char);
                    lhs.eq(rhs)
                }
                Err(_) => false,
            },
        };
        if ok {
            *self = fork;
        }
        ok
    }

    fn expect_tt(&mut self) -> Result<tt::TokenTree, ()> {
        if let Some(tt::TokenTree::Leaf(tt::Leaf::Punct(punct))) = self.peek_n(0) {
            if punct.char == '\'' {
                self.expect_lifetime()
            } else {
                let puncts = self.expect_glued_punct()?;
                let token_trees = puncts.into_iter().map(|p| tt::Leaf::Punct(p).into()).collect();
                Ok(tt::TokenTree::Subtree(tt::Subtree {
                    delimiter: tt::Delimiter::unspecified(),
                    token_trees,
                }))
            }
        } else {
            self.next().ok_or(()).cloned()
        }
    }

    fn expect_lifetime(&mut self) -> Result<tt::TokenTree, ()> {
        let punct = self.expect_single_punct()?;
        if punct.char != '\'' {
            return Err(());
        }
        let ident = self.expect_ident_or_underscore()?;

        Ok(tt::Subtree {
            delimiter: tt::Delimiter::unspecified(),
            token_trees: vec![
                tt::Leaf::Punct(*punct).into(),
                tt::Leaf::Ident(ident.clone()).into(),
            ],
        }
        .into())
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
