// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Formatting of chained expressions, i.e. expressions which are chained by
//! dots: struct and enum field access, method calls, and try shorthand (?).
//!
//! Instead of walking these subexpressions one-by-one, as is our usual strategy
//! for expression formatting, we collect maximal sequences of these expressions
//! and handle them simultaneously.
//!
//! Whenever possible, the entire chain is put on a single line. If that fails,
//! we put each subexpression on a separate, much like the (default) function
//! argument function argument strategy.
//!
//! Depends on config options: `chain_indent` is the indent to use for
//! blocks in the parent/root/base of the chain (and the rest of the chain's
//! alignment).
//! E.g., `let foo = { aaaa; bbb; ccc }.bar.baz();`, we would layout for the
//! following values of `chain_indent`:
//! Block:
//!
//! ```ignore
//! let foo = {
//!     aaaa;
//!     bbb;
//!     ccc
//! }.bar
//!     .baz();
//! ```
//!
//! Visual:
//!
//! ```ignore
//! let foo = {
//!               aaaa;
//!               bbb;
//!               ccc
//!           }
//!           .bar
//!           .baz();
//! ```
//!
//! If the first item in the chain is a block expression, we align the dots with
//! the braces.
//! Block:
//!
//! ```ignore
//! let a = foo.bar
//!     .baz()
//!     .qux
//! ```
//!
//! Visual:
//!
//! ```ignore
//! let a = foo.bar
//!            .baz()
//!            .qux
//! ```

use comment::rewrite_comment;
use config::IndentStyle;
use expr::rewrite_call;
use lists::{extract_post_comment, extract_pre_comment, get_comment_end};
use macros::convert_try_mac;
use rewrite::{Rewrite, RewriteContext};
use shape::Shape;
use source_map::SpanUtils;
use utils::{
    first_line_width, last_line_extendable, last_line_width, mk_sp, trimmed_last_line_width,
    wrap_str,
};

use std::borrow::Cow;
use std::cmp::min;
use std::iter;

use syntax::source_map::{BytePos, Span};
use syntax::{ast, ptr};

pub fn rewrite_chain(expr: &ast::Expr, context: &RewriteContext, shape: Shape) -> Option<String> {
    let chain = Chain::from_ast(expr, context);
    debug!("rewrite_chain {:?} {:?}", chain, shape);

    // If this is just an expression with some `?`s, then format it trivially and
    // return early.
    if chain.children.is_empty() {
        return chain.parent.rewrite(context, shape);
    }

    chain.rewrite(context, shape)
}

#[derive(Debug)]
enum CommentPosition {
    Back,
    Top,
}

// An expression plus trailing `?`s to be formatted together.
#[derive(Debug)]
struct ChainItem {
    kind: ChainItemKind,
    tries: usize,
    span: Span,
}

// FIXME: we can't use a reference here because to convert `try!` to `?` we
// synthesise the AST node. However, I think we could use `Cow` and that
// would remove a lot of cloning.
#[derive(Debug)]
enum ChainItemKind {
    Parent(ast::Expr),
    MethodCall(
        ast::PathSegment,
        Vec<ast::GenericArg>,
        Vec<ptr::P<ast::Expr>>,
    ),
    StructField(ast::Ident),
    TupleField(ast::Ident, bool),
    Comment(String, CommentPosition),
}

impl ChainItemKind {
    fn is_block_like(&self, context: &RewriteContext, reps: &str) -> bool {
        match self {
            ChainItemKind::Parent(ref expr) => is_block_expr(context, expr, reps),
            ChainItemKind::MethodCall(..) => reps.contains('\n'),
            ChainItemKind::StructField(..)
            | ChainItemKind::TupleField(..)
            | ChainItemKind::Comment(..) => false,
        }
    }

    fn is_tup_field_access(expr: &ast::Expr) -> bool {
        match expr.node {
            ast::ExprKind::Field(_, ref field) => {
                field.name.to_string().chars().all(|c| c.is_digit(10))
            }
            _ => false,
        }
    }

    fn from_ast(context: &RewriteContext, expr: &ast::Expr) -> (ChainItemKind, Span) {
        let (kind, span) = match expr.node {
            ast::ExprKind::MethodCall(ref segment, ref expressions) => {
                let types = if let Some(ref generic_args) = segment.args {
                    if let ast::GenericArgs::AngleBracketed(ref data) = **generic_args {
                        data.args.clone()
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                };
                let span = mk_sp(expressions[0].span.hi(), expr.span.hi());
                let kind = ChainItemKind::MethodCall(segment.clone(), types, expressions.clone());
                (kind, span)
            }
            ast::ExprKind::Field(ref nested, field) => {
                let kind = if Self::is_tup_field_access(expr) {
                    ChainItemKind::TupleField(field, Self::is_tup_field_access(nested))
                } else {
                    ChainItemKind::StructField(field)
                };
                let span = mk_sp(nested.span.hi(), field.span.hi());
                (kind, span)
            }
            _ => return (ChainItemKind::Parent(expr.clone()), expr.span),
        };

        // Remove comments from the span.
        let lo = context.snippet_provider.span_before(span, ".");
        (kind, mk_sp(lo, span.hi()))
    }
}

impl Rewrite for ChainItem {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        let shape = shape.sub_width(self.tries)?;
        let rewrite = match self.kind {
            ChainItemKind::Parent(ref expr) => expr.rewrite(context, shape)?,
            ChainItemKind::MethodCall(ref segment, ref types, ref exprs) => {
                Self::rewrite_method_call(segment.ident, types, exprs, self.span, context, shape)?
            }
            ChainItemKind::StructField(ident) => format!(".{}", ident.name),
            ChainItemKind::TupleField(ident, nested) => {
                format!("{}.{}", if nested { " " } else { "" }, ident.name)
            }
            ChainItemKind::Comment(ref comment, _) => {
                rewrite_comment(comment, false, shape, context.config)?
            }
        };
        Some(format!("{}{}", rewrite, "?".repeat(self.tries)))
    }
}

impl ChainItem {
    fn new(context: &RewriteContext, expr: &ast::Expr, tries: usize) -> ChainItem {
        let (kind, span) = ChainItemKind::from_ast(context, expr);
        ChainItem { kind, tries, span }
    }

    fn comment(span: Span, comment: String, pos: CommentPosition) -> ChainItem {
        ChainItem {
            kind: ChainItemKind::Comment(comment, pos),
            tries: 0,
            span,
        }
    }

    fn is_comment(&self) -> bool {
        match self.kind {
            ChainItemKind::Comment(..) => true,
            _ => false,
        }
    }

    fn rewrite_method_call(
        method_name: ast::Ident,
        types: &[ast::GenericArg],
        args: &[ptr::P<ast::Expr>],
        span: Span,
        context: &RewriteContext,
        shape: Shape,
    ) -> Option<String> {
        let type_str = if types.is_empty() {
            String::new()
        } else {
            let type_list = types
                .iter()
                .map(|ty| ty.rewrite(context, shape))
                .collect::<Option<Vec<_>>>()?;

            format!("::<{}>", type_list.join(", "))
        };
        let callee_str = format!(".{}{}", method_name, type_str);
        rewrite_call(context, &callee_str, &args[1..], span, shape)
    }
}

#[derive(Debug)]
struct Chain {
    parent: ChainItem,
    children: Vec<ChainItem>,
}

impl Chain {
    fn from_ast(expr: &ast::Expr, context: &RewriteContext) -> Chain {
        let subexpr_list = Self::make_subexpr_list(expr, context);

        // Un-parse the expression tree into ChainItems
        let mut rev_children = vec![];
        let mut sub_tries = 0;
        for subexpr in &subexpr_list {
            match subexpr.node {
                ast::ExprKind::Try(_) => sub_tries += 1,
                _ => {
                    rev_children.push(ChainItem::new(context, subexpr, sub_tries));
                    sub_tries = 0;
                }
            }
        }

        fn is_tries(s: &str) -> bool {
            s.chars().all(|c| c == '?')
        }

        fn handle_post_comment(
            post_comment_span: Span,
            post_comment_snippet: &str,
            prev_span_end: &mut BytePos,
            children: &mut Vec<ChainItem>,
        ) {
            let white_spaces: &[_] = &[' ', '\t'];
            if post_comment_snippet
                .trim_matches(white_spaces)
                .starts_with('\n')
            {
                // No post comment.
                return;
            }
            // HACK: Treat `?`s as separators.
            let trimmed_snippet = post_comment_snippet.trim_matches('?');
            let comment_end = get_comment_end(trimmed_snippet, "?", "", false);
            let maybe_post_comment = extract_post_comment(trimmed_snippet, comment_end, "?")
                .and_then(|comment| {
                    if comment.is_empty() {
                        None
                    } else {
                        Some((comment, comment_end))
                    }
                });

            if let Some((post_comment, comment_end)) = maybe_post_comment {
                children.push(ChainItem::comment(
                    post_comment_span,
                    post_comment,
                    CommentPosition::Back,
                ));
                *prev_span_end = *prev_span_end + BytePos(comment_end as u32);
            }
        }

        let parent = rev_children.pop().unwrap();
        let mut children = vec![];
        let mut prev_span_end = parent.span.hi();
        let mut iter = rev_children.into_iter().rev().peekable();
        if let Some(first_chain_item) = iter.peek() {
            let comment_span = mk_sp(prev_span_end, first_chain_item.span.lo());
            let comment_snippet = context.snippet(comment_span);
            if !is_tries(comment_snippet.trim()) {
                handle_post_comment(
                    comment_span,
                    comment_snippet,
                    &mut prev_span_end,
                    &mut children,
                );
            }
        }
        while let Some(chain_item) = iter.next() {
            let comment_snippet = context.snippet(chain_item.span);
            // FIXME: Figure out the way to get a correct span when converting `try!` to `?`.
            let handle_comment =
                !(context.config.use_try_shorthand() || is_tries(comment_snippet.trim()));

            // Pre-comment
            if handle_comment {
                let pre_comment_span = mk_sp(prev_span_end, chain_item.span.lo());
                let pre_comment_snippet = context.snippet(pre_comment_span);
                let pre_comment_snippet = pre_comment_snippet.trim().trim_matches('?');
                let (pre_comment, _) = extract_pre_comment(pre_comment_snippet);
                match pre_comment {
                    Some(ref comment) if !comment.is_empty() => {
                        children.push(ChainItem::comment(
                            pre_comment_span,
                            comment.to_owned(),
                            CommentPosition::Top,
                        ));
                    }
                    _ => (),
                }
            }

            prev_span_end = chain_item.span.hi();
            children.push(chain_item);

            // Post-comment
            if !handle_comment || iter.peek().is_none() {
                continue;
            }

            let next_lo = iter.peek().unwrap().span.lo();
            let post_comment_span = mk_sp(prev_span_end, next_lo);
            let post_comment_snippet = context.snippet(post_comment_span);
            handle_post_comment(
                post_comment_span,
                post_comment_snippet,
                &mut prev_span_end,
                &mut children,
            );
        }

        Chain { parent, children }
    }

    // Returns a Vec of the prefixes of the chain.
    // E.g., for input `a.b.c` we return [`a.b.c`, `a.b`, 'a']
    fn make_subexpr_list(expr: &ast::Expr, context: &RewriteContext) -> Vec<ast::Expr> {
        let mut subexpr_list = vec![expr.clone()];

        while let Some(subexpr) = Self::pop_expr_chain(subexpr_list.last().unwrap(), context) {
            subexpr_list.push(subexpr.clone());
        }

        subexpr_list
    }

    // Returns the expression's subexpression, if it exists. When the subexpr
    // is a try! macro, we'll convert it to shorthand when the option is set.
    fn pop_expr_chain(expr: &ast::Expr, context: &RewriteContext) -> Option<ast::Expr> {
        match expr.node {
            ast::ExprKind::MethodCall(_, ref expressions) => {
                Some(Self::convert_try(&expressions[0], context))
            }
            ast::ExprKind::Field(ref subexpr, _) | ast::ExprKind::Try(ref subexpr) => {
                Some(Self::convert_try(subexpr, context))
            }
            _ => None,
        }
    }

    fn convert_try(expr: &ast::Expr, context: &RewriteContext) -> ast::Expr {
        match expr.node {
            ast::ExprKind::Mac(ref mac) if context.config.use_try_shorthand() => {
                if let Some(subexpr) = convert_try_mac(mac, context) {
                    subexpr
                } else {
                    expr.clone()
                }
            }
            _ => expr.clone(),
        }
    }
}

impl Rewrite for Chain {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        debug!("rewrite chain {:?} {:?}", self, shape);

        let mut formatter = match context.config.indent_style() {
            IndentStyle::Block => Box::new(ChainFormatterBlock::new(self)) as Box<ChainFormatter>,
            IndentStyle::Visual => Box::new(ChainFormatterVisual::new(self)) as Box<ChainFormatter>,
        };

        formatter.format_root(&self.parent, context, shape)?;
        if let Some(result) = formatter.pure_root() {
            return wrap_str(result, context.config.max_width(), shape);
        }

        // Decide how to layout the rest of the chain.
        let child_shape = formatter.child_shape(context, shape)?;

        formatter.format_children(context, child_shape)?;
        formatter.format_last_child(context, shape, child_shape)?;

        let result = formatter.join_rewrites(context, child_shape)?;
        wrap_str(result, context.config.max_width(), shape)
    }
}

// There are a few types for formatting chains. This is because there is a lot
// in common between formatting with block vs visual indent, but they are
// different enough that branching on the indent all over the place gets ugly.
// Anything that can format a chain is a ChainFormatter.
trait ChainFormatter {
    // Parent is the first item in the chain, e.g., `foo` in `foo.bar.baz()`.
    // Root is the parent plus any other chain items placed on the first line to
    // avoid an orphan. E.g.,
    // ```
    // foo.bar
    //     .baz()
    // ```
    // If `bar` were not part of the root, then foo would be orphaned and 'float'.
    fn format_root(
        &mut self,
        parent: &ChainItem,
        context: &RewriteContext,
        shape: Shape,
    ) -> Option<()>;
    fn child_shape(&self, context: &RewriteContext, shape: Shape) -> Option<Shape>;
    fn format_children(&mut self, context: &RewriteContext, child_shape: Shape) -> Option<()>;
    fn format_last_child(
        &mut self,
        context: &RewriteContext,
        shape: Shape,
        child_shape: Shape,
    ) -> Option<()>;
    fn join_rewrites(&self, context: &RewriteContext, child_shape: Shape) -> Option<String>;
    // Returns `Some` if the chain is only a root, None otherwise.
    fn pure_root(&mut self) -> Option<String>;
}

// Data and behaviour that is shared by both chain formatters. The concrete
// formatters can delegate much behaviour to `ChainFormatterShared`.
struct ChainFormatterShared<'a> {
    // The current working set of child items.
    children: &'a [ChainItem],
    // The current rewrites of items (includes trailing `?`s, but not any way to
    // connect the rewrites together).
    rewrites: Vec<String>,
    // Whether the chain can fit on one line.
    fits_single_line: bool,
    // The number of children in the chain. This is not equal to `self.children.len()`
    // because `self.children` will change size as we process the chain.
    child_count: usize,
}

impl<'a> ChainFormatterShared<'a> {
    fn new(chain: &'a Chain) -> ChainFormatterShared<'a> {
        ChainFormatterShared {
            children: &chain.children,
            rewrites: Vec::with_capacity(chain.children.len() + 1),
            fits_single_line: false,
            child_count: chain.children.len(),
        }
    }

    fn pure_root(&mut self) -> Option<String> {
        if self.children.is_empty() {
            assert_eq!(self.rewrites.len(), 1);
            Some(self.rewrites.pop().unwrap())
        } else {
            None
        }
    }

    // Rewrite the last child. The last child of a chain requires special treatment. We need to
    // know whether 'overflowing' the last child make a better formatting:
    //
    // A chain with overflowing the last child:
    // ```
    // parent.child1.child2.last_child(
    //     a,
    //     b,
    //     c,
    // )
    // ```
    //
    // A chain without overflowing the last child (in vertical layout):
    // ```
    // parent
    //     .child1
    //     .child2
    //     .last_child(a, b, c)
    // ```
    //
    // In particular, overflowing is effective when the last child is a method with a multi-lined
    // block-like argument (e.g. closure):
    // ```
    // parent.child1.child2.last_child(|a, b, c| {
    //     let x = foo(a, b, c);
    //     let y = bar(a, b, c);
    //
    //     // ...
    //
    //     result
    // })
    // ```
    fn format_last_child(
        &mut self,
        may_extend: bool,
        context: &RewriteContext,
        shape: Shape,
        child_shape: Shape,
    ) -> Option<()> {
        let last = self.children.last()?;
        let extendable = may_extend && last_line_extendable(&self.rewrites[0]);
        let prev_last_line_width = last_line_width(&self.rewrites[0]);

        // Total of all items excluding the last.
        let almost_total = if extendable {
            prev_last_line_width
        } else {
            self.rewrites.iter().fold(0, |a, b| a + b.len())
        } + last.tries;
        let one_line_budget = if self.child_count == 1 {
            shape.width
        } else {
            min(shape.width, context.config.width_heuristics().chain_width)
        }.saturating_sub(almost_total);

        let all_in_one_line = !self.children.iter().any(ChainItem::is_comment)
            && self.rewrites.iter().all(|s| !s.contains('\n'))
            && one_line_budget > 0;
        let last_shape = if all_in_one_line {
            shape.sub_width(last.tries)?
        } else if extendable {
            child_shape.sub_width(last.tries)?
        } else {
            child_shape.sub_width(shape.rhs_overhead(context.config) + last.tries)?
        };

        let mut last_subexpr_str = None;
        if all_in_one_line || extendable {
            // First we try to 'overflow' the last child and see if it looks better than using
            // vertical layout.
            if let Some(one_line_shape) = last_shape.offset_left(almost_total) {
                if let Some(rw) = last.rewrite(context, one_line_shape) {
                    // We allow overflowing here only if both of the following conditions match:
                    // 1. The entire chain fits in a single line except the last child.
                    // 2. `last_child_str.lines().count() >= 5`.
                    let line_count = rw.lines().count();
                    let could_fit_single_line = first_line_width(&rw) <= one_line_budget;
                    if could_fit_single_line && line_count >= 5 {
                        last_subexpr_str = Some(rw);
                        self.fits_single_line = all_in_one_line;
                    } else {
                        // We could not know whether overflowing is better than using vertical
                        // layout, just by looking at the overflowed rewrite. Now we rewrite the
                        // last child on its own line, and compare two rewrites to choose which is
                        // better.
                        let last_shape = child_shape
                            .sub_width(shape.rhs_overhead(context.config) + last.tries)?;
                        match last.rewrite(context, last_shape) {
                            Some(ref new_rw) if !could_fit_single_line => {
                                last_subexpr_str = Some(new_rw.clone());
                            }
                            Some(ref new_rw) if new_rw.lines().count() >= line_count => {
                                last_subexpr_str = Some(rw);
                                self.fits_single_line = could_fit_single_line && all_in_one_line;
                            }
                            new_rw @ Some(..) => {
                                last_subexpr_str = new_rw;
                            }
                            _ => {
                                last_subexpr_str = Some(rw);
                                self.fits_single_line = could_fit_single_line && all_in_one_line;
                            }
                        }
                    }
                }
            }
        }

        last_subexpr_str = last_subexpr_str.or_else(|| last.rewrite(context, last_shape));
        self.rewrites.push(last_subexpr_str?);
        Some(())
    }

    fn join_rewrites(
        &self,
        context: &RewriteContext,
        child_shape: Shape,
        block_like_iter: impl Iterator<Item = bool>,
    ) -> Option<String> {
        let connector = if self.fits_single_line {
            // Yay, we can put everything on one line.
            Cow::from("")
        } else {
            // Use new lines.
            if *context.force_one_line_chain.borrow() {
                return None;
            }
            child_shape.to_string_with_newline(context.config)
        };

        let mut rewrite_iter = self.rewrites.iter();
        let mut result = rewrite_iter.next().unwrap().clone();
        let children_iter = self.children.iter();
        let iter = rewrite_iter.zip(block_like_iter).zip(children_iter);

        for ((rewrite, prev_is_block_like), chain_item) in iter {
            match chain_item.kind {
                ChainItemKind::Comment(_, CommentPosition::Back) => result.push(' '),
                ChainItemKind::Comment(_, CommentPosition::Top) => result.push_str(&connector),
                _ => {
                    if !prev_is_block_like {
                        result.push_str(&connector);
                    }
                }
            }
            result.push_str(&rewrite);
        }

        Some(result)
    }
}

// Formats a chain using block indent.
struct ChainFormatterBlock<'a> {
    shared: ChainFormatterShared<'a>,
    // For each rewrite, whether the corresponding item is block-like.
    is_block_like: Vec<bool>,
}

impl<'a> ChainFormatterBlock<'a> {
    fn new(chain: &'a Chain) -> ChainFormatterBlock<'a> {
        ChainFormatterBlock {
            shared: ChainFormatterShared::new(chain),
            is_block_like: Vec::with_capacity(chain.children.len() + 1),
        }
    }
}

impl<'a> ChainFormatter for ChainFormatterBlock<'a> {
    fn format_root(
        &mut self,
        parent: &ChainItem,
        context: &RewriteContext,
        shape: Shape,
    ) -> Option<()> {
        let mut root_rewrite: String = parent.rewrite(context, shape)?;

        let mut root_ends_with_block = parent.kind.is_block_like(context, &root_rewrite);
        let tab_width = context.config.tab_spaces().saturating_sub(shape.offset);

        while root_rewrite.len() <= tab_width && !root_rewrite.contains('\n') {
            let item = &self.shared.children[0];
            if let ChainItemKind::Comment(..) = item.kind {
                break;
            }
            let shape = shape.offset_left(root_rewrite.len())?;
            match &item.rewrite(context, shape) {
                Some(rewrite) => root_rewrite.push_str(rewrite),
                None => break,
            }

            root_ends_with_block = item.kind.is_block_like(context, &root_rewrite);

            self.shared.children = &self.shared.children[1..];
            if self.shared.children.is_empty() {
                break;
            }
        }
        self.is_block_like.push(root_ends_with_block);
        self.shared.rewrites.push(root_rewrite);
        Some(())
    }

    fn child_shape(&self, context: &RewriteContext, shape: Shape) -> Option<Shape> {
        Some(
            if self.is_block_like[0] {
                shape.block_indent(0)
            } else {
                shape.block_indent(context.config.tab_spaces())
            }.with_max_width(context.config),
        )
    }

    fn format_children(&mut self, context: &RewriteContext, child_shape: Shape) -> Option<()> {
        for item in &self.shared.children[..self.shared.children.len() - 1] {
            let rewrite = item.rewrite(context, child_shape)?;
            self.is_block_like
                .push(item.kind.is_block_like(context, &rewrite));
            self.shared.rewrites.push(rewrite);
        }
        Some(())
    }

    fn format_last_child(
        &mut self,
        context: &RewriteContext,
        shape: Shape,
        child_shape: Shape,
    ) -> Option<()> {
        self.shared
            .format_last_child(true, context, shape, child_shape)
    }

    fn join_rewrites(&self, context: &RewriteContext, child_shape: Shape) -> Option<String> {
        self.shared
            .join_rewrites(context, child_shape, self.is_block_like.iter().cloned())
    }

    fn pure_root(&mut self) -> Option<String> {
        self.shared.pure_root()
    }
}

// Format a chain using visual indent.
struct ChainFormatterVisual<'a> {
    shared: ChainFormatterShared<'a>,
    // The extra offset from the chain's shape to the position of the `.`
    offset: usize,
}

impl<'a> ChainFormatterVisual<'a> {
    fn new(chain: &'a Chain) -> ChainFormatterVisual<'a> {
        ChainFormatterVisual {
            shared: ChainFormatterShared::new(chain),
            offset: 0,
        }
    }
}

impl<'a> ChainFormatter for ChainFormatterVisual<'a> {
    fn format_root(
        &mut self,
        parent: &ChainItem,
        context: &RewriteContext,
        shape: Shape,
    ) -> Option<()> {
        let parent_shape = shape.visual_indent(0);
        let mut root_rewrite = parent.rewrite(context, parent_shape)?;
        let multiline = root_rewrite.contains('\n');
        self.offset = if multiline {
            last_line_width(&root_rewrite).saturating_sub(shape.used_width())
        } else {
            trimmed_last_line_width(&root_rewrite)
        };

        if !multiline || parent.kind.is_block_like(context, &root_rewrite) {
            let item = &self.shared.children[0];
            if let ChainItemKind::Comment(..) = item.kind {
                self.shared.rewrites.push(root_rewrite);
                return Some(());
            }
            let child_shape = parent_shape
                .visual_indent(self.offset)
                .sub_width(self.offset)?;
            let rewrite = item.rewrite(context, child_shape)?;
            match wrap_str(rewrite, context.config.max_width(), shape) {
                Some(rewrite) => root_rewrite.push_str(&rewrite),
                None => {
                    // We couldn't fit in at the visual indent, try the last
                    // indent.
                    let rewrite = item.rewrite(context, parent_shape)?;
                    root_rewrite.push_str(&rewrite);
                    self.offset = 0;
                }
            }

            self.shared.children = &self.shared.children[1..];
        }

        self.shared.rewrites.push(root_rewrite);
        Some(())
    }

    fn child_shape(&self, context: &RewriteContext, shape: Shape) -> Option<Shape> {
        shape
            .with_max_width(context.config)
            .offset_left(self.offset)
            .map(|s| s.visual_indent(0))
    }

    fn format_children(&mut self, context: &RewriteContext, child_shape: Shape) -> Option<()> {
        for item in &self.shared.children[..self.shared.children.len() - 1] {
            let rewrite = item.rewrite(context, child_shape)?;
            self.shared.rewrites.push(rewrite);
        }
        Some(())
    }

    fn format_last_child(
        &mut self,
        context: &RewriteContext,
        shape: Shape,
        child_shape: Shape,
    ) -> Option<()> {
        self.shared
            .format_last_child(false, context, shape, child_shape)
    }

    fn join_rewrites(&self, context: &RewriteContext, child_shape: Shape) -> Option<String> {
        self.shared
            .join_rewrites(context, child_shape, iter::repeat(false))
    }

    fn pure_root(&mut self) -> Option<String> {
        self.shared.pure_root()
    }
}

// States whether an expression's last line exclusively consists of closing
// parens, braces, and brackets in its idiomatic formatting.
fn is_block_expr(context: &RewriteContext, expr: &ast::Expr, repr: &str) -> bool {
    match expr.node {
        ast::ExprKind::Mac(..)
        | ast::ExprKind::Call(..)
        | ast::ExprKind::MethodCall(..)
        | ast::ExprKind::Struct(..)
        | ast::ExprKind::While(..)
        | ast::ExprKind::WhileLet(..)
        | ast::ExprKind::If(..)
        | ast::ExprKind::IfLet(..)
        | ast::ExprKind::Block(..)
        | ast::ExprKind::Loop(..)
        | ast::ExprKind::ForLoop(..)
        | ast::ExprKind::Match(..) => repr.contains('\n'),
        ast::ExprKind::Paren(ref expr)
        | ast::ExprKind::Binary(_, _, ref expr)
        | ast::ExprKind::Index(_, ref expr)
        | ast::ExprKind::Unary(_, ref expr)
        | ast::ExprKind::Closure(_, _, _, _, ref expr, _)
        | ast::ExprKind::Try(ref expr)
        | ast::ExprKind::Yield(Some(ref expr)) => is_block_expr(context, expr, repr),
        // This can only be a string lit
        ast::ExprKind::Lit(_) => {
            repr.contains('\n') && trimmed_last_line_width(repr) <= context.config.tab_spaces()
        }
        _ => false,
    }
}
