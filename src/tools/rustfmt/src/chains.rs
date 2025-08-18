//! Formatting of chained expressions, i.e., expressions that are chained by
//! dots: struct and enum field access, method calls, and try shorthand (`?`).
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
//! ```text
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
//! ```text
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
//! ```text
//! let a = foo.bar
//!     .baz()
//!     .qux
//! ```
//!
//! Visual:
//!
//! ```text
//! let a = foo.bar
//!            .baz()
//!            .qux
//! ```

use std::borrow::Cow;
use std::cmp::min;

use rustc_ast::ast;
use rustc_span::{BytePos, Span, symbol};
use tracing::debug;

use crate::comment::{CharClasses, FullCodeCharKind, RichChar, rewrite_comment};
use crate::config::{IndentStyle, StyleEdition};
use crate::expr::rewrite_call;
use crate::lists::extract_pre_comment;
use crate::macros::convert_try_mac;
use crate::rewrite::{Rewrite, RewriteContext, RewriteError, RewriteErrorExt, RewriteResult};
use crate::shape::Shape;
use crate::source_map::SpanUtils;
use crate::utils::{
    self, filtered_str_fits, first_line_width, last_line_extendable, last_line_width, mk_sp,
    rewrite_ident, trimmed_last_line_width, wrap_str,
};

use thin_vec::ThinVec;

/// Provides the original input contents from the span
/// of a chain element with trailing spaces trimmed.
fn format_overflow_style(span: Span, context: &RewriteContext<'_>) -> Option<String> {
    // TODO(ding-young): Currently returning None when the given span is out of the range
    // covered by the snippet provider. If this is a common cause for internal
    // rewrite failure, add a new enum variant and return RewriteError instead of None
    context.snippet_provider.span_to_snippet(span).map(|s| {
        s.lines()
            .map(|l| l.trim_end())
            .collect::<Vec<_>>()
            .join("\n")
    })
}

fn format_chain_item(
    item: &ChainItem,
    context: &RewriteContext<'_>,
    rewrite_shape: Shape,
    allow_overflow: bool,
) -> RewriteResult {
    if allow_overflow {
        // TODO(ding-young): Consider calling format_overflow_style()
        // only when item.rewrite_result() returns RewriteError::ExceedsMaxWidth.
        // It may be inappropriate to call format_overflow_style on other RewriteError
        // since the current approach retries formatting if allow_overflow is true
        item.rewrite_result(context, rewrite_shape)
            .or_else(|_| format_overflow_style(item.span, context).unknown_error())
    } else {
        item.rewrite_result(context, rewrite_shape)
    }
}

fn get_block_child_shape(
    prev_ends_with_block: bool,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> Shape {
    if prev_ends_with_block {
        shape.block_indent(0)
    } else {
        shape.block_indent(context.config.tab_spaces())
    }
    .with_max_width(context.config)
}

fn get_visual_style_child_shape(
    context: &RewriteContext<'_>,
    shape: Shape,
    offset: usize,
    parent_overflowing: bool,
) -> Option<Shape> {
    if !parent_overflowing {
        shape
            .with_max_width(context.config)
            .offset_left(offset)
            .map(|s| s.visual_indent(0))
    } else {
        Some(shape.visual_indent(offset))
    }
}

pub(crate) fn rewrite_chain(
    expr: &ast::Expr,
    context: &RewriteContext<'_>,
    shape: Shape,
) -> RewriteResult {
    let chain = Chain::from_ast(expr, context);
    debug!("rewrite_chain {:?} {:?}", chain, shape);

    // If this is just an expression with some `?`s, then format it trivially and
    // return early.
    if chain.children.is_empty() {
        return chain.parent.rewrite_result(context, shape);
    }

    chain.rewrite_result(context, shape)
}

#[derive(Debug)]
enum CommentPosition {
    Back,
    Top,
}

/// Information about an expression in a chain.
struct SubExpr {
    expr: ast::Expr,
    is_method_call_receiver: bool,
}

/// An expression plus trailing `?`s to be formatted together.
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
    Parent {
        expr: ast::Expr,
        parens: bool,
    },
    MethodCall(
        ast::PathSegment,
        Vec<ast::GenericArg>,
        ThinVec<Box<ast::Expr>>,
    ),
    StructField(symbol::Ident),
    TupleField(symbol::Ident, bool),
    Await,
    Yield,
    Comment(String, CommentPosition),
}

impl ChainItemKind {
    fn is_block_like(&self, context: &RewriteContext<'_>, reps: &str) -> bool {
        match self {
            ChainItemKind::Parent { expr, .. } => utils::is_block_expr(context, expr, reps),
            ChainItemKind::MethodCall(..)
            | ChainItemKind::StructField(..)
            | ChainItemKind::TupleField(..)
            | ChainItemKind::Await
            | ChainItemKind::Yield
            | ChainItemKind::Comment(..) => false,
        }
    }

    fn is_tup_field_access(expr: &ast::Expr) -> bool {
        match expr.kind {
            ast::ExprKind::Field(_, ref field) => {
                field.name.as_str().chars().all(|c| c.is_digit(10))
            }
            _ => false,
        }
    }

    fn from_ast(
        context: &RewriteContext<'_>,
        expr: &ast::Expr,
        is_method_call_receiver: bool,
    ) -> (ChainItemKind, Span) {
        let (kind, span) = match expr.kind {
            ast::ExprKind::MethodCall(ref call) => {
                let types = if let Some(ref generic_args) = call.seg.args {
                    if let ast::GenericArgs::AngleBracketed(ref data) = **generic_args {
                        data.args
                            .iter()
                            .filter_map(|x| match x {
                                ast::AngleBracketedArg::Arg(ref generic_arg) => {
                                    Some(generic_arg.clone())
                                }
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                };
                let span = mk_sp(call.receiver.span.hi(), expr.span.hi());
                let kind = ChainItemKind::MethodCall(call.seg.clone(), types, call.args.clone());
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
            ast::ExprKind::Await(ref nested, _) => {
                let span = mk_sp(nested.span.hi(), expr.span.hi());
                (ChainItemKind::Await, span)
            }
            ast::ExprKind::Yield(ast::YieldKind::Postfix(ref nested)) => {
                let span = mk_sp(nested.span.hi(), expr.span.hi());
                (ChainItemKind::Yield, span)
            }
            _ => {
                return (
                    ChainItemKind::Parent {
                        expr: expr.clone(),
                        parens: is_method_call_receiver && should_add_parens(expr),
                    },
                    expr.span,
                );
            }
        };

        // Remove comments from the span.
        let lo = context.snippet_provider.span_before(span, ".");
        (kind, mk_sp(lo, span.hi()))
    }
}

impl Rewrite for ChainItem {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        let shape = shape
            .sub_width(self.tries)
            .max_width_error(shape.width, self.span)?;
        let rewrite = match self.kind {
            ChainItemKind::Parent {
                ref expr,
                parens: true,
            } => crate::expr::rewrite_paren(context, &expr, shape, expr.span)?,
            ChainItemKind::Parent {
                ref expr,
                parens: false,
            } => expr.rewrite_result(context, shape)?,
            ChainItemKind::MethodCall(ref segment, ref types, ref exprs) => {
                Self::rewrite_method_call(segment.ident, types, exprs, self.span, context, shape)?
            }
            ChainItemKind::StructField(ident) => format!(".{}", rewrite_ident(context, ident)),
            ChainItemKind::TupleField(ident, nested) => format!(
                "{}.{}",
                if nested && context.config.style_edition() <= StyleEdition::Edition2021 {
                    " "
                } else {
                    ""
                },
                rewrite_ident(context, ident)
            ),
            ChainItemKind::Await => ".await".to_owned(),
            ChainItemKind::Yield => ".yield".to_owned(),
            ChainItemKind::Comment(ref comment, _) => {
                rewrite_comment(comment, false, shape, context.config)?
            }
        };
        Ok(format!("{rewrite}{}", "?".repeat(self.tries)))
    }
}

impl ChainItem {
    fn new(context: &RewriteContext<'_>, expr: &SubExpr, tries: usize) -> ChainItem {
        let (kind, span) =
            ChainItemKind::from_ast(context, &expr.expr, expr.is_method_call_receiver);
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
        matches!(self.kind, ChainItemKind::Comment(..))
    }

    fn rewrite_method_call(
        method_name: symbol::Ident,
        types: &[ast::GenericArg],
        args: &[Box<ast::Expr>],
        span: Span,
        context: &RewriteContext<'_>,
        shape: Shape,
    ) -> RewriteResult {
        let type_str = if types.is_empty() {
            String::new()
        } else {
            let type_list = types
                .iter()
                .map(|ty| ty.rewrite_result(context, shape))
                .collect::<Result<Vec<_>, RewriteError>>()?;

            format!("::<{}>", type_list.join(", "))
        };
        let callee_str = format!(".{}{}", rewrite_ident(context, method_name), type_str);
        rewrite_call(context, &callee_str, &args, span, shape)
    }
}

#[derive(Debug)]
struct Chain {
    parent: ChainItem,
    children: Vec<ChainItem>,
}

impl Chain {
    fn from_ast(expr: &ast::Expr, context: &RewriteContext<'_>) -> Chain {
        let subexpr_list = Self::make_subexpr_list(expr, context);

        // Un-parse the expression tree into ChainItems
        let mut rev_children = vec![];
        let mut sub_tries = 0;
        for subexpr in &subexpr_list {
            match subexpr.expr.kind {
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

        fn is_post_comment(s: &str) -> bool {
            let comment_start_index = s.chars().position(|c| c == '/');
            if comment_start_index.is_none() {
                return false;
            }

            let newline_index = s.chars().position(|c| c == '\n');
            if newline_index.is_none() {
                return true;
            }

            comment_start_index.unwrap() < newline_index.unwrap()
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
            let trimmed_snippet = trim_tries(post_comment_snippet);
            if is_post_comment(&trimmed_snippet) {
                children.push(ChainItem::comment(
                    post_comment_span,
                    trimmed_snippet.trim().to_owned(),
                    CommentPosition::Back,
                ));
                *prev_span_end = post_comment_span.hi();
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
                let pre_comment_snippet = trim_tries(context.snippet(pre_comment_span));
                let (pre_comment, _) = extract_pre_comment(&pre_comment_snippet);
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
    fn make_subexpr_list(expr: &ast::Expr, context: &RewriteContext<'_>) -> Vec<SubExpr> {
        let mut subexpr_list = vec![SubExpr {
            expr: expr.clone(),
            is_method_call_receiver: false,
        }];

        while let Some(subexpr) = Self::pop_expr_chain(subexpr_list.last().unwrap(), context) {
            subexpr_list.push(subexpr);
        }

        subexpr_list
    }

    // Returns the expression's subexpression, if it exists. When the subexpr
    // is a try! macro, we'll convert it to shorthand when the option is set.
    fn pop_expr_chain(expr: &SubExpr, context: &RewriteContext<'_>) -> Option<SubExpr> {
        match expr.expr.kind {
            ast::ExprKind::MethodCall(ref call) => Some(SubExpr {
                expr: Self::convert_try(&call.receiver, context),
                is_method_call_receiver: true,
            }),
            ast::ExprKind::Field(ref subexpr, _)
            | ast::ExprKind::Try(ref subexpr)
            | ast::ExprKind::Await(ref subexpr, _)
            | ast::ExprKind::Yield(ast::YieldKind::Postfix(ref subexpr)) => Some(SubExpr {
                expr: Self::convert_try(subexpr, context),
                is_method_call_receiver: false,
            }),
            _ => None,
        }
    }

    fn convert_try(expr: &ast::Expr, context: &RewriteContext<'_>) -> ast::Expr {
        match expr.kind {
            ast::ExprKind::MacCall(ref mac) if context.config.use_try_shorthand() => {
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
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        debug!("rewrite chain {:?} {:?}", self, shape);

        let mut formatter = match context.config.indent_style() {
            IndentStyle::Block => {
                Box::new(ChainFormatterBlock::new(self)) as Box<dyn ChainFormatter>
            }
            IndentStyle::Visual => {
                Box::new(ChainFormatterVisual::new(self)) as Box<dyn ChainFormatter>
            }
        };

        formatter.format_root(&self.parent, context, shape)?;
        if let Some(result) = formatter.pure_root() {
            return wrap_str(result, context.config.max_width(), shape)
                .max_width_error(shape.width, self.parent.span);
        }

        let first = self.children.first().unwrap_or(&self.parent);
        let last = self.children.last().unwrap_or(&self.parent);
        let children_span = mk_sp(first.span.lo(), last.span.hi());
        let full_span = self.parent.span.with_hi(children_span.hi());

        // Decide how to layout the rest of the chain.
        let child_shape = formatter
            .child_shape(context, shape)
            .max_width_error(shape.width, children_span)?;

        formatter.format_children(context, child_shape)?;
        formatter.format_last_child(context, shape, child_shape)?;

        let result = formatter.join_rewrites(context, child_shape)?;
        wrap_str(result, context.config.max_width(), shape).max_width_error(shape.width, full_span)
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
    // ```text
    // foo.bar
    //     .baz()
    // ```
    // If `bar` were not part of the root, then foo would be orphaned and 'float'.
    fn format_root(
        &mut self,
        parent: &ChainItem,
        context: &RewriteContext<'_>,
        shape: Shape,
    ) -> Result<(), RewriteError>;
    fn child_shape(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<Shape>;
    fn format_children(
        &mut self,
        context: &RewriteContext<'_>,
        child_shape: Shape,
    ) -> Result<(), RewriteError>;
    fn format_last_child(
        &mut self,
        context: &RewriteContext<'_>,
        shape: Shape,
        child_shape: Shape,
    ) -> Result<(), RewriteError>;
    fn join_rewrites(&self, context: &RewriteContext<'_>, child_shape: Shape) -> RewriteResult;
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
    // Whether elements are allowed to overflow past the max_width limit
    allow_overflow: bool,
}

impl<'a> ChainFormatterShared<'a> {
    fn new(chain: &'a Chain) -> ChainFormatterShared<'a> {
        ChainFormatterShared {
            children: &chain.children,
            rewrites: Vec::with_capacity(chain.children.len() + 1),
            fits_single_line: false,
            child_count: chain.children.len(),
            // TODO(calebcartwright)
            allow_overflow: false,
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

    fn format_children(
        &mut self,
        context: &RewriteContext<'_>,
        child_shape: Shape,
    ) -> Result<(), RewriteError> {
        for item in &self.children[..self.children.len() - 1] {
            let rewrite = format_chain_item(item, context, child_shape, self.allow_overflow)?;
            self.rewrites.push(rewrite);
        }
        Ok(())
    }

    // Rewrite the last child. The last child of a chain requires special treatment. We need to
    // know whether 'overflowing' the last child make a better formatting:
    //
    // A chain with overflowing the last child:
    // ```text
    // parent.child1.child2.last_child(
    //     a,
    //     b,
    //     c,
    // )
    // ```
    //
    // A chain without overflowing the last child (in vertical layout):
    // ```text
    // parent
    //     .child1
    //     .child2
    //     .last_child(a, b, c)
    // ```
    //
    // In particular, overflowing is effective when the last child is a method with a multi-lined
    // block-like argument (e.g., closure):
    // ```text
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
        context: &RewriteContext<'_>,
        shape: Shape,
        child_shape: Shape,
    ) -> Result<(), RewriteError> {
        let last = self.children.last().unknown_error()?;
        let extendable = may_extend && last_line_extendable(&self.rewrites[0]);
        let prev_last_line_width = last_line_width(&self.rewrites[0]);

        // Total of all items excluding the last.
        let almost_total = if extendable {
            prev_last_line_width
        } else {
            self.rewrites
                .iter()
                .map(|rw| utils::unicode_str_width(rw))
                .sum()
        } + last.tries;
        let one_line_budget = if self.child_count == 1 {
            shape.width
        } else {
            min(shape.width, context.config.chain_width())
        }
        .saturating_sub(almost_total);

        let all_in_one_line = !self.children.iter().any(ChainItem::is_comment)
            && self.rewrites.iter().all(|s| !s.contains('\n'))
            && one_line_budget > 0;
        let last_shape = if all_in_one_line {
            shape
                .sub_width(last.tries)
                .max_width_error(shape.width, last.span)?
        } else if extendable {
            child_shape
                .sub_width(last.tries)
                .max_width_error(child_shape.width, last.span)?
        } else {
            child_shape
                .sub_width(shape.rhs_overhead(context.config) + last.tries)
                .max_width_error(child_shape.width, last.span)?
        };

        let mut last_subexpr_str = None;
        if all_in_one_line || extendable {
            // First we try to 'overflow' the last child and see if it looks better than using
            // vertical layout.
            let one_line_shape = if context.use_block_indent() {
                last_shape.offset_left(almost_total)
            } else {
                last_shape
                    .visual_indent(almost_total)
                    .sub_width(almost_total)
            };

            if let Some(one_line_shape) = one_line_shape {
                if let Ok(rw) = last.rewrite_result(context, one_line_shape) {
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
                            .sub_width(shape.rhs_overhead(context.config) + last.tries)
                            .max_width_error(child_shape.width, last.span)?;
                        match last.rewrite_result(context, last_shape) {
                            Ok(ref new_rw) if !could_fit_single_line => {
                                last_subexpr_str = Some(new_rw.clone());
                            }
                            Ok(ref new_rw) if new_rw.lines().count() >= line_count => {
                                last_subexpr_str = Some(rw);
                                self.fits_single_line = could_fit_single_line && all_in_one_line;
                            }
                            Ok(new_rw) => {
                                last_subexpr_str = Some(new_rw);
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

        let last_shape = if context.use_block_indent() {
            last_shape
        } else {
            child_shape
                .sub_width(shape.rhs_overhead(context.config) + last.tries)
                .max_width_error(child_shape.width, last.span)?
        };

        let last_subexpr_str =
            last_subexpr_str.unwrap_or(last.rewrite_result(context, last_shape)?);
        self.rewrites.push(last_subexpr_str);
        Ok(())
    }

    fn join_rewrites(&self, context: &RewriteContext<'_>, child_shape: Shape) -> RewriteResult {
        let connector = if self.fits_single_line {
            // Yay, we can put everything on one line.
            Cow::from("")
        } else {
            // Use new lines.
            if context.force_one_line_chain.get() {
                return Err(RewriteError::ExceedsMaxWidth {
                    configured_width: child_shape.width,
                    span: self.children.last().unknown_error()?.span,
                });
            }
            child_shape.to_string_with_newline(context.config)
        };

        let mut rewrite_iter = self.rewrites.iter();
        let mut result = rewrite_iter.next().unwrap().clone();
        let children_iter = self.children.iter();
        let iter = rewrite_iter.zip(children_iter);

        for (rewrite, chain_item) in iter {
            match chain_item.kind {
                ChainItemKind::Comment(_, CommentPosition::Back) => result.push(' '),
                ChainItemKind::Comment(_, CommentPosition::Top) => result.push_str(&connector),
                _ => result.push_str(&connector),
            }
            result.push_str(rewrite);
        }

        Ok(result)
    }
}

// Formats a chain using block indent.
struct ChainFormatterBlock<'a> {
    shared: ChainFormatterShared<'a>,
    root_ends_with_block: bool,
}

impl<'a> ChainFormatterBlock<'a> {
    fn new(chain: &'a Chain) -> ChainFormatterBlock<'a> {
        ChainFormatterBlock {
            shared: ChainFormatterShared::new(chain),
            root_ends_with_block: false,
        }
    }
}

impl<'a> ChainFormatter for ChainFormatterBlock<'a> {
    fn format_root(
        &mut self,
        parent: &ChainItem,
        context: &RewriteContext<'_>,
        shape: Shape,
    ) -> Result<(), RewriteError> {
        let mut root_rewrite: String = parent.rewrite_result(context, shape)?;

        let mut root_ends_with_block = parent.kind.is_block_like(context, &root_rewrite);
        let tab_width = context.config.tab_spaces().saturating_sub(shape.offset);

        while root_rewrite.len() <= tab_width && !root_rewrite.contains('\n') {
            let item = &self.shared.children[0];
            if let ChainItemKind::Comment(..) = item.kind {
                break;
            }
            let shape = shape
                .offset_left(root_rewrite.len())
                .max_width_error(shape.width, item.span)?;
            match &item.rewrite_result(context, shape) {
                Ok(rewrite) => root_rewrite.push_str(rewrite),
                Err(_) => break,
            }

            root_ends_with_block = last_line_extendable(&root_rewrite);

            self.shared.children = &self.shared.children[1..];
            if self.shared.children.is_empty() {
                break;
            }
        }
        self.shared.rewrites.push(root_rewrite);
        self.root_ends_with_block = root_ends_with_block;
        Ok(())
    }

    fn child_shape(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<Shape> {
        let block_end = self.root_ends_with_block;
        Some(get_block_child_shape(block_end, context, shape))
    }

    fn format_children(
        &mut self,
        context: &RewriteContext<'_>,
        child_shape: Shape,
    ) -> Result<(), RewriteError> {
        self.shared.format_children(context, child_shape)
    }

    fn format_last_child(
        &mut self,
        context: &RewriteContext<'_>,
        shape: Shape,
        child_shape: Shape,
    ) -> Result<(), RewriteError> {
        self.shared
            .format_last_child(true, context, shape, child_shape)
    }

    fn join_rewrites(&self, context: &RewriteContext<'_>, child_shape: Shape) -> RewriteResult {
        self.shared.join_rewrites(context, child_shape)
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
        context: &RewriteContext<'_>,
        shape: Shape,
    ) -> Result<(), RewriteError> {
        let parent_shape = shape.visual_indent(0);
        let mut root_rewrite = parent.rewrite_result(context, parent_shape)?;
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
                return Ok(());
            }
            let child_shape = parent_shape
                .visual_indent(self.offset)
                .sub_width(self.offset)
                .max_width_error(parent_shape.width, item.span)?;
            let rewrite = item.rewrite_result(context, child_shape)?;
            if filtered_str_fits(&rewrite, context.config.max_width(), shape) {
                root_rewrite.push_str(&rewrite);
            } else {
                // We couldn't fit in at the visual indent, try the last
                // indent.
                let rewrite = item.rewrite_result(context, parent_shape)?;
                root_rewrite.push_str(&rewrite);
                self.offset = 0;
            }

            self.shared.children = &self.shared.children[1..];
        }

        self.shared.rewrites.push(root_rewrite);
        Ok(())
    }

    fn child_shape(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<Shape> {
        get_visual_style_child_shape(
            context,
            shape,
            self.offset,
            // TODO(calebcartwright): self.shared.permissibly_overflowing_parent,
            false,
        )
    }

    fn format_children(
        &mut self,
        context: &RewriteContext<'_>,
        child_shape: Shape,
    ) -> Result<(), RewriteError> {
        self.shared.format_children(context, child_shape)
    }

    fn format_last_child(
        &mut self,
        context: &RewriteContext<'_>,
        shape: Shape,
        child_shape: Shape,
    ) -> Result<(), RewriteError> {
        self.shared
            .format_last_child(false, context, shape, child_shape)
    }

    fn join_rewrites(&self, context: &RewriteContext<'_>, child_shape: Shape) -> RewriteResult {
        self.shared.join_rewrites(context, child_shape)
    }

    fn pure_root(&mut self) -> Option<String> {
        self.shared.pure_root()
    }
}

/// Removes try operators (`?`s) that appear in the given string. If removing
/// them leaves an empty line, remove that line as well unless it is the first
/// line (we need the first newline for detecting pre/post comment).
fn trim_tries(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut line_buffer = String::with_capacity(s.len());
    for (kind, rich_char) in CharClasses::new(s.chars()) {
        match rich_char.get_char() {
            '\n' => {
                if result.is_empty() || !line_buffer.trim().is_empty() {
                    result.push_str(&line_buffer);
                    result.push('\n')
                }
                line_buffer.clear();
            }
            '?' if kind == FullCodeCharKind::Normal => continue,
            c => line_buffer.push(c),
        }
    }
    if !line_buffer.trim().is_empty() {
        result.push_str(&line_buffer);
    }
    result
}

/// Whether a method call's receiver needs parenthesis, like
/// ```rust,ignore
/// || .. .method();
/// || 1.. .method();
/// 1. .method();
/// ```
/// Which all need parenthesis or a space before `.method()`.
fn should_add_parens(expr: &ast::Expr) -> bool {
    match expr.kind {
        ast::ExprKind::Lit(ref lit) => crate::expr::lit_ends_in_dot(lit),
        ast::ExprKind::Closure(ref cl) => match cl.body.kind {
            ast::ExprKind::Range(_, _, ast::RangeLimits::HalfOpen) => true,
            ast::ExprKind::Lit(ref lit) => crate::expr::lit_ends_in_dot(lit),
            _ => false,
        },
        _ => false,
    }
}
