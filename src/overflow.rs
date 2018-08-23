// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Rewrite a list some items with overflow.
// FIXME: Replace `ToExpr` with some enum.

use config::lists::*;
use syntax::ast;
use syntax::parse::token::DelimToken;
use syntax::source_map::Span;

use closures;
use expr::{is_every_expr_simple, is_method_call, is_nested_call, maybe_get_args_offset, ToExpr};
use lists::{definitive_tactic, itemize_list, write_list, ListFormatting, ListItem, Separator};
use rewrite::{Rewrite, RewriteContext};
use shape::Shape;
use source_map::SpanUtils;
use spanned::Spanned;
use utils::{count_newlines, extra_offset, first_line_width, last_line_width, mk_sp};

use std::cmp::min;

const SHORT_ITEM_THRESHOLD: usize = 10;

pub fn rewrite_with_parens<T>(
    context: &RewriteContext,
    ident: &str,
    items: &[&T],
    shape: Shape,
    span: Span,
    item_max_width: usize,
    force_separator_tactic: Option<SeparatorTactic>,
) -> Option<String>
where
    T: Rewrite + ToExpr + Spanned,
{
    Context::new(
        context,
        items,
        ident,
        shape,
        span,
        "(",
        ")",
        item_max_width,
        force_separator_tactic,
        None,
    ).rewrite(shape)
}

pub fn rewrite_with_angle_brackets<T>(
    context: &RewriteContext,
    ident: &str,
    items: &[&T],
    shape: Shape,
    span: Span,
) -> Option<String>
where
    T: Rewrite + ToExpr + Spanned,
{
    Context::new(
        context,
        items,
        ident,
        shape,
        span,
        "<",
        ">",
        context.config.max_width(),
        None,
        None,
    ).rewrite(shape)
}

pub fn rewrite_with_square_brackets<T>(
    context: &RewriteContext,
    name: &str,
    items: &[&T],
    shape: Shape,
    span: Span,
    force_separator_tactic: Option<SeparatorTactic>,
    delim_token: Option<DelimToken>,
) -> Option<String>
where
    T: Rewrite + ToExpr + Spanned,
{
    let (lhs, rhs) = match delim_token {
        Some(DelimToken::Paren) => ("(", ")"),
        Some(DelimToken::Brace) => ("{", "}"),
        _ => ("[", "]"),
    };
    Context::new(
        context,
        items,
        name,
        shape,
        span,
        lhs,
        rhs,
        context.config.width_heuristics().array_width,
        force_separator_tactic,
        Some(("[", "]")),
    ).rewrite(shape)
}

struct Context<'a, T: 'a> {
    context: &'a RewriteContext<'a>,
    items: &'a [&'a T],
    ident: &'a str,
    prefix: &'static str,
    suffix: &'static str,
    one_line_shape: Shape,
    nested_shape: Shape,
    span: Span,
    item_max_width: usize,
    one_line_width: usize,
    force_separator_tactic: Option<SeparatorTactic>,
    custom_delims: Option<(&'a str, &'a str)>,
}

impl<'a, T: 'a + Rewrite + ToExpr + Spanned> Context<'a, T> {
    pub fn new(
        context: &'a RewriteContext,
        items: &'a [&'a T],
        ident: &'a str,
        shape: Shape,
        span: Span,
        prefix: &'static str,
        suffix: &'static str,
        item_max_width: usize,
        force_separator_tactic: Option<SeparatorTactic>,
        custom_delims: Option<(&'a str, &'a str)>,
    ) -> Context<'a, T> {
        let used_width = extra_offset(ident, shape);
        // 1 = `()`
        let one_line_width = shape.width.saturating_sub(used_width + 2);

        // 1 = "(" or ")"
        let one_line_shape = shape
            .offset_left(last_line_width(ident) + 1)
            .and_then(|shape| shape.sub_width(1))
            .unwrap_or(Shape { width: 0, ..shape });
        let nested_shape = shape_from_indent_style(context, shape, used_width + 2, used_width + 1);
        Context {
            context,
            items,
            ident,
            one_line_shape,
            nested_shape,
            span,
            prefix,
            suffix,
            item_max_width,
            one_line_width,
            force_separator_tactic,
            custom_delims,
        }
    }

    fn last_item(&self) -> Option<&&T> {
        self.items.last()
    }

    fn items_span(&self) -> Span {
        let span_lo = self
            .context
            .snippet_provider
            .span_after(self.span, self.prefix);
        mk_sp(span_lo, self.span.hi())
    }

    fn rewrite_last_item_with_overflow(
        &self,
        last_list_item: &mut ListItem,
        shape: Shape,
    ) -> Option<String> {
        let last_item = self.last_item()?;
        let rewrite = if let Some(expr) = last_item.to_expr() {
            match expr.node {
                // When overflowing the closure which consists of a single control flow expression,
                // force to use block if its condition uses multi line.
                ast::ExprKind::Closure(..) => {
                    // If the argument consists of multiple closures, we do not overflow
                    // the last closure.
                    if closures::args_have_many_closure(self.items) {
                        None
                    } else {
                        closures::rewrite_last_closure(self.context, expr, shape)
                    }
                }
                _ => expr.rewrite(self.context, shape),
            }
        } else {
            last_item.rewrite(self.context, shape)
        };

        if let Some(rewrite) = rewrite {
            let rewrite_first_line = Some(rewrite[..first_line_width(&rewrite)].to_owned());
            last_list_item.item = rewrite_first_line;
            Some(rewrite)
        } else {
            None
        }
    }

    fn default_tactic(&self, list_items: &[ListItem]) -> DefinitiveListTactic {
        definitive_tactic(
            list_items,
            ListTactic::LimitedHorizontalVertical(self.item_max_width),
            Separator::Comma,
            self.one_line_width,
        )
    }

    fn try_overflow_last_item(&self, list_items: &mut Vec<ListItem>) -> DefinitiveListTactic {
        // 1 = "("
        let combine_arg_with_callee = self.items.len() == 1
            && self.items[0].to_expr().is_some()
            && self.ident.len() + 1 <= self.context.config.tab_spaces();
        let overflow_last = combine_arg_with_callee || can_be_overflowed(self.context, self.items);

        // Replace the last item with its first line to see if it fits with
        // first arguments.
        let placeholder = if overflow_last {
            let old_value = *self.context.force_one_line_chain.borrow();
            if !combine_arg_with_callee {
                if let Some(ref expr) = self.last_item().and_then(|item| item.to_expr()) {
                    if is_method_call(expr) {
                        self.context.force_one_line_chain.replace(true);
                    }
                }
            }
            let result = last_item_shape(
                self.items,
                list_items,
                self.one_line_shape,
                self.item_max_width,
            ).and_then(|arg_shape| {
                self.rewrite_last_item_with_overflow(
                    &mut list_items[self.items.len() - 1],
                    arg_shape,
                )
            });
            self.context.force_one_line_chain.replace(old_value);
            result
        } else {
            None
        };

        let mut tactic = definitive_tactic(
            &*list_items,
            ListTactic::LimitedHorizontalVertical(self.item_max_width),
            Separator::Comma,
            self.one_line_width,
        );

        // Replace the stub with the full overflowing last argument if the rewrite
        // succeeded and its first line fits with the other arguments.
        match (overflow_last, tactic, placeholder) {
            (true, DefinitiveListTactic::Horizontal, Some(ref overflowed))
                if self.items.len() == 1 =>
            {
                // When we are rewriting a nested function call, we restrict the
                // budget for the inner function to avoid them being deeply nested.
                // However, when the inner function has a prefix or a suffix
                // (e.g. `foo() as u32`), this budget reduction may produce poorly
                // formatted code, where a prefix or a suffix being left on its own
                // line. Here we explicitlly check those cases.
                if count_newlines(overflowed) == 1 {
                    let rw = self
                        .items
                        .last()
                        .and_then(|last_item| last_item.rewrite(self.context, self.nested_shape));
                    let no_newline = rw.as_ref().map_or(false, |s| !s.contains('\n'));
                    if no_newline {
                        list_items[self.items.len() - 1].item = rw;
                    } else {
                        list_items[self.items.len() - 1].item = Some(overflowed.to_owned());
                    }
                } else {
                    list_items[self.items.len() - 1].item = Some(overflowed.to_owned());
                }
            }
            (true, DefinitiveListTactic::Horizontal, placeholder @ Some(..)) => {
                list_items[self.items.len() - 1].item = placeholder;
            }
            _ if self.items.len() >= 1 => {
                list_items[self.items.len() - 1].item = self
                    .items
                    .last()
                    .and_then(|last_item| last_item.rewrite(self.context, self.nested_shape));

                // Use horizontal layout for a function with a single argument as long as
                // everything fits in a single line.
                // `self.one_line_width == 0` means vertical layout is forced.
                if self.items.len() == 1
                    && self.one_line_width != 0
                    && !list_items[0].has_comment()
                    && !list_items[0].inner_as_ref().contains('\n')
                    && ::lists::total_item_width(&list_items[0]) <= self.one_line_width
                {
                    tactic = DefinitiveListTactic::Horizontal;
                } else {
                    tactic = self.default_tactic(list_items);

                    if tactic == DefinitiveListTactic::Vertical {
                        if let Some((all_simple, num_args_before)) =
                            maybe_get_args_offset(self.ident, self.items)
                        {
                            let one_line = all_simple
                                && definitive_tactic(
                                    &list_items[..num_args_before],
                                    ListTactic::HorizontalVertical,
                                    Separator::Comma,
                                    self.nested_shape.width,
                                ) == DefinitiveListTactic::Horizontal
                                && definitive_tactic(
                                    &list_items[num_args_before + 1..],
                                    ListTactic::HorizontalVertical,
                                    Separator::Comma,
                                    self.nested_shape.width,
                                ) == DefinitiveListTactic::Horizontal;

                            if one_line {
                                tactic = DefinitiveListTactic::SpecialMacro(num_args_before);
                            };
                        } else if is_every_expr_simple(self.items) && no_long_items(list_items) {
                            tactic = DefinitiveListTactic::Mixed;
                        }
                    }
                }
            }
            _ => (),
        }

        tactic
    }

    fn rewrite_items(&self) -> Option<(bool, String)> {
        let span = self.items_span();
        let items = itemize_list(
            self.context.snippet_provider,
            self.items.iter(),
            self.suffix,
            ",",
            |item| item.span().lo(),
            |item| item.span().hi(),
            |item| item.rewrite(self.context, self.nested_shape),
            span.lo(),
            span.hi(),
            true,
        );
        let mut list_items: Vec<_> = items.collect();

        // Try letting the last argument overflow to the next line with block
        // indentation. If its first line fits on one line with the other arguments,
        // we format the function arguments horizontally.
        let tactic = self.try_overflow_last_item(&mut list_items);
        let trailing_separator = if let Some(tactic) = self.force_separator_tactic {
            tactic
        } else if !self.context.use_block_indent() {
            SeparatorTactic::Never
        } else if tactic == DefinitiveListTactic::Mixed {
            // We are using mixed layout because everything did not fit within a single line.
            SeparatorTactic::Always
        } else {
            self.context.config.trailing_comma()
        };
        let ends_with_newline = match tactic {
            DefinitiveListTactic::Vertical | DefinitiveListTactic::Mixed => {
                self.context.use_block_indent()
            }
            _ => false,
        };

        let fmt = ListFormatting::new(self.nested_shape, self.context.config)
            .tactic(tactic)
            .trailing_separator(trailing_separator)
            .ends_with_newline(ends_with_newline);

        write_list(&list_items, &fmt)
            .map(|items_str| (tactic == DefinitiveListTactic::Horizontal, items_str))
    }

    fn wrap_items(&self, items_str: &str, shape: Shape, is_extendable: bool) -> String {
        let shape = Shape {
            width: shape.width.saturating_sub(last_line_width(self.ident)),
            ..shape
        };

        let (prefix, suffix) = match self.custom_delims {
            Some((lhs, rhs)) => (lhs, rhs),
            _ => (self.prefix, self.suffix),
        };

        // 2 = `()`
        let fits_one_line = items_str.len() + 2 <= shape.width;
        let extend_width = if items_str.is_empty() {
            2
        } else {
            first_line_width(items_str) + 1
        };
        let nested_indent_str = self
            .nested_shape
            .indent
            .to_string_with_newline(self.context.config);
        let indent_str = shape
            .block()
            .indent
            .to_string_with_newline(self.context.config);
        let mut result = String::with_capacity(
            self.ident.len() + items_str.len() + 2 + indent_str.len() + nested_indent_str.len(),
        );
        result.push_str(self.ident);
        result.push_str(prefix);
        if !self.context.use_block_indent()
            || (self.context.inside_macro() && !items_str.contains('\n') && fits_one_line)
            || (is_extendable && extend_width <= shape.width)
        {
            result.push_str(items_str);
        } else {
            if !items_str.is_empty() {
                result.push_str(&nested_indent_str);
                result.push_str(items_str);
            }
            result.push_str(&indent_str);
        }
        result.push_str(suffix);
        result
    }

    fn rewrite(&self, shape: Shape) -> Option<String> {
        let (extendable, items_str) = self.rewrite_items()?;

        // If we are using visual indent style and failed to format, retry with block indent.
        if !self.context.use_block_indent()
            && need_block_indent(&items_str, self.nested_shape)
            && !extendable
        {
            self.context.use_block.replace(true);
            let result = self.rewrite(shape);
            self.context.use_block.replace(false);
            return result;
        }

        Some(self.wrap_items(&items_str, shape, extendable))
    }
}

fn need_block_indent(s: &str, shape: Shape) -> bool {
    s.lines().skip(1).any(|s| {
        s.find(|c| !char::is_whitespace(c))
            .map_or(false, |w| w + 1 < shape.indent.width())
    })
}

fn can_be_overflowed<'a, T>(context: &RewriteContext, items: &[&T]) -> bool
where
    T: Rewrite + Spanned + ToExpr + 'a,
{
    items
        .last()
        .map_or(false, |x| x.can_be_overflowed(context, items.len()))
}

/// Returns a shape for the last argument which is going to be overflowed.
fn last_item_shape<T>(
    lists: &[&T],
    items: &[ListItem],
    shape: Shape,
    args_max_width: usize,
) -> Option<Shape>
where
    T: Rewrite + Spanned + ToExpr,
{
    let is_nested_call = lists
        .iter()
        .next()
        .and_then(|item| item.to_expr())
        .map_or(false, is_nested_call);
    if items.len() == 1 && !is_nested_call {
        return Some(shape);
    }
    let offset = items.iter().rev().skip(1).fold(0, |acc, i| {
        // 2 = ", "
        acc + 2 + i.inner_as_ref().len()
    });
    Shape {
        width: min(args_max_width, shape.width),
        ..shape
    }.offset_left(offset)
}

fn shape_from_indent_style(
    context: &RewriteContext,
    shape: Shape,
    overhead: usize,
    offset: usize,
) -> Shape {
    let (shape, overhead) = if context.use_block_indent() {
        let shape = shape
            .block()
            .block_indent(context.config.tab_spaces())
            .with_max_width(context.config);
        (shape, 1) // 1 = ","
    } else {
        (shape.visual_indent(offset), overhead)
    };
    Shape {
        width: shape.width.saturating_sub(overhead),
        ..shape
    }
}

fn no_long_items(list: &[ListItem]) -> bool {
    list.iter()
        .all(|item| item.inner_as_ref().len() <= SHORT_ITEM_THRESHOLD)
}
