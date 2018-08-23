// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Format with vertical alignment.

use std::cmp;

use config::lists::*;
use syntax::ast;
use syntax::source_map::{BytePos, Span};

use comment::{combine_strs_with_missing_comments, contains_comment};
use expr::rewrite_field;
use items::{rewrite_struct_field, rewrite_struct_field_prefix};
use lists::{definitive_tactic, itemize_list, write_list, ListFormatting, Separator};
use rewrite::{Rewrite, RewriteContext};
use shape::{Indent, Shape};
use source_map::SpanUtils;
use spanned::Spanned;
use utils::{contains_skip, is_attributes_extendable, mk_sp, rewrite_ident};

pub trait AlignedItem {
    fn skip(&self) -> bool;
    fn get_span(&self) -> Span;
    fn rewrite_prefix(&self, context: &RewriteContext, shape: Shape) -> Option<String>;
    fn rewrite_aligned_item(
        &self,
        context: &RewriteContext,
        shape: Shape,
        prefix_max_width: usize,
    ) -> Option<String>;
}

impl AlignedItem for ast::StructField {
    fn skip(&self) -> bool {
        contains_skip(&self.attrs)
    }

    fn get_span(&self) -> Span {
        self.span()
    }

    fn rewrite_prefix(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        let attrs_str = self.attrs.rewrite(context, shape)?;
        let missing_span = if self.attrs.is_empty() {
            mk_sp(self.span.lo(), self.span.lo())
        } else {
            mk_sp(self.attrs.last().unwrap().span.hi(), self.span.lo())
        };
        let attrs_extendable = self.ident.is_none() && is_attributes_extendable(&attrs_str);
        rewrite_struct_field_prefix(context, self).and_then(|field_str| {
            combine_strs_with_missing_comments(
                context,
                &attrs_str,
                &field_str,
                missing_span,
                shape,
                attrs_extendable,
            )
        })
    }

    fn rewrite_aligned_item(
        &self,
        context: &RewriteContext,
        shape: Shape,
        prefix_max_width: usize,
    ) -> Option<String> {
        rewrite_struct_field(context, self, shape, prefix_max_width)
    }
}

impl AlignedItem for ast::Field {
    fn skip(&self) -> bool {
        contains_skip(&self.attrs)
    }

    fn get_span(&self) -> Span {
        self.span()
    }

    fn rewrite_prefix(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        let attrs_str = self.attrs.rewrite(context, shape)?;
        let name = rewrite_ident(context, self.ident);
        let missing_span = if self.attrs.is_empty() {
            mk_sp(self.span.lo(), self.span.lo())
        } else {
            mk_sp(self.attrs.last().unwrap().span.hi(), self.span.lo())
        };
        combine_strs_with_missing_comments(
            context,
            &attrs_str,
            name,
            missing_span,
            shape,
            is_attributes_extendable(&attrs_str),
        )
    }

    fn rewrite_aligned_item(
        &self,
        context: &RewriteContext,
        shape: Shape,
        prefix_max_width: usize,
    ) -> Option<String> {
        rewrite_field(context, self, shape, prefix_max_width)
    }
}

pub fn rewrite_with_alignment<T: AlignedItem>(
    fields: &[T],
    context: &RewriteContext,
    shape: Shape,
    span: Span,
    one_line_width: usize,
) -> Option<String> {
    let (spaces, group_index) = if context.config.struct_field_align_threshold() > 0 {
        group_aligned_items(context, fields)
    } else {
        ("", fields.len() - 1)
    };
    let init = &fields[0..group_index + 1];
    let rest = &fields[group_index + 1..];
    let init_last_pos = if rest.is_empty() {
        span.hi()
    } else {
        // Decide whether the missing comments should stick to init or rest.
        let init_hi = init[init.len() - 1].get_span().hi();
        let rest_lo = rest[0].get_span().lo();
        let missing_span = mk_sp(init_hi, rest_lo);
        let missing_span = mk_sp(
            context.snippet_provider.span_after(missing_span, ","),
            missing_span.hi(),
        );

        let snippet = context.snippet(missing_span);
        if snippet.trim_left().starts_with("//") {
            let offset = snippet.lines().next().map_or(0, |l| l.len());
            // 2 = "," + "\n"
            init_hi + BytePos(offset as u32 + 2)
        } else if snippet.trim_left().starts_with("/*") {
            let comment_lines = snippet
                .lines()
                .position(|line| line.trim_right().ends_with("*/"))
                .unwrap_or(0);

            let offset = snippet
                .lines()
                .take(comment_lines + 1)
                .collect::<Vec<_>>()
                .join("\n")
                .len();

            init_hi + BytePos(offset as u32 + 2)
        } else {
            missing_span.lo()
        }
    };
    let init_span = mk_sp(span.lo(), init_last_pos);
    let one_line_width = if rest.is_empty() { one_line_width } else { 0 };
    let result =
        rewrite_aligned_items_inner(context, init, init_span, shape.indent, one_line_width)?;
    if rest.is_empty() {
        Some(result + spaces)
    } else {
        let rest_span = mk_sp(init_last_pos, span.hi());
        let rest_str = rewrite_with_alignment(rest, context, shape, rest_span, one_line_width)?;
        Some(
            result
                + spaces
                + "\n"
                + &shape
                    .indent
                    .block_indent(context.config)
                    .to_string(context.config)
                + &rest_str,
        )
    }
}

fn struct_field_prefix_max_min_width<T: AlignedItem>(
    context: &RewriteContext,
    fields: &[T],
    shape: Shape,
) -> (usize, usize) {
    fields
        .iter()
        .map(|field| {
            field.rewrite_prefix(context, shape).and_then(|field_str| {
                if field_str.contains('\n') {
                    None
                } else {
                    Some(field_str.len())
                }
            })
        }).fold(Some((0, ::std::usize::MAX)), |acc, len| match (acc, len) {
            (Some((max_len, min_len)), Some(len)) => {
                Some((cmp::max(max_len, len), cmp::min(min_len, len)))
            }
            _ => None,
        }).unwrap_or((0, 0))
}

fn rewrite_aligned_items_inner<T: AlignedItem>(
    context: &RewriteContext,
    fields: &[T],
    span: Span,
    offset: Indent,
    one_line_width: usize,
) -> Option<String> {
    let item_indent = offset.block_indent(context.config);
    // 1 = ","
    let item_shape = Shape::indented(item_indent, context.config).sub_width(1)?;
    let (mut field_prefix_max_width, field_prefix_min_width) =
        struct_field_prefix_max_min_width(context, fields, item_shape);
    let max_diff = field_prefix_max_width.saturating_sub(field_prefix_min_width);
    if max_diff > context.config.struct_field_align_threshold() {
        field_prefix_max_width = 0;
    }

    let items = itemize_list(
        context.snippet_provider,
        fields.iter(),
        "}",
        ",",
        |field| field.get_span().lo(),
        |field| field.get_span().hi(),
        |field| field.rewrite_aligned_item(context, item_shape, field_prefix_max_width),
        span.lo(),
        span.hi(),
        false,
    ).collect::<Vec<_>>();

    let tactic = definitive_tactic(
        &items,
        ListTactic::HorizontalVertical,
        Separator::Comma,
        one_line_width,
    );

    let fmt = ListFormatting::new(item_shape, context.config)
        .tactic(tactic)
        .trailing_separator(context.config.trailing_comma())
        .preserve_newline(true);
    write_list(&items, &fmt)
}

fn group_aligned_items<T: AlignedItem>(
    context: &RewriteContext,
    fields: &[T],
) -> (&'static str, usize) {
    let mut index = 0;
    for i in 0..fields.len() - 1 {
        if fields[i].skip() {
            return ("", index);
        }
        // See if there are comments or empty lines between fields.
        let span = mk_sp(fields[i].get_span().hi(), fields[i + 1].get_span().lo());
        let snippet = context
            .snippet(span)
            .lines()
            .skip(1)
            .collect::<Vec<_>>()
            .join("\n");
        let spacings = if snippet.lines().rev().skip(1).any(|l| l.trim().is_empty()) {
            "\n"
        } else {
            ""
        };
        if contains_comment(&snippet) || snippet.lines().count() > 1 {
            return (spacings, index);
        }
        index += 1;
    }
    ("", index)
}
