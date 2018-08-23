// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Format attributes and meta items.

use comment::{contains_comment, rewrite_doc_comment};
use config::lists::*;
use config::IndentStyle;
use expr::rewrite_literal;
use lists::{definitive_tactic, itemize_list, write_list, ListFormatting, Separator};
use rewrite::{Rewrite, RewriteContext};
use shape::Shape;
use types::{rewrite_path, PathContext};
use utils::{count_newlines, mk_sp};

use std::borrow::Cow;
use syntax::ast;
use syntax::source_map::{BytePos, Span, DUMMY_SP};

/// Returns attributes on the given statement.
pub fn get_attrs_from_stmt(stmt: &ast::Stmt) -> &[ast::Attribute] {
    match stmt.node {
        ast::StmtKind::Local(ref local) => &local.attrs,
        ast::StmtKind::Item(ref item) => &item.attrs,
        ast::StmtKind::Expr(ref expr) | ast::StmtKind::Semi(ref expr) => &expr.attrs,
        ast::StmtKind::Mac(ref mac) => &mac.2,
    }
}

/// Returns attributes that are within `outer_span`.
pub fn filter_inline_attrs(attrs: &[ast::Attribute], outer_span: Span) -> Vec<ast::Attribute> {
    attrs
        .iter()
        .filter(|a| outer_span.lo() <= a.span.lo() && a.span.hi() <= outer_span.hi())
        .cloned()
        .collect()
}

fn is_derive(attr: &ast::Attribute) -> bool {
    attr.check_name("derive")
}

/// Returns the arguments of `#[derive(...)]`.
fn get_derive_spans<'a>(attr: &ast::Attribute) -> Option<Vec<Span>> {
    attr.meta_item_list().map(|meta_item_list| {
        meta_item_list
            .iter()
            .map(|nested_meta_item| nested_meta_item.span)
            .collect()
    })
}

// The shape of the arguments to a function-like attribute.
fn argument_shape(
    left: usize,
    right: usize,
    combine: bool,
    shape: Shape,
    context: &RewriteContext,
) -> Option<Shape> {
    match context.config.indent_style() {
        IndentStyle::Block => {
            if combine {
                shape.offset_left(left)
            } else {
                Some(
                    shape
                        .block_indent(context.config.tab_spaces())
                        .with_max_width(context.config),
                )
            }
        }
        IndentStyle::Visual => shape
            .visual_indent(0)
            .shrink_left(left)
            .and_then(|s| s.sub_width(right)),
    }
}

fn format_derive(
    derive_args: &[Span],
    prefix: &str,
    shape: Shape,
    context: &RewriteContext,
) -> Option<String> {
    let mut result = String::with_capacity(128);
    result.push_str(prefix);
    result.push_str("[derive(");

    let argument_shape = argument_shape(10 + prefix.len(), 2, false, shape, context)?;
    let item_str = format_arg_list(
        derive_args.iter(),
        |_| DUMMY_SP.lo(),
        |_| DUMMY_SP.hi(),
        |sp| Some(context.snippet(**sp).to_owned()),
        DUMMY_SP,
        context,
        argument_shape,
        // 10 = "[derive()]", 3 = "()" and "]"
        shape.offset_left(10 + prefix.len())?.sub_width(3)?,
        None,
        false,
    )?;

    result.push_str(&item_str);
    if item_str.starts_with('\n') {
        result.push(',');
        result.push_str(&shape.indent.to_string_with_newline(context.config));
    }
    result.push_str(")]");
    Some(result)
}

/// Returns the first group of attributes that fills the given predicate.
/// We consider two doc comments are in different group if they are separated by normal comments.
fn take_while_with_pred<'a, P>(
    context: &RewriteContext,
    attrs: &'a [ast::Attribute],
    pred: P,
) -> &'a [ast::Attribute]
where
    P: Fn(&ast::Attribute) -> bool,
{
    let mut len = 0;
    let mut iter = attrs.iter().peekable();

    while let Some(attr) = iter.next() {
        if pred(attr) {
            len += 1;
        } else {
            break;
        }
        if let Some(next_attr) = iter.peek() {
            // Extract comments between two attributes.
            let span_between_attr = mk_sp(attr.span.hi(), next_attr.span.lo());
            let snippet = context.snippet(span_between_attr);
            if count_newlines(snippet) >= 2 || snippet.contains('/') {
                break;
            }
        }
    }

    &attrs[..len]
}

/// Rewrite the any doc comments which come before any other attributes.
fn rewrite_initial_doc_comments(
    context: &RewriteContext,
    attrs: &[ast::Attribute],
    shape: Shape,
) -> Option<(usize, Option<String>)> {
    if attrs.is_empty() {
        return Some((0, None));
    }
    // Rewrite doc comments
    let sugared_docs = take_while_with_pred(context, attrs, |a| a.is_sugared_doc);
    if !sugared_docs.is_empty() {
        let snippet = sugared_docs
            .iter()
            .map(|a| context.snippet(a.span))
            .collect::<Vec<_>>()
            .join("\n");
        return Some((
            sugared_docs.len(),
            Some(rewrite_doc_comment(
                &snippet,
                shape.comment(context.config),
                context.config,
            )?),
        ));
    }

    Some((0, None))
}

impl Rewrite for ast::NestedMetaItem {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        match self.node {
            ast::NestedMetaItemKind::MetaItem(ref meta_item) => meta_item.rewrite(context, shape),
            ast::NestedMetaItemKind::Literal(ref l) => rewrite_literal(context, l, shape),
        }
    }
}

fn has_newlines_before_after_comment(comment: &str) -> (&str, &str) {
    // Look at before and after comment and see if there are any empty lines.
    let comment_begin = comment.find('/');
    let len = comment_begin.unwrap_or_else(|| comment.len());
    let mlb = count_newlines(&comment[..len]) > 1;
    let mla = if comment_begin.is_none() {
        mlb
    } else {
        comment
            .chars()
            .rev()
            .take_while(|c| c.is_whitespace())
            .filter(|&c| c == '\n')
            .count()
            > 1
    };
    (if mlb { "\n" } else { "" }, if mla { "\n" } else { "" })
}

impl Rewrite for ast::MetaItem {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        Some(match self.node {
            ast::MetaItemKind::Word => {
                rewrite_path(context, PathContext::Type, None, &self.ident, shape)?
            }
            ast::MetaItemKind::List(ref list) => {
                let path = rewrite_path(context, PathContext::Type, None, &self.ident, shape)?;

                let snippet = context.snippet(self.span);
                // 2 = )] (this might go wrong if there is whitespace between the brackets, but
                // it's close enough).
                let snippet = snippet[..snippet.len() - 2].trim();
                let trailing_comma = if snippet.ends_with(',') { "," } else { "" };
                let combine = list.len() == 1 && match list[0].node {
                    ast::NestedMetaItemKind::Literal(..) => false,
                    ast::NestedMetaItemKind::MetaItem(ref inner_meta_item) => {
                        match inner_meta_item.node {
                            ast::MetaItemKind::List(..) => rewrite_path(
                                context,
                                PathContext::Type,
                                None,
                                &inner_meta_item.ident,
                                shape,
                            ).map_or(false, |s| s.len() + path.len() + 2 <= shape.width),
                            _ => false,
                        }
                    }
                };

                let argument_shape = argument_shape(
                    path.len() + 1,
                    2 + trailing_comma.len(),
                    combine,
                    shape,
                    context,
                )?;
                let item_str = format_arg_list(
                    list.iter(),
                    |nested_meta_item| nested_meta_item.span.lo(),
                    |nested_meta_item| nested_meta_item.span.hi(),
                    |nested_meta_item| nested_meta_item.rewrite(context, argument_shape),
                    self.span,
                    context,
                    argument_shape,
                    // 3 = "()" and "]"
                    shape
                        .offset_left(path.len())?
                        .sub_width(3 + trailing_comma.len())?,
                    Some(context.config.width_heuristics().fn_call_width),
                    combine,
                )?;

                let indent = if item_str.starts_with('\n') {
                    shape.indent.to_string_with_newline(context.config)
                } else {
                    Cow::Borrowed("")
                };

                format!("{}({}{}{})", path, item_str, trailing_comma, indent)
            }
            ast::MetaItemKind::NameValue(ref literal) => {
                let path = rewrite_path(context, PathContext::Type, None, &self.ident, shape)?;
                // 3 = ` = `
                let lit_shape = shape.shrink_left(path.len() + 3)?;
                // `rewrite_literal` returns `None` when `literal` exceeds max
                // width. Since a literal is basically unformattable unless it
                // is a string literal (and only if `format_strings` is set),
                // we might be better off ignoring the fact that the attribute
                // is longer than the max width and contiue on formatting.
                // See #2479 for example.
                let value = rewrite_literal(context, literal, lit_shape)
                    .unwrap_or_else(|| context.snippet(literal.span).to_owned());
                format!("{} = {}", path, value)
            }
        })
    }
}

fn format_arg_list<I, T, F1, F2, F3>(
    list: I,
    get_lo: F1,
    get_hi: F2,
    get_item_string: F3,
    span: Span,
    context: &RewriteContext,
    shape: Shape,
    one_line_shape: Shape,
    one_line_limit: Option<usize>,
    combine: bool,
) -> Option<String>
where
    I: Iterator<Item = T>,
    F1: Fn(&T) -> BytePos,
    F2: Fn(&T) -> BytePos,
    F3: Fn(&T) -> Option<String>,
{
    let items = itemize_list(
        context.snippet_provider,
        list,
        ")",
        ",",
        get_lo,
        get_hi,
        get_item_string,
        span.lo(),
        span.hi(),
        false,
    );
    let item_vec = items.collect::<Vec<_>>();
    let tactic = if let Some(limit) = one_line_limit {
        ListTactic::LimitedHorizontalVertical(limit)
    } else {
        ListTactic::HorizontalVertical
    };

    let tactic = definitive_tactic(&item_vec, tactic, Separator::Comma, shape.width);
    let fmt = ListFormatting::new(shape, context.config)
        .tactic(tactic)
        .ends_with_newline(false);
    let item_str = write_list(&item_vec, &fmt)?;

    let one_line_budget = one_line_shape.width;
    if context.config.indent_style() == IndentStyle::Visual
        || combine
        || (!item_str.contains('\n') && item_str.len() <= one_line_budget)
    {
        Some(item_str)
    } else {
        let nested_indent = shape.indent.to_string_with_newline(context.config);
        Some(format!("{}{}", nested_indent, item_str))
    }
}

impl Rewrite for ast::Attribute {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        let snippet = context.snippet(self.span);
        if self.is_sugared_doc {
            rewrite_doc_comment(snippet, shape.comment(context.config), context.config)
        } else {
            let prefix = attr_prefix(self);

            if contains_comment(snippet) {
                return Some(snippet.to_owned());
            }
            // 1 = `[`
            let shape = shape.offset_left(prefix.len() + 1)?;
            Some(
                self.meta()
                    .and_then(|meta| meta.rewrite(context, shape))
                    .map_or_else(|| snippet.to_owned(), |rw| format!("{}[{}]", prefix, rw)),
            )
        }
    }
}

impl<'a> Rewrite for [ast::Attribute] {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        if self.is_empty() {
            return Some(String::new());
        }

        // The current remaining attributes.
        let mut attrs = self;
        let mut result = String::new();

        // This is not just a simple map because we need to handle doc comments
        // (where we take as many doc comment attributes as possible) and possibly
        // merging derives into a single attribute.
        loop {
            if attrs.is_empty() {
                return Some(result);
            }

            // Handle doc comments.
            let (doc_comment_len, doc_comment_str) =
                rewrite_initial_doc_comments(context, attrs, shape)?;
            if doc_comment_len > 0 {
                let doc_comment_str = doc_comment_str.expect("doc comments, but no result");
                result.push_str(&doc_comment_str);

                let missing_span = attrs
                    .get(doc_comment_len)
                    .map(|next| mk_sp(attrs[doc_comment_len - 1].span.hi(), next.span.lo()));
                if let Some(missing_span) = missing_span {
                    let snippet = context.snippet(missing_span);
                    let (mla, mlb) = has_newlines_before_after_comment(snippet);
                    let comment = ::comment::recover_missing_comment_in_span(
                        missing_span,
                        shape.with_max_width(context.config),
                        context,
                        0,
                    )?;
                    let comment = if comment.is_empty() {
                        format!("\n{}", mlb)
                    } else {
                        format!("{}{}\n{}", mla, comment, mlb)
                    };
                    result.push_str(&comment);
                    result.push_str(&shape.indent.to_string(context.config));
                }

                attrs = &attrs[doc_comment_len..];

                continue;
            }

            // Handle derives if we will merge them.
            if context.config.merge_derives() && is_derive(&attrs[0]) {
                let derives = take_while_with_pred(context, attrs, is_derive);
                let mut derive_spans = vec![];
                for derive in derives {
                    derive_spans.append(&mut get_derive_spans(derive)?);
                }
                let derive_str =
                    format_derive(&derive_spans, attr_prefix(&attrs[0]), shape, context)?;
                result.push_str(&derive_str);

                let missing_span = attrs
                    .get(derives.len())
                    .map(|next| mk_sp(attrs[derives.len() - 1].span.hi(), next.span.lo()));
                if let Some(missing_span) = missing_span {
                    let comment = ::comment::recover_missing_comment_in_span(
                        missing_span,
                        shape.with_max_width(context.config),
                        context,
                        0,
                    )?;
                    result.push_str(&comment);
                    if let Some(next) = attrs.get(derives.len()) {
                        if next.is_sugared_doc {
                            let snippet = context.snippet(missing_span);
                            let (_, mlb) = has_newlines_before_after_comment(snippet);
                            result.push_str(&mlb);
                        }
                    }
                    result.push('\n');
                    result.push_str(&shape.indent.to_string(context.config));
                }

                attrs = &attrs[derives.len()..];

                continue;
            }

            // If we get here, then we have a regular attribute, just handle one
            // at a time.

            let formatted_attr = attrs[0].rewrite(context, shape)?;
            result.push_str(&formatted_attr);

            let missing_span = attrs
                .get(1)
                .map(|next| mk_sp(attrs[0].span.hi(), next.span.lo()));
            if let Some(missing_span) = missing_span {
                let comment = ::comment::recover_missing_comment_in_span(
                    missing_span,
                    shape.with_max_width(context.config),
                    context,
                    0,
                )?;
                result.push_str(&comment);
                if let Some(next) = attrs.get(1) {
                    if next.is_sugared_doc {
                        let snippet = context.snippet(missing_span);
                        let (_, mlb) = has_newlines_before_after_comment(snippet);
                        result.push_str(&mlb);
                    }
                }
                result.push('\n');
                result.push_str(&shape.indent.to_string(context.config));
            }

            attrs = &attrs[1..];
        }
    }
}

fn attr_prefix(attr: &ast::Attribute) -> &'static str {
    match attr.style {
        ast::AttrStyle::Inner => "#!",
        ast::AttrStyle::Outer => "#",
    }
}
