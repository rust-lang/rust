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

use config::lists::*;
use config::IndentStyle;
use syntax::ast;
use syntax::codemap::Span;

use comment::{combine_strs_with_missing_comments, contains_comment, rewrite_doc_comment};
use expr::rewrite_literal;
use lists::{itemize_list, write_list, ListFormatting};
use rewrite::{Rewrite, RewriteContext};
use shape::Shape;
use types::{rewrite_path, PathContext};
use utils::{count_newlines, mk_sp};

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
fn get_derive_args<'a>(context: &'a RewriteContext, attr: &ast::Attribute) -> Option<Vec<&'a str>> {
    attr.meta_item_list().map(|meta_item_list| {
        meta_item_list
            .iter()
            .map(|nested_meta_item| context.snippet(nested_meta_item.span))
            .collect()
    })
}

// Format `#[derive(..)]`, using visual indent & mixed style when we need to go multiline.
fn format_derive(context: &RewriteContext, derive_args: &[&str], shape: Shape) -> Option<String> {
    let mut result = String::with_capacity(128);
    result.push_str("#[derive(");
    // 11 = `#[derive()]`
    let initial_budget = shape.width.checked_sub(11)?;
    let mut budget = initial_budget;
    let num = derive_args.len();
    for (i, a) in derive_args.iter().enumerate() {
        // 2 = `, ` or `)]`
        let width = a.len() + 2;
        if width > budget {
            if i > 0 {
                // Remove trailing whitespace.
                result.pop();
            }
            result.push('\n');
            // 9 = `#[derive(`
            result.push_str(&(shape.indent + 9).to_string(context.config));
            budget = initial_budget;
        } else {
            budget = budget.saturating_sub(width);
        }
        result.push_str(a);
        if i != num - 1 {
            result.push_str(", ")
        }
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

/// Rewrite the same kind of attributes at the same time. This includes doc
/// comments and derives.
fn rewrite_first_group_attrs(
    context: &RewriteContext,
    attrs: &[ast::Attribute],
    shape: Shape,
) -> Option<(usize, String)> {
    if attrs.is_empty() {
        return Some((0, String::new()));
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
            rewrite_doc_comment(&snippet, shape.comment(context.config), context.config)?,
        ));
    }
    // Rewrite `#[derive(..)]`s.
    if context.config.merge_derives() {
        let derives = take_while_with_pred(context, attrs, is_derive);
        if !derives.is_empty() {
            let mut derive_args = vec![];
            for derive in derives {
                derive_args.append(&mut get_derive_args(context, derive)?);
            }
            return Some((derives.len(), format_derive(context, &derive_args, shape)?));
        }
    }
    // Rewrite the first attribute.
    Some((1, attrs[0].rewrite(context, shape)?))
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
            .count() > 1
    };
    (if mlb { "\n" } else { "" }, if mla { "\n" } else { "" })
}

fn allow_mixed_tactic_for_nested_metaitem_list(list: &[ast::NestedMetaItem]) -> bool {
    list.iter().all(|nested_metaitem| {
        if let ast::NestedMetaItemKind::MetaItem(ref inner_metaitem) = nested_metaitem.node {
            match inner_metaitem.node {
                ast::MetaItemKind::List(..) | ast::MetaItemKind::NameValue(..) => false,
                _ => true,
            }
        } else {
            true
        }
    })
}

impl Rewrite for ast::MetaItem {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        Some(match self.node {
            ast::MetaItemKind::Word => {
                rewrite_path(context, PathContext::Type, None, &self.ident, shape)?
            }
            ast::MetaItemKind::List(ref list) => {
                let path = rewrite_path(context, PathContext::Type, None, &self.ident, shape)?;
                let item_shape = match context.config.indent_style() {
                    IndentStyle::Block => shape
                        .block_indent(context.config.tab_spaces())
                        .with_max_width(context.config),
                    // 1 = `(`, 2 = `]` and `)`
                    IndentStyle::Visual => shape
                        .visual_indent(0)
                        .shrink_left(path.len() + 1)
                        .and_then(|s| s.sub_width(2))?,
                };
                let items = itemize_list(
                    context.snippet_provider,
                    list.iter(),
                    ")",
                    ",",
                    |nested_meta_item| nested_meta_item.span.lo(),
                    |nested_meta_item| nested_meta_item.span.hi(),
                    |nested_meta_item| nested_meta_item.rewrite(context, item_shape),
                    self.span.lo(),
                    self.span.hi(),
                    false,
                );
                let item_vec = items.collect::<Vec<_>>();
                let tactic = if allow_mixed_tactic_for_nested_metaitem_list(list) {
                    DefinitiveListTactic::Mixed
                } else {
                    ::lists::definitive_tactic(
                        &item_vec,
                        ListTactic::HorizontalVertical,
                        ::lists::Separator::Comma,
                        item_shape.width,
                    )
                };
                let fmt = ListFormatting {
                    tactic,
                    separator: ",",
                    trailing_separator: SeparatorTactic::Never,
                    separator_place: SeparatorPlace::Back,
                    shape: item_shape,
                    ends_with_newline: false,
                    preserve_newline: false,
                    nested: false,
                    config: context.config,
                };
                let item_str = write_list(&item_vec, &fmt)?;
                // 3 = "()" and "]"
                let one_line_budget = shape.offset_left(path.len())?.sub_width(3)?.width;
                if context.config.indent_style() == IndentStyle::Visual
                    || (!item_str.contains('\n') && item_str.len() <= one_line_budget)
                {
                    format!("{}({})", path, item_str)
                } else {
                    let indent = shape.indent.to_string_with_newline(context.config);
                    let nested_indent = item_shape.indent.to_string_with_newline(context.config);
                    format!("{}({}{}{})", path, nested_indent, item_str, indent)
                }
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

impl Rewrite for ast::Attribute {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        let prefix = match self.style {
            ast::AttrStyle::Inner => "#!",
            ast::AttrStyle::Outer => "#",
        };
        let snippet = context.snippet(self.span);
        if self.is_sugared_doc {
            rewrite_doc_comment(snippet, shape.comment(context.config), context.config)
        } else {
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
        let (first_group_len, first_group_str) = rewrite_first_group_attrs(context, self, shape)?;
        if self.len() == 1 || first_group_len == self.len() {
            Some(first_group_str)
        } else {
            let rest_str = self[first_group_len..].rewrite(context, shape)?;
            let missing_span = mk_sp(
                self[first_group_len - 1].span.hi(),
                self[first_group_len].span.lo(),
            );
            // Preserve an empty line before/after doc comments.
            if self[0].is_sugared_doc || self[first_group_len].is_sugared_doc {
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
                Some(format!(
                    "{}{}{}{}",
                    first_group_str,
                    comment,
                    shape.indent.to_string(context.config),
                    rest_str
                ))
            } else {
                combine_strs_with_missing_comments(
                    context,
                    &first_group_str,
                    &rest_str,
                    missing_span,
                    shape,
                    false,
                )
            }
        }
    }
}
