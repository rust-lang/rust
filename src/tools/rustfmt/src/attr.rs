//! Format attributes and meta items.

use rustc_ast::HasAttrs;
use rustc_ast::ast;
use rustc_span::{Span, symbol::sym};
use tracing::debug;

use self::doc_comment::DocCommentFormatter;
use crate::comment::{CommentStyle, contains_comment, rewrite_doc_comment};
use crate::config::IndentStyle;
use crate::config::lists::*;
use crate::expr::rewrite_literal;
use crate::lists::{ListFormatting, Separator, definitive_tactic, itemize_list, write_list};
use crate::overflow;
use crate::rewrite::{Rewrite, RewriteContext, RewriteError, RewriteErrorExt, RewriteResult};
use crate::shape::Shape;
use crate::source_map::SpanUtils;
use crate::types::{PathContext, rewrite_path};
use crate::utils::{count_newlines, mk_sp};

mod doc_comment;

/// Returns attributes on the given statement.
pub(crate) fn get_attrs_from_stmt(stmt: &ast::Stmt) -> &[ast::Attribute] {
    stmt.attrs()
}

pub(crate) fn get_span_without_attrs(stmt: &ast::Stmt) -> Span {
    match stmt.kind {
        ast::StmtKind::Let(ref local) => local.span,
        ast::StmtKind::Item(ref item) => item.span,
        ast::StmtKind::Expr(ref expr) | ast::StmtKind::Semi(ref expr) => expr.span,
        ast::StmtKind::MacCall(ref mac_stmt) => mac_stmt.mac.span(),
        ast::StmtKind::Empty => stmt.span,
    }
}

/// Returns attributes that are within `outer_span`.
pub(crate) fn filter_inline_attrs(attrs: &[ast::Attribute], outer_span: Span) -> ast::AttrVec {
    attrs
        .iter()
        .filter(|a| outer_span.lo() <= a.span.lo() && a.span.hi() <= outer_span.hi())
        .cloned()
        .collect()
}

fn is_derive(attr: &ast::Attribute) -> bool {
    attr.has_name(sym::derive)
}

// The shape of the arguments to a function-like attribute.
fn argument_shape(
    left: usize,
    right: usize,
    combine: bool,
    shape: Shape,
    context: &RewriteContext<'_>,
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
    derives: &[ast::Attribute],
    shape: Shape,
    context: &RewriteContext<'_>,
) -> Option<String> {
    // Collect all items from all attributes
    let all_items = derives
        .iter()
        .map(|attr| {
            // Parse the derive items and extract the span for each item; if any
            // attribute is not parseable, none of the attributes will be
            // reformatted.
            let item_spans = attr.meta_item_list().map(|meta_item_list| {
                meta_item_list
                    .into_iter()
                    .map(|meta_item_inner| meta_item_inner.span())
            })?;

            let items = itemize_list(
                context.snippet_provider,
                item_spans,
                ")",
                ",",
                |span| span.lo(),
                |span| span.hi(),
                |span| Ok(context.snippet(*span).to_owned()),
                // We update derive attribute spans to start after the opening '('
                // This helps us focus parsing to just what's inside #[derive(...)]
                context.snippet_provider.span_after(attr.span, "("),
                attr.span.hi(),
                false,
            );

            Some(items)
        })
        // Fail if any attribute failed.
        .collect::<Option<Vec<_>>>()?
        // Collect the results into a single, flat, Vec.
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    // Collect formatting parameters.
    let prefix = attr_prefix(&derives[0]);
    let argument_shape = argument_shape(
        "[derive()]".len() + prefix.len(),
        ")]".len(),
        false,
        shape,
        context,
    )?;
    let one_line_shape = shape
        .offset_left("[derive()]".len() + prefix.len())?
        .sub_width("()]".len())?;
    let one_line_budget = one_line_shape.width;

    let tactic = definitive_tactic(
        &all_items,
        ListTactic::HorizontalVertical,
        Separator::Comma,
        argument_shape.width,
    );
    let trailing_separator = match context.config.indent_style() {
        // We always add the trailing comma and remove it if it is not needed.
        IndentStyle::Block => SeparatorTactic::Always,
        IndentStyle::Visual => SeparatorTactic::Never,
    };

    // Format the collection of items.
    let fmt = ListFormatting::new(argument_shape, context.config)
        .tactic(tactic)
        .trailing_separator(trailing_separator)
        .ends_with_newline(false);
    let item_str = write_list(&all_items, &fmt).ok()?;

    debug!("item_str: '{}'", item_str);

    // Determine if the result will be nested, i.e. if we're using the block
    // indent style and either the items are on multiple lines or we've exceeded
    // our budget to fit on a single line.
    let nested = context.config.indent_style() == IndentStyle::Block
        && (item_str.contains('\n') || item_str.len() > one_line_budget);

    // Format the final result.
    let mut result = String::with_capacity(128);
    result.push_str(prefix);
    result.push_str("[derive(");
    if nested {
        let nested_indent = argument_shape.indent.to_string_with_newline(context.config);
        result.push_str(&nested_indent);
        result.push_str(&item_str);
        result.push_str(&shape.indent.to_string_with_newline(context.config));
    } else if let SeparatorTactic::Always = context.config.trailing_comma() {
        // Retain the trailing comma.
        result.push_str(&item_str);
    } else if item_str.ends_with(',') {
        // Remove the trailing comma.
        result.push_str(&item_str[..item_str.len() - 1]);
    } else {
        result.push_str(&item_str);
    }
    result.push_str(")]");

    Some(result)
}

/// Returns the first group of attributes that fills the given predicate.
/// We consider two doc comments are in different group if they are separated by normal comments.
fn take_while_with_pred<'a, P>(
    context: &RewriteContext<'_>,
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
    context: &RewriteContext<'_>,
    attrs: &[ast::Attribute],
    shape: Shape,
) -> Result<(usize, Option<String>), RewriteError> {
    if attrs.is_empty() {
        return Ok((0, None));
    }
    // Rewrite doc comments
    let sugared_docs = take_while_with_pred(context, attrs, |a| a.is_doc_comment());
    if !sugared_docs.is_empty() {
        let snippet = sugared_docs
            .iter()
            .map(|a| context.snippet(a.span))
            .collect::<Vec<_>>()
            .join("\n");
        return Ok((
            sugared_docs.len(),
            Some(rewrite_doc_comment(
                &snippet,
                shape.comment(context.config),
                context.config,
            )?),
        ));
    }

    Ok((0, None))
}

impl Rewrite for ast::MetaItemInner {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        match self {
            ast::MetaItemInner::MetaItem(ref meta_item) => meta_item.rewrite_result(context, shape),
            ast::MetaItemInner::Lit(ref l) => {
                rewrite_literal(context, l.as_token_lit(), l.span, shape)
            }
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
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        Ok(match self.kind {
            ast::MetaItemKind::Word => {
                rewrite_path(context, PathContext::Type, &None, &self.path, shape)?
            }
            ast::MetaItemKind::List(ref list) => {
                let path = rewrite_path(context, PathContext::Type, &None, &self.path, shape)?;
                let has_trailing_comma = crate::expr::span_ends_with_comma(context, self.span);
                overflow::rewrite_with_parens(
                    context,
                    &path,
                    list.iter(),
                    // 1 = "]"
                    shape.sub_width(1).max_width_error(shape.width, self.span)?,
                    self.span,
                    context.config.attr_fn_like_width(),
                    Some(if has_trailing_comma {
                        SeparatorTactic::Always
                    } else {
                        SeparatorTactic::Never
                    }),
                )?
            }
            ast::MetaItemKind::NameValue(ref lit) => {
                let path = rewrite_path(context, PathContext::Type, &None, &self.path, shape)?;
                // 3 = ` = `
                let lit_shape = shape
                    .shrink_left(path.len() + 3)
                    .max_width_error(shape.width, self.span)?;
                // `rewrite_literal` returns `None` when `lit` exceeds max
                // width. Since a literal is basically unformattable unless it
                // is a string literal (and only if `format_strings` is set),
                // we might be better off ignoring the fact that the attribute
                // is longer than the max width and continue on formatting.
                // See #2479 for example.
                let value = rewrite_literal(context, lit.as_token_lit(), lit.span, lit_shape)
                    .unwrap_or_else(|_| context.snippet(lit.span).to_owned());
                format!("{path} = {value}")
            }
        })
    }
}

impl Rewrite for ast::Attribute {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        let snippet = context.snippet(self.span);
        if self.is_doc_comment() {
            rewrite_doc_comment(snippet, shape.comment(context.config), context.config)
        } else {
            let should_skip = self
                .ident()
                .map(|s| context.skip_context.attributes.skip(s.name.as_str()))
                .unwrap_or(false);
            let prefix = attr_prefix(self);

            if should_skip || contains_comment(snippet) {
                return Ok(snippet.to_owned());
            }

            if let Some(ref meta) = self.meta() {
                // This attribute is possibly a doc attribute needing normalization to a doc comment
                if context.config.normalize_doc_attributes() && meta.has_name(sym::doc) {
                    if let Some(ref literal) = meta.value_str() {
                        let comment_style = match self.style {
                            ast::AttrStyle::Inner => CommentStyle::Doc,
                            ast::AttrStyle::Outer => CommentStyle::TripleSlash,
                        };

                        let literal_str = literal.as_str();
                        let doc_comment_formatter =
                            DocCommentFormatter::new(literal_str, comment_style);
                        let doc_comment = format!("{doc_comment_formatter}");
                        return rewrite_doc_comment(
                            &doc_comment,
                            shape.comment(context.config),
                            context.config,
                        );
                    }
                }

                // 1 = `[`
                let shape = shape
                    .offset_left(prefix.len() + 1)
                    .max_width_error(shape.width, self.span)?;
                Ok(meta.rewrite_result(context, shape).map_or_else(
                    |_| snippet.to_owned(),
                    |rw| match &self.kind {
                        ast::AttrKind::Normal(normal_attr) => match normal_attr.item.unsafety {
                            // For #![feature(unsafe_attributes)]
                            // See https://github.com/rust-lang/rust/issues/123757
                            ast::Safety::Unsafe(_) => format!("{}[unsafe({})]", prefix, rw),
                            _ => format!("{}[{}]", prefix, rw),
                        },
                        _ => format!("{}[{}]", prefix, rw),
                    },
                ))
            } else {
                Ok(snippet.to_owned())
            }
        }
    }
}

impl Rewrite for [ast::Attribute] {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.rewrite_result(context, shape).ok()
    }

    fn rewrite_result(&self, context: &RewriteContext<'_>, shape: Shape) -> RewriteResult {
        if self.is_empty() {
            return Ok(String::new());
        }

        // The current remaining attributes.
        let mut attrs = self;
        let mut result = String::new();

        // Determine if the source text is annotated with `#[rustfmt::skip::attributes(derive)]`
        // or `#![rustfmt::skip::attributes(derive)]`
        let skip_derives = context.skip_context.attributes.skip("derive");

        // This is not just a simple map because we need to handle doc comments
        // (where we take as many doc comment attributes as possible) and possibly
        // merging derives into a single attribute.
        loop {
            if attrs.is_empty() {
                return Ok(result);
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
                    let comment = crate::comment::recover_missing_comment_in_span(
                        missing_span,
                        shape.with_max_width(context.config),
                        context,
                        0,
                    )?;
                    let comment = if comment.is_empty() {
                        format!("\n{mlb}")
                    } else {
                        format!("{mla}{comment}\n{mlb}")
                    };
                    result.push_str(&comment);
                    result.push_str(&shape.indent.to_string(context.config));
                }

                attrs = &attrs[doc_comment_len..];

                continue;
            }

            // Handle derives if we will merge them.
            if !skip_derives && context.config.merge_derives() && is_derive(&attrs[0]) {
                let derives = take_while_with_pred(context, attrs, is_derive);
                let derive_str = format_derive(derives, shape, context).unknown_error()?;
                result.push_str(&derive_str);

                let missing_span = attrs
                    .get(derives.len())
                    .map(|next| mk_sp(attrs[derives.len() - 1].span.hi(), next.span.lo()));
                if let Some(missing_span) = missing_span {
                    let comment = crate::comment::recover_missing_comment_in_span(
                        missing_span,
                        shape.with_max_width(context.config),
                        context,
                        0,
                    )?;
                    result.push_str(&comment);
                    if let Some(next) = attrs.get(derives.len()) {
                        if next.is_doc_comment() {
                            let snippet = context.snippet(missing_span);
                            let (_, mlb) = has_newlines_before_after_comment(snippet);
                            result.push_str(mlb);
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

            let formatted_attr = attrs[0].rewrite_result(context, shape)?;
            result.push_str(&formatted_attr);

            let missing_span = attrs
                .get(1)
                .map(|next| mk_sp(attrs[0].span.hi(), next.span.lo()));
            if let Some(missing_span) = missing_span {
                let comment = crate::comment::recover_missing_comment_in_span(
                    missing_span,
                    shape.with_max_width(context.config),
                    context,
                    0,
                )?;
                result.push_str(&comment);
                if let Some(next) = attrs.get(1) {
                    if next.is_doc_comment() {
                        let snippet = context.snippet(missing_span);
                        let (_, mlb) = has_newlines_before_after_comment(snippet);
                        result.push_str(mlb);
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

pub(crate) trait MetaVisitor<'ast> {
    fn visit_meta_item(&mut self, meta_item: &'ast ast::MetaItem) {
        match meta_item.kind {
            ast::MetaItemKind::Word => self.visit_meta_word(meta_item),
            ast::MetaItemKind::List(ref list) => self.visit_meta_list(meta_item, list),
            ast::MetaItemKind::NameValue(ref lit) => self.visit_meta_name_value(meta_item, lit),
        }
    }

    fn visit_meta_list(
        &mut self,
        _meta_item: &'ast ast::MetaItem,
        list: &'ast [ast::MetaItemInner],
    ) {
        for nm in list {
            self.visit_meta_item_inner(nm);
        }
    }

    fn visit_meta_word(&mut self, _meta_item: &'ast ast::MetaItem) {}

    fn visit_meta_name_value(
        &mut self,
        _meta_item: &'ast ast::MetaItem,
        _lit: &'ast ast::MetaItemLit,
    ) {
    }

    fn visit_meta_item_inner(&mut self, nm: &'ast ast::MetaItemInner) {
        match nm {
            ast::MetaItemInner::MetaItem(ref meta_item) => self.visit_meta_item(meta_item),
            ast::MetaItemInner::Lit(ref lit) => self.visit_meta_item_lit(lit),
        }
    }

    fn visit_meta_item_lit(&mut self, _lit: &'ast ast::MetaItemLit) {}
}
