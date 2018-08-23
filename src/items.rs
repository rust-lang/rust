// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Formatting top-level items - functions, structs, enums, traits, impls.

use std::borrow::Cow;
use std::cmp::{min, Ordering};

use config::lists::*;
use regex::Regex;
use rustc_target::spec::abi;
use syntax::source_map::{self, BytePos, Span};
use syntax::visit;
use syntax::{ast, ptr, symbol};

use comment::{
    combine_strs_with_missing_comments, contains_comment, recover_comment_removed,
    recover_missing_comment_in_span, rewrite_missing_comment, FindUncommented,
};
use config::{BraceStyle, Config, Density, IndentStyle};
use expr::{
    format_expr, is_empty_block, is_simple_block_stmt, rewrite_assign_rhs, rewrite_assign_rhs_with,
    ExprType, RhsTactics,
};
use lists::{definitive_tactic, itemize_list, write_list, ListFormatting, ListItem, Separator};
use macros::{rewrite_macro, MacroPosition};
use overflow;
use rewrite::{Rewrite, RewriteContext};
use shape::{Indent, Shape};
use source_map::{LineRangeUtils, SpanUtils};
use spanned::Spanned;
use utils::*;
use vertical::rewrite_with_alignment;
use visitor::FmtVisitor;

const DEFAULT_VISIBILITY: ast::Visibility = source_map::Spanned {
    node: ast::VisibilityKind::Inherited,
    span: source_map::DUMMY_SP,
};

fn type_annotation_separator(config: &Config) -> &str {
    colon_spaces(config.space_before_colon(), config.space_after_colon())
}

// Statements of the form
// let pat: ty = init;
impl Rewrite for ast::Local {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        debug!(
            "Local::rewrite {:?} {} {:?}",
            self, shape.width, shape.indent
        );

        skip_out_of_file_lines_range!(context, self.span);

        if contains_skip(&self.attrs) {
            return None;
        }

        let attrs_str = self.attrs.rewrite(context, shape)?;
        let mut result = if attrs_str.is_empty() {
            "let ".to_owned()
        } else {
            combine_strs_with_missing_comments(
                context,
                &attrs_str,
                "let ",
                mk_sp(
                    self.attrs.last().map(|a| a.span.hi()).unwrap(),
                    self.span.lo(),
                ),
                shape,
                false,
            )?
        };

        // 4 = "let ".len()
        let pat_shape = shape.offset_left(4)?;
        // 1 = ;
        let pat_shape = pat_shape.sub_width(1)?;
        let pat_str = self.pat.rewrite(context, pat_shape)?;
        result.push_str(&pat_str);

        // String that is placed within the assignment pattern and expression.
        let infix = {
            let mut infix = String::with_capacity(32);

            if let Some(ref ty) = self.ty {
                let separator = type_annotation_separator(context.config);
                let indent = shape.indent + last_line_width(&result) + separator.len();
                // 1 = ;
                let budget = shape.width.checked_sub(indent.width() + 1)?;
                let rewrite = ty.rewrite(context, Shape::legacy(budget, indent))?;

                infix.push_str(separator);
                infix.push_str(&rewrite);
            }

            if self.init.is_some() {
                infix.push_str(" =");
            }

            infix
        };

        result.push_str(&infix);

        if let Some(ref ex) = self.init {
            // 1 = trailing semicolon;
            let nested_shape = shape.sub_width(1)?;

            result = rewrite_assign_rhs(context, result, &**ex, nested_shape)?;
        }

        result.push(';');
        Some(result)
    }
}

// FIXME convert to using rewrite style rather than visitor
// FIXME format modules in this style
#[allow(dead_code)]
struct Item<'a> {
    keyword: &'static str,
    abi: Cow<'static, str>,
    vis: Option<&'a ast::Visibility>,
    body: Vec<BodyElement<'a>>,
    span: Span,
}

impl<'a> Item<'a> {
    fn from_foreign_mod(fm: &'a ast::ForeignMod, span: Span, config: &Config) -> Item<'a> {
        Item {
            keyword: "",
            abi: format_abi(fm.abi, config.force_explicit_abi(), true),
            vis: None,
            body: fm
                .items
                .iter()
                .map(|i| BodyElement::ForeignItem(i))
                .collect(),
            span,
        }
    }
}

enum BodyElement<'a> {
    // Stmt(&'a ast::Stmt),
    // Field(&'a ast::Field),
    // Variant(&'a ast::Variant),
    // Item(&'a ast::Item),
    ForeignItem(&'a ast::ForeignItem),
}

/// Represents a fn's signature.
pub struct FnSig<'a> {
    decl: &'a ast::FnDecl,
    generics: &'a ast::Generics,
    abi: abi::Abi,
    is_async: ast::IsAsync,
    constness: ast::Constness,
    defaultness: ast::Defaultness,
    unsafety: ast::Unsafety,
    visibility: ast::Visibility,
}

impl<'a> FnSig<'a> {
    pub fn new(
        decl: &'a ast::FnDecl,
        generics: &'a ast::Generics,
        vis: ast::Visibility,
    ) -> FnSig<'a> {
        FnSig {
            decl,
            generics,
            abi: abi::Abi::Rust,
            is_async: ast::IsAsync::NotAsync,
            constness: ast::Constness::NotConst,
            defaultness: ast::Defaultness::Final,
            unsafety: ast::Unsafety::Normal,
            visibility: vis,
        }
    }

    pub fn from_method_sig(
        method_sig: &'a ast::MethodSig,
        generics: &'a ast::Generics,
    ) -> FnSig<'a> {
        FnSig {
            unsafety: method_sig.header.unsafety,
            is_async: method_sig.header.asyncness,
            constness: method_sig.header.constness.node,
            defaultness: ast::Defaultness::Final,
            abi: method_sig.header.abi,
            decl: &*method_sig.decl,
            generics,
            visibility: DEFAULT_VISIBILITY,
        }
    }

    pub fn from_fn_kind(
        fn_kind: &'a visit::FnKind,
        generics: &'a ast::Generics,
        decl: &'a ast::FnDecl,
        defaultness: ast::Defaultness,
    ) -> FnSig<'a> {
        match *fn_kind {
            visit::FnKind::ItemFn(_, fn_header, visibility, _) => FnSig {
                decl,
                generics,
                abi: fn_header.abi,
                constness: fn_header.constness.node,
                is_async: fn_header.asyncness,
                defaultness,
                unsafety: fn_header.unsafety,
                visibility: visibility.clone(),
            },
            visit::FnKind::Method(_, method_sig, vis, _) => {
                let mut fn_sig = FnSig::from_method_sig(method_sig, generics);
                fn_sig.defaultness = defaultness;
                if let Some(vis) = vis {
                    fn_sig.visibility = vis.clone();
                }
                fn_sig
            }
            _ => unreachable!(),
        }
    }

    fn to_str(&self, context: &RewriteContext) -> String {
        let mut result = String::with_capacity(128);
        // Vis defaultness constness unsafety abi.
        result.push_str(&*format_visibility(context, &self.visibility));
        result.push_str(format_defaultness(self.defaultness));
        result.push_str(format_constness(self.constness));
        result.push_str(format_unsafety(self.unsafety));
        result.push_str(format_async(self.is_async));
        result.push_str(&format_abi(
            self.abi,
            context.config.force_explicit_abi(),
            false,
        ));
        result
    }
}

impl<'a> FmtVisitor<'a> {
    fn format_item(&mut self, item: &Item) {
        self.buffer.push_str(&item.abi);

        let snippet = self.snippet(item.span);
        let brace_pos = snippet.find_uncommented("{").unwrap();

        self.push_str("{");
        if !item.body.is_empty() || contains_comment(&snippet[brace_pos..]) {
            // FIXME: this skips comments between the extern keyword and the opening
            // brace.
            self.last_pos = item.span.lo() + BytePos(brace_pos as u32 + 1);
            self.block_indent = self.block_indent.block_indent(self.config);

            if item.body.is_empty() {
                self.format_missing_no_indent(item.span.hi() - BytePos(1));
                self.block_indent = self.block_indent.block_unindent(self.config);
                let indent_str = self.block_indent.to_string(self.config);
                self.push_str(&indent_str);
            } else {
                for item in &item.body {
                    self.format_body_element(item);
                }

                self.block_indent = self.block_indent.block_unindent(self.config);
                self.format_missing_with_indent(item.span.hi() - BytePos(1));
            }
        }

        self.push_str("}");
        self.last_pos = item.span.hi();
    }

    fn format_body_element(&mut self, element: &BodyElement) {
        match *element {
            BodyElement::ForeignItem(item) => self.format_foreign_item(item),
        }
    }

    pub fn format_foreign_mod(&mut self, fm: &ast::ForeignMod, span: Span) {
        let item = Item::from_foreign_mod(fm, span, self.config);
        self.format_item(&item);
    }

    fn format_foreign_item(&mut self, item: &ast::ForeignItem) {
        let rewrite = item.rewrite(&self.get_context(), self.shape());
        self.push_rewrite(item.span(), rewrite);
        self.last_pos = item.span.hi();
    }

    pub fn rewrite_fn(
        &mut self,
        indent: Indent,
        ident: ast::Ident,
        fn_sig: &FnSig,
        span: Span,
        block: &ast::Block,
        inner_attrs: Option<&[ast::Attribute]>,
    ) -> Option<String> {
        let context = self.get_context();

        let mut newline_brace = newline_for_brace(self.config, &fn_sig.generics.where_clause);

        let (mut result, force_newline_brace) =
            rewrite_fn_base(&context, indent, ident, fn_sig, span, newline_brace, true)?;

        // 2 = ` {`
        if self.config.brace_style() == BraceStyle::AlwaysNextLine
            || force_newline_brace
            || last_line_width(&result) + 2 > self.shape().width
        {
            newline_brace = true;
        } else if !result.contains('\n') {
            newline_brace = false;
        }

        // Prepare for the function body by possibly adding a newline and
        // indent.
        // FIXME we'll miss anything between the end of the signature and the
        // start of the body, but we need more spans from the compiler to solve
        // this.
        if newline_brace {
            result.push_str(&indent.to_string_with_newline(self.config));
        } else {
            result.push(' ');
        }

        self.single_line_fn(&result, block, inner_attrs)
            .or_else(|| Some(result))
    }

    pub fn rewrite_required_fn(
        &mut self,
        indent: Indent,
        ident: ast::Ident,
        sig: &ast::MethodSig,
        generics: &ast::Generics,
        span: Span,
    ) -> Option<String> {
        // Drop semicolon or it will be interpreted as comment.
        let span = mk_sp(span.lo(), span.hi() - BytePos(1));
        let context = self.get_context();

        let (mut result, _) = rewrite_fn_base(
            &context,
            indent,
            ident,
            &FnSig::from_method_sig(sig, generics),
            span,
            false,
            false,
        )?;

        // Re-attach semicolon
        result.push(';');

        Some(result)
    }

    fn single_line_fn(
        &self,
        fn_str: &str,
        block: &ast::Block,
        inner_attrs: Option<&[ast::Attribute]>,
    ) -> Option<String> {
        if fn_str.contains('\n') || inner_attrs.map_or(false, |a| !a.is_empty()) {
            return None;
        }

        let source_map = self.get_context().source_map;

        if self.config.empty_item_single_line()
            && is_empty_block(block, None, source_map)
            && self.block_indent.width() + fn_str.len() + 2 <= self.config.max_width()
        {
            return Some(format!("{}{{}}", fn_str));
        }

        if self.config.fn_single_line() && is_simple_block_stmt(block, None, source_map) {
            let rewrite = {
                if let Some(stmt) = block.stmts.first() {
                    match stmt_expr(stmt) {
                        Some(e) => {
                            let suffix = if semicolon_for_expr(&self.get_context(), e) {
                                ";"
                            } else {
                                ""
                            };

                            format_expr(e, ExprType::Statement, &self.get_context(), self.shape())
                                .map(|s| s + suffix)
                                .or_else(|| Some(self.snippet(e.span).to_owned()))
                        }
                        None => stmt.rewrite(&self.get_context(), self.shape()),
                    }
                } else {
                    None
                }
            };

            if let Some(res) = rewrite {
                let width = self.block_indent.width() + fn_str.len() + res.len() + 4;
                if !res.contains('\n') && width <= self.config.max_width() {
                    return Some(format!("{}{{ {} }}", fn_str, res));
                }
            }
        }

        None
    }

    pub fn visit_static(&mut self, static_parts: &StaticParts) {
        let rewrite = rewrite_static(&self.get_context(), static_parts, self.block_indent);
        self.push_rewrite(static_parts.span, rewrite);
    }

    pub fn visit_struct(&mut self, struct_parts: &StructParts) {
        let is_tuple = struct_parts.def.is_tuple();
        let rewrite = format_struct(&self.get_context(), struct_parts, self.block_indent, None)
            .map(|s| if is_tuple { s + ";" } else { s });
        self.push_rewrite(struct_parts.span, rewrite);
    }

    pub fn visit_enum(
        &mut self,
        ident: ast::Ident,
        vis: &ast::Visibility,
        enum_def: &ast::EnumDef,
        generics: &ast::Generics,
        span: Span,
    ) {
        let enum_header = format_header(&self.get_context(), "enum ", ident, vis);
        self.push_str(&enum_header);

        let enum_snippet = self.snippet(span);
        let brace_pos = enum_snippet.find_uncommented("{").unwrap();
        let body_start = span.lo() + BytePos(brace_pos as u32 + 1);
        let generics_str = format_generics(
            &self.get_context(),
            generics,
            self.config.brace_style(),
            if enum_def.variants.is_empty() {
                BracePos::ForceSameLine
            } else {
                BracePos::Auto
            },
            self.block_indent,
            mk_sp(span.lo(), body_start),
            last_line_width(&enum_header),
        ).unwrap();
        self.push_str(&generics_str);

        self.last_pos = body_start;

        match self.format_variant_list(enum_def, body_start, span.hi()) {
            Some(ref s) if enum_def.variants.is_empty() => self.push_str(s),
            rw => {
                self.push_rewrite(mk_sp(body_start, span.hi()), rw);
                self.block_indent = self.block_indent.block_unindent(self.config);
            }
        }
    }

    // Format the body of an enum definition
    fn format_variant_list(
        &mut self,
        enum_def: &ast::EnumDef,
        body_lo: BytePos,
        body_hi: BytePos,
    ) -> Option<String> {
        if enum_def.variants.is_empty() {
            let mut buffer = String::with_capacity(128);
            // 1 = "}"
            let span = mk_sp(body_lo, body_hi - BytePos(1));
            format_empty_struct_or_tuple(
                &self.get_context(),
                span,
                self.block_indent,
                &mut buffer,
                "",
                "}",
            );
            return Some(buffer);
        }
        let mut result = String::with_capacity(1024);
        let original_offset = self.block_indent;
        self.block_indent = self.block_indent.block_indent(self.config);

        let itemize_list_with = |one_line_width: usize| {
            itemize_list(
                self.snippet_provider,
                enum_def.variants.iter(),
                "}",
                ",",
                |f| {
                    if !f.node.attrs.is_empty() {
                        f.node.attrs[0].span.lo()
                    } else {
                        f.span.lo()
                    }
                },
                |f| f.span.hi(),
                |f| self.format_variant(f, one_line_width),
                body_lo,
                body_hi,
                false,
            ).collect()
        };
        let mut items: Vec<_> =
            itemize_list_with(self.config.width_heuristics().struct_variant_width);
        // If one of the variants use multiple lines, use multi-lined formatting for all variants.
        let has_multiline_variant = items.iter().any(|item| item.inner_as_ref().contains('\n'));
        let has_single_line_variant = items.iter().any(|item| !item.inner_as_ref().contains('\n'));
        if has_multiline_variant && has_single_line_variant {
            items = itemize_list_with(0);
        }

        let shape = self.shape().sub_width(2)?;
        let fmt = ListFormatting::new(shape, self.config)
            .trailing_separator(self.config.trailing_comma())
            .preserve_newline(true);

        let list = write_list(&items, &fmt)?;
        result.push_str(&list);
        result.push_str(&original_offset.to_string_with_newline(self.config));
        result.push('}');
        Some(result)
    }

    // Variant of an enum.
    fn format_variant(&self, field: &ast::Variant, one_line_width: usize) -> Option<String> {
        if contains_skip(&field.node.attrs) {
            let lo = field.node.attrs[0].span.lo();
            let span = mk_sp(lo, field.span.hi());
            return Some(self.snippet(span).to_owned());
        }

        let context = self.get_context();
        // 1 = ','
        let shape = self.shape().sub_width(1)?;
        let attrs_str = field.node.attrs.rewrite(&context, shape)?;
        let lo = field
            .node
            .attrs
            .last()
            .map_or(field.span.lo(), |attr| attr.span.hi());
        let span = mk_sp(lo, field.span.lo());

        let variant_body = match field.node.data {
            ast::VariantData::Tuple(..) | ast::VariantData::Struct(..) => format_struct(
                &context,
                &StructParts::from_variant(field),
                self.block_indent,
                Some(one_line_width),
            )?,
            ast::VariantData::Unit(..) => {
                if let Some(ref expr) = field.node.disr_expr {
                    let lhs = format!("{} =", rewrite_ident(&context, field.node.ident));
                    rewrite_assign_rhs(&context, lhs, &*expr.value, shape)?
                } else {
                    rewrite_ident(&context, field.node.ident).to_owned()
                }
            }
        };

        combine_strs_with_missing_comments(&context, &attrs_str, &variant_body, span, shape, false)
    }

    fn visit_impl_items(&mut self, items: &[ast::ImplItem]) {
        if self.get_context().config.reorder_impl_items() {
            // Create visitor for each items, then reorder them.
            let mut buffer = vec![];
            for item in items {
                self.visit_impl_item(item);
                buffer.push((self.buffer.clone(), item.clone()));
                self.buffer.clear();
            }
            // type -> existential -> const -> macro -> method
            use ast::ImplItemKind::*;
            fn need_empty_line(a: &ast::ImplItemKind, b: &ast::ImplItemKind) -> bool {
                match (a, b) {
                    (Type(..), Type(..))
                    | (Const(..), Const(..))
                    | (Existential(..), Existential(..)) => false,
                    _ => true,
                }
            }

            buffer.sort_by(|(_, a), (_, b)| match (&a.node, &b.node) {
                (Type(..), Type(..))
                | (Const(..), Const(..))
                | (Macro(..), Macro(..))
                | (Existential(..), Existential(..)) => a.ident.as_str().cmp(&b.ident.as_str()),
                (Method(..), Method(..)) => a.span.lo().cmp(&b.span.lo()),
                (Type(..), _) => Ordering::Less,
                (_, Type(..)) => Ordering::Greater,
                (Existential(..), _) => Ordering::Less,
                (_, Existential(..)) => Ordering::Greater,
                (Const(..), _) => Ordering::Less,
                (_, Const(..)) => Ordering::Greater,
                (Macro(..), _) => Ordering::Less,
                (_, Macro(..)) => Ordering::Greater,
            });
            let mut prev_kind = None;
            for (buf, item) in buffer {
                // Make sure that there are at least a single empty line between
                // different impl items.
                if prev_kind
                    .as_ref()
                    .map_or(false, |prev_kind| need_empty_line(prev_kind, &item.node))
                {
                    self.push_str("\n");
                }
                let indent_str = self.block_indent.to_string_with_newline(self.config);
                self.push_str(&indent_str);
                self.push_str(buf.trim());
                prev_kind = Some(item.node.clone());
            }
        } else {
            for item in items {
                self.visit_impl_item(item);
            }
        }
    }
}

pub fn format_impl(
    context: &RewriteContext,
    item: &ast::Item,
    offset: Indent,
    where_span_end: Option<BytePos>,
) -> Option<String> {
    if let ast::ItemKind::Impl(_, _, _, ref generics, _, ref self_ty, ref items) = item.node {
        let mut result = String::with_capacity(128);
        let ref_and_type = format_impl_ref_and_type(context, item, offset)?;
        let sep = offset.to_string_with_newline(context.config);
        result.push_str(&ref_and_type);

        let where_budget = if result.contains('\n') {
            context.config.max_width()
        } else {
            context.budget(last_line_width(&result))
        };

        let mut option = WhereClauseOption::snuggled(&ref_and_type);
        let snippet = context.snippet(item.span);
        let open_pos = snippet.find_uncommented("{")? + 1;
        if !contains_comment(&snippet[open_pos..])
            && items.is_empty()
            && generics.where_clause.predicates.len() == 1
            && !result.contains('\n')
        {
            option.suppress_comma();
            option.snuggle();
            option.compress_where();
        }

        let where_clause_str = rewrite_where_clause(
            context,
            &generics.where_clause,
            context.config.brace_style(),
            Shape::legacy(where_budget, offset.block_only()),
            Density::Vertical,
            "{",
            where_span_end,
            self_ty.span.hi(),
            option,
            false,
        )?;

        // If there is no where clause, we may have missing comments between the trait name and
        // the opening brace.
        if generics.where_clause.predicates.is_empty() {
            if let Some(hi) = where_span_end {
                match recover_missing_comment_in_span(
                    mk_sp(self_ty.span.hi(), hi),
                    Shape::indented(offset, context.config),
                    context,
                    last_line_width(&result),
                ) {
                    Some(ref missing_comment) if !missing_comment.is_empty() => {
                        result.push_str(missing_comment);
                    }
                    _ => (),
                }
            }
        }

        if is_impl_single_line(context, items, &result, &where_clause_str, item)? {
            result.push_str(&where_clause_str);
            if where_clause_str.contains('\n') || last_line_contains_single_line_comment(&result) {
                // if the where_clause contains extra comments AND
                // there is only one where clause predicate
                // recover the suppressed comma in single line where_clause formatting
                if generics.where_clause.predicates.len() == 1 {
                    result.push_str(",");
                }
                result.push_str(&format!("{}{{{}}}", &sep, &sep));
            } else {
                result.push_str(" {}");
            }
            return Some(result);
        }

        result.push_str(&where_clause_str);

        let need_newline = last_line_contains_single_line_comment(&result) || result.contains('\n');
        match context.config.brace_style() {
            _ if need_newline => result.push_str(&sep),
            BraceStyle::AlwaysNextLine => result.push_str(&sep),
            BraceStyle::PreferSameLine => result.push(' '),
            BraceStyle::SameLineWhere => {
                if !where_clause_str.is_empty() {
                    result.push_str(&sep);
                } else {
                    result.push(' ');
                }
            }
        }

        result.push('{');

        let snippet = context.snippet(item.span);
        let open_pos = snippet.find_uncommented("{")? + 1;

        if !items.is_empty() || contains_comment(&snippet[open_pos..]) {
            let mut visitor = FmtVisitor::from_context(context);
            let item_indent = offset.block_only().block_indent(context.config);
            visitor.block_indent = item_indent;
            visitor.last_pos = item.span.lo() + BytePos(open_pos as u32);

            visitor.visit_attrs(&item.attrs, ast::AttrStyle::Inner);
            visitor.visit_impl_items(items);

            visitor.format_missing(item.span.hi() - BytePos(1));

            let inner_indent_str = visitor.block_indent.to_string_with_newline(context.config);
            let outer_indent_str = offset.block_only().to_string_with_newline(context.config);

            result.push_str(&inner_indent_str);
            result.push_str(visitor.buffer.to_string().trim());
            result.push_str(&outer_indent_str);
        }

        if result.ends_with('{') && !context.config.empty_item_single_line() {
            result.push_str(&sep);
        }
        result.push('}');

        Some(result)
    } else {
        unreachable!();
    }
}

fn is_impl_single_line(
    context: &RewriteContext,
    items: &[ast::ImplItem],
    result: &str,
    where_clause_str: &str,
    item: &ast::Item,
) -> Option<bool> {
    let snippet = context.snippet(item.span);
    let open_pos = snippet.find_uncommented("{")? + 1;

    Some(
        context.config.empty_item_single_line()
            && items.is_empty()
            && !result.contains('\n')
            && result.len() + where_clause_str.len() <= context.config.max_width()
            && !contains_comment(&snippet[open_pos..]),
    )
}

fn format_impl_ref_and_type(
    context: &RewriteContext,
    item: &ast::Item,
    offset: Indent,
) -> Option<String> {
    if let ast::ItemKind::Impl(
        unsafety,
        polarity,
        defaultness,
        ref generics,
        ref trait_ref,
        ref self_ty,
        _,
    ) = item.node
    {
        let mut result = String::with_capacity(128);

        result.push_str(&format_visibility(context, &item.vis));
        result.push_str(format_defaultness(defaultness));
        result.push_str(format_unsafety(unsafety));

        let shape = generics_shape_from_config(
            context.config,
            Shape::indented(offset + last_line_width(&result), context.config),
            0,
        )?;
        let generics_str = rewrite_generics(context, "impl", generics, shape)?;
        result.push_str(&generics_str);

        let polarity_str = if polarity == ast::ImplPolarity::Negative {
            "!"
        } else {
            ""
        };

        if let Some(ref trait_ref) = *trait_ref {
            let result_len = last_line_width(&result);
            result.push_str(&rewrite_trait_ref(
                context,
                trait_ref,
                offset,
                polarity_str,
                result_len,
            )?);
        }

        // Try to put the self type in a single line.
        // ` for`
        let trait_ref_overhead = if trait_ref.is_some() { 4 } else { 0 };
        let curly_brace_overhead = if generics.where_clause.predicates.is_empty() {
            // If there is no where clause adapt budget for type formatting to take space and curly
            // brace into account.
            match context.config.brace_style() {
                BraceStyle::AlwaysNextLine => 0,
                _ => 2,
            }
        } else {
            0
        };
        let used_space = last_line_width(&result) + trait_ref_overhead + curly_brace_overhead;
        // 1 = space before the type.
        let budget = context.budget(used_space + 1);
        if let Some(self_ty_str) = self_ty.rewrite(context, Shape::legacy(budget, offset)) {
            if !self_ty_str.contains('\n') {
                if trait_ref.is_some() {
                    result.push_str(" for ");
                } else {
                    result.push(' ');
                }
                result.push_str(&self_ty_str);
                return Some(result);
            }
        }

        // Couldn't fit the self type on a single line, put it on a new line.
        result.push('\n');
        // Add indentation of one additional tab.
        let new_line_offset = offset.block_indent(context.config);
        result.push_str(&new_line_offset.to_string(context.config));
        if trait_ref.is_some() {
            result.push_str("for ");
        }
        let budget = context.budget(last_line_width(&result));
        let type_offset = match context.config.indent_style() {
            IndentStyle::Visual => new_line_offset + trait_ref_overhead,
            IndentStyle::Block => new_line_offset,
        };
        result.push_str(&*self_ty.rewrite(context, Shape::legacy(budget, type_offset))?);
        Some(result)
    } else {
        unreachable!();
    }
}

fn rewrite_trait_ref(
    context: &RewriteContext,
    trait_ref: &ast::TraitRef,
    offset: Indent,
    polarity_str: &str,
    result_len: usize,
) -> Option<String> {
    // 1 = space between generics and trait_ref
    let used_space = 1 + polarity_str.len() + result_len;
    let shape = Shape::indented(offset + used_space, context.config);
    if let Some(trait_ref_str) = trait_ref.rewrite(context, shape) {
        if !trait_ref_str.contains('\n') {
            return Some(format!(" {}{}", polarity_str, &trait_ref_str));
        }
    }
    // We could not make enough space for trait_ref, so put it on new line.
    let offset = offset.block_indent(context.config);
    let shape = Shape::indented(offset, context.config);
    let trait_ref_str = trait_ref.rewrite(context, shape)?;
    Some(format!(
        "{}{}{}",
        &offset.to_string_with_newline(context.config),
        polarity_str,
        &trait_ref_str
    ))
}

pub struct StructParts<'a> {
    prefix: &'a str,
    ident: ast::Ident,
    vis: &'a ast::Visibility,
    def: &'a ast::VariantData,
    generics: Option<&'a ast::Generics>,
    span: Span,
}

impl<'a> StructParts<'a> {
    fn format_header(&self, context: &RewriteContext) -> String {
        format_header(context, self.prefix, self.ident, self.vis)
    }

    fn from_variant(variant: &'a ast::Variant) -> Self {
        StructParts {
            prefix: "",
            ident: variant.node.ident,
            vis: &DEFAULT_VISIBILITY,
            def: &variant.node.data,
            generics: None,
            span: variant.span,
        }
    }

    pub fn from_item(item: &'a ast::Item) -> Self {
        let (prefix, def, generics) = match item.node {
            ast::ItemKind::Struct(ref def, ref generics) => ("struct ", def, generics),
            ast::ItemKind::Union(ref def, ref generics) => ("union ", def, generics),
            _ => unreachable!(),
        };
        StructParts {
            prefix,
            ident: item.ident,
            vis: &item.vis,
            def,
            generics: Some(generics),
            span: item.span,
        }
    }
}

fn format_struct(
    context: &RewriteContext,
    struct_parts: &StructParts,
    offset: Indent,
    one_line_width: Option<usize>,
) -> Option<String> {
    match *struct_parts.def {
        ast::VariantData::Unit(..) => format_unit_struct(context, struct_parts, offset),
        ast::VariantData::Tuple(ref fields, _) => {
            format_tuple_struct(context, struct_parts, fields, offset)
        }
        ast::VariantData::Struct(ref fields, _) => {
            format_struct_struct(context, struct_parts, fields, offset, one_line_width)
        }
    }
}

pub fn format_trait(context: &RewriteContext, item: &ast::Item, offset: Indent) -> Option<String> {
    if let ast::ItemKind::Trait(
        is_auto,
        unsafety,
        ref generics,
        ref generic_bounds,
        ref trait_items,
    ) = item.node
    {
        let mut result = String::with_capacity(128);
        let header = format!(
            "{}{}{}trait ",
            format_visibility(context, &item.vis),
            format_unsafety(unsafety),
            format_auto(is_auto),
        );
        result.push_str(&header);

        let body_lo = context.snippet_provider.span_after(item.span, "{");

        let shape = Shape::indented(offset, context.config).offset_left(result.len())?;
        let generics_str =
            rewrite_generics(context, rewrite_ident(context, item.ident), generics, shape)?;
        result.push_str(&generics_str);

        // FIXME(#2055): rustfmt fails to format when there are comments between trait bounds.
        if !generic_bounds.is_empty() {
            let ident_hi = context
                .snippet_provider
                .span_after(item.span, &item.ident.as_str());
            let bound_hi = generic_bounds.last().unwrap().span().hi();
            let snippet = context.snippet(mk_sp(ident_hi, bound_hi));
            if contains_comment(snippet) {
                return None;
            }

            result = rewrite_assign_rhs_with(
                context,
                result + ":",
                generic_bounds,
                shape,
                RhsTactics::ForceNextLineWithoutIndent,
            )?;
        }

        // Rewrite where clause.
        if !generics.where_clause.predicates.is_empty() {
            let where_density = if context.config.indent_style() == IndentStyle::Block {
                Density::Compressed
            } else {
                Density::Tall
            };

            let where_budget = context.budget(last_line_width(&result));
            let pos_before_where = if generic_bounds.is_empty() {
                generics.where_clause.span.lo()
            } else {
                generic_bounds[generic_bounds.len() - 1].span().hi()
            };
            let option = WhereClauseOption::snuggled(&generics_str);
            let where_clause_str = rewrite_where_clause(
                context,
                &generics.where_clause,
                context.config.brace_style(),
                Shape::legacy(where_budget, offset.block_only()),
                where_density,
                "{",
                None,
                pos_before_where,
                option,
                false,
            )?;
            // If the where clause cannot fit on the same line,
            // put the where clause on a new line
            if !where_clause_str.contains('\n')
                && last_line_width(&result) + where_clause_str.len() + offset.width()
                    > context.config.comment_width()
            {
                let width = offset.block_indent + context.config.tab_spaces() - 1;
                let where_indent = Indent::new(0, width);
                result.push_str(&where_indent.to_string_with_newline(context.config));
            }
            result.push_str(&where_clause_str);
        } else {
            let item_snippet = context.snippet(item.span);
            if let Some(lo) = item_snippet.find('/') {
                // 1 = `{`
                let comment_hi = body_lo - BytePos(1);
                let comment_lo = item.span.lo() + BytePos(lo as u32);
                if comment_lo < comment_hi {
                    match recover_missing_comment_in_span(
                        mk_sp(comment_lo, comment_hi),
                        Shape::indented(offset, context.config),
                        context,
                        last_line_width(&result),
                    ) {
                        Some(ref missing_comment) if !missing_comment.is_empty() => {
                            result.push_str(missing_comment);
                        }
                        _ => (),
                    }
                }
            }
        }

        match context.config.brace_style() {
            _ if last_line_contains_single_line_comment(&result)
                || last_line_width(&result) + 2 > context.budget(offset.width()) =>
            {
                result.push_str(&offset.to_string_with_newline(context.config));
            }
            BraceStyle::AlwaysNextLine => {
                result.push_str(&offset.to_string_with_newline(context.config));
            }
            BraceStyle::PreferSameLine => result.push(' '),
            BraceStyle::SameLineWhere => {
                if result.contains('\n')
                    || (!generics.where_clause.predicates.is_empty() && !trait_items.is_empty())
                {
                    result.push_str(&offset.to_string_with_newline(context.config));
                } else {
                    result.push(' ');
                }
            }
        }
        result.push('{');

        let snippet = context.snippet(item.span);
        let open_pos = snippet.find_uncommented("{")? + 1;

        if !trait_items.is_empty() || contains_comment(&snippet[open_pos..]) {
            let mut visitor = FmtVisitor::from_context(context);
            visitor.block_indent = offset.block_only().block_indent(context.config);
            visitor.last_pos = item.span.lo() + BytePos(open_pos as u32);

            for item in trait_items {
                visitor.visit_trait_item(item);
            }

            visitor.format_missing(item.span.hi() - BytePos(1));

            let inner_indent_str = visitor.block_indent.to_string_with_newline(context.config);
            let outer_indent_str = offset.block_only().to_string_with_newline(context.config);

            result.push_str(&inner_indent_str);
            result.push_str(visitor.buffer.to_string().trim());
            result.push_str(&outer_indent_str);
        } else if result.contains('\n') {
            result.push('\n');
        }

        result.push('}');
        Some(result)
    } else {
        unreachable!();
    }
}

pub fn format_trait_alias(
    context: &RewriteContext,
    ident: ast::Ident,
    generics: &ast::Generics,
    generic_bounds: &ast::GenericBounds,
    shape: Shape,
) -> Option<String> {
    let alias = rewrite_ident(context, ident);
    // 6 = "trait ", 2 = " ="
    let g_shape = shape.offset_left(6)?.sub_width(2)?;
    let generics_str = rewrite_generics(context, &alias, generics, g_shape)?;
    let lhs = format!("trait {} =", generics_str);
    // 1 = ";"
    rewrite_assign_rhs(context, lhs, generic_bounds, shape.sub_width(1)?).map(|s| s + ";")
}

fn format_unit_struct(context: &RewriteContext, p: &StructParts, offset: Indent) -> Option<String> {
    let header_str = format_header(context, p.prefix, p.ident, p.vis);
    let generics_str = if let Some(generics) = p.generics {
        let hi = if generics.where_clause.predicates.is_empty() {
            generics.span.hi()
        } else {
            generics.where_clause.span.hi()
        };
        format_generics(
            context,
            generics,
            context.config.brace_style(),
            BracePos::None,
            offset,
            mk_sp(generics.span.lo(), hi),
            last_line_width(&header_str),
        )?
    } else {
        String::new()
    };
    Some(format!("{}{};", header_str, generics_str))
}

pub fn format_struct_struct(
    context: &RewriteContext,
    struct_parts: &StructParts,
    fields: &[ast::StructField],
    offset: Indent,
    one_line_width: Option<usize>,
) -> Option<String> {
    let mut result = String::with_capacity(1024);
    let span = struct_parts.span;

    let header_str = struct_parts.format_header(context);
    result.push_str(&header_str);

    let header_hi = span.lo() + BytePos(header_str.len() as u32);
    let body_lo = context.snippet_provider.span_after(span, "{");

    let generics_str = match struct_parts.generics {
        Some(g) => format_generics(
            context,
            g,
            context.config.brace_style(),
            if fields.is_empty() {
                BracePos::ForceSameLine
            } else {
                BracePos::Auto
            },
            offset,
            mk_sp(header_hi, body_lo),
            last_line_width(&result),
        )?,
        None => {
            // 3 = ` {}`, 2 = ` {`.
            let overhead = if fields.is_empty() { 3 } else { 2 };
            if (context.config.brace_style() == BraceStyle::AlwaysNextLine && !fields.is_empty())
                || context.config.max_width() < overhead + result.len()
            {
                format!("\n{}{{", offset.block_only().to_string(context.config))
            } else {
                " {".to_owned()
            }
        }
    };
    // 1 = `}`
    let overhead = if fields.is_empty() { 1 } else { 0 };
    let total_width = result.len() + generics_str.len() + overhead;
    if !generics_str.is_empty()
        && !generics_str.contains('\n')
        && total_width > context.config.max_width()
    {
        result.push('\n');
        result.push_str(&offset.to_string(context.config));
        result.push_str(generics_str.trim_left());
    } else {
        result.push_str(&generics_str);
    }

    if fields.is_empty() {
        let inner_span = mk_sp(body_lo, span.hi() - BytePos(1));
        format_empty_struct_or_tuple(context, inner_span, offset, &mut result, "", "}");
        return Some(result);
    }

    // 3 = ` ` and ` }`
    let one_line_budget = context.budget(result.len() + 3 + offset.width());
    let one_line_budget =
        one_line_width.map_or(0, |one_line_width| min(one_line_width, one_line_budget));

    let items_str = rewrite_with_alignment(
        fields,
        context,
        Shape::indented(offset, context.config).sub_width(1)?,
        mk_sp(body_lo, span.hi()),
        one_line_budget,
    )?;

    if !items_str.contains('\n')
        && !result.contains('\n')
        && items_str.len() <= one_line_budget
        && !last_line_contains_single_line_comment(&items_str)
    {
        Some(format!("{} {} }}", result, items_str))
    } else {
        Some(format!(
            "{}\n{}{}\n{}}}",
            result,
            offset
                .block_indent(context.config)
                .to_string(context.config),
            items_str,
            offset.to_string(context.config)
        ))
    }
}

fn get_bytepos_after_visibility(vis: &ast::Visibility, default_span: Span) -> BytePos {
    match vis.node {
        ast::VisibilityKind::Crate(..) | ast::VisibilityKind::Restricted { .. } => vis.span.hi(),
        _ => default_span.lo(),
    }
}

// Format tuple or struct without any fields. We need to make sure that the comments
// inside the delimiters are preserved.
fn format_empty_struct_or_tuple(
    context: &RewriteContext,
    span: Span,
    offset: Indent,
    result: &mut String,
    opener: &str,
    closer: &str,
) {
    // 3 = " {}" or "();"
    let used_width = last_line_used_width(&result, offset.width()) + 3;
    if used_width > context.config.max_width() {
        result.push_str(&offset.to_string_with_newline(context.config))
    }
    result.push_str(opener);
    match rewrite_missing_comment(span, Shape::indented(offset, context.config), context) {
        Some(ref s) if s.is_empty() => (),
        Some(ref s) => {
            if !is_single_line(s) || first_line_contains_single_line_comment(s) {
                let nested_indent_str = offset
                    .block_indent(context.config)
                    .to_string_with_newline(context.config);
                result.push_str(&nested_indent_str);
            }
            result.push_str(s);
            if last_line_contains_single_line_comment(s) {
                result.push_str(&offset.to_string_with_newline(context.config));
            }
        }
        None => result.push_str(context.snippet(span)),
    }
    result.push_str(closer);
}

fn format_tuple_struct(
    context: &RewriteContext,
    struct_parts: &StructParts,
    fields: &[ast::StructField],
    offset: Indent,
) -> Option<String> {
    let mut result = String::with_capacity(1024);
    let span = struct_parts.span;

    let header_str = struct_parts.format_header(context);
    result.push_str(&header_str);

    let body_lo = if fields.is_empty() {
        let lo = get_bytepos_after_visibility(struct_parts.vis, span);
        context
            .snippet_provider
            .span_after(mk_sp(lo, span.hi()), "(")
    } else {
        fields[0].span.lo()
    };
    let body_hi = if fields.is_empty() {
        context
            .snippet_provider
            .span_after(mk_sp(body_lo, span.hi()), ")")
    } else {
        // This is a dirty hack to work around a missing `)` from the span of the last field.
        let last_arg_span = fields[fields.len() - 1].span;
        context
            .snippet_provider
            .opt_span_after(mk_sp(last_arg_span.hi(), span.hi()), ")")
            .unwrap_or(last_arg_span.hi())
    };

    let where_clause_str = match struct_parts.generics {
        Some(generics) => {
            let budget = context.budget(last_line_width(&header_str));
            let shape = Shape::legacy(budget, offset);
            let generics_str = rewrite_generics(context, "", generics, shape)?;
            result.push_str(&generics_str);

            let where_budget = context.budget(last_line_width(&result));
            let option = WhereClauseOption::new(true, false);
            rewrite_where_clause(
                context,
                &generics.where_clause,
                context.config.brace_style(),
                Shape::legacy(where_budget, offset.block_only()),
                Density::Compressed,
                ";",
                None,
                body_hi,
                option,
                false,
            )?
        }
        None => "".to_owned(),
    };

    if fields.is_empty() {
        let body_hi = context
            .snippet_provider
            .span_before(mk_sp(body_lo, span.hi()), ")");
        let inner_span = mk_sp(body_lo, body_hi);
        format_empty_struct_or_tuple(context, inner_span, offset, &mut result, "(", ")");
    } else {
        let shape = Shape::indented(offset, context.config).sub_width(1)?;
        let fields = &fields.iter().collect::<Vec<_>>();
        result = overflow::rewrite_with_parens(
            context,
            &result,
            fields,
            shape,
            span,
            context.config.width_heuristics().fn_call_width,
            None,
        )?;
    }

    if !where_clause_str.is_empty()
        && !where_clause_str.contains('\n')
        && (result.contains('\n')
            || offset.block_indent + result.len() + where_clause_str.len() + 1
                > context.config.max_width())
    {
        // We need to put the where clause on a new line, but we didn't
        // know that earlier, so the where clause will not be indented properly.
        result.push('\n');
        result.push_str(
            &(offset.block_only() + (context.config.tab_spaces() - 1)).to_string(context.config),
        );
    }
    result.push_str(&where_clause_str);

    Some(result)
}

fn rewrite_type_prefix(
    context: &RewriteContext,
    indent: Indent,
    prefix: &str,
    ident: ast::Ident,
    generics: &ast::Generics,
) -> Option<String> {
    let mut result = String::with_capacity(128);
    result.push_str(prefix);
    let ident_str = rewrite_ident(context, ident);

    // 2 = `= `
    if generics.params.is_empty() {
        result.push_str(ident_str)
    } else {
        let g_shape = Shape::indented(indent, context.config)
            .offset_left(result.len())?
            .sub_width(2)?;
        let generics_str = rewrite_generics(context, ident_str, generics, g_shape)?;
        result.push_str(&generics_str);
    }

    let where_budget = context.budget(last_line_width(&result));
    let option = WhereClauseOption::snuggled(&result);
    let where_clause_str = rewrite_where_clause(
        context,
        &generics.where_clause,
        context.config.brace_style(),
        Shape::legacy(where_budget, indent),
        Density::Vertical,
        "=",
        None,
        generics.span.hi(),
        option,
        false,
    )?;
    result.push_str(&where_clause_str);

    Some(result)
}

fn rewrite_type_item<R: Rewrite>(
    context: &RewriteContext,
    indent: Indent,
    prefix: &str,
    suffix: &str,
    ident: ast::Ident,
    rhs: &R,
    generics: &ast::Generics,
    vis: &ast::Visibility,
) -> Option<String> {
    let mut result = String::with_capacity(128);
    result.push_str(&rewrite_type_prefix(
        context,
        indent,
        &format!("{}{} ", format_visibility(context, vis), prefix),
        ident,
        generics,
    )?);

    if generics.where_clause.predicates.is_empty() {
        result.push_str(suffix);
    } else {
        result.push_str(&indent.to_string_with_newline(context.config));
        result.push_str(suffix.trim_left());
    }

    // 1 = ";"
    let rhs_shape = Shape::indented(indent, context.config).sub_width(1)?;
    rewrite_assign_rhs(context, result, rhs, rhs_shape).map(|s| s + ";")
}

pub fn rewrite_type_alias(
    context: &RewriteContext,
    indent: Indent,
    ident: ast::Ident,
    ty: &ast::Ty,
    generics: &ast::Generics,
    vis: &ast::Visibility,
) -> Option<String> {
    rewrite_type_item(context, indent, "type", " =", ident, ty, generics, vis)
}

pub fn rewrite_existential_type(
    context: &RewriteContext,
    indent: Indent,
    ident: ast::Ident,
    generic_bounds: &ast::GenericBounds,
    generics: &ast::Generics,
    vis: &ast::Visibility,
) -> Option<String> {
    rewrite_type_item(
        context,
        indent,
        "existential type",
        ":",
        ident,
        generic_bounds,
        generics,
        vis,
    )
}

fn type_annotation_spacing(config: &Config) -> (&str, &str) {
    (
        if config.space_before_colon() { " " } else { "" },
        if config.space_after_colon() { " " } else { "" },
    )
}

pub fn rewrite_struct_field_prefix(
    context: &RewriteContext,
    field: &ast::StructField,
) -> Option<String> {
    let vis = format_visibility(context, &field.vis);
    let type_annotation_spacing = type_annotation_spacing(context.config);
    Some(match field.ident {
        Some(name) => format!(
            "{}{}{}:",
            vis,
            rewrite_ident(context, name),
            type_annotation_spacing.0
        ),
        None => format!("{}", vis),
    })
}

impl Rewrite for ast::StructField {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        rewrite_struct_field(context, self, shape, 0)
    }
}

pub fn rewrite_struct_field(
    context: &RewriteContext,
    field: &ast::StructField,
    shape: Shape,
    lhs_max_width: usize,
) -> Option<String> {
    if contains_skip(&field.attrs) {
        return Some(context.snippet(field.span()).to_owned());
    }

    let type_annotation_spacing = type_annotation_spacing(context.config);
    let prefix = rewrite_struct_field_prefix(context, field)?;

    let attrs_str = field.attrs.rewrite(context, shape)?;
    let attrs_extendable = field.ident.is_none() && is_attributes_extendable(&attrs_str);
    let missing_span = if field.attrs.is_empty() {
        mk_sp(field.span.lo(), field.span.lo())
    } else {
        mk_sp(field.attrs.last().unwrap().span.hi(), field.span.lo())
    };
    let mut spacing = String::from(if field.ident.is_some() {
        type_annotation_spacing.1
    } else {
        ""
    });
    // Try to put everything on a single line.
    let attr_prefix = combine_strs_with_missing_comments(
        context,
        &attrs_str,
        &prefix,
        missing_span,
        shape,
        attrs_extendable,
    )?;
    let overhead = last_line_width(&attr_prefix);
    let lhs_offset = lhs_max_width.saturating_sub(overhead);
    for _ in 0..lhs_offset {
        spacing.push(' ');
    }
    // In this extreme case we will be missing a space betweeen an attribute and a field.
    if prefix.is_empty() && !attrs_str.is_empty() && attrs_extendable && spacing.is_empty() {
        spacing.push(' ');
    }
    let orig_ty = shape
        .offset_left(overhead + spacing.len())
        .and_then(|ty_shape| field.ty.rewrite(context, ty_shape));
    if let Some(ref ty) = orig_ty {
        if !ty.contains('\n') {
            return Some(attr_prefix + &spacing + ty);
        }
    }

    let is_prefix_empty = prefix.is_empty();
    // We must use multiline. We are going to put attributes and a field on different lines.
    let field_str = rewrite_assign_rhs(context, prefix, &*field.ty, shape)?;
    // Remove a leading white-space from `rewrite_assign_rhs()` when rewriting a tuple struct.
    let field_str = if is_prefix_empty {
        field_str.trim_left()
    } else {
        &field_str
    };
    combine_strs_with_missing_comments(context, &attrs_str, field_str, missing_span, shape, false)
}

pub struct StaticParts<'a> {
    prefix: &'a str,
    vis: &'a ast::Visibility,
    ident: ast::Ident,
    ty: &'a ast::Ty,
    mutability: ast::Mutability,
    expr_opt: Option<&'a ptr::P<ast::Expr>>,
    defaultness: Option<ast::Defaultness>,
    span: Span,
}

impl<'a> StaticParts<'a> {
    pub fn from_item(item: &'a ast::Item) -> Self {
        let (prefix, ty, mutability, expr) = match item.node {
            ast::ItemKind::Static(ref ty, mutability, ref expr) => ("static", ty, mutability, expr),
            ast::ItemKind::Const(ref ty, ref expr) => {
                ("const", ty, ast::Mutability::Immutable, expr)
            }
            _ => unreachable!(),
        };
        StaticParts {
            prefix,
            vis: &item.vis,
            ident: item.ident,
            ty,
            mutability,
            expr_opt: Some(expr),
            defaultness: None,
            span: item.span,
        }
    }

    pub fn from_trait_item(ti: &'a ast::TraitItem) -> Self {
        let (ty, expr_opt) = match ti.node {
            ast::TraitItemKind::Const(ref ty, ref expr_opt) => (ty, expr_opt),
            _ => unreachable!(),
        };
        StaticParts {
            prefix: "const",
            vis: &DEFAULT_VISIBILITY,
            ident: ti.ident,
            ty,
            mutability: ast::Mutability::Immutable,
            expr_opt: expr_opt.as_ref(),
            defaultness: None,
            span: ti.span,
        }
    }

    pub fn from_impl_item(ii: &'a ast::ImplItem) -> Self {
        let (ty, expr) = match ii.node {
            ast::ImplItemKind::Const(ref ty, ref expr) => (ty, expr),
            _ => unreachable!(),
        };
        StaticParts {
            prefix: "const",
            vis: &ii.vis,
            ident: ii.ident,
            ty,
            mutability: ast::Mutability::Immutable,
            expr_opt: Some(expr),
            defaultness: Some(ii.defaultness),
            span: ii.span,
        }
    }
}

fn rewrite_static(
    context: &RewriteContext,
    static_parts: &StaticParts,
    offset: Indent,
) -> Option<String> {
    let colon = colon_spaces(
        context.config.space_before_colon(),
        context.config.space_after_colon(),
    );
    let mut prefix = format!(
        "{}{}{} {}{}{}",
        format_visibility(context, static_parts.vis),
        static_parts.defaultness.map_or("", format_defaultness),
        static_parts.prefix,
        format_mutability(static_parts.mutability),
        static_parts.ident,
        colon,
    );
    // 2 = " =".len()
    let ty_shape =
        Shape::indented(offset.block_only(), context.config).offset_left(prefix.len() + 2)?;
    let ty_str = match static_parts.ty.rewrite(context, ty_shape) {
        Some(ty_str) => ty_str,
        None => {
            if prefix.ends_with(' ') {
                prefix.pop();
            }
            let nested_indent = offset.block_indent(context.config);
            let nested_shape = Shape::indented(nested_indent, context.config);
            let ty_str = static_parts.ty.rewrite(context, nested_shape)?;
            format!(
                "{}{}",
                nested_indent.to_string_with_newline(context.config),
                ty_str
            )
        }
    };

    if let Some(expr) = static_parts.expr_opt {
        let lhs = format!("{}{} =", prefix, ty_str);
        // 1 = ;
        let remaining_width = context.budget(offset.block_indent + 1);
        rewrite_assign_rhs(
            context,
            lhs,
            &**expr,
            Shape::legacy(remaining_width, offset.block_only()),
        ).and_then(|res| recover_comment_removed(res, static_parts.span, context))
        .map(|s| if s.ends_with(';') { s } else { s + ";" })
    } else {
        Some(format!("{}{};", prefix, ty_str))
    }
}

pub fn rewrite_associated_type(
    ident: ast::Ident,
    ty_opt: Option<&ptr::P<ast::Ty>>,
    generic_bounds_opt: Option<&ast::GenericBounds>,
    context: &RewriteContext,
    indent: Indent,
) -> Option<String> {
    let prefix = format!("type {}", rewrite_ident(context, ident));

    let type_bounds_str = if let Some(bounds) = generic_bounds_opt {
        if bounds.is_empty() {
            String::new()
        } else {
            // 2 = ": ".len()
            let shape = Shape::indented(indent, context.config).offset_left(prefix.len() + 2)?;
            bounds.rewrite(context, shape).map(|s| format!(": {}", s))?
        }
    } else {
        String::new()
    };

    if let Some(ty) = ty_opt {
        // 1 = `;`
        let shape = Shape::indented(indent, context.config).sub_width(1)?;
        let lhs = format!("{}{} =", prefix, type_bounds_str);
        rewrite_assign_rhs(context, lhs, &**ty, shape).map(|s| s + ";")
    } else {
        Some(format!("{}{};", prefix, type_bounds_str))
    }
}

pub fn rewrite_existential_impl_type(
    context: &RewriteContext,
    ident: ast::Ident,
    generic_bounds: &ast::GenericBounds,
    indent: Indent,
) -> Option<String> {
    rewrite_associated_type(ident, None, Some(generic_bounds), context, indent)
        .map(|s| format!("existential {}", s))
}

pub fn rewrite_associated_impl_type(
    ident: ast::Ident,
    defaultness: ast::Defaultness,
    ty_opt: Option<&ptr::P<ast::Ty>>,
    context: &RewriteContext,
    indent: Indent,
) -> Option<String> {
    let result = rewrite_associated_type(ident, ty_opt, None, context, indent)?;

    match defaultness {
        ast::Defaultness::Default => Some(format!("default {}", result)),
        _ => Some(result),
    }
}

impl Rewrite for ast::FunctionRetTy {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        match *self {
            ast::FunctionRetTy::Default(_) => Some(String::new()),
            ast::FunctionRetTy::Ty(ref ty) => {
                let inner_width = shape.width.checked_sub(3)?;
                ty.rewrite(context, Shape::legacy(inner_width, shape.indent + 3))
                    .map(|r| format!("-> {}", r))
            }
        }
    }
}

fn is_empty_infer(context: &RewriteContext, ty: &ast::Ty) -> bool {
    match ty.node {
        ast::TyKind::Infer => {
            let original = context.snippet(ty.span);
            original != "_"
        }
        _ => false,
    }
}

impl Rewrite for ast::Arg {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        if is_named_arg(self) {
            let mut result = self
                .pat
                .rewrite(context, Shape::legacy(shape.width, shape.indent))?;

            if !is_empty_infer(context, &*self.ty) {
                if context.config.space_before_colon() {
                    result.push_str(" ");
                }
                result.push_str(":");
                if context.config.space_after_colon() {
                    result.push_str(" ");
                }
                let overhead = last_line_width(&result);
                let max_width = shape.width.checked_sub(overhead)?;
                let ty_str = self
                    .ty
                    .rewrite(context, Shape::legacy(max_width, shape.indent))?;
                result.push_str(&ty_str);
            }

            Some(result)
        } else {
            self.ty.rewrite(context, shape)
        }
    }
}

fn rewrite_explicit_self(
    explicit_self: &ast::ExplicitSelf,
    args: &[ast::Arg],
    context: &RewriteContext,
) -> Option<String> {
    match explicit_self.node {
        ast::SelfKind::Region(lt, m) => {
            let mut_str = format_mutability(m);
            match lt {
                Some(ref l) => {
                    let lifetime_str = l.rewrite(
                        context,
                        Shape::legacy(context.config.max_width(), Indent::empty()),
                    )?;
                    Some(format!("&{} {}self", lifetime_str, mut_str))
                }
                None => Some(format!("&{}self", mut_str)),
            }
        }
        ast::SelfKind::Explicit(ref ty, _) => {
            assert!(!args.is_empty(), "&[ast::Arg] shouldn't be empty.");

            let mutability = explicit_self_mutability(&args[0]);
            let type_str = ty.rewrite(
                context,
                Shape::legacy(context.config.max_width(), Indent::empty()),
            )?;

            Some(format!(
                "{}self: {}",
                format_mutability(mutability),
                type_str
            ))
        }
        ast::SelfKind::Value(_) => {
            assert!(!args.is_empty(), "&[ast::Arg] shouldn't be empty.");

            let mutability = explicit_self_mutability(&args[0]);

            Some(format!("{}self", format_mutability(mutability)))
        }
    }
}

// Hacky solution caused by absence of `Mutability` in `SelfValue` and
// `SelfExplicit` variants of `ast::ExplicitSelf_`.
fn explicit_self_mutability(arg: &ast::Arg) -> ast::Mutability {
    if let ast::PatKind::Ident(ast::BindingMode::ByValue(mutability), _, _) = arg.pat.node {
        mutability
    } else {
        unreachable!()
    }
}

pub fn span_lo_for_arg(arg: &ast::Arg) -> BytePos {
    if is_named_arg(arg) {
        arg.pat.span.lo()
    } else {
        arg.ty.span.lo()
    }
}

pub fn span_hi_for_arg(context: &RewriteContext, arg: &ast::Arg) -> BytePos {
    match arg.ty.node {
        ast::TyKind::Infer if context.snippet(arg.ty.span) == "_" => arg.ty.span.hi(),
        ast::TyKind::Infer if is_named_arg(arg) => arg.pat.span.hi(),
        _ => arg.ty.span.hi(),
    }
}

pub fn is_named_arg(arg: &ast::Arg) -> bool {
    if let ast::PatKind::Ident(_, ident, _) = arg.pat.node {
        ident != symbol::keywords::Invalid.ident()
    } else {
        true
    }
}

// Return type is (result, force_new_line_for_brace)
fn rewrite_fn_base(
    context: &RewriteContext,
    indent: Indent,
    ident: ast::Ident,
    fn_sig: &FnSig,
    span: Span,
    newline_brace: bool,
    has_body: bool,
) -> Option<(String, bool)> {
    let mut force_new_line_for_brace = false;

    let where_clause = &fn_sig.generics.where_clause;

    let mut result = String::with_capacity(1024);
    result.push_str(&fn_sig.to_str(context));

    // fn foo
    result.push_str("fn ");

    // Generics.
    let overhead = if has_body && !newline_brace {
        // 4 = `() {`
        4
    } else {
        // 2 = `()`
        2
    };
    let used_width = last_line_used_width(&result, indent.width());
    let one_line_budget = context.budget(used_width + overhead);
    let shape = Shape {
        width: one_line_budget,
        indent,
        offset: used_width,
    };
    let fd = fn_sig.decl;
    let generics_str = rewrite_generics(
        context,
        rewrite_ident(context, ident),
        fn_sig.generics,
        shape,
    )?;
    result.push_str(&generics_str);

    let snuggle_angle_bracket = generics_str
        .lines()
        .last()
        .map_or(false, |l| l.trim_left().len() == 1);

    // Note that the width and indent don't really matter, we'll re-layout the
    // return type later anyway.
    let ret_str = fd
        .output
        .rewrite(context, Shape::indented(indent, context.config))?;

    let multi_line_ret_str = ret_str.contains('\n');
    let ret_str_len = if multi_line_ret_str { 0 } else { ret_str.len() };

    // Args.
    let (one_line_budget, multi_line_budget, mut arg_indent) = compute_budgets_for_args(
        context,
        &result,
        indent,
        ret_str_len,
        newline_brace,
        has_body,
        multi_line_ret_str,
    )?;

    debug!(
        "rewrite_fn_base: one_line_budget: {}, multi_line_budget: {}, arg_indent: {:?}",
        one_line_budget, multi_line_budget, arg_indent
    );

    // Check if vertical layout was forced.
    if one_line_budget == 0 {
        if snuggle_angle_bracket {
            result.push('(');
        } else {
            result.push_str("(");
            if context.config.indent_style() == IndentStyle::Visual {
                result.push_str(&arg_indent.to_string_with_newline(context.config));
            }
        }
    } else {
        result.push('(');
    }

    // Skip `pub(crate)`.
    let lo_after_visibility = get_bytepos_after_visibility(&fn_sig.visibility, span);
    // A conservative estimation, to goal is to be over all parens in generics
    let args_start = fn_sig
        .generics
        .params
        .iter()
        .last()
        .map_or(lo_after_visibility, |param| param.span().hi());
    let args_end = if fd.inputs.is_empty() {
        context
            .snippet_provider
            .span_after(mk_sp(args_start, span.hi()), ")")
    } else {
        let last_span = mk_sp(fd.inputs[fd.inputs.len() - 1].span().hi(), span.hi());
        context.snippet_provider.span_after(last_span, ")")
    };
    let args_span = mk_sp(
        context
            .snippet_provider
            .span_after(mk_sp(args_start, span.hi()), "("),
        args_end,
    );
    let arg_str = rewrite_args(
        context,
        &fd.inputs,
        fd.get_self().as_ref(),
        one_line_budget,
        multi_line_budget,
        indent,
        arg_indent,
        args_span,
        fd.variadic,
        generics_str.contains('\n'),
    )?;

    let put_args_in_block = match context.config.indent_style() {
        IndentStyle::Block => arg_str.contains('\n') || arg_str.len() > one_line_budget,
        _ => false,
    } && !fd.inputs.is_empty();

    let mut args_last_line_contains_comment = false;
    if put_args_in_block {
        arg_indent = indent.block_indent(context.config);
        result.push_str(&arg_indent.to_string_with_newline(context.config));
        result.push_str(&arg_str);
        result.push_str(&indent.to_string_with_newline(context.config));
        result.push(')');
    } else {
        result.push_str(&arg_str);
        let used_width = last_line_used_width(&result, indent.width()) + first_line_width(&ret_str);
        // Put the closing brace on the next line if it overflows the max width.
        // 1 = `)`
        if fd.inputs.is_empty() && used_width + 1 > context.config.max_width() {
            result.push('\n');
        }
        // If the last line of args contains comment, we cannot put the closing paren
        // on the same line.
        if arg_str
            .lines()
            .last()
            .map_or(false, |last_line| last_line.contains("//"))
        {
            args_last_line_contains_comment = true;
            result.push_str(&arg_indent.to_string_with_newline(context.config));
        }
        result.push(')');
    }

    // Return type.
    if let ast::FunctionRetTy::Ty(..) = fd.output {
        let ret_should_indent = match context.config.indent_style() {
            // If our args are block layout then we surely must have space.
            IndentStyle::Block if put_args_in_block || fd.inputs.is_empty() => false,
            _ if args_last_line_contains_comment => false,
            _ if result.contains('\n') || multi_line_ret_str => true,
            _ => {
                // If the return type would push over the max width, then put the return type on
                // a new line. With the +1 for the signature length an additional space between
                // the closing parenthesis of the argument and the arrow '->' is considered.
                let mut sig_length = result.len() + indent.width() + ret_str_len + 1;

                // If there is no where clause, take into account the space after the return type
                // and the brace.
                if where_clause.predicates.is_empty() {
                    sig_length += 2;
                }

                sig_length > context.config.max_width()
            }
        };
        let ret_indent = if ret_should_indent {
            let indent = if arg_str.is_empty() {
                // Aligning with non-existent args looks silly.
                force_new_line_for_brace = true;
                indent + 4
            } else {
                // FIXME: we might want to check that using the arg indent
                // doesn't blow our budget, and if it does, then fallback to
                // the where clause indent.
                arg_indent
            };

            result.push_str(&indent.to_string_with_newline(context.config));
            indent
        } else {
            result.push(' ');
            Indent::new(indent.block_indent, last_line_width(&result))
        };

        if multi_line_ret_str || ret_should_indent {
            // Now that we know the proper indent and width, we need to
            // re-layout the return type.
            let ret_str = fd
                .output
                .rewrite(context, Shape::indented(ret_indent, context.config))?;
            result.push_str(&ret_str);
        } else {
            result.push_str(&ret_str);
        }

        // Comment between return type and the end of the decl.
        let snippet_lo = fd.output.span().hi();
        if where_clause.predicates.is_empty() {
            let snippet_hi = span.hi();
            let snippet = context.snippet(mk_sp(snippet_lo, snippet_hi));
            // Try to preserve the layout of the original snippet.
            let original_starts_with_newline = snippet
                .find(|c| c != ' ')
                .map_or(false, |i| starts_with_newline(&snippet[i..]));
            let original_ends_with_newline = snippet
                .rfind(|c| c != ' ')
                .map_or(false, |i| snippet[i..].ends_with('\n'));
            let snippet = snippet.trim();
            if !snippet.is_empty() {
                result.push(if original_starts_with_newline {
                    '\n'
                } else {
                    ' '
                });
                result.push_str(snippet);
                if original_ends_with_newline {
                    force_new_line_for_brace = true;
                }
            }
        }
    }

    let pos_before_where = match fd.output {
        ast::FunctionRetTy::Default(..) => args_span.hi(),
        ast::FunctionRetTy::Ty(ref ty) => ty.span.hi(),
    };

    let is_args_multi_lined = arg_str.contains('\n');

    let option = WhereClauseOption::new(!has_body, put_args_in_block && ret_str.is_empty());
    let where_clause_str = rewrite_where_clause(
        context,
        where_clause,
        context.config.brace_style(),
        Shape::indented(indent, context.config),
        Density::Tall,
        "{",
        Some(span.hi()),
        pos_before_where,
        option,
        is_args_multi_lined,
    )?;
    // If there are neither where clause nor return type, we may be missing comments between
    // args and `{`.
    if where_clause_str.is_empty() {
        if let ast::FunctionRetTy::Default(ret_span) = fd.output {
            match recover_missing_comment_in_span(
                mk_sp(args_span.hi(), ret_span.hi()),
                shape,
                context,
                last_line_width(&result),
            ) {
                Some(ref missing_comment) if !missing_comment.is_empty() => {
                    result.push_str(missing_comment);
                    force_new_line_for_brace = true;
                }
                _ => (),
            }
        }
    }

    result.push_str(&where_clause_str);

    force_new_line_for_brace |= last_line_contains_single_line_comment(&result);
    force_new_line_for_brace |= is_args_multi_lined && context.config.where_single_line();
    Some((result, force_new_line_for_brace))
}

#[derive(Copy, Clone)]
struct WhereClauseOption {
    suppress_comma: bool, // Force no trailing comma
    snuggle: bool,        // Do not insert newline before `where`
    compress_where: bool, // Try single line where clause instead of vertical layout
}

impl WhereClauseOption {
    pub fn new(suppress_comma: bool, snuggle: bool) -> WhereClauseOption {
        WhereClauseOption {
            suppress_comma,
            snuggle,
            compress_where: false,
        }
    }

    pub fn snuggled(current: &str) -> WhereClauseOption {
        WhereClauseOption {
            suppress_comma: false,
            snuggle: last_line_width(current) == 1,
            compress_where: false,
        }
    }

    pub fn suppress_comma(&mut self) {
        self.suppress_comma = true
    }

    pub fn compress_where(&mut self) {
        self.compress_where = true
    }

    pub fn snuggle(&mut self) {
        self.snuggle = true
    }
}

fn rewrite_args(
    context: &RewriteContext,
    args: &[ast::Arg],
    explicit_self: Option<&ast::ExplicitSelf>,
    one_line_budget: usize,
    multi_line_budget: usize,
    indent: Indent,
    arg_indent: Indent,
    span: Span,
    variadic: bool,
    generics_str_contains_newline: bool,
) -> Option<String> {
    let mut arg_item_strs = args
        .iter()
        .map(|arg| arg.rewrite(context, Shape::legacy(multi_line_budget, arg_indent)))
        .collect::<Option<Vec<_>>>()?;

    // Account for sugary self.
    // FIXME: the comment for the self argument is dropped. This is blocked
    // on rust issue #27522.
    let min_args = explicit_self
        .and_then(|explicit_self| rewrite_explicit_self(explicit_self, args, context))
        .map_or(1, |self_str| {
            arg_item_strs[0] = self_str;
            2
        });

    // Comments between args.
    let mut arg_items = Vec::new();
    if min_args == 2 {
        arg_items.push(ListItem::from_str(""));
    }

    // FIXME(#21): if there are no args, there might still be a comment, but
    // without spans for the comment or parens, there is no chance of
    // getting it right. You also don't get to put a comment on self, unless
    // it is explicit.
    if args.len() >= min_args || variadic {
        let comment_span_start = if min_args == 2 {
            let second_arg_start = if arg_has_pattern(&args[1]) {
                args[1].pat.span.lo()
            } else {
                args[1].ty.span.lo()
            };
            let reduced_span = mk_sp(span.lo(), second_arg_start);

            context.snippet_provider.span_after_last(reduced_span, ",")
        } else {
            span.lo()
        };

        enum ArgumentKind<'a> {
            Regular(&'a ast::Arg),
            Variadic(BytePos),
        }

        let variadic_arg = if variadic {
            let variadic_span = mk_sp(args.last().unwrap().ty.span.hi(), span.hi());
            let variadic_start =
                context.snippet_provider.span_after(variadic_span, "...") - BytePos(3);
            Some(ArgumentKind::Variadic(variadic_start))
        } else {
            None
        };

        let more_items = itemize_list(
            context.snippet_provider,
            args[min_args - 1..]
                .iter()
                .map(ArgumentKind::Regular)
                .chain(variadic_arg),
            ")",
            ",",
            |arg| match *arg {
                ArgumentKind::Regular(arg) => span_lo_for_arg(arg),
                ArgumentKind::Variadic(start) => start,
            },
            |arg| match *arg {
                ArgumentKind::Regular(arg) => arg.ty.span.hi(),
                ArgumentKind::Variadic(start) => start + BytePos(3),
            },
            |arg| match *arg {
                ArgumentKind::Regular(..) => None,
                ArgumentKind::Variadic(..) => Some("...".to_owned()),
            },
            comment_span_start,
            span.hi(),
            false,
        );

        arg_items.extend(more_items);
    }

    let fits_in_one_line = !generics_str_contains_newline
        && (arg_items.is_empty()
            || arg_items.len() == 1 && arg_item_strs[0].len() <= one_line_budget);

    for (item, arg) in arg_items.iter_mut().zip(arg_item_strs) {
        item.item = Some(arg);
    }

    let last_line_ends_with_comment = arg_items
        .iter()
        .last()
        .and_then(|item| item.post_comment.as_ref())
        .map_or(false, |s| s.trim().starts_with("//"));

    let (indent, trailing_comma) = match context.config.indent_style() {
        IndentStyle::Block if fits_in_one_line => {
            (indent.block_indent(context.config), SeparatorTactic::Never)
        }
        IndentStyle::Block => (
            indent.block_indent(context.config),
            context.config.trailing_comma(),
        ),
        IndentStyle::Visual if last_line_ends_with_comment => {
            (arg_indent, context.config.trailing_comma())
        }
        IndentStyle::Visual => (arg_indent, SeparatorTactic::Never),
    };

    let tactic = definitive_tactic(
        &arg_items,
        context.config.fn_args_density().to_list_tactic(),
        Separator::Comma,
        one_line_budget,
    );
    let budget = match tactic {
        DefinitiveListTactic::Horizontal => one_line_budget,
        _ => multi_line_budget,
    };

    debug!("rewrite_args: budget: {}, tactic: {:?}", budget, tactic);

    let trailing_separator = if variadic {
        SeparatorTactic::Never
    } else {
        trailing_comma
    };
    let fmt = ListFormatting::new(Shape::legacy(budget, indent), context.config)
        .tactic(tactic)
        .trailing_separator(trailing_separator)
        .ends_with_newline(tactic.ends_with_newline(context.config.indent_style()))
        .preserve_newline(true);
    write_list(&arg_items, &fmt)
}

fn arg_has_pattern(arg: &ast::Arg) -> bool {
    if let ast::PatKind::Ident(_, ident, _) = arg.pat.node {
        ident != symbol::keywords::Invalid.ident()
    } else {
        true
    }
}

fn compute_budgets_for_args(
    context: &RewriteContext,
    result: &str,
    indent: Indent,
    ret_str_len: usize,
    newline_brace: bool,
    has_braces: bool,
    force_vertical_layout: bool,
) -> Option<((usize, usize, Indent))> {
    debug!(
        "compute_budgets_for_args {} {:?}, {}, {}",
        result.len(),
        indent,
        ret_str_len,
        newline_brace
    );
    // Try keeping everything on the same line.
    if !result.contains('\n') && !force_vertical_layout {
        // 2 = `()`, 3 = `() `, space is before ret_string.
        let overhead = if ret_str_len == 0 { 2 } else { 3 };
        let mut used_space = indent.width() + result.len() + ret_str_len + overhead;
        if has_braces {
            if !newline_brace {
                // 2 = `{}`
                used_space += 2;
            }
        } else {
            // 1 = `;`
            used_space += 1;
        }
        let one_line_budget = context.budget(used_space);

        if one_line_budget > 0 {
            // 4 = "() {".len()
            let (indent, multi_line_budget) = match context.config.indent_style() {
                IndentStyle::Block => {
                    let indent = indent.block_indent(context.config);
                    (indent, context.budget(indent.width() + 1))
                }
                IndentStyle::Visual => {
                    let indent = indent + result.len() + 1;
                    let multi_line_overhead = indent.width() + if newline_brace { 2 } else { 4 };
                    (indent, context.budget(multi_line_overhead))
                }
            };

            return Some((one_line_budget, multi_line_budget, indent));
        }
    }

    // Didn't work. we must force vertical layout and put args on a newline.
    let new_indent = indent.block_indent(context.config);
    let used_space = match context.config.indent_style() {
        // 1 = `,`
        IndentStyle::Block => new_indent.width() + 1,
        // Account for `)` and possibly ` {`.
        IndentStyle::Visual => new_indent.width() + if ret_str_len == 0 { 1 } else { 3 },
    };
    Some((0, context.budget(used_space), new_indent))
}

fn newline_for_brace(config: &Config, where_clause: &ast::WhereClause) -> bool {
    let predicate_count = where_clause.predicates.len();

    if config.where_single_line() && predicate_count == 1 {
        return false;
    }
    let brace_style = config.brace_style();

    brace_style == BraceStyle::AlwaysNextLine
        || (brace_style == BraceStyle::SameLineWhere && predicate_count > 0)
}

fn rewrite_generics(
    context: &RewriteContext,
    ident: &str,
    generics: &ast::Generics,
    shape: Shape,
) -> Option<String> {
    // FIXME: convert bounds to where clauses where they get too big or if
    // there is a where clause at all.

    if generics.params.is_empty() {
        return Some(ident.to_owned());
    }

    let params = &generics.params.iter().map(|e| &*e).collect::<Vec<_>>();
    overflow::rewrite_with_angle_brackets(context, ident, params, shape, generics.span)
}

pub fn generics_shape_from_config(config: &Config, shape: Shape, offset: usize) -> Option<Shape> {
    match config.indent_style() {
        IndentStyle::Visual => shape.visual_indent(1 + offset).sub_width(offset + 2),
        IndentStyle::Block => {
            // 1 = ","
            shape
                .block()
                .block_indent(config.tab_spaces())
                .with_max_width(config)
                .sub_width(1)
        }
    }
}

fn rewrite_where_clause_rfc_style(
    context: &RewriteContext,
    where_clause: &ast::WhereClause,
    shape: Shape,
    terminator: &str,
    span_end: Option<BytePos>,
    span_end_before_where: BytePos,
    where_clause_option: WhereClauseOption,
    is_args_multi_line: bool,
) -> Option<String> {
    let block_shape = shape.block().with_max_width(context.config);

    let (span_before, span_after) =
        missing_span_before_after_where(span_end_before_where, where_clause);
    let (comment_before, comment_after) =
        rewrite_comments_before_after_where(context, span_before, span_after, shape)?;

    let starting_newline = if where_clause_option.snuggle && comment_before.is_empty() {
        Cow::from(" ")
    } else {
        block_shape.indent.to_string_with_newline(context.config)
    };

    let clause_shape = block_shape.block_left(context.config.tab_spaces())?;
    // 1 = `,`
    let clause_shape = clause_shape.sub_width(1)?;
    // each clause on one line, trailing comma (except if suppress_comma)
    let span_start = where_clause.predicates[0].span().lo();
    // If we don't have the start of the next span, then use the end of the
    // predicates, but that means we miss comments.
    let len = where_clause.predicates.len();
    let end_of_preds = where_clause.predicates[len - 1].span().hi();
    let span_end = span_end.unwrap_or(end_of_preds);
    let items = itemize_list(
        context.snippet_provider,
        where_clause.predicates.iter(),
        terminator,
        ",",
        |pred| pred.span().lo(),
        |pred| pred.span().hi(),
        |pred| pred.rewrite(context, clause_shape),
        span_start,
        span_end,
        false,
    );
    let where_single_line = context.config.where_single_line() && len == 1 && !is_args_multi_line;
    let comma_tactic = if where_clause_option.suppress_comma || where_single_line {
        SeparatorTactic::Never
    } else {
        context.config.trailing_comma()
    };

    // shape should be vertical only and only if we have `where_single_line` option enabled
    // and the number of items of the where clause is equal to 1
    let shape_tactic = if where_single_line {
        DefinitiveListTactic::Horizontal
    } else {
        DefinitiveListTactic::Vertical
    };

    let fmt = ListFormatting::new(clause_shape, context.config)
        .tactic(shape_tactic)
        .trailing_separator(comma_tactic)
        .preserve_newline(true);
    let preds_str = write_list(&items.collect::<Vec<_>>(), &fmt)?;

    let comment_separator = |comment: &str, shape: Shape| {
        if comment.is_empty() {
            Cow::from("")
        } else {
            shape.indent.to_string_with_newline(context.config)
        }
    };
    let newline_before_where = comment_separator(&comment_before, shape);
    let newline_after_where = comment_separator(&comment_after, clause_shape);

    // 6 = `where `
    let clause_sep = if where_clause_option.compress_where
        && comment_before.is_empty()
        && comment_after.is_empty()
        && !preds_str.contains('\n')
        && 6 + preds_str.len() <= shape.width
        || where_single_line
    {
        Cow::from(" ")
    } else {
        clause_shape.indent.to_string_with_newline(context.config)
    };
    Some(format!(
        "{}{}{}where{}{}{}{}",
        starting_newline,
        comment_before,
        newline_before_where,
        newline_after_where,
        comment_after,
        clause_sep,
        preds_str
    ))
}

fn rewrite_where_clause(
    context: &RewriteContext,
    where_clause: &ast::WhereClause,
    brace_style: BraceStyle,
    shape: Shape,
    density: Density,
    terminator: &str,
    span_end: Option<BytePos>,
    span_end_before_where: BytePos,
    where_clause_option: WhereClauseOption,
    is_args_multi_line: bool,
) -> Option<String> {
    if where_clause.predicates.is_empty() {
        return Some(String::new());
    }

    if context.config.indent_style() == IndentStyle::Block {
        return rewrite_where_clause_rfc_style(
            context,
            where_clause,
            shape,
            terminator,
            span_end,
            span_end_before_where,
            where_clause_option,
            is_args_multi_line,
        );
    }

    let extra_indent = Indent::new(context.config.tab_spaces(), 0);

    let offset = match context.config.indent_style() {
        IndentStyle::Block => shape.indent + extra_indent.block_indent(context.config),
        // 6 = "where ".len()
        IndentStyle::Visual => shape.indent + extra_indent + 6,
    };
    // FIXME: if indent_style != Visual, then the budgets below might
    // be out by a char or two.

    let budget = context.config.max_width() - offset.width();
    let span_start = where_clause.predicates[0].span().lo();
    // If we don't have the start of the next span, then use the end of the
    // predicates, but that means we miss comments.
    let len = where_clause.predicates.len();
    let end_of_preds = where_clause.predicates[len - 1].span().hi();
    let span_end = span_end.unwrap_or(end_of_preds);
    let items = itemize_list(
        context.snippet_provider,
        where_clause.predicates.iter(),
        terminator,
        ",",
        |pred| pred.span().lo(),
        |pred| pred.span().hi(),
        |pred| pred.rewrite(context, Shape::legacy(budget, offset)),
        span_start,
        span_end,
        false,
    );
    let item_vec = items.collect::<Vec<_>>();
    // FIXME: we don't need to collect here
    let tactic = definitive_tactic(&item_vec, ListTactic::Vertical, Separator::Comma, budget);

    let mut comma_tactic = context.config.trailing_comma();
    // Kind of a hack because we don't usually have trailing commas in where clauses.
    if comma_tactic == SeparatorTactic::Vertical || where_clause_option.suppress_comma {
        comma_tactic = SeparatorTactic::Never;
    }

    let fmt = ListFormatting::new(Shape::legacy(budget, offset), context.config)
        .tactic(tactic)
        .trailing_separator(comma_tactic)
        .ends_with_newline(tactic.ends_with_newline(context.config.indent_style()))
        .preserve_newline(true);
    let preds_str = write_list(&item_vec, &fmt)?;

    let end_length = if terminator == "{" {
        // If the brace is on the next line we don't need to count it otherwise it needs two
        // characters " {"
        match brace_style {
            BraceStyle::AlwaysNextLine | BraceStyle::SameLineWhere => 0,
            BraceStyle::PreferSameLine => 2,
        }
    } else if terminator == "=" {
        2
    } else {
        terminator.len()
    };
    if density == Density::Tall
        || preds_str.contains('\n')
        || shape.indent.width() + " where ".len() + preds_str.len() + end_length > shape.width
    {
        Some(format!(
            "\n{}where {}",
            (shape.indent + extra_indent).to_string(context.config),
            preds_str
        ))
    } else {
        Some(format!(" where {}", preds_str))
    }
}

fn missing_span_before_after_where(
    before_item_span_end: BytePos,
    where_clause: &ast::WhereClause,
) -> (Span, Span) {
    let missing_span_before = mk_sp(before_item_span_end, where_clause.span.lo());
    // 5 = `where`
    let pos_after_where = where_clause.span.lo() + BytePos(5);
    let missing_span_after = mk_sp(pos_after_where, where_clause.predicates[0].span().lo());
    (missing_span_before, missing_span_after)
}

fn rewrite_comments_before_after_where(
    context: &RewriteContext,
    span_before_where: Span,
    span_after_where: Span,
    shape: Shape,
) -> Option<(String, String)> {
    let before_comment = rewrite_missing_comment(span_before_where, shape, context)?;
    let after_comment = rewrite_missing_comment(
        span_after_where,
        shape.block_indent(context.config.tab_spaces()),
        context,
    )?;
    Some((before_comment, after_comment))
}

fn format_header(
    context: &RewriteContext,
    item_name: &str,
    ident: ast::Ident,
    vis: &ast::Visibility,
) -> String {
    format!(
        "{}{}{}",
        format_visibility(context, vis),
        item_name,
        rewrite_ident(context, ident)
    )
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum BracePos {
    None,
    Auto,
    ForceSameLine,
}

fn format_generics(
    context: &RewriteContext,
    generics: &ast::Generics,
    brace_style: BraceStyle,
    brace_pos: BracePos,
    offset: Indent,
    span: Span,
    used_width: usize,
) -> Option<String> {
    let shape = Shape::legacy(context.budget(used_width + offset.width()), offset);
    let mut result = rewrite_generics(context, "", generics, shape)?;

    let same_line_brace = if !generics.where_clause.predicates.is_empty() || result.contains('\n') {
        let budget = context.budget(last_line_used_width(&result, offset.width()));
        let mut option = WhereClauseOption::snuggled(&result);
        if brace_pos == BracePos::None {
            option.suppress_comma = true;
        }
        // If the generics are not parameterized then generics.span.hi() == 0,
        // so we use span.lo(), which is the position after `struct Foo`.
        let span_end_before_where = if !generics.params.is_empty() {
            generics.span.hi()
        } else {
            span.lo()
        };
        let where_clause_str = rewrite_where_clause(
            context,
            &generics.where_clause,
            brace_style,
            Shape::legacy(budget, offset.block_only()),
            Density::Tall,
            "{",
            Some(span.hi()),
            span_end_before_where,
            option,
            false,
        )?;
        result.push_str(&where_clause_str);
        brace_pos == BracePos::ForceSameLine
            || brace_style == BraceStyle::PreferSameLine
            || (generics.where_clause.predicates.is_empty()
                && trimmed_last_line_width(&result) == 1)
    } else {
        brace_pos == BracePos::ForceSameLine
            || trimmed_last_line_width(&result) == 1
            || brace_style != BraceStyle::AlwaysNextLine
    };
    if brace_pos == BracePos::None {
        return Some(result);
    }
    let total_used_width = last_line_used_width(&result, used_width);
    let remaining_budget = context.budget(total_used_width);
    // If the same line brace if forced, it indicates that we are rewriting an item with empty body,
    // and hence we take the closer into account as well for one line budget.
    // We assume that the closer has the same length as the opener.
    let overhead = if brace_pos == BracePos::ForceSameLine {
        // 3 = ` {}`
        3
    } else {
        // 2 = ` {`
        2
    };
    let forbid_same_line_brace = overhead > remaining_budget;
    if !forbid_same_line_brace && same_line_brace {
        result.push(' ');
    } else {
        result.push('\n');
        result.push_str(&offset.block_only().to_string(context.config));
    }
    result.push('{');

    Some(result)
}

impl Rewrite for ast::ForeignItem {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        let attrs_str = self.attrs.rewrite(context, shape)?;
        // Drop semicolon or it will be interpreted as comment.
        // FIXME: this may be a faulty span from libsyntax.
        let span = mk_sp(self.span.lo(), self.span.hi() - BytePos(1));

        let item_str = match self.node {
            ast::ForeignItemKind::Fn(ref fn_decl, ref generics) => rewrite_fn_base(
                context,
                shape.indent,
                self.ident,
                &FnSig::new(fn_decl, generics, self.vis.clone()),
                span,
                false,
                false,
            ).map(|(s, _)| format!("{};", s)),
            ast::ForeignItemKind::Static(ref ty, is_mutable) => {
                // FIXME(#21): we're dropping potential comments in between the
                // function keywords here.
                let vis = format_visibility(context, &self.vis);
                let mut_str = if is_mutable { "mut " } else { "" };
                let prefix = format!(
                    "{}static {}{}:",
                    vis,
                    mut_str,
                    rewrite_ident(context, self.ident)
                );
                // 1 = ;
                rewrite_assign_rhs(context, prefix, &**ty, shape.sub_width(1)?).map(|s| s + ";")
            }
            ast::ForeignItemKind::Ty => {
                let vis = format_visibility(context, &self.vis);
                Some(format!(
                    "{}type {};",
                    vis,
                    rewrite_ident(context, self.ident)
                ))
            }
            ast::ForeignItemKind::Macro(ref mac) => {
                rewrite_macro(mac, None, context, shape, MacroPosition::Item)
            }
        }?;

        let missing_span = if self.attrs.is_empty() {
            mk_sp(self.span.lo(), self.span.lo())
        } else {
            mk_sp(self.attrs[self.attrs.len() - 1].span.hi(), self.span.lo())
        };
        combine_strs_with_missing_comments(
            context,
            &attrs_str,
            &item_str,
            missing_span,
            shape,
            false,
        )
    }
}

/// Rewrite an inline mod.
pub fn rewrite_mod(context: &RewriteContext, item: &ast::Item) -> String {
    let mut result = String::with_capacity(32);
    result.push_str(&*format_visibility(context, &item.vis));
    result.push_str("mod ");
    result.push_str(rewrite_ident(context, item.ident));
    result.push(';');
    result
}

/// Rewrite `extern crate foo;` WITHOUT attributes.
pub fn rewrite_extern_crate(context: &RewriteContext, item: &ast::Item) -> Option<String> {
    assert!(is_extern_crate(item));
    let new_str = context.snippet(item.span);
    Some(if contains_comment(new_str) {
        new_str.to_owned()
    } else {
        let no_whitespace = &new_str.split_whitespace().collect::<Vec<&str>>().join(" ");
        String::from(&*Regex::new(r"\s;").unwrap().replace(no_whitespace, ";"))
    })
}

/// Returns true for `mod foo;`, false for `mod foo { .. }`.
pub fn is_mod_decl(item: &ast::Item) -> bool {
    match item.node {
        ast::ItemKind::Mod(ref m) => m.inner.hi() != item.span.hi(),
        _ => false,
    }
}

pub fn is_use_item(item: &ast::Item) -> bool {
    match item.node {
        ast::ItemKind::Use(_) => true,
        _ => false,
    }
}

pub fn is_extern_crate(item: &ast::Item) -> bool {
    match item.node {
        ast::ItemKind::ExternCrate(..) => true,
        _ => false,
    }
}
