// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::attr::HasAttrs;
use syntax::parse::ParseSess;
use syntax::source_map::{self, BytePos, Pos, SourceMap, Span};
use syntax::{ast, visit};

use attr::*;
use comment::{CodeCharKind, CommentCodeSlices, FindUncommented};
use config::{BraceStyle, Config};
use items::{
    format_impl, format_trait, format_trait_alias, is_mod_decl, is_use_item,
    rewrite_associated_impl_type, rewrite_associated_type, rewrite_existential_impl_type,
    rewrite_existential_type, rewrite_extern_crate, rewrite_type_alias, FnSig, StaticParts,
    StructParts,
};
use macros::{rewrite_macro, rewrite_macro_def, MacroPosition};
use rewrite::{Rewrite, RewriteContext};
use shape::{Indent, Shape};
use source_map::{LineRangeUtils, SpanUtils};
use spanned::Spanned;
use utils::{
    self, contains_skip, count_newlines, inner_attributes, mk_sp, ptr_vec_to_ref_vec,
    rewrite_ident, DEPR_SKIP_ANNOTATION,
};
use {ErrorKind, FormatReport, FormattingError};

use std::cell::RefCell;

/// Creates a string slice corresponding to the specified span.
pub struct SnippetProvider<'a> {
    /// A pointer to the content of the file we are formatting.
    big_snippet: &'a str,
    /// A position of the start of `big_snippet`, used as an offset.
    start_pos: usize,
}

impl<'a> SnippetProvider<'a> {
    pub fn span_to_snippet(&self, span: Span) -> Option<&str> {
        let start_index = span.lo().to_usize().checked_sub(self.start_pos)?;
        let end_index = span.hi().to_usize().checked_sub(self.start_pos)?;
        Some(&self.big_snippet[start_index..end_index])
    }

    pub fn new(start_pos: BytePos, big_snippet: &'a str) -> Self {
        let start_pos = start_pos.to_usize();
        SnippetProvider {
            big_snippet,
            start_pos,
        }
    }
}

pub struct FmtVisitor<'a> {
    pub parse_session: &'a ParseSess,
    pub source_map: &'a SourceMap,
    pub buffer: String,
    pub last_pos: BytePos,
    // FIXME: use an RAII util or closure for indenting
    pub block_indent: Indent,
    pub config: &'a Config,
    pub is_if_else_block: bool,
    pub snippet_provider: &'a SnippetProvider<'a>,
    pub line_number: usize,
    pub skipped_range: Vec<(usize, usize)>,
    pub macro_rewrite_failure: bool,
    pub(crate) report: FormatReport,
}

impl<'b, 'a: 'b> FmtVisitor<'a> {
    pub fn shape(&self) -> Shape {
        Shape::indented(self.block_indent, self.config)
    }

    fn visit_stmt(&mut self, stmt: &ast::Stmt) {
        debug!(
            "visit_stmt: {:?} {:?}",
            self.source_map.lookup_char_pos(stmt.span.lo()),
            self.source_map.lookup_char_pos(stmt.span.hi())
        );

        match stmt.node {
            ast::StmtKind::Item(ref item) => {
                self.visit_item(item);
                // Handle potential `;` after the item.
                self.format_missing(stmt.span.hi());
            }
            ast::StmtKind::Local(..) | ast::StmtKind::Expr(..) | ast::StmtKind::Semi(..) => {
                if contains_skip(get_attrs_from_stmt(stmt)) {
                    self.push_skipped_with_span(stmt.span());
                } else {
                    let rewrite = stmt.rewrite(&self.get_context(), self.shape());
                    self.push_rewrite(stmt.span(), rewrite)
                }
            }
            ast::StmtKind::Mac(ref mac) => {
                let (ref mac, _macro_style, ref attrs) = **mac;
                if self.visit_attrs(attrs, ast::AttrStyle::Outer) {
                    self.push_skipped_with_span(stmt.span());
                } else {
                    self.visit_mac(mac, None, MacroPosition::Statement);
                }
                self.format_missing(stmt.span.hi());
            }
        }
    }

    pub fn visit_block(
        &mut self,
        b: &ast::Block,
        inner_attrs: Option<&[ast::Attribute]>,
        has_braces: bool,
    ) {
        debug!(
            "visit_block: {:?} {:?}",
            self.source_map.lookup_char_pos(b.span.lo()),
            self.source_map.lookup_char_pos(b.span.hi())
        );

        // Check if this block has braces.
        let brace_compensation = BytePos(if has_braces { 1 } else { 0 });

        self.last_pos = self.last_pos + brace_compensation;
        self.block_indent = self.block_indent.block_indent(self.config);
        self.push_str("{");

        if let Some(first_stmt) = b.stmts.first() {
            let attr_lo = inner_attrs
                .and_then(|attrs| inner_attributes(attrs).first().map(|attr| attr.span.lo()))
                .or_else(|| {
                    // Attributes for an item in a statement position
                    // do not belong to the statement. (rust-lang/rust#34459)
                    if let ast::StmtKind::Item(ref item) = first_stmt.node {
                        item.attrs.first()
                    } else {
                        first_stmt.attrs().first()
                    }.and_then(|attr| {
                        // Some stmts can have embedded attributes.
                        // e.g. `match { #![attr] ... }`
                        let attr_lo = attr.span.lo();
                        if attr_lo < first_stmt.span.lo() {
                            Some(attr_lo)
                        } else {
                            None
                        }
                    })
                });

            let snippet = self.snippet(mk_sp(
                self.last_pos,
                attr_lo.unwrap_or_else(|| first_stmt.span.lo()),
            ));
            let len = CommentCodeSlices::new(snippet)
                .nth(0)
                .and_then(|(kind, _, s)| {
                    if kind == CodeCharKind::Normal {
                        s.rfind('\n')
                    } else {
                        None
                    }
                });
            if let Some(len) = len {
                self.last_pos = self.last_pos + BytePos::from_usize(len);
            }
        }

        // Format inner attributes if available.
        let skip_rewrite = if let Some(attrs) = inner_attrs {
            self.visit_attrs(attrs, ast::AttrStyle::Inner)
        } else {
            false
        };

        if skip_rewrite {
            self.push_rewrite(b.span, None);
            self.close_block(false);
            self.last_pos = source!(self, b.span).hi();
            return;
        }

        self.walk_block_stmts(b);

        if !b.stmts.is_empty() {
            if let Some(expr) = utils::stmt_expr(&b.stmts[b.stmts.len() - 1]) {
                if utils::semicolon_for_expr(&self.get_context(), expr) {
                    self.push_str(";");
                }
            }
        }

        let mut remove_len = BytePos(0);
        if let Some(stmt) = b.stmts.last() {
            let snippet = self.snippet(mk_sp(
                stmt.span.hi(),
                source!(self, b.span).hi() - brace_compensation,
            ));
            let len = CommentCodeSlices::new(snippet)
                .last()
                .and_then(|(kind, _, s)| {
                    if kind == CodeCharKind::Normal && s.trim().is_empty() {
                        Some(s.len())
                    } else {
                        None
                    }
                });
            if let Some(len) = len {
                remove_len = BytePos::from_usize(len);
            }
        }

        let unindent_comment = (self.is_if_else_block && !b.stmts.is_empty()) && {
            let end_pos = source!(self, b.span).hi() - brace_compensation - remove_len;
            let snippet = self.snippet(mk_sp(self.last_pos, end_pos));
            snippet.contains("//") || snippet.contains("/*")
        };
        // FIXME: we should compress any newlines here to just one
        if unindent_comment {
            self.block_indent = self.block_indent.block_unindent(self.config);
        }
        self.format_missing_with_indent(
            source!(self, b.span).hi() - brace_compensation - remove_len,
        );
        if unindent_comment {
            self.block_indent = self.block_indent.block_indent(self.config);
        }
        self.close_block(unindent_comment);
        self.last_pos = source!(self, b.span).hi();
    }

    // FIXME: this is a terrible hack to indent the comments between the last
    // item in the block and the closing brace to the block's level.
    // The closing brace itself, however, should be indented at a shallower
    // level.
    fn close_block(&mut self, unindent_comment: bool) {
        let total_len = self.buffer.len();
        let chars_too_many = if unindent_comment {
            0
        } else if self.config.hard_tabs() {
            1
        } else {
            self.config.tab_spaces()
        };
        self.buffer.truncate(total_len - chars_too_many);
        self.push_str("}");
        self.block_indent = self.block_indent.block_unindent(self.config);
    }

    // Note that this only gets called for function definitions. Required methods
    // on traits do not get handled here.
    fn visit_fn(
        &mut self,
        fk: visit::FnKind,
        generics: &ast::Generics,
        fd: &ast::FnDecl,
        s: Span,
        defaultness: ast::Defaultness,
        inner_attrs: Option<&[ast::Attribute]>,
    ) {
        let indent = self.block_indent;
        let block;
        let rewrite = match fk {
            visit::FnKind::ItemFn(ident, _, _, b) | visit::FnKind::Method(ident, _, _, b) => {
                block = b;
                self.rewrite_fn(
                    indent,
                    ident,
                    &FnSig::from_fn_kind(&fk, generics, fd, defaultness),
                    mk_sp(s.lo(), b.span.lo()),
                    b,
                    inner_attrs,
                )
            }
            visit::FnKind::Closure(_) => unreachable!(),
        };

        if let Some(fn_str) = rewrite {
            self.format_missing_with_indent(source!(self, s).lo());
            self.push_str(&fn_str);
            if let Some(c) = fn_str.chars().last() {
                if c == '}' {
                    self.last_pos = source!(self, block.span).hi();
                    return;
                }
            }
        } else {
            self.format_missing(source!(self, block.span).lo());
        }

        self.last_pos = source!(self, block.span).lo();
        self.visit_block(block, inner_attrs, true)
    }

    pub fn visit_item(&mut self, item: &ast::Item) {
        skip_out_of_file_lines_range_visitor!(self, item.span);

        // This is where we bail out if there is a skip attribute. This is only
        // complex in the module case. It is complex because the module could be
        // in a separate file and there might be attributes in both files, but
        // the AST lumps them all together.
        let filtered_attrs;
        let mut attrs = &item.attrs;
        match item.node {
            // For use items, skip rewriting attributes. Just check for a skip attribute.
            ast::ItemKind::Use(..) => {
                if contains_skip(attrs) {
                    self.push_skipped_with_span(item.span());
                    return;
                }
            }
            // Module is inline, in this case we treat it like any other item.
            _ if !is_mod_decl(item) => {
                if self.visit_attrs(&item.attrs, ast::AttrStyle::Outer) {
                    self.push_skipped_with_span(item.span());
                    return;
                }
            }
            // Module is not inline, but should be skipped.
            ast::ItemKind::Mod(..) if contains_skip(&item.attrs) => {
                return;
            }
            // Module is not inline and should not be skipped. We want
            // to process only the attributes in the current file.
            ast::ItemKind::Mod(..) => {
                filtered_attrs = filter_inline_attrs(&item.attrs, item.span());
                // Assert because if we should skip it should be caught by
                // the above case.
                assert!(!self.visit_attrs(&filtered_attrs, ast::AttrStyle::Outer));
                attrs = &filtered_attrs;
            }
            _ => {
                if self.visit_attrs(&item.attrs, ast::AttrStyle::Outer) {
                    self.push_skipped_with_span(item.span());
                    return;
                }
            }
        }

        match item.node {
            ast::ItemKind::Use(ref tree) => self.format_import(item, tree),
            ast::ItemKind::Impl(..) => {
                let snippet = self.snippet(item.span);
                let where_span_end = snippet
                    .find_uncommented("{")
                    .map(|x| BytePos(x as u32) + source!(self, item.span).lo());
                let rw = format_impl(&self.get_context(), item, self.block_indent, where_span_end);
                self.push_rewrite(item.span, rw);
            }
            ast::ItemKind::Trait(..) => {
                let rw = format_trait(&self.get_context(), item, self.block_indent);
                self.push_rewrite(item.span, rw);
            }
            ast::ItemKind::TraitAlias(ref generics, ref generic_bounds) => {
                let shape = Shape::indented(self.block_indent, self.config);
                let rw = format_trait_alias(
                    &self.get_context(),
                    item.ident,
                    generics,
                    generic_bounds,
                    shape,
                );
                self.push_rewrite(item.span, rw);
            }
            ast::ItemKind::ExternCrate(_) => {
                let rw = rewrite_extern_crate(&self.get_context(), item);
                self.push_rewrite(item.span, rw);
            }
            ast::ItemKind::Struct(..) | ast::ItemKind::Union(..) => {
                self.visit_struct(&StructParts::from_item(item));
            }
            ast::ItemKind::Enum(ref def, ref generics) => {
                self.format_missing_with_indent(source!(self, item.span).lo());
                self.visit_enum(item.ident, &item.vis, def, generics, item.span);
                self.last_pos = source!(self, item.span).hi();
            }
            ast::ItemKind::Mod(ref module) => {
                let is_inline = !is_mod_decl(item);
                self.format_missing_with_indent(source!(self, item.span).lo());
                self.format_mod(module, &item.vis, item.span, item.ident, attrs, is_inline);
            }
            ast::ItemKind::Mac(ref mac) => {
                self.visit_mac(mac, Some(item.ident), MacroPosition::Item);
            }
            ast::ItemKind::ForeignMod(ref foreign_mod) => {
                self.format_missing_with_indent(source!(self, item.span).lo());
                self.format_foreign_mod(foreign_mod, item.span);
            }
            ast::ItemKind::Static(..) | ast::ItemKind::Const(..) => {
                self.visit_static(&StaticParts::from_item(item));
            }
            ast::ItemKind::Fn(ref decl, fn_header, ref generics, ref body) => {
                let inner_attrs = inner_attributes(&item.attrs);
                self.visit_fn(
                    visit::FnKind::ItemFn(item.ident, fn_header, &item.vis, body),
                    generics,
                    decl,
                    item.span,
                    ast::Defaultness::Final,
                    Some(&inner_attrs),
                )
            }
            ast::ItemKind::Ty(ref ty, ref generics) => {
                let rewrite = rewrite_type_alias(
                    &self.get_context(),
                    self.block_indent,
                    item.ident,
                    ty,
                    generics,
                    &item.vis,
                );
                self.push_rewrite(item.span, rewrite);
            }
            ast::ItemKind::Existential(ref generic_bounds, ref generics) => {
                let rewrite = rewrite_existential_type(
                    &self.get_context(),
                    self.block_indent,
                    item.ident,
                    generic_bounds,
                    generics,
                    &item.vis,
                );
                self.push_rewrite(item.span, rewrite);
            }
            ast::ItemKind::GlobalAsm(..) => {
                let snippet = Some(self.snippet(item.span).to_owned());
                self.push_rewrite(item.span, snippet);
            }
            ast::ItemKind::MacroDef(ref def) => {
                let rewrite = rewrite_macro_def(
                    &self.get_context(),
                    self.shape(),
                    self.block_indent,
                    def,
                    item.ident,
                    &item.vis,
                    item.span,
                );
                self.push_rewrite(item.span, rewrite);
            }
        }
    }

    pub fn visit_trait_item(&mut self, ti: &ast::TraitItem) {
        skip_out_of_file_lines_range_visitor!(self, ti.span);

        if self.visit_attrs(&ti.attrs, ast::AttrStyle::Outer) {
            self.push_skipped_with_span(ti.span());
            return;
        }

        match ti.node {
            ast::TraitItemKind::Const(..) => self.visit_static(&StaticParts::from_trait_item(ti)),
            ast::TraitItemKind::Method(ref sig, None) => {
                let indent = self.block_indent;
                let rewrite =
                    self.rewrite_required_fn(indent, ti.ident, sig, &ti.generics, ti.span);
                self.push_rewrite(ti.span, rewrite);
            }
            ast::TraitItemKind::Method(ref sig, Some(ref body)) => {
                let inner_attrs = inner_attributes(&ti.attrs);
                self.visit_fn(
                    visit::FnKind::Method(ti.ident, sig, None, body),
                    &ti.generics,
                    &sig.decl,
                    ti.span,
                    ast::Defaultness::Final,
                    Some(&inner_attrs),
                );
            }
            ast::TraitItemKind::Type(ref generic_bounds, ref type_default) => {
                let rewrite = rewrite_associated_type(
                    ti.ident,
                    type_default.as_ref(),
                    Some(generic_bounds),
                    &self.get_context(),
                    self.block_indent,
                );
                self.push_rewrite(ti.span, rewrite);
            }
            ast::TraitItemKind::Macro(ref mac) => {
                self.visit_mac(mac, Some(ti.ident), MacroPosition::Item);
            }
        }
    }

    pub fn visit_impl_item(&mut self, ii: &ast::ImplItem) {
        skip_out_of_file_lines_range_visitor!(self, ii.span);

        if self.visit_attrs(&ii.attrs, ast::AttrStyle::Outer) {
            self.push_skipped_with_span(ii.span());
            return;
        }

        match ii.node {
            ast::ImplItemKind::Method(ref sig, ref body) => {
                let inner_attrs = inner_attributes(&ii.attrs);
                self.visit_fn(
                    visit::FnKind::Method(ii.ident, sig, Some(&ii.vis), body),
                    &ii.generics,
                    &sig.decl,
                    ii.span,
                    ii.defaultness,
                    Some(&inner_attrs),
                );
            }
            ast::ImplItemKind::Const(..) => self.visit_static(&StaticParts::from_impl_item(ii)),
            ast::ImplItemKind::Type(ref ty) => {
                let rewrite = rewrite_associated_impl_type(
                    ii.ident,
                    ii.defaultness,
                    Some(ty),
                    &self.get_context(),
                    self.block_indent,
                );
                self.push_rewrite(ii.span, rewrite);
            }
            ast::ImplItemKind::Existential(ref generic_bounds) => {
                let rewrite = rewrite_existential_impl_type(
                    &self.get_context(),
                    ii.ident,
                    generic_bounds,
                    self.block_indent,
                );
                self.push_rewrite(ii.span, rewrite);
            }
            ast::ImplItemKind::Macro(ref mac) => {
                self.visit_mac(mac, Some(ii.ident), MacroPosition::Item);
            }
        }
    }

    fn visit_mac(&mut self, mac: &ast::Mac, ident: Option<ast::Ident>, pos: MacroPosition) {
        skip_out_of_file_lines_range_visitor!(self, mac.span);

        // 1 = ;
        let shape = self.shape().sub_width(1).unwrap();
        let rewrite = self.with_context(|ctx| rewrite_macro(mac, ident, ctx, shape, pos));
        self.push_rewrite(mac.span, rewrite);
    }

    pub fn push_str(&mut self, s: &str) {
        self.line_number += count_newlines(s);
        self.buffer.push_str(s);
    }

    #[cfg_attr(feature = "cargo-clippy", allow(needless_pass_by_value))]
    fn push_rewrite_inner(&mut self, span: Span, rewrite: Option<String>) {
        if let Some(ref s) = rewrite {
            self.push_str(s);
        } else {
            let snippet = self.snippet(span);
            self.push_str(snippet);
        }
        self.last_pos = source!(self, span).hi();
    }

    pub fn push_rewrite(&mut self, span: Span, rewrite: Option<String>) {
        self.format_missing_with_indent(source!(self, span).lo());
        self.push_rewrite_inner(span, rewrite);
    }

    pub fn push_skipped_with_span(&mut self, span: Span) {
        self.format_missing_with_indent(source!(self, span).lo());
        let lo = self.line_number + 1;
        self.push_rewrite_inner(span, None);
        let hi = self.line_number + 1;
        self.skipped_range.push((lo, hi));
    }

    pub fn from_context(ctx: &'a RewriteContext) -> FmtVisitor<'a> {
        FmtVisitor::from_source_map(
            ctx.parse_session,
            ctx.config,
            ctx.snippet_provider,
            ctx.report.clone(),
        )
    }

    pub(crate) fn from_source_map(
        parse_session: &'a ParseSess,
        config: &'a Config,
        snippet_provider: &'a SnippetProvider,
        report: FormatReport,
    ) -> FmtVisitor<'a> {
        FmtVisitor {
            parse_session,
            source_map: parse_session.source_map(),
            buffer: String::with_capacity(snippet_provider.big_snippet.len() * 2),
            last_pos: BytePos(0),
            block_indent: Indent::empty(),
            config,
            is_if_else_block: false,
            snippet_provider,
            line_number: 0,
            skipped_range: vec![],
            macro_rewrite_failure: false,
            report,
        }
    }

    pub fn opt_snippet(&'b self, span: Span) -> Option<&'a str> {
        self.snippet_provider.span_to_snippet(span)
    }

    pub fn snippet(&'b self, span: Span) -> &'a str {
        self.opt_snippet(span).unwrap()
    }

    // Returns true if we should skip the following item.
    pub fn visit_attrs(&mut self, attrs: &[ast::Attribute], style: ast::AttrStyle) -> bool {
        for attr in attrs {
            if attr.name() == DEPR_SKIP_ANNOTATION {
                let file_name = self.source_map.span_to_filename(attr.span).into();
                self.report.append(
                    file_name,
                    vec![FormattingError::from_span(
                        &attr.span,
                        &self.source_map,
                        ErrorKind::DeprecatedAttr,
                    )],
                );
            } else if attr.path.segments[0].ident.to_string() == "rustfmt" {
                if attr.path.segments.len() == 1
                    || attr.path.segments[1].ident.to_string() != "skip"
                {
                    let file_name = self.source_map.span_to_filename(attr.span).into();
                    self.report.append(
                        file_name,
                        vec![FormattingError::from_span(
                            &attr.span,
                            &self.source_map,
                            ErrorKind::BadAttr,
                        )],
                    );
                }
            }
        }
        if contains_skip(attrs) {
            return true;
        }

        let attrs: Vec<_> = attrs.iter().filter(|a| a.style == style).cloned().collect();
        if attrs.is_empty() {
            return false;
        }

        let rewrite = attrs.rewrite(&self.get_context(), self.shape());
        let span = mk_sp(attrs[0].span.lo(), attrs[attrs.len() - 1].span.hi());
        self.push_rewrite(span, rewrite);

        false
    }

    fn walk_mod_items(&mut self, m: &ast::Mod) {
        self.visit_items_with_reordering(&ptr_vec_to_ref_vec(&m.items));
    }

    fn walk_stmts(&mut self, stmts: &[ast::Stmt]) {
        fn to_stmt_item(stmt: &ast::Stmt) -> Option<&ast::Item> {
            match stmt.node {
                ast::StmtKind::Item(ref item) => Some(&**item),
                _ => None,
            }
        }

        if stmts.is_empty() {
            return;
        }

        // Extract leading `use ...;`.
        let items: Vec<_> = stmts
            .iter()
            .take_while(|stmt| to_stmt_item(stmt).map_or(false, is_use_item))
            .filter_map(|stmt| to_stmt_item(stmt))
            .collect();

        if items.is_empty() {
            self.visit_stmt(&stmts[0]);
            self.walk_stmts(&stmts[1..]);
        } else {
            self.visit_items_with_reordering(&items);
            self.walk_stmts(&stmts[items.len()..]);
        }
    }

    fn walk_block_stmts(&mut self, b: &ast::Block) {
        self.walk_stmts(&b.stmts)
    }

    fn format_mod(
        &mut self,
        m: &ast::Mod,
        vis: &ast::Visibility,
        s: Span,
        ident: ast::Ident,
        attrs: &[ast::Attribute],
        is_internal: bool,
    ) {
        let vis_str = utils::format_visibility(&self.get_context(), vis);
        self.push_str(&*vis_str);
        self.push_str("mod ");
        // Calling `to_owned()` to work around borrow checker.
        let ident_str = rewrite_ident(&self.get_context(), ident).to_owned();
        self.push_str(&ident_str);

        if is_internal {
            match self.config.brace_style() {
                BraceStyle::AlwaysNextLine => {
                    let indent_str = self.block_indent.to_string_with_newline(self.config);
                    self.push_str(&indent_str);
                    self.push_str("{");
                }
                _ => self.push_str(" {"),
            }
            // Hackery to account for the closing }.
            let mod_lo = self.snippet_provider.span_after(source!(self, s), "{");
            let body_snippet =
                self.snippet(mk_sp(mod_lo, source!(self, m.inner).hi() - BytePos(1)));
            let body_snippet = body_snippet.trim();
            if body_snippet.is_empty() {
                self.push_str("}");
            } else {
                self.last_pos = mod_lo;
                self.block_indent = self.block_indent.block_indent(self.config);
                self.visit_attrs(attrs, ast::AttrStyle::Inner);
                self.walk_mod_items(m);
                self.format_missing_with_indent(source!(self, m.inner).hi() - BytePos(1));
                self.close_block(false);
            }
            self.last_pos = source!(self, m.inner).hi();
        } else {
            self.push_str(";");
            self.last_pos = source!(self, s).hi();
        }
    }

    pub fn format_separate_mod(&mut self, m: &ast::Mod, source_file: &source_map::SourceFile) {
        self.block_indent = Indent::empty();
        self.walk_mod_items(m);
        self.format_missing_with_indent(source_file.end_pos);
    }

    pub fn skip_empty_lines(&mut self, end_pos: BytePos) {
        while let Some(pos) = self
            .snippet_provider
            .opt_span_after(mk_sp(self.last_pos, end_pos), "\n")
        {
            if let Some(snippet) = self.opt_snippet(mk_sp(self.last_pos, pos)) {
                if snippet.trim().is_empty() {
                    self.last_pos = pos;
                } else {
                    return;
                }
            }
        }
    }

    pub fn with_context<F>(&mut self, f: F) -> Option<String>
    where
        F: Fn(&RewriteContext) -> Option<String>,
    {
        let result;
        let macro_rewrite_failure = {
            let context = self.get_context();
            result = f(&context);
            unsafe { *context.macro_rewrite_failure.as_ptr() }
        };
        self.macro_rewrite_failure |= macro_rewrite_failure;
        result
    }

    pub fn get_context(&self) -> RewriteContext {
        RewriteContext {
            parse_session: self.parse_session,
            source_map: self.source_map,
            config: self.config,
            inside_macro: RefCell::new(false),
            use_block: RefCell::new(false),
            is_if_else_block: RefCell::new(false),
            force_one_line_chain: RefCell::new(false),
            snippet_provider: self.snippet_provider,
            macro_rewrite_failure: RefCell::new(false),
            report: self.report.clone(),
        }
    }
}
