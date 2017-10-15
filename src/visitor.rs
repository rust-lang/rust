// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp;

use strings::string_buffer::StringBuffer;
use syntax::{ast, visit};
use syntax::attr::HasAttrs;
use syntax::codemap::{self, BytePos, CodeMap, Pos, Span};
use syntax::parse::ParseSess;

use expr::rewrite_literal;
use spanned::Spanned;
use codemap::{LineRangeUtils, SpanUtils};
use comment::{contains_comment, recover_missing_comment_in_span, remove_trailing_white_spaces,
              CodeCharKind, CommentCodeSlices, FindUncommented};
use comment::rewrite_comment;
use config::{BraceStyle, Config};
use items::{format_impl, format_struct, format_struct_struct, format_trait,
            rewrite_associated_impl_type, rewrite_associated_type, rewrite_static,
            rewrite_type_alias, FnSig};
use lists::{itemize_list, write_list, DefinitiveListTactic, ListFormatting, SeparatorPlace,
            SeparatorTactic};
use macros::{rewrite_macro, MacroPosition};
use regex::Regex;
use rewrite::{Rewrite, RewriteContext};
use shape::{Indent, Shape};
use utils::{self, contains_skip, inner_attributes, mk_sp, ptr_vec_to_ref_vec};

fn is_use_item(item: &ast::Item) -> bool {
    match item.node {
        ast::ItemKind::Use(_) => true,
        _ => false,
    }
}

fn is_extern_crate(item: &ast::Item) -> bool {
    match item.node {
        ast::ItemKind::ExternCrate(..) => true,
        _ => false,
    }
}

pub struct FmtVisitor<'a> {
    pub parse_session: &'a ParseSess,
    pub codemap: &'a CodeMap,
    pub buffer: StringBuffer,
    pub last_pos: BytePos,
    // FIXME: use an RAII util or closure for indenting
    pub block_indent: Indent,
    pub config: &'a Config,
    pub is_if_else_block: bool,
}

impl<'a> FmtVisitor<'a> {
    pub fn shape(&self) -> Shape {
        Shape::indented(self.block_indent, self.config)
    }

    fn visit_stmt(&mut self, stmt: &ast::Stmt) {
        debug!(
            "visit_stmt: {:?} {:?}",
            self.codemap.lookup_char_pos(stmt.span.lo()),
            self.codemap.lookup_char_pos(stmt.span.hi())
        );

        match stmt.node {
            ast::StmtKind::Item(ref item) => {
                self.visit_item(item);
            }
            ast::StmtKind::Local(..) | ast::StmtKind::Expr(..) | ast::StmtKind::Semi(..) => {
                let rewrite = stmt.rewrite(&self.get_context(), self.shape());
                self.push_rewrite(stmt.span(), rewrite)
            }
            ast::StmtKind::Mac(ref mac) => {
                let (ref mac, _macro_style, ref attrs) = **mac;
                if self.visit_attrs(attrs, ast::AttrStyle::Outer) {
                    self.push_rewrite(stmt.span(), None);
                } else {
                    self.visit_mac(mac, None, MacroPosition::Statement);
                }
                self.format_missing(stmt.span.hi());
            }
        }
    }

    pub fn visit_block(&mut self, b: &ast::Block, inner_attrs: Option<&[ast::Attribute]>) {
        debug!(
            "visit_block: {:?} {:?}",
            self.codemap.lookup_char_pos(b.span.lo()),
            self.codemap.lookup_char_pos(b.span.hi())
        );

        // Check if this block has braces.
        let snippet = self.snippet(b.span);
        let has_braces = snippet.starts_with('{') || snippet.starts_with("unsafe");
        let brace_compensation = if has_braces { BytePos(1) } else { BytePos(0) };

        self.last_pos = self.last_pos + brace_compensation;
        self.block_indent = self.block_indent.block_indent(self.config);
        self.buffer.push_str("{");

        if self.config.remove_blank_lines_at_start_or_end_of_block() {
            if let Some(first_stmt) = b.stmts.first() {
                let attr_lo = inner_attrs
                    .and_then(|attrs| {
                        inner_attributes(attrs).first().map(|attr| attr.span.lo())
                    })
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
                    attr_lo.unwrap_or(first_stmt.span.lo()),
                ));
                let len = CommentCodeSlices::new(&snippet).nth(0).and_then(
                    |(kind, _, s)| if kind == CodeCharKind::Normal {
                        s.rfind('\n')
                    } else {
                        None
                    },
                );
                if let Some(len) = len {
                    self.last_pos = self.last_pos + BytePos::from_usize(len);
                }
            }
        }

        // Format inner attributes if available.
        if let Some(attrs) = inner_attrs {
            self.visit_attrs(attrs, ast::AttrStyle::Inner);
        }

        self.walk_block_stmts(b);

        if !b.stmts.is_empty() {
            if let Some(expr) = utils::stmt_expr(&b.stmts[b.stmts.len() - 1]) {
                if utils::semicolon_for_expr(&self.get_context(), expr) {
                    self.buffer.push_str(";");
                }
            }
        }

        let mut remove_len = BytePos(0);
        if self.config.remove_blank_lines_at_start_or_end_of_block() {
            if let Some(stmt) = b.stmts.last() {
                let snippet = self.snippet(mk_sp(
                    stmt.span.hi(),
                    source!(self, b.span).hi() - brace_compensation,
                ));
                let len = CommentCodeSlices::new(&snippet)
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
        }

        let mut unindent_comment = self.is_if_else_block && !b.stmts.is_empty();
        if unindent_comment {
            let end_pos = source!(self, b.span).hi() - brace_compensation - remove_len;
            let snippet = self.snippet(mk_sp(self.last_pos, end_pos));
            unindent_comment = snippet.contains("//") || snippet.contains("/*");
        }
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
        let total_len = self.buffer.len;
        let chars_too_many = if unindent_comment {
            0
        } else if self.config.hard_tabs() {
            1
        } else {
            self.config.tab_spaces()
        };
        self.buffer.truncate(total_len - chars_too_many);
        self.buffer.push_str("}");
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
        _: ast::NodeId,
        defaultness: ast::Defaultness,
        inner_attrs: Option<&[ast::Attribute]>,
    ) {
        let indent = self.block_indent;
        let block;
        let rewrite = match fk {
            visit::FnKind::ItemFn(ident, _, _, _, _, b) => {
                block = b;
                self.rewrite_fn(
                    indent,
                    ident,
                    &FnSig::from_fn_kind(&fk, generics, fd, defaultness),
                    mk_sp(s.lo(), b.span.lo()),
                    b,
                )
            }
            visit::FnKind::Method(ident, _, _, b) => {
                block = b;
                self.rewrite_fn(
                    indent,
                    ident,
                    &FnSig::from_fn_kind(&fk, generics, fd, defaultness),
                    mk_sp(s.lo(), b.span.lo()),
                    b,
                )
            }
            visit::FnKind::Closure(_) => unreachable!(),
        };

        if let Some(fn_str) = rewrite {
            self.format_missing_with_indent(source!(self, s).lo());
            self.buffer.push_str(&fn_str);
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
        self.visit_block(block, inner_attrs)
    }

    pub fn visit_item(&mut self, item: &ast::Item) {
        skip_out_of_file_lines_range_visitor!(self, item.span);

        // This is where we bail out if there is a skip attribute. This is only
        // complex in the module case. It is complex because the module could be
        // in a separate file and there might be attributes in both files, but
        // the AST lumps them all together.
        let filterd_attrs;
        let mut attrs = &item.attrs;
        match item.node {
            ast::ItemKind::Mod(ref m) => {
                let outer_file = self.codemap.lookup_char_pos(item.span.lo()).file;
                let inner_file = self.codemap.lookup_char_pos(m.inner.lo()).file;
                if outer_file.name == inner_file.name {
                    // Module is inline, in this case we treat modules like any
                    // other item.
                    if self.visit_attrs(&item.attrs, ast::AttrStyle::Outer) {
                        self.push_rewrite(item.span, None);
                        return;
                    }
                } else if contains_skip(&item.attrs) {
                    // Module is not inline, but should be skipped.
                    return;
                } else {
                    // Module is not inline and should not be skipped. We want
                    // to process only the attributes in the current file.
                    filterd_attrs = item.attrs
                        .iter()
                        .filter_map(|a| {
                            let attr_file = self.codemap.lookup_char_pos(a.span.lo()).file;
                            if attr_file.name == outer_file.name {
                                Some(a.clone())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();
                    // Assert because if we should skip it should be caught by
                    // the above case.
                    assert!(!self.visit_attrs(&filterd_attrs, ast::AttrStyle::Outer));
                    attrs = &filterd_attrs;
                }
            }
            _ => if self.visit_attrs(&item.attrs, ast::AttrStyle::Outer) {
                self.push_rewrite(item.span, None);
                return;
            },
        }

        match item.node {
            ast::ItemKind::Use(ref vp) => self.format_import(item, vp),
            ast::ItemKind::Impl(..) => {
                let snippet = self.snippet(item.span);
                let where_span_end = snippet
                    .find_uncommented("{")
                    .map(|x| (BytePos(x as u32)) + source!(self, item.span).lo());
                let rw = format_impl(&self.get_context(), item, self.block_indent, where_span_end);
                self.push_rewrite(item.span, rw);
            }
            ast::ItemKind::Trait(..) => {
                let rw = format_trait(&self.get_context(), item, self.block_indent);
                self.push_rewrite(item.span, rw);
            }
            ast::ItemKind::ExternCrate(_) => {
                let rw = rewrite_extern_crate(&self.get_context(), item);
                self.push_rewrite(item.span, rw);
            }
            ast::ItemKind::Struct(ref def, ref generics) => {
                let rewrite = format_struct(
                    &self.get_context(),
                    "struct ",
                    item.ident,
                    &item.vis,
                    def,
                    Some(generics),
                    item.span,
                    self.block_indent,
                    None,
                ).map(|s| match *def {
                    ast::VariantData::Tuple(..) => s + ";",
                    _ => s,
                });
                self.push_rewrite(item.span, rewrite);
            }
            ast::ItemKind::Enum(ref def, ref generics) => {
                self.format_missing_with_indent(source!(self, item.span).lo());
                self.visit_enum(item.ident, &item.vis, def, generics, item.span);
                self.last_pos = source!(self, item.span).hi();
            }
            ast::ItemKind::Mod(ref module) => {
                self.format_missing_with_indent(source!(self, item.span).lo());
                self.format_mod(module, &item.vis, item.span, item.ident, attrs);
            }
            ast::ItemKind::Mac(ref mac) => {
                self.visit_mac(mac, Some(item.ident), MacroPosition::Item);
            }
            ast::ItemKind::ForeignMod(ref foreign_mod) => {
                self.format_missing_with_indent(source!(self, item.span).lo());
                self.format_foreign_mod(foreign_mod, item.span);
            }
            ast::ItemKind::Static(ref ty, mutability, ref expr) => {
                let rewrite = rewrite_static(
                    "static",
                    &item.vis,
                    item.ident,
                    ty,
                    mutability,
                    Some(expr),
                    self.block_indent,
                    item.span,
                    &self.get_context(),
                );
                self.push_rewrite(item.span, rewrite);
            }
            ast::ItemKind::Const(ref ty, ref expr) => {
                let rewrite = rewrite_static(
                    "const",
                    &item.vis,
                    item.ident,
                    ty,
                    ast::Mutability::Immutable,
                    Some(expr),
                    self.block_indent,
                    item.span,
                    &self.get_context(),
                );
                self.push_rewrite(item.span, rewrite);
            }
            ast::ItemKind::DefaultImpl(..) => {
                // FIXME(#78): format impl definitions.
            }
            ast::ItemKind::Fn(ref decl, unsafety, constness, abi, ref generics, ref body) => {
                self.visit_fn(
                    visit::FnKind::ItemFn(item.ident, unsafety, constness, abi, &item.vis, body),
                    generics,
                    decl,
                    item.span,
                    item.id,
                    ast::Defaultness::Final,
                    Some(&item.attrs),
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
                    item.span,
                );
                self.push_rewrite(item.span, rewrite);
            }
            ast::ItemKind::Union(ref def, ref generics) => {
                let rewrite = format_struct_struct(
                    &self.get_context(),
                    "union ",
                    item.ident,
                    &item.vis,
                    def.fields(),
                    Some(generics),
                    item.span,
                    self.block_indent,
                    None,
                );
                self.push_rewrite(item.span, rewrite);
            }
            ast::ItemKind::GlobalAsm(..) => {
                let snippet = Some(self.snippet(item.span));
                self.push_rewrite(item.span, snippet);
            }
            ast::ItemKind::MacroDef(..) => {
                // FIXME(#1539): macros 2.0
                let mac_snippet = Some(remove_trailing_white_spaces(&self.snippet(item.span)));
                self.push_rewrite(item.span, mac_snippet);
            }
        }
    }

    pub fn visit_trait_item(&mut self, ti: &ast::TraitItem) {
        skip_out_of_file_lines_range_visitor!(self, ti.span);

        if self.visit_attrs(&ti.attrs, ast::AttrStyle::Outer) {
            self.push_rewrite(ti.span, None);
            return;
        }

        match ti.node {
            ast::TraitItemKind::Const(ref ty, ref expr_opt) => {
                let rewrite = rewrite_static(
                    "const",
                    &ast::Visibility::Inherited,
                    ti.ident,
                    ty,
                    ast::Mutability::Immutable,
                    expr_opt.as_ref(),
                    self.block_indent,
                    ti.span,
                    &self.get_context(),
                );
                self.push_rewrite(ti.span, rewrite);
            }
            ast::TraitItemKind::Method(ref sig, None) => {
                let indent = self.block_indent;
                let rewrite =
                    self.rewrite_required_fn(indent, ti.ident, sig, &ti.generics, ti.span);
                self.push_rewrite(ti.span, rewrite);
            }
            ast::TraitItemKind::Method(ref sig, Some(ref body)) => {
                self.visit_fn(
                    visit::FnKind::Method(ti.ident, sig, None, body),
                    &ti.generics,
                    &sig.decl,
                    ti.span,
                    ti.id,
                    ast::Defaultness::Final,
                    Some(&ti.attrs),
                );
            }
            ast::TraitItemKind::Type(ref type_param_bounds, ref type_default) => {
                let rewrite = rewrite_associated_type(
                    ti.ident,
                    type_default.as_ref(),
                    Some(type_param_bounds),
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
            self.push_rewrite(ii.span, None);
            return;
        }

        match ii.node {
            ast::ImplItemKind::Method(ref sig, ref body) => {
                self.visit_fn(
                    visit::FnKind::Method(ii.ident, sig, Some(&ii.vis), body),
                    &ii.generics,
                    &sig.decl,
                    ii.span,
                    ii.id,
                    ii.defaultness,
                    Some(&ii.attrs),
                );
            }
            ast::ImplItemKind::Const(ref ty, ref expr) => {
                let rewrite = rewrite_static(
                    "const",
                    &ii.vis,
                    ii.ident,
                    ty,
                    ast::Mutability::Immutable,
                    Some(expr),
                    self.block_indent,
                    ii.span,
                    &self.get_context(),
                );
                self.push_rewrite(ii.span, rewrite);
            }
            ast::ImplItemKind::Type(ref ty) => {
                let rewrite = rewrite_associated_impl_type(
                    ii.ident,
                    ii.defaultness,
                    Some(ty),
                    None,
                    &self.get_context(),
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
        let rewrite = rewrite_macro(mac, ident, &self.get_context(), shape, pos);
        self.push_rewrite(mac.span, rewrite);
    }

    pub fn push_rewrite(&mut self, span: Span, rewrite: Option<String>) {
        self.format_missing_with_indent(source!(self, span).lo());
        let result = rewrite.unwrap_or_else(|| self.snippet(span));
        self.buffer.push_str(&result);
        self.last_pos = source!(self, span).hi();
    }

    pub fn from_codemap(parse_session: &'a ParseSess, config: &'a Config) -> FmtVisitor<'a> {
        FmtVisitor {
            parse_session: parse_session,
            codemap: parse_session.codemap(),
            buffer: StringBuffer::new(),
            last_pos: BytePos(0),
            block_indent: Indent::empty(),
            config: config,
            is_if_else_block: false,
        }
    }

    pub fn snippet(&self, span: Span) -> String {
        match self.codemap.span_to_snippet(span) {
            Ok(s) => s,
            Err(_) => {
                println!(
                    "Couldn't make snippet for span {:?}->{:?}",
                    self.codemap.lookup_char_pos(span.lo()),
                    self.codemap.lookup_char_pos(span.hi())
                );
                "".to_owned()
            }
        }
    }

    // Returns true if we should skip the following item.
    pub fn visit_attrs(&mut self, attrs: &[ast::Attribute], style: ast::AttrStyle) -> bool {
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

    fn reorder_items<F>(&mut self, items_left: &[&ast::Item], is_item: &F, in_group: bool) -> usize
    where
        F: Fn(&ast::Item) -> bool,
    {
        let mut last = self.codemap.lookup_line_range(items_left[0].span());
        let item_length = items_left
            .iter()
            .take_while(|ppi| {
                is_item(&***ppi) && (!in_group || {
                    let current = self.codemap.lookup_line_range(ppi.span());
                    let in_same_group = current.lo < last.hi + 2;
                    last = current;
                    in_same_group
                })
            })
            .count();
        let items = &items_left[..item_length];

        let at_least_one_in_file_lines = items
            .iter()
            .any(|item| !out_of_file_lines_range!(self, item.span));

        if at_least_one_in_file_lines {
            self.format_imports(items);
        } else {
            for item in items {
                self.push_rewrite(item.span, None);
            }
        }

        item_length
    }

    fn walk_items(&mut self, mut items_left: &[&ast::Item]) {
        while !items_left.is_empty() {
            // If the next item is a `use` declaration, then extract it and any subsequent `use`s
            // to be potentially reordered within `format_imports`. Otherwise, just format the
            // next item for output.
            if self.config.reorder_imports() && is_use_item(&*items_left[0]) {
                let used_items_len = self.reorder_items(
                    items_left,
                    &is_use_item,
                    self.config.reorder_imports_in_group(),
                );
                let (_, rest) = items_left.split_at(used_items_len);
                items_left = rest;
            } else if self.config.reorder_extern_crates() && is_extern_crate(&*items_left[0]) {
                let used_items_len = self.reorder_items(
                    items_left,
                    &is_extern_crate,
                    self.config.reorder_extern_crates_in_group(),
                );
                let (_, rest) = items_left.split_at(used_items_len);
                items_left = rest;
            } else {
                // `unwrap()` is safe here because we know `items_left`
                // has elements from the loop condition
                let (item, rest) = items_left.split_first().unwrap();
                self.visit_item(item);
                items_left = rest;
            }
        }
    }

    fn walk_mod_items(&mut self, m: &ast::Mod) {
        self.walk_items(&ptr_vec_to_ref_vec(&m.items));
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
            .take_while(|stmt| to_stmt_item(stmt).is_some())
            .filter_map(|stmt| to_stmt_item(stmt))
            .take_while(|item| is_use_item(item))
            .collect();

        if items.is_empty() {
            self.visit_stmt(&stmts[0]);
            self.walk_stmts(&stmts[1..]);
        } else {
            self.walk_items(&items);
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
    ) {
        // Decide whether this is an inline mod or an external mod.
        let local_file_name = self.codemap.span_to_filename(s);
        let inner_span = source!(self, m.inner);
        let is_internal = !(inner_span.lo().0 == 0 && inner_span.hi().0 == 0)
            && local_file_name == self.codemap.span_to_filename(inner_span);

        self.buffer.push_str(&*utils::format_visibility(vis));
        self.buffer.push_str("mod ");
        self.buffer.push_str(&ident.to_string());

        if is_internal {
            match self.config.item_brace_style() {
                BraceStyle::AlwaysNextLine => self.buffer
                    .push_str(&format!("\n{}{{", self.block_indent.to_string(self.config))),
                _ => self.buffer.push_str(" {"),
            }
            // Hackery to account for the closing }.
            let mod_lo = self.codemap.span_after(source!(self, s), "{");
            let body_snippet =
                self.snippet(mk_sp(mod_lo, source!(self, m.inner).hi() - BytePos(1)));
            let body_snippet = body_snippet.trim();
            if body_snippet.is_empty() {
                self.buffer.push_str("}");
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
            self.buffer.push_str(";");
            self.last_pos = source!(self, s).hi();
        }
    }

    pub fn format_separate_mod(&mut self, m: &ast::Mod, filemap: &codemap::FileMap) {
        self.block_indent = Indent::empty();
        self.walk_mod_items(m);
        self.format_missing_with_indent(filemap.end_pos);
    }

    pub fn get_context(&self) -> RewriteContext {
        RewriteContext {
            parse_session: self.parse_session,
            codemap: self.codemap,
            config: self.config,
            inside_macro: false,
            use_block: false,
            is_if_else_block: false,
            force_one_line_chain: false,
        }
    }
}

impl Rewrite for ast::NestedMetaItem {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        match self.node {
            ast::NestedMetaItemKind::MetaItem(ref meta_item) => meta_item.rewrite(context, shape),
            ast::NestedMetaItemKind::Literal(ref l) => rewrite_literal(context, l, shape),
        }
    }
}

impl Rewrite for ast::MetaItem {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        Some(match self.node {
            ast::MetaItemKind::Word => String::from(&*self.name.as_str()),
            ast::MetaItemKind::List(ref list) => {
                let name = self.name.as_str();
                // 1 = `(`, 2 = `]` and `)`
                let item_shape = shape
                    .visual_indent(0)
                    .shrink_left(name.len() + 1)
                    .and_then(|s| s.sub_width(2))?;
                let items = itemize_list(
                    context.codemap,
                    list.iter(),
                    ")",
                    |nested_meta_item| nested_meta_item.span.lo(),
                    |nested_meta_item| nested_meta_item.span.hi(),
                    |nested_meta_item| nested_meta_item.rewrite(context, item_shape),
                    self.span.lo(),
                    self.span.hi(),
                    false,
                );
                let item_vec = items.collect::<Vec<_>>();
                let fmt = ListFormatting {
                    tactic: DefinitiveListTactic::Mixed,
                    separator: ",",
                    trailing_separator: SeparatorTactic::Never,
                    separator_place: SeparatorPlace::Back,
                    shape: item_shape,
                    ends_with_newline: false,
                    preserve_newline: false,
                    config: context.config,
                };
                format!("{}({})", name, write_list(&item_vec, &fmt)?)
            }
            ast::MetaItemKind::NameValue(ref literal) => {
                let name = self.name.as_str();
                // 3 = ` = `
                let lit_shape = shape.shrink_left(name.len() + 3)?;
                let value = rewrite_literal(context, literal, lit_shape)?;
                format!("{} = {}", name, value)
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
            let doc_shape = Shape {
                width: cmp::min(shape.width, context.config.comment_width())
                    .checked_sub(shape.indent.width())
                    .unwrap_or(0),
                ..shape
            };
            rewrite_comment(&snippet, false, doc_shape, context.config)
        } else {
            if contains_comment(&snippet) {
                return Some(snippet);
            }
            // 1 = `[`
            let shape = shape.offset_left(prefix.len() + 1)?;
            self.meta()?
                .rewrite(context, shape)
                .map(|rw| format!("{}[{}]", prefix, rw))
        }
    }
}

impl<'a> Rewrite for [ast::Attribute] {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        if self.is_empty() {
            return Some(String::new());
        }
        let mut result = String::with_capacity(128);
        let indent = shape.indent.to_string(context.config);

        let mut derive_args = Vec::new();

        let mut iter = self.iter().enumerate().peekable();
        let mut insert_new_line = true;
        let mut is_prev_sugared_doc = false;
        while let Some((i, a)) = iter.next() {
            let a_str = a.rewrite(context, shape)?;

            // Write comments and blank lines between attributes.
            if i > 0 {
                let comment = context.snippet(mk_sp(self[i - 1].span.hi(), a.span.lo()));
                // This particular horror show is to preserve line breaks in between doc
                // comments. An alternative would be to force such line breaks to start
                // with the usual doc comment token.
                let (multi_line_before, multi_line_after) = if a.is_sugared_doc
                    || is_prev_sugared_doc
                {
                    // Look at before and after comment and see if there are any empty lines.
                    let comment_begin = comment.chars().position(|c| c == '/');
                    let len = comment_begin.unwrap_or_else(|| comment.len());
                    let mlb = comment.chars().take(len).filter(|c| *c == '\n').count() > 1;
                    let mla = if comment_begin.is_none() {
                        mlb
                    } else {
                        let comment_end = comment.chars().rev().position(|c| !c.is_whitespace());
                        let len = comment_end.unwrap();
                        comment
                            .chars()
                            .rev()
                            .take(len)
                            .filter(|c| *c == '\n')
                            .count() > 1
                    };
                    (mlb, mla)
                } else {
                    (false, false)
                };

                let comment = recover_missing_comment_in_span(
                    mk_sp(self[i - 1].span.hi(), a.span.lo()),
                    shape.with_max_width(context.config),
                    context,
                    0,
                )?;

                if !comment.is_empty() {
                    if multi_line_before {
                        result.push('\n');
                    }
                    result.push_str(&comment);
                    result.push('\n');
                    if multi_line_after {
                        result.push('\n')
                    }
                } else if insert_new_line {
                    result.push('\n');
                    if multi_line_after {
                        result.push('\n')
                    }
                }

                if derive_args.is_empty() {
                    result.push_str(&indent);
                }

                insert_new_line = true;
            }

            // Write the attribute itself.
            if context.config.merge_derives() {
                // If the attribute is `#[derive(...)]`, take the arguments.
                if let Some(mut args) = get_derive_args(context, a) {
                    derive_args.append(&mut args);
                    match iter.peek() {
                        // If the next attribute is `#[derive(...)]` as well, skip rewriting.
                        Some(&(_, next_attr)) if is_derive(next_attr) => insert_new_line = false,
                        // If not, rewrite the merged derives.
                        _ => {
                            result.push_str(&format_derive(context, &derive_args, shape)?);
                            derive_args.clear();
                        }
                    }
                } else {
                    result.push_str(&a_str);
                }
            } else {
                result.push_str(&a_str);
            }

            is_prev_sugared_doc = a.is_sugared_doc;
        }
        Some(result)
    }
}

// Format `#[derive(..)]`, using visual indent & mixed style when we need to go multiline.
fn format_derive(context: &RewriteContext, derive_args: &[String], shape: Shape) -> Option<String> {
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
            budget = budget.checked_sub(width).unwrap_or(0);
        }
        result.push_str(a);
        if i != num - 1 {
            result.push_str(", ")
        }
    }
    result.push_str(")]");
    Some(result)
}

fn is_derive(attr: &ast::Attribute) -> bool {
    match attr.meta() {
        Some(meta_item) => match meta_item.node {
            ast::MetaItemKind::List(..) => meta_item.name.as_str() == "derive",
            _ => false,
        },
        _ => false,
    }
}

/// Returns the arguments of `#[derive(...)]`.
fn get_derive_args(context: &RewriteContext, attr: &ast::Attribute) -> Option<Vec<String>> {
    attr.meta().and_then(|meta_item| match meta_item.node {
        ast::MetaItemKind::List(ref args) if meta_item.name.as_str() == "derive" => {
            // Every argument of `derive` should be `NestedMetaItemKind::Literal`.
            Some(
                args.iter()
                    .map(|a| context.snippet(a.span))
                    .collect::<Vec<_>>(),
            )
        }
        _ => None,
    })
}

// Rewrite `extern crate foo;` WITHOUT attributes.
pub fn rewrite_extern_crate(context: &RewriteContext, item: &ast::Item) -> Option<String> {
    assert!(is_extern_crate(item));
    let new_str = context.snippet(item.span);
    Some(if contains_comment(&new_str) {
        new_str
    } else {
        let no_whitespace = &new_str.split_whitespace().collect::<Vec<&str>>().join(" ");
        String::from(&*Regex::new(r"\s;").unwrap().replace(no_whitespace, ";"))
    })
}
