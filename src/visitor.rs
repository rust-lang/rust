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

use syntax::{ast, ptr, visit};
use syntax::codemap::{CodeMap, Span, BytePos};
use syntax::parse::ParseSess;

use strings::string_buffer::StringBuffer;

use {Indent, Shape};
use utils::{self, mk_sp};
use codemap::{LineRangeUtils, SpanUtils};
use comment::FindUncommented;
use config::Config;
use rewrite::{Rewrite, RewriteContext};
use comment::rewrite_comment;
use macros::{rewrite_macro, MacroPosition};
use items::{rewrite_static, rewrite_associated_type, rewrite_associated_impl_type,
            rewrite_type_alias, format_impl, format_trait};
use lists::{itemize_list, write_list, DefinitiveListTactic, ListFormatting, SeparatorTactic};

fn is_use_item(item: &ast::Item) -> bool {
    match item.node {
        ast::ItemKind::Use(_) => true,
        _ => false,
    }
}

fn item_bound(item: &ast::Item) -> Span {
    item.attrs.iter().map(|attr| attr.span).fold(
        item.span,
        |bound, span| {
            Span {
                lo: cmp::min(bound.lo, span.lo),
                hi: cmp::max(bound.hi, span.hi),
                ctxt: span.ctxt,
            }
        },
    )
}

pub struct FmtVisitor<'a> {
    pub parse_session: &'a ParseSess,
    pub codemap: &'a CodeMap,
    pub buffer: StringBuffer,
    pub last_pos: BytePos,
    // FIXME: use an RAII util or closure for indenting
    pub block_indent: Indent,
    pub config: &'a Config,
    pub failed: bool,
    pub is_if_else_block: bool,
}

impl<'a> FmtVisitor<'a> {
    fn visit_stmt(&mut self, stmt: &ast::Stmt) {
        debug!(
            "visit_stmt: {:?} {:?}",
            self.codemap.lookup_char_pos(stmt.span.lo),
            self.codemap.lookup_char_pos(stmt.span.hi)
        );

        // FIXME(#434): Move this check to somewhere more central, eg Rewrite.
        if !self.config.file_lines().intersects(
            &self.codemap.lookup_line_range(
                stmt.span,
            ),
        )
        {
            return;
        }

        match stmt.node {
            ast::StmtKind::Item(ref item) => {
                self.visit_item(item);
            }
            ast::StmtKind::Local(..) => {
                let rewrite = stmt.rewrite(
                    &self.get_context(),
                    Shape::indented(self.block_indent, self.config),
                );
                self.push_rewrite(stmt.span, rewrite);
            }
            ast::StmtKind::Expr(ref expr) |
            ast::StmtKind::Semi(ref expr) => {
                let rewrite = stmt.rewrite(
                    &self.get_context(),
                    Shape::indented(self.block_indent, self.config),
                );
                let span = if expr.attrs.is_empty() {
                    stmt.span
                } else {
                    mk_sp(expr.attrs[0].span.lo, stmt.span.hi)
                };
                self.push_rewrite(span, rewrite)
            }
            ast::StmtKind::Mac(ref mac) => {
                let (ref mac, _macro_style, _) = **mac;
                self.visit_mac(mac, None, MacroPosition::Statement);
                self.format_missing(stmt.span.hi);
            }
        }
    }

    pub fn visit_block(&mut self, b: &ast::Block) {
        debug!(
            "visit_block: {:?} {:?}",
            self.codemap.lookup_char_pos(b.span.lo),
            self.codemap.lookup_char_pos(b.span.hi)
        );

        // Check if this block has braces.
        let snippet = self.snippet(b.span);
        let has_braces = snippet.starts_with('{') || snippet.starts_with("unsafe");
        let brace_compensation = if has_braces { BytePos(1) } else { BytePos(0) };

        self.last_pos = self.last_pos + brace_compensation;
        self.block_indent = self.block_indent.block_indent(self.config);
        self.buffer.push_str("{");

        for stmt in &b.stmts {
            self.visit_stmt(stmt)
        }

        if !b.stmts.is_empty() {
            if let Some(expr) = utils::stmt_expr(&b.stmts[b.stmts.len() - 1]) {
                if utils::semicolon_for_expr(expr) {
                    self.buffer.push_str(";");
                }
            }
        }

        let mut unindent_comment = self.is_if_else_block && !b.stmts.is_empty();
        if unindent_comment {
            let end_pos = source!(self, b.span).hi - brace_compensation;
            let snippet = self.get_context().snippet(mk_sp(self.last_pos, end_pos));
            unindent_comment = snippet.contains("//") || snippet.contains("/*");
        }
        // FIXME: we should compress any newlines here to just one
        if unindent_comment {
            self.block_indent = self.block_indent.block_unindent(self.config);
        }
        self.format_missing_with_indent(source!(self, b.span).hi - brace_compensation);
        if unindent_comment {
            self.block_indent = self.block_indent.block_indent(self.config);
        }
        self.close_block(unindent_comment);
        self.last_pos = source!(self, b.span).hi;
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
        fd: &ast::FnDecl,
        s: Span,
        _: ast::NodeId,
        defaultness: ast::Defaultness,
    ) {
        let indent = self.block_indent;
        let block;
        let rewrite = match fk {
            visit::FnKind::ItemFn(ident, generics, unsafety, constness, abi, vis, b) => {
                block = b;
                self.rewrite_fn(
                    indent,
                    ident,
                    fd,
                    generics,
                    unsafety,
                    constness.node,
                    defaultness,
                    abi,
                    vis,
                    mk_sp(s.lo, b.span.lo),
                    &b,
                )
            }
            visit::FnKind::Method(ident, sig, vis, b) => {
                block = b;
                self.rewrite_fn(
                    indent,
                    ident,
                    fd,
                    &sig.generics,
                    sig.unsafety,
                    sig.constness.node,
                    defaultness,
                    sig.abi,
                    vis.unwrap_or(&ast::Visibility::Inherited),
                    mk_sp(s.lo, b.span.lo),
                    &b,
                )
            }
            visit::FnKind::Closure(_) => unreachable!(),
        };

        if let Some(fn_str) = rewrite {
            self.format_missing_with_indent(source!(self, s).lo);
            self.buffer.push_str(&fn_str);
            if let Some(c) = fn_str.chars().last() {
                if c == '}' {
                    self.last_pos = source!(self, block.span).hi;
                    return;
                }
            }
        } else {
            self.format_missing(source!(self, block.span).lo);
        }

        self.last_pos = source!(self, block.span).lo;
        self.visit_block(block)
    }

    pub fn visit_item(&mut self, item: &ast::Item) {
        // This is where we bail out if there is a skip attribute. This is only
        // complex in the module case. It is complex because the module could be
        // in a separate file and there might be attributes in both files, but
        // the AST lumps them all together.
        match item.node {
            ast::ItemKind::Mod(ref m) => {
                let outer_file = self.codemap.lookup_char_pos(item.span.lo).file;
                let inner_file = self.codemap.lookup_char_pos(m.inner.lo).file;
                if outer_file.name == inner_file.name {
                    // Module is inline, in this case we treat modules like any
                    // other item.
                    if self.visit_attrs(&item.attrs) {
                        self.push_rewrite(item.span, None);
                        return;
                    }
                } else if utils::contains_skip(&item.attrs) {
                    // Module is not inline, but should be skipped.
                    return;
                } else {
                    // Module is not inline and should not be skipped. We want
                    // to process only the attributes in the current file.
                    let attrs = item.attrs
                        .iter()
                        .filter_map(|a| {
                            let attr_file = self.codemap.lookup_char_pos(a.span.lo).file;
                            if attr_file.name == outer_file.name {
                                Some(a.clone())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();
                    // Assert because if we should skip it should be caught by
                    // the above case.
                    assert!(!self.visit_attrs(&attrs));
                }
            }
            _ => {
                if self.visit_attrs(&item.attrs) {
                    self.push_rewrite(item.span, None);
                    return;
                }
            }
        }

        match item.node {
            ast::ItemKind::Use(ref vp) => {
                self.format_import(&item.vis, vp, item.span);
            }
            ast::ItemKind::Impl(..) => {
                self.format_missing_with_indent(source!(self, item.span).lo);
                let snippet = self.get_context().snippet(item.span);
                let where_span_end = snippet.find_uncommented("{").map(|x| {
                    (BytePos(x as u32)) + source!(self, item.span).lo
                });
                if let Some(impl_str) = format_impl(
                    &self.get_context(),
                    item,
                    self.block_indent,
                    where_span_end,
                )
                {
                    self.buffer.push_str(&impl_str);
                    self.last_pos = source!(self, item.span).hi;
                }
            }
            ast::ItemKind::Trait(..) => {
                self.format_missing_with_indent(item.span.lo);
                if let Some(trait_str) = format_trait(
                    &self.get_context(),
                    item,
                    self.block_indent,
                )
                {
                    self.buffer.push_str(&trait_str);
                    self.last_pos = source!(self, item.span).hi;
                }
            }
            ast::ItemKind::ExternCrate(_) => {
                self.format_missing_with_indent(source!(self, item.span).lo);
                let new_str = self.snippet(item.span);
                self.buffer.push_str(&new_str);
                self.last_pos = source!(self, item.span).hi;
            }
            ast::ItemKind::Struct(ref def, ref generics) => {
                let rewrite = {
                    let indent = self.block_indent;
                    let context = self.get_context();
                    ::items::format_struct(
                        &context,
                        "struct ",
                        item.ident,
                        &item.vis,
                        def,
                        Some(generics),
                        item.span,
                        indent,
                        None,
                    ).map(|s| match *def {
                        ast::VariantData::Tuple(..) => s + ";",
                        _ => s,
                    })
                };
                self.push_rewrite(item.span, rewrite);
            }
            ast::ItemKind::Enum(ref def, ref generics) => {
                self.format_missing_with_indent(source!(self, item.span).lo);
                self.visit_enum(item.ident, &item.vis, def, generics, item.span);
                self.last_pos = source!(self, item.span).hi;
            }
            ast::ItemKind::Mod(ref module) => {
                self.format_missing_with_indent(source!(self, item.span).lo);
                self.format_mod(module, &item.vis, item.span, item.ident);
            }
            ast::ItemKind::Mac(ref mac) => {
                self.visit_mac(mac, Some(item.ident), MacroPosition::Item);
            }
            ast::ItemKind::ForeignMod(ref foreign_mod) => {
                self.format_missing_with_indent(source!(self, item.span).lo);
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
                    visit::FnKind::ItemFn(
                        item.ident,
                        generics,
                        unsafety,
                        constness,
                        abi,
                        &item.vis,
                        body,
                    ),
                    decl,
                    item.span,
                    item.id,
                    ast::Defaultness::Final,
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
            ast::ItemKind::Union(..) => {
                // FIXME(#1157): format union definitions.
            }
            ast::ItemKind::GlobalAsm(..) => {
                let snippet = Some(self.snippet(item.span));
                self.push_rewrite(item.span, snippet);
            }
            ast::ItemKind::MacroDef(..) => {
                // FIXME(#1539): macros 2.0
                let snippet = Some(self.snippet(item.span));
                self.push_rewrite(item.span, snippet);
            }
        }
    }

    pub fn visit_trait_item(&mut self, ti: &ast::TraitItem) {
        if self.visit_attrs(&ti.attrs) {
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
                let rewrite = self.rewrite_required_fn(indent, ti.ident, sig, ti.span);
                self.push_rewrite(ti.span, rewrite);
            }
            ast::TraitItemKind::Method(ref sig, Some(ref body)) => {
                self.visit_fn(
                    visit::FnKind::Method(ti.ident, sig, None, body),
                    &sig.decl,
                    ti.span,
                    ti.id,
                    ast::Defaultness::Final,
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
        if self.visit_attrs(&ii.attrs) {
            self.push_rewrite(ii.span, None);
            return;
        }

        match ii.node {
            ast::ImplItemKind::Method(ref sig, ref body) => {
                self.visit_fn(
                    visit::FnKind::Method(ii.ident, sig, Some(&ii.vis), body),
                    &sig.decl,
                    ii.span,
                    ii.id,
                    ii.defaultness,
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
        // 1 = ;
        let shape = Shape::indented(self.block_indent, self.config)
            .sub_width(1)
            .unwrap();
        let rewrite = rewrite_macro(mac, ident, &self.get_context(), shape, pos);
        self.push_rewrite(mac.span, rewrite);
    }

    fn push_rewrite(&mut self, span: Span, rewrite: Option<String>) {
        self.format_missing_with_indent(source!(self, span).lo);
        self.failed = match rewrite {
            Some(ref s)
                if s.rewrite(
                    &self.get_context(),
                    Shape::indented(self.block_indent, self.config),
                ).is_none() => true,
            None => true,
            _ => self.failed,
        };
        let result = rewrite.unwrap_or_else(|| self.snippet(span));
        self.buffer.push_str(&result);
        self.last_pos = source!(self, span).hi;
    }

    pub fn from_codemap(parse_session: &'a ParseSess, config: &'a Config) -> FmtVisitor<'a> {
        FmtVisitor {
            parse_session: parse_session,
            codemap: parse_session.codemap(),
            buffer: StringBuffer::new(),
            last_pos: BytePos(0),
            block_indent: Indent::empty(),
            config: config,
            failed: false,
            is_if_else_block: false,
        }
    }

    pub fn snippet(&self, span: Span) -> String {
        match self.codemap.span_to_snippet(span) {
            Ok(s) => s,
            Err(_) => {
                println!(
                    "Couldn't make snippet for span {:?}->{:?}",
                    self.codemap.lookup_char_pos(span.lo),
                    self.codemap.lookup_char_pos(span.hi)
                );
                "".to_owned()
            }
        }
    }

    // Returns true if we should skip the following item.
    pub fn visit_attrs(&mut self, attrs: &[ast::Attribute]) -> bool {
        if utils::contains_skip(attrs) {
            return true;
        }

        let outers: Vec<_> = attrs
            .iter()
            .filter(|a| a.style == ast::AttrStyle::Outer)
            .cloned()
            .collect();
        if outers.is_empty() {
            return false;
        }

        let first = &outers[0];
        self.format_missing_with_indent(source!(self, first.span).lo);

        let rewrite = outers
            .rewrite(
                &self.get_context(),
                Shape::indented(self.block_indent, self.config),
            )
            .unwrap();
        self.buffer.push_str(&rewrite);
        let last = outers.last().unwrap();
        self.last_pos = source!(self, last.span).hi;
        false
    }

    fn walk_mod_items(&mut self, m: &ast::Mod) {
        let mut items_left: &[ptr::P<ast::Item>] = &m.items;
        while !items_left.is_empty() {
            // If the next item is a `use` declaration, then extract it and any subsequent `use`s
            // to be potentially reordered within `format_imports`. Otherwise, just format the
            // next item for output.
            if self.config.reorder_imports() && is_use_item(&*items_left[0]) {
                let reorder_imports_in_group = self.config.reorder_imports_in_group();
                let mut last = self.codemap.lookup_line_range(item_bound(&items_left[0]));
                let use_item_length = items_left
                    .iter()
                    .take_while(|ppi| {
                        is_use_item(&***ppi) &&
                            (!reorder_imports_in_group ||
                                 {
                                     let current = self.codemap.lookup_line_range(item_bound(&ppi));
                                     let in_same_group = current.lo < last.hi + 2;
                                     last = current;
                                     in_same_group
                                 })
                    })
                    .count();
                let (use_items, rest) = items_left.split_at(use_item_length);
                self.format_imports(use_items);
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

    fn format_mod(&mut self, m: &ast::Mod, vis: &ast::Visibility, s: Span, ident: ast::Ident) {
        // Decide whether this is an inline mod or an external mod.
        let local_file_name = self.codemap.span_to_filename(s);
        let inner_span = source!(self, m.inner);
        let is_internal = !(inner_span.lo.0 == 0 && inner_span.hi.0 == 0) &&
            local_file_name == self.codemap.span_to_filename(inner_span);

        self.buffer.push_str(&*utils::format_visibility(vis));
        self.buffer.push_str("mod ");
        self.buffer.push_str(&ident.to_string());

        if is_internal {
            self.buffer.push_str(" {");
            // Hackery to account for the closing }.
            let mod_lo = self.codemap.span_after(source!(self, s), "{");
            let body_snippet = self.snippet(mk_sp(mod_lo, source!(self, m.inner).hi - BytePos(1)));
            let body_snippet = body_snippet.trim();
            if body_snippet.is_empty() {
                self.buffer.push_str("}");
            } else {
                self.last_pos = mod_lo;
                self.block_indent = self.block_indent.block_indent(self.config);
                self.walk_mod_items(m);
                self.format_missing_with_indent(source!(self, m.inner).hi - BytePos(1));
                self.close_block(false);
            }
            self.last_pos = source!(self, m.inner).hi;
        } else {
            self.buffer.push_str(";");
            self.last_pos = source!(self, s).hi;
        }
    }

    pub fn format_separate_mod(&mut self, m: &ast::Mod) {
        let filemap = self.codemap.lookup_char_pos(m.inner.lo).file;
        self.last_pos = filemap.start_pos;
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
            ast::NestedMetaItemKind::Literal(..) => Some(context.snippet(self.span)),
        }
    }
}

fn count_missing_closing_parens(s: &str) -> u32 {
    let op_parens = s.chars().filter(|c| *c == '(').count();
    let cl_parens = s.chars().filter(|c| *c == ')').count();
    op_parens.checked_sub(cl_parens).unwrap_or(0) as u32
}

impl Rewrite for ast::MetaItem {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        Some(match self.node {
            ast::MetaItemKind::Word => String::from(&*self.name.as_str()),
            ast::MetaItemKind::List(ref list) => {
                let name = self.name.as_str();
                // 3 = `#[` and `(`, 2 = `]` and `)`
                let item_shape = try_opt!(shape.shrink_left(name.len() + 3).and_then(
                    |s| s.sub_width(2),
                ));
                let hi = self.span.hi +
                    BytePos(count_missing_closing_parens(&context.snippet(self.span)));
                let items = itemize_list(
                    context.codemap,
                    list.iter(),
                    ")",
                    |nested_meta_item| nested_meta_item.span.lo,
                    // FIXME: Span from MetaItem is missing closing parens.
                    |nested_meta_item| {
                        let snippet = context.snippet(nested_meta_item.span);
                        nested_meta_item.span.hi + BytePos(count_missing_closing_parens(&snippet))
                    },
                    |nested_meta_item| nested_meta_item.rewrite(context, item_shape),
                    self.span.lo,
                    hi,
                );
                let item_vec = items.collect::<Vec<_>>();
                let fmt = ListFormatting {
                    tactic: DefinitiveListTactic::Mixed,
                    separator: ",",
                    trailing_separator: SeparatorTactic::Never,
                    shape: item_shape,
                    ends_with_newline: false,
                    config: context.config,
                };
                format!("{}({})", name, try_opt!(write_list(&item_vec, &fmt)))
            }
            ast::MetaItemKind::NameValue(ref literal) => {
                let name = self.name.as_str();
                let value = context.snippet(literal.span);
                if &*name == "doc" && value.starts_with("///") {
                    let doc_shape = Shape {
                        width: cmp::min(shape.width, context.config.comment_width())
                            .checked_sub(shape.indent.width())
                            .unwrap_or(0),
                        ..shape
                    };
                    format!(
                        "{}",
                        try_opt!(rewrite_comment(&value, false, doc_shape, context.config))
                    )
                } else {
                    format!("{} = {}", name, value)
                }
            }
        })
    }
}

impl Rewrite for ast::Attribute {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        try_opt!(self.meta()).rewrite(context, shape).map(|rw| {
            if rw.starts_with("///") {
                rw
            } else {
                format!("#[{}]", rw)
            }
        })
    }
}

impl<'a> Rewrite for [ast::Attribute] {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        let mut result = String::new();
        if self.is_empty() {
            return Some(result);
        }
        let indent = shape.indent.to_string(context.config);

        for (i, a) in self.iter().enumerate() {
            let a_str = try_opt!(a.rewrite(context, shape));

            // Write comments and blank lines between attributes.
            if i > 0 {
                let comment = context.snippet(mk_sp(self[i - 1].span.hi, a.span.lo));
                // This particular horror show is to preserve line breaks in between doc
                // comments. An alternative would be to force such line breaks to start
                // with the usual doc comment token.
                let multi_line = a_str.starts_with("//") && comment.matches('\n').count() > 1;
                let comment = comment.trim();
                if !comment.is_empty() {
                    let comment = try_opt!(rewrite_comment(
                        comment,
                        false,
                        Shape::legacy(
                            context.config.comment_width() - shape.indent.width(),
                            shape.indent,
                        ),
                        context.config,
                    ));
                    result.push_str(&indent);
                    result.push_str(&comment);
                    result.push('\n');
                } else if multi_line {
                    result.push('\n');
                }
                result.push_str(&indent);
            }

            // Write the attribute itself.
            result.push_str(&a_str);

            if i < self.len() - 1 {
                result.push('\n');
            }
        }

        Some(result)
    }
}
