// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;
use syntax::codemap::{self, CodeMap, Span, BytePos};
use syntax::parse::ParseSess;
use syntax::visit;

use strings::string_buffer::StringBuffer;

use {Indent, WriteMode};
use utils;
use config::Config;
use rewrite::{Rewrite, RewriteContext};
use comment::rewrite_comment;
use macros::rewrite_macro;
use items::{rewrite_static, rewrite_type_alias, format_impl};

pub struct FmtVisitor<'a> {
    pub parse_session: &'a ParseSess,
    pub codemap: &'a CodeMap,
    pub buffer: StringBuffer,
    pub last_pos: BytePos,
    // FIXME: use an RAII util or closure for indenting
    pub block_indent: Indent,
    pub config: &'a Config,
    pub write_mode: Option<WriteMode>,
}

impl<'a> FmtVisitor<'a> {
    fn visit_stmt(&mut self, stmt: &ast::Stmt) {
        match stmt.node {
            ast::Stmt_::StmtDecl(ref decl, _) => {
                if let ast::Decl_::DeclItem(ref item) = decl.node {
                    self.visit_item(item);
                } else {
                    let rewrite = stmt.rewrite(&self.get_context(),
                                               self.config.max_width - self.block_indent.width(),
                                               self.block_indent);

                    self.push_rewrite(stmt.span, rewrite);
                }
            }
            ast::Stmt_::StmtExpr(..) | ast::Stmt_::StmtSemi(..) => {
                let rewrite = stmt.rewrite(&self.get_context(),
                                           self.config.max_width - self.block_indent.width(),
                                           self.block_indent);

                self.push_rewrite(stmt.span, rewrite);
            }
            ast::Stmt_::StmtMac(ref mac, _macro_style) => {
                self.format_missing_with_indent(stmt.span.lo);
                self.visit_mac(mac);
            }
        }
    }

    pub fn visit_block(&mut self, b: &ast::Block) {
        debug!("visit_block: {:?} {:?}",
               self.codemap.lookup_char_pos(b.span.lo),
               self.codemap.lookup_char_pos(b.span.hi));

        // Check if this block has braces.
        let snippet = self.snippet(b.span);
        let has_braces = snippet.starts_with("{") || snippet.starts_with("unsafe");
        let brace_compensation = if has_braces {
            BytePos(1)
        } else {
            BytePos(0)
        };

        self.last_pos = self.last_pos + brace_compensation;
        self.block_indent = self.block_indent.block_indent(self.config);
        self.buffer.push_str("{");

        for stmt in &b.stmts {
            self.visit_stmt(&stmt)
        }

        if let Some(ref e) = b.expr {
            self.format_missing_with_indent(e.span.lo);
            let rewrite = e.rewrite(&self.get_context(),
                                    self.config.max_width - self.block_indent.width(),
                                    self.block_indent)
                           .unwrap_or_else(|| self.snippet(e.span));

            self.buffer.push_str(&rewrite);
            self.last_pos = e.span.hi;

            if utils::semicolon_for_expr(e) {
                self.buffer.push_str(";");
            }
        }

        // FIXME: we should compress any newlines here to just one
        self.format_missing_with_indent(b.span.hi - brace_compensation);
        self.close_block();
        self.last_pos = b.span.hi;
    }

    // FIXME: this is a terrible hack to indent the comments between the last
    // item in the block and the closing brace to the block's level.
    // The closing brace itself, however, should be indented at a shallower
    // level.
    fn close_block(&mut self) {
        let total_len = self.buffer.len;
        let chars_too_many = if self.config.hard_tabs {
            1
        } else {
            self.config.tab_spaces
        };
        self.buffer.truncate(total_len - chars_too_many);
        self.buffer.push_str("}");
        self.block_indent = self.block_indent.block_unindent(self.config);
    }

    // Note that this only gets called for function definitions. Required methods
    // on traits do not get handled here.
    fn visit_fn(&mut self,
                fk: visit::FnKind,
                fd: &ast::FnDecl,
                b: &ast::Block,
                s: Span,
                _: ast::NodeId) {
        let indent = self.block_indent;
        let rewrite = match fk {
            visit::FnKind::ItemFn(ident, ref generics, unsafety, constness, abi, vis) => {
                self.rewrite_fn(indent,
                                ident,
                                fd,
                                None,
                                generics,
                                unsafety,
                                constness,
                                abi,
                                vis,
                                codemap::mk_sp(s.lo, b.span.lo),
                                &b)
            }
            visit::FnKind::Method(ident, ref sig, vis) => {
                self.rewrite_fn(indent,
                                ident,
                                fd,
                                Some(&sig.explicit_self),
                                &sig.generics,
                                sig.unsafety,
                                sig.constness,
                                sig.abi,
                                vis.unwrap_or(ast::Visibility::Inherited),
                                codemap::mk_sp(s.lo, b.span.lo),
                                &b)
            }
            visit::FnKind::Closure => None,
        };

        if let Some(fn_str) = rewrite {
            self.format_missing_with_indent(s.lo);
            self.buffer.push_str(&fn_str);
            if let Some(c) = fn_str.chars().last() {
                if c == '}' {
                    self.last_pos = b.span.hi;
                    return;
                }
            }
        } else {
            self.format_missing(b.span.lo);
        }

        self.last_pos = b.span.lo;
        self.visit_block(b)
    }

    fn visit_item(&mut self, item: &ast::Item) {
        // Don't look at attributes for modules (except for rustfmt_skip).
        // We want to avoid looking at attributes in another file, which the AST
        // doesn't distinguish.
        // FIXME This is overly conservative and means we miss attributes on
        // inline modules.
        match item.node {
            ast::Item_::ItemMod(_) => {
                if utils::contains_skip(&item.attrs) {
                    return;
                }
            }
            _ => {
                if self.visit_attrs(&item.attrs) {
                    return;
                }
            }
        }

        match item.node {
            ast::Item_::ItemUse(ref vp) => {
                self.format_import(item.vis, vp, item.span);
            }
            ast::Item_::ItemImpl(..) => {
                self.format_missing_with_indent(item.span.lo);
                if let Some(impl_str) = format_impl(&self.get_context(), item, self.block_indent) {
                    self.buffer.push_str(&impl_str);
                    self.last_pos = item.span.hi;
                }
            }
            // FIXME(#78): format traits.
            ast::Item_::ItemTrait(_, _, _, ref trait_items) => {
                self.format_missing_with_indent(item.span.lo);
                self.block_indent = self.block_indent.block_indent(self.config);
                for item in trait_items {
                    self.visit_trait_item(&item);
                }
                self.block_indent = self.block_indent.block_unindent(self.config);
            }
            ast::Item_::ItemExternCrate(_) => {
                self.format_missing_with_indent(item.span.lo);
                let new_str = self.snippet(item.span);
                self.buffer.push_str(&new_str);
                self.last_pos = item.span.hi;
            }
            ast::Item_::ItemStruct(ref def, ref generics) => {
                let rewrite = {
                    let indent = self.block_indent;
                    let context = self.get_context();
                    ::items::format_struct(&context,
                                           "struct ",
                                           item.ident,
                                           item.vis,
                                           def,
                                           Some(generics),
                                           item.span,
                                           indent)
                        .map(|s| {
                            match *def {
                                ast::VariantData::Tuple(..) => s + ";",
                                _ => s,
                            }
                        })
                };
                self.push_rewrite(item.span, rewrite);
            }
            ast::Item_::ItemEnum(ref def, ref generics) => {
                self.format_missing_with_indent(item.span.lo);
                self.visit_enum(item.ident, item.vis, def, generics, item.span);
                self.last_pos = item.span.hi;
            }
            ast::Item_::ItemMod(ref module) => {
                self.format_missing_with_indent(item.span.lo);
                self.format_mod(module, item.vis, item.span, item.ident);
            }
            ast::Item_::ItemMac(..) => {
                self.format_missing_with_indent(item.span.lo);
                let snippet = self.snippet(item.span);
                self.buffer.push_str(&snippet);
                self.last_pos = item.span.hi;
                // FIXME: we cannot format these yet, because of a bad span.
                // See rust lang issue #28424.
            }
            ast::Item_::ItemForeignMod(ref foreign_mod) => {
                self.format_missing_with_indent(item.span.lo);
                self.format_foreign_mod(foreign_mod, item.span);
            }
            ast::Item_::ItemStatic(ref ty, mutability, ref expr) => {
                let rewrite = rewrite_static("static",
                                             item.vis,
                                             item.ident,
                                             ty,
                                             mutability,
                                             expr,
                                             &self.get_context());
                self.push_rewrite(item.span, rewrite);
            }
            ast::Item_::ItemConst(ref ty, ref expr) => {
                let rewrite = rewrite_static("const",
                                             item.vis,
                                             item.ident,
                                             ty,
                                             ast::Mutability::MutImmutable,
                                             expr,
                                             &self.get_context());
                self.push_rewrite(item.span, rewrite);
            }
            ast::Item_::ItemDefaultImpl(..) => {
                // FIXME(#78): format impl definitions.
            }
            ast::ItemFn(ref declaration, unsafety, constness, abi, ref generics, ref body) => {
                self.visit_fn(visit::FnKind::ItemFn(item.ident,
                                                    generics,
                                                    unsafety,
                                                    constness,
                                                    abi,
                                                    item.vis),
                              declaration,
                              body,
                              item.span,
                              item.id)
            }
            ast::Item_::ItemTy(ref ty, ref generics) => {
                let rewrite = rewrite_type_alias(&self.get_context(),
                                                 self.block_indent,
                                                 item.ident,
                                                 ty,
                                                 generics,
                                                 item.vis,
                                                 item.span);
                self.push_rewrite(item.span, rewrite);
            }
        }
    }

    fn visit_trait_item(&mut self, ti: &ast::TraitItem) {
        if self.visit_attrs(&ti.attrs) {
            return;
        }

        match ti.node {
            ast::ConstTraitItem(..) => {
                // FIXME: Implement
            }
            ast::MethodTraitItem(ref sig, None) => {
                let indent = self.block_indent;
                let rewrite = self.rewrite_required_fn(indent, ti.ident, sig, ti.span);
                self.push_rewrite(ti.span, rewrite);
            }
            ast::MethodTraitItem(ref sig, Some(ref body)) => {
                self.visit_fn(visit::FnKind::Method(ti.ident, sig, None),
                              &sig.decl,
                              &body,
                              ti.span,
                              ti.id);
            }
            ast::TypeTraitItem(..) => {
                // FIXME: Implement
            }
        }
    }

    pub fn visit_impl_item(&mut self, ii: &ast::ImplItem) {
        if self.visit_attrs(&ii.attrs) {
            return;
        }

        match ii.node {
            ast::ImplItemKind::Method(ref sig, ref body) => {
                self.visit_fn(visit::FnKind::Method(ii.ident, sig, Some(ii.vis)),
                              &sig.decl,
                              body,
                              ii.span,
                              ii.id);
            }
            ast::ImplItemKind::Const(..) => {
                // FIXME: Implement
            }
            ast::ImplItemKind::Type(_) => {
                // FIXME: Implement
            }
            ast::ImplItemKind::Macro(ref mac) => {
                self.visit_mac(mac);
            }
        }
    }

    fn visit_mac(&mut self, mac: &ast::Mac) {
        // 1 = ;
        let width = self.config.max_width - self.block_indent.width() - 1;
        let rewrite = rewrite_macro(mac, &self.get_context(), width, self.block_indent);

        if let Some(res) = rewrite {
            self.buffer.push_str(&res);
            self.last_pos = mac.span.hi;
        }
    }

    fn push_rewrite(&mut self, span: Span, rewrite: Option<String>) {
        self.format_missing_with_indent(span.lo);

        if let Some(res) = rewrite {
            self.buffer.push_str(&res);
            self.last_pos = span.hi;
        }
    }

    pub fn from_codemap(parse_session: &'a ParseSess,
                        config: &'a Config,
                        mode: Option<WriteMode>)
                        -> FmtVisitor<'a> {
        FmtVisitor {
            parse_session: parse_session,
            codemap: parse_session.codemap(),
            buffer: StringBuffer::new(),
            last_pos: BytePos(0),
            block_indent: Indent {
                block_indent: 0,
                alignment: 0,
            },
            config: config,
            write_mode: mode,
        }
    }

    pub fn snippet(&self, span: Span) -> String {
        match self.codemap.span_to_snippet(span) {
            Ok(s) => s,
            Err(_) => {
                println!("Couldn't make snippet for span {:?}->{:?}",
                         self.codemap.lookup_char_pos(span.lo),
                         self.codemap.lookup_char_pos(span.hi));
                "".to_owned()
            }
        }
    }

    // Returns true if we should skip the following item.
    pub fn visit_attrs(&mut self, attrs: &[ast::Attribute]) -> bool {
        if utils::contains_skip(attrs) {
            return true;
        }

        let outers: Vec<_> = attrs.iter()
                                  .filter(|a| a.node.style == ast::AttrStyle::Outer)
                                  .cloned()
                                  .collect();
        if outers.is_empty() {
            return false;
        }

        let first = &outers[0];
        self.format_missing_with_indent(first.span.lo);

        let rewrite = outers.rewrite(&self.get_context(),
                                     self.config.max_width - self.block_indent.width(),
                                     self.block_indent)
                            .unwrap();
        self.buffer.push_str(&rewrite);
        let last = outers.last().unwrap();
        self.last_pos = last.span.hi;
        false
    }

    fn walk_mod_items(&mut self, m: &ast::Mod) {
        for item in &m.items {
            self.visit_item(&item);
        }
    }

    fn format_mod(&mut self, m: &ast::Mod, vis: ast::Visibility, s: Span, ident: ast::Ident) {
        // Decide whether this is an inline mod or an external mod.
        let local_file_name = self.codemap.span_to_filename(s);
        let is_internal = local_file_name == self.codemap.span_to_filename(m.inner);

        self.buffer.push_str(utils::format_visibility(vis));
        self.buffer.push_str("mod ");
        self.buffer.push_str(&ident.to_string());

        if is_internal {
            self.buffer.push_str(" {");
            self.last_pos = ::utils::span_after(s, "{", self.codemap);
            self.block_indent = self.block_indent.block_indent(self.config);
            self.walk_mod_items(m);
            self.format_missing_with_indent(m.inner.hi - BytePos(1));
            self.close_block();
            self.last_pos = m.inner.hi;
        } else {
            self.buffer.push_str(";");
            self.last_pos = s.hi;
        }
    }

    pub fn format_separate_mod(&mut self, m: &ast::Mod) {
        let filemap = self.codemap.lookup_char_pos(m.inner.lo).file;
        self.last_pos = filemap.start_pos;
        self.block_indent = Indent::empty();
        self.walk_mod_items(m);
        self.format_missing(filemap.end_pos);
    }

    fn format_import(&mut self, vis: ast::Visibility, vp: &ast::ViewPath, span: Span) {
        let vis = utils::format_visibility(vis);
        let mut offset = self.block_indent;
        offset.alignment += vis.len() + "use ".len();
        // 1 = ";"
        match vp.rewrite(&self.get_context(),
                         self.config.max_width - offset.width() - 1,
                         offset) {
            Some(ref s) if s.is_empty() => {
                // Format up to last newline
                let prev_span = codemap::mk_sp(self.last_pos, span.lo);
                let span_end = match self.snippet(prev_span).rfind('\n') {
                    Some(offset) => self.last_pos + BytePos(offset as u32),
                    None => span.lo,
                };
                self.format_missing(span_end);
                self.last_pos = span.hi;
            }
            Some(ref s) => {
                let s = format!("{}use {};", vis, s);
                self.format_missing_with_indent(span.lo);
                self.buffer.push_str(&s);
                self.last_pos = span.hi;
            }
            None => {
                self.format_missing_with_indent(span.lo);
                self.format_missing(span.hi);
            }
        }
    }

    pub fn get_context(&self) -> RewriteContext {
        RewriteContext {
            parse_session: self.parse_session,
            codemap: self.codemap,
            config: self.config,
            block_indent: self.block_indent,
        }
    }
}

impl<'a> Rewrite for [ast::Attribute] {
    fn rewrite(&self, context: &RewriteContext, _: usize, offset: Indent) -> Option<String> {
        let mut result = String::new();
        if self.is_empty() {
            return Some(result);
        }
        let indent = offset.to_string(context.config);

        for (i, a) in self.iter().enumerate() {
            let a_str = context.snippet(a.span);

            if i > 0 {
                let comment = context.snippet(codemap::mk_sp(self[i - 1].span.hi, a.span.lo));
                // This particular horror show is to preserve line breaks in between doc
                // comments. An alternative would be to force such line breaks to start
                // with the usual doc comment token.
                let multi_line = a_str.starts_with("//") && comment.matches('\n').count() > 1;
                let comment = comment.trim();
                if !comment.is_empty() {
                    let comment = try_opt!(rewrite_comment(comment,
                                                           false,
                                                           context.config.max_width -
                                                           offset.width(),
                                                           offset,
                                                           context.config));
                    result.push_str(&indent);
                    result.push_str(&comment);
                    result.push('\n');
                } else if multi_line {
                    result.push('\n');
                }
                result.push_str(&indent);
            }

            result.push_str(&a_str);

            if i < self.len() - 1 {
                result.push('\n');
            }
        }

        Some(result)
    }
}
