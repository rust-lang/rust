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
use syntax::visit;

use strings::string_buffer::StringBuffer;

use utils;
use config::Config;
use rewrite::{Rewrite, RewriteContext};
use comment::rewrite_comment;

pub struct FmtVisitor<'a> {
    pub codemap: &'a CodeMap,
    pub buffer: StringBuffer,
    pub last_pos: BytePos,
    // TODO RAII util for indenting
    pub block_indent: usize,
    pub config: &'a Config,
}

impl<'a, 'v> visit::Visitor<'v> for FmtVisitor<'a> {
    // FIXME: We'd rather not format expressions here, as we have little
    // context. How are we still reaching this?
    fn visit_expr(&mut self, ex: &'v ast::Expr) {
        debug!("visit_expr: {:?} {:?}",
               self.codemap.lookup_char_pos(ex.span.lo),
               self.codemap.lookup_char_pos(ex.span.hi));
        self.format_missing(ex.span.lo);

        let offset = self.buffer.cur_offset();
        let rewrite = ex.rewrite(&self.get_context(), self.config.max_width - offset, offset);

        if let Some(new_str) = rewrite {
            self.buffer.push_str(&new_str);
            self.last_pos = ex.span.hi;
        }
    }

    fn visit_stmt(&mut self, stmt: &'v ast::Stmt) {
        match stmt.node {
            ast::Stmt_::StmtDecl(ref decl, _) => {
                match decl.node {
                    ast::Decl_::DeclLocal(ref local) => self.visit_let(local, stmt.span),
                    ast::Decl_::DeclItem(..) => visit::walk_stmt(self, stmt),
                }
            }
            ast::Stmt_::StmtExpr(ref ex, _) | ast::Stmt_::StmtSemi(ref ex, _) => {
                self.format_missing_with_indent(stmt.span.lo);
                let suffix = if let ast::Stmt_::StmtExpr(..) = stmt.node {
                    ""
                } else {
                    ";"
                };

                // 1 = trailing semicolon;
                let rewrite = ex.rewrite(&self.get_context(),
                                         self.config.max_width - self.block_indent - suffix.len(),
                                         self.block_indent);

                if let Some(new_str) = rewrite {
                    self.buffer.push_str(&new_str);
                    self.buffer.push_str(suffix);
                    self.last_pos = stmt.span.hi;
                }
            }
            ast::Stmt_::StmtMac(..) => {
                self.format_missing_with_indent(stmt.span.lo);
                visit::walk_stmt(self, stmt);
            }
        }
    }

    fn visit_block(&mut self, b: &'v ast::Block) {
        debug!("visit_block: {:?} {:?}",
               self.codemap.lookup_char_pos(b.span.lo),
               self.codemap.lookup_char_pos(b.span.hi));

        // Check if this block has braces.
        let snippet = self.snippet(b.span);
        let has_braces = &snippet[..1] == "{" || &snippet[..6] == "unsafe";
        let brace_compensation = if has_braces {
            BytePos(1)
        } else {
            BytePos(0)
        };

        self.last_pos = self.last_pos + brace_compensation;
        self.block_indent += self.config.tab_spaces;
        self.buffer.push_str("{");

        for stmt in &b.stmts {
            self.visit_stmt(&stmt)
        }

        match b.expr {
            Some(ref e) => {
                self.format_missing_with_indent(e.span.lo);
                self.visit_expr(e);
            }
            None => {}
        }

        self.block_indent -= self.config.tab_spaces;
        // TODO we should compress any newlines here to just one
        self.format_missing_with_indent(b.span.hi - brace_compensation);
        self.buffer.push_str("}");
        self.last_pos = b.span.hi;
    }

    // Note that this only gets called for function definitions. Required methods
    // on traits do not get handled here.
    fn visit_fn(&mut self,
                fk: visit::FnKind<'v>,
                fd: &'v ast::FnDecl,
                b: &'v ast::Block,
                s: Span,
                _: ast::NodeId) {
        let indent = self.block_indent;
        let rewrite = match fk {
            visit::FnKind::ItemFn(ident,
                                  ref generics,
                                  ref unsafety,
                                  ref constness,
                                  ref abi,
                                  vis) => {
                self.rewrite_fn(indent,
                                ident,
                                fd,
                                None,
                                generics,
                                unsafety,
                                constness,
                                abi,
                                vis,
                                codemap::mk_sp(s.lo, b.span.lo))
            }
            visit::FnKind::Method(ident, ref sig, vis) => {
                self.rewrite_fn(indent,
                                ident,
                                fd,
                                Some(&sig.explicit_self),
                                &sig.generics,
                                &sig.unsafety,
                                &sig.constness,
                                &sig.abi,
                                vis.unwrap_or(ast::Visibility::Inherited),
                                codemap::mk_sp(s.lo, b.span.lo))
            }
            visit::FnKind::Closure => None,
        };

        if let Some(fn_str) = rewrite {
            self.format_missing_with_indent(s.lo);
            self.buffer.push_str(&fn_str);
        } else {
            self.format_missing(b.span.lo);
        }

        self.last_pos = b.span.lo;
        self.visit_block(b)
    }

    fn visit_item(&mut self, item: &'v ast::Item) {
        // Don't look at attributes for modules.
        // We want to avoid looking at attributes in another file, which the AST
        // doesn't distinguish. FIXME This is overly conservative and means we miss
        // attributes on inline modules.
        match item.node {
            ast::Item_::ItemMod(_) => {}
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
            ast::Item_::ItemImpl(..) |
            ast::Item_::ItemTrait(..) => {
                self.block_indent += self.config.tab_spaces;
                visit::walk_item(self, item);
                self.block_indent -= self.config.tab_spaces;
            }
            ast::Item_::ItemExternCrate(_) => {
                self.format_missing_with_indent(item.span.lo);
                let new_str = self.snippet(item.span);
                self.buffer.push_str(&new_str);
                self.last_pos = item.span.hi;
            }
            ast::Item_::ItemStruct(ref def, ref generics) => {
                self.format_missing_with_indent(item.span.lo);
                self.visit_struct(item.ident, item.vis, def, generics, item.span);
            }
            ast::Item_::ItemEnum(ref def, ref generics) => {
                self.format_missing_with_indent(item.span.lo);
                self.visit_enum(item.ident, item.vis, def, generics, item.span);
                self.last_pos = item.span.hi;
            }
            ast::Item_::ItemMod(ref module) => {
                self.format_missing_with_indent(item.span.lo);
                self.format_mod(module, item.span, item.ident);
            }
            _ => {
                visit::walk_item(self, item);
            }
        }
    }

    fn visit_trait_item(&mut self, ti: &'v ast::TraitItem) {
        if self.visit_attrs(&ti.attrs) {
            return;
        }

        if let ast::TraitItem_::MethodTraitItem(ref sig, None) = ti.node {
            self.format_missing_with_indent(ti.span.lo);

            let indent = self.block_indent;
            let new_fn = self.rewrite_required_fn(indent, ti.ident, sig, ti.span);


            if let Some(fn_str) = new_fn {
                self.buffer.push_str(&fn_str);
                self.last_pos = ti.span.hi;
            }
        }
        // TODO format trait types

        visit::walk_trait_item(self, ti)
    }

    fn visit_impl_item(&mut self, ii: &'v ast::ImplItem) {
        if self.visit_attrs(&ii.attrs) {
            return;
        }
        visit::walk_impl_item(self, ii)
    }

    fn visit_mac(&mut self, mac: &'v ast::Mac) {
        visit::walk_mac(self, mac)
    }
}

impl<'a> FmtVisitor<'a> {
    pub fn from_codemap<'b>(codemap: &'b CodeMap, config: &'b Config) -> FmtVisitor<'b> {
        FmtVisitor {
            codemap: codemap,
            buffer: StringBuffer::new(),
            last_pos: BytePos(0),
            block_indent: 0,
            config: config,
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
        if attrs.is_empty() {
            return false;
        }

        let first = &attrs[0];
        self.format_missing_with_indent(first.span.lo);

        if utils::contains_skip(attrs) {
            true
        } else {
            let rewrite = attrs.rewrite(&self.get_context(),
                                        self.config.max_width - self.block_indent,
                                        self.block_indent)
                               .unwrap();
            self.buffer.push_str(&rewrite);
            let last = attrs.last().unwrap();
            self.last_pos = last.span.hi;
            false
        }
    }

    fn format_mod(&mut self, m: &ast::Mod, s: Span, ident: ast::Ident) {
        debug!("FmtVisitor::format_mod: ident: {:?}, span: {:?}", ident, s);

        // Decide whether this is an inline mod or an external mod.
        let local_file_name = self.codemap.span_to_filename(s);
        let is_internal = local_file_name == self.codemap.span_to_filename(m.inner);

        // TODO Should rewrite properly `mod X;`

        if is_internal {
            debug!("FmtVisitor::format_mod: internal mod");
            self.block_indent += self.config.tab_spaces;
            visit::walk_mod(self, m);
            debug!("... last_pos after: {:?}", self.last_pos);
            self.block_indent -= self.config.tab_spaces;
        }
    }

    pub fn format_separate_mod(&mut self, m: &ast::Mod, filename: &str) {
        let filemap = self.codemap.get_filemap(filename);
        self.last_pos = filemap.start_pos;
        self.block_indent = 0;
        visit::walk_mod(self, m);
        self.format_missing(filemap.end_pos);
    }

    fn format_import(&mut self, vis: ast::Visibility, vp: &ast::ViewPath, span: Span) {
        let vis = utils::format_visibility(vis);
        let offset = self.block_indent + vis.len() + "use ".len();
        let context = RewriteContext {
            codemap: self.codemap,
            config: self.config,
            block_indent: self.block_indent,
            overflow_indent: 0,
        };
        // 1 = ";"
        match vp.rewrite(&context, self.config.max_width - offset - 1, offset) {
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
            codemap: self.codemap,
            config: self.config,
            block_indent: self.block_indent,
            overflow_indent: 0,
        }
    }
}

impl<'a> Rewrite for [ast::Attribute] {
    fn rewrite(&self, context: &RewriteContext, _: usize, offset: usize) -> Option<String> {
        let mut result = String::new();
        if self.is_empty() {
            return Some(result);
        }
        let indent = utils::make_indent(offset);

        for (i, a) in self.iter().enumerate() {
            let a_str = context.snippet(a.span);

            if i > 0 {
                let comment = context.snippet(codemap::mk_sp(self[i-1].span.hi, a.span.lo));
                // This particular horror show is to preserve line breaks in between doc
                // comments. An alternative would be to force such line breaks to start
                // with the usual doc comment token.
                let multi_line = a_str.starts_with("//") && comment.matches('\n').count() > 1;
                let comment = comment.trim();
                if !comment.is_empty() {
                    let comment = rewrite_comment(comment,
                                                  false,
                                                  context.config.max_width - offset,
                                                  offset);
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
