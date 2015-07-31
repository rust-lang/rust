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
use syntax::parse::parser;
use std::path::PathBuf;

use utils;
use config::Config;

use changes::ChangeSet;
use rewrite::{Rewrite, RewriteContext};

pub struct FmtVisitor<'a> {
    pub codemap: &'a CodeMap,
    pub changes: ChangeSet<'a>,
    pub last_pos: BytePos,
    // TODO RAII util for indenting
    pub block_indent: usize,
    pub config: &'a Config,
}

impl<'a, 'v> visit::Visitor<'v> for FmtVisitor<'a> {
    fn visit_expr(&mut self, ex: &'v ast::Expr) {
        debug!("visit_expr: {:?} {:?}",
               self.codemap.lookup_char_pos(ex.span.lo),
               self.codemap.lookup_char_pos(ex.span.hi));
        self.format_missing(ex.span.lo);
        let offset = self.changes.cur_offset_span(ex.span);
        let context = RewriteContext {
            codemap: self.codemap,
            config: self.config,
            block_indent: self.block_indent,
        };
        let rewrite = ex.rewrite(&context, self.config.max_width - offset, offset);

        if let Some(new_str) = rewrite {
            self.changes.push_str_span(ex.span, &new_str);
            self.last_pos = ex.span.hi;
        }
    }

    fn visit_stmt(&mut self, stmt: &'v ast::Stmt) {
        // If the stmt is actually an item, then we'll handle any missing spans
        // there. This is important because of annotations.
        // Although it might make more sense for the statement span to include
        // any annotations on the item.
        let skip_missing = match stmt.node {
            ast::Stmt_::StmtDecl(ref decl, _) => {
                match decl.node {
                    ast::Decl_::DeclItem(_) => true,
                    _ => false,
                }
            }
            _ => false,
        };
        if !skip_missing {
            self.format_missing_with_indent(stmt.span.lo);
        }
        visit::walk_stmt(self, stmt);
    }

    fn visit_block(&mut self, b: &'v ast::Block) {
        debug!("visit_block: {:?} {:?}",
               self.codemap.lookup_char_pos(b.span.lo),
               self.codemap.lookup_char_pos(b.span.hi));
        self.format_missing(b.span.lo);

        self.changes.push_str_span(b.span, "{");
        self.last_pos = self.last_pos + BytePos(1);
        self.block_indent += self.config.tab_spaces;

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
        self.format_missing_with_indent(b.span.hi - BytePos(1));
        self.changes.push_str_span(b.span, "}");
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
        self.format_missing_with_indent(s.lo);
        self.last_pos = s.lo;

        let indent = self.block_indent;
        match fk {
            visit::FkItemFn(ident,
                            ref generics,
                            ref unsafety,
                            ref constness,
                            ref abi,
                            vis) => {
                let new_fn = self.rewrite_fn(indent,
                                             ident,
                                             fd,
                                             None,
                                             generics,
                                             unsafety,
                                             constness,
                                             abi,
                                             vis,
                                             codemap::mk_sp(s.lo, b.span.lo));
                self.changes.push_str_span(s, &new_fn);
            }
            visit::FkMethod(ident, ref sig, vis) => {
                let new_fn = self.rewrite_fn(indent,
                                             ident,
                                             fd,
                                             Some(&sig.explicit_self),
                                             &sig.generics,
                                             &sig.unsafety,
                                             &sig.constness,
                                             &sig.abi,
                                             vis.unwrap_or(ast::Visibility::Inherited),
                                             codemap::mk_sp(s.lo, b.span.lo));
                self.changes.push_str_span(s, &new_fn);
            }
            visit::FkFnBlock(..) => {}
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
                self.changes.push_str_span(item.span, &new_str);
                self.last_pos = item.span.hi;
            }
            ast::Item_::ItemStruct(ref def, ref generics) => {
                self.format_missing_with_indent(item.span.lo);
                self.visit_struct(item.ident,
                                  item.vis,
                                  def,
                                  generics,
                                  item.span);
            }
            ast::Item_::ItemEnum(ref def, ref generics) => {
                self.format_missing_with_indent(item.span.lo);
                self.visit_enum(item.ident,
                                item.vis,
                                def,
                                generics,
                                item.span);
                self.last_pos = item.span.hi;
            }
            ast::Item_::ItemMod(ref module) => {
                self.format_missing_with_indent(item.span.lo);
                self.format_mod(module, item.span, item.ident, &item.attrs);
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
            let new_fn = self.rewrite_required_fn(indent,
                                                  ti.ident,
                                                  sig,
                                                  ti.span);

            self.changes.push_str_span(ti.span, &new_fn);
            self.last_pos = ti.span.hi;
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

    fn visit_mod(&mut self, m: &'v ast::Mod, s: Span, _: ast::NodeId) {
        // This is only called for the root module
        let filename = self.codemap.span_to_filename(s);
        self.format_separate_mod(m, &filename);
    }
}

impl<'a> FmtVisitor<'a> {
    pub fn from_codemap<'b>(codemap: &'b CodeMap, config: &'b Config) -> FmtVisitor<'b> {
        FmtVisitor {
            codemap: codemap,
            changes: ChangeSet::from_codemap(codemap),
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
        if attrs.len() == 0 {
            return false;
        }

        let first = &attrs[0];
        self.format_missing_with_indent(first.span.lo);

        if utils::contains_skip(attrs) {
            true
        } else {
            let rewrite = self.rewrite_attrs(attrs, self.block_indent);
            self.changes.push_str_span(first.span, &rewrite);
            let last = attrs.last().unwrap();
            self.last_pos = last.span.hi;
            false
        }
    }

    pub fn rewrite_attrs(&self, attrs: &[ast::Attribute], indent: usize) -> String {
        let mut result = String::new();
        let indent = utils::make_indent(indent);

        for (i, a) in attrs.iter().enumerate() {
            let a_str = self.snippet(a.span);

            if i > 0 {
                let comment = self.snippet(codemap::mk_sp(attrs[i-1].span.hi, a.span.lo));
                // This particular horror show is to preserve line breaks in between doc
                // comments. An alternative would be to force such line breaks to start
                // with the usual doc comment token.
                let multi_line = a_str.starts_with("//") && comment.matches('\n').count() > 1;
                let comment = comment.trim();
                if comment.len() > 0 {
                    result.push_str(&indent);
                    result.push_str(comment);
                    result.push('\n');
                } else if multi_line {
                    result.push('\n');
                }
                result.push_str(&indent);
            }

            result.push_str(&a_str);

            if i < attrs.len() - 1 {
                result.push('\n');
            }
        }

        result
    }

    fn format_mod(&mut self, m: &ast::Mod, s: Span, ident: ast::Ident, attrs: &[ast::Attribute]) {
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
        } else {
            debug!("FmtVisitor::format_mod: external mod");
            let file_path = self.module_file(ident, attrs, local_file_name);
            let filename = file_path.to_str().unwrap();
            if self.changes.is_changed(filename) {
                // The file has already been reformatted, do nothing
            } else {
                self.format_separate_mod(m, filename);
            }
        }

        debug!("FmtVisitor::format_mod: exit");
    }

    /// Find the file corresponding to an external mod
    fn module_file(&self, id: ast::Ident, attrs: &[ast::Attribute], filename: String) -> PathBuf {
        let dir_path = {
            let mut path = PathBuf::from(&filename);
            path.pop();
            path
        };

        if let Some(path) = parser::Parser::submod_path_from_attr(attrs, &dir_path) {
            return path;
        }

        match parser::Parser::default_submod_path(id, &dir_path, &self.codemap).result {
            Ok(parser::ModulePathSuccess { path, .. }) => path,
            _ => panic!("Couldn't find module {}", id)
        }
    }

    /// Format the content of a module into a separate file
    fn format_separate_mod(&mut self, m: &ast::Mod, filename: &str) {
        let last_pos = self.last_pos;
        let block_indent = self.block_indent;
        let filemap = self.codemap.get_filemap(filename);
        self.last_pos = filemap.start_pos;
        self.block_indent = 0;
        visit::walk_mod(self, m);
        self.format_missing(filemap.end_pos);
        self.last_pos = last_pos;
        self.block_indent = block_indent;
    }

    fn format_import(&mut self, vis: ast::Visibility, vp: &ast::ViewPath, span: Span) {
        let vis = utils::format_visibility(vis);
        let offset = self.block_indent + vis.len() + "use ".len();
        let context = RewriteContext {
            codemap: self.codemap,
            config: self.config,
            block_indent: self.block_indent,
        };
        // 1 = ";"
        match vp.rewrite(&context, self.config.max_width - offset - 1, offset) {
            Some(ref s) if s.len() == 0 => {
                // Format up to last newline
                let prev_span = codemap::mk_sp(self.last_pos, span.lo);
                let span_end = match self.snippet(prev_span).rfind('\n') {
                    Some(offset) => self.last_pos + BytePos(offset as u32),
                    None => span.lo
                };
                self.format_missing(span_end);
                self.last_pos = span.hi;
            }
            Some(ref s) => {
                let s = format!("{}use {};", vis, s);
                self.format_missing_with_indent(span.lo);
                self.changes.push_str_span(span, &s);
                self.last_pos = span.hi;
            }
            None => {
                self.format_missing_with_indent(span.lo);
                self.format_missing(span.hi);
            }
        }
    }
}
