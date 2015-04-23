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
use syntax::codemap::{CodeMap, Span, BytePos};
use syntax::visit;

use {MAX_WIDTH, TAB_SPACES, SKIP_ANNOTATION};
use changes::ChangeSet;

pub struct FmtVisitor<'a> {
    pub codemap: &'a CodeMap,
    pub changes: ChangeSet<'a>,
    pub last_pos: BytePos,
    // TODO RAII util for indenting
    pub block_indent: usize,
}

impl<'a, 'v> visit::Visitor<'v> for FmtVisitor<'a> {
    fn visit_expr(&mut self, ex: &'v ast::Expr) {
        debug!("visit_expr: {:?} {:?}",
               self.codemap.lookup_char_pos(ex.span.lo),
               self.codemap.lookup_char_pos(ex.span.hi));
        self.format_missing(ex.span.lo);
        let offset = self.changes.cur_offset_span(ex.span);
        let new_str = self.rewrite_expr(ex, MAX_WIDTH - offset, offset);
        self.changes.push_str_span(ex.span, &new_str);
        self.last_pos = ex.span.hi;
    }

    fn visit_block(&mut self, b: &'v ast::Block) {
        debug!("visit_block: {:?} {:?}",
               self.codemap.lookup_char_pos(b.span.lo),
               self.codemap.lookup_char_pos(b.span.hi));
        self.format_missing(b.span.lo);

        self.changes.push_str_span(b.span, "{");
        self.last_pos = self.last_pos + BytePos(1);
        self.block_indent += TAB_SPACES;

        for stmt in &b.stmts {
            self.format_missing_with_indent(stmt.span.lo);
            self.visit_stmt(&stmt)
        }
        match b.expr {
            Some(ref e) => {
                self.format_missing_with_indent(e.span.lo);
                self.visit_expr(e);
            }
            None => {}
        }

        self.block_indent -= TAB_SPACES;
        // TODO we should compress any newlines here to just one
        self.format_missing_with_indent(b.span.hi - BytePos(1));
        self.changes.push_str_span(b.span, "}");
        self.last_pos = b.span.hi;
    }

    // Note that this only gets called for function defintions. Required methods
    // on traits do not get handled here.
    fn visit_fn(&mut self,
                fk: visit::FnKind<'v>,
                fd: &'v ast::FnDecl,
                b: &'v ast::Block,
                s: Span,
                _: ast::NodeId) {
        self.format_missing(s.lo);
        self.last_pos = s.lo;

        // TODO need to check against expected indent
        let indent = self.codemap.lookup_char_pos(s.lo).col.0;
        match fk {
            visit::FkItemFn(ident, ref generics, ref unsafety, ref abi, vis) => {
                let new_fn = self.rewrite_fn(indent,
                                             ident,
                                             fd,
                                             None,
                                             generics,
                                             unsafety,
                                             abi,
                                             vis,
                                             b.span);
                self.changes.push_str_span(s, &new_fn);
            }
            visit::FkMethod(ident, ref sig, vis) => {
                let new_fn = self.rewrite_fn(indent,
                                             ident,
                                             fd,
                                             Some(&sig.explicit_self),
                                             &sig.generics,
                                             &sig.unsafety,
                                             &sig.abi,
                                             vis.unwrap_or(ast::Visibility::Inherited),
                                             b.span);
                self.changes.push_str_span(s, &new_fn);
            }
            visit::FkFnBlock(..) => {}
        }

        self.last_pos = b.span.lo;
        self.visit_block(b)
    }

    fn visit_item(&mut self, item: &'v ast::Item) {
        if item.attrs.iter().any(|a| is_skip(&a.node.value)) {
            return;
        }

        match item.node {
            ast::Item_::ItemUse(ref vp) => {
                match vp.node {
                    ast::ViewPath_::ViewPathList(ref path, ref path_list) => {
                        self.format_missing(item.span.lo);
                        let new_str = self.rewrite_use_list(path, path_list, vp.span);
                        self.changes.push_str_span(item.span, &new_str);
                        self.last_pos = item.span.hi;
                    }
                    ast::ViewPath_::ViewPathGlob(_) => {
                        // FIXME convert to list?
                    }
                    _ => {}
                }
                visit::walk_item(self, item);
            }
            ast::Item_::ItemImpl(..) => {
                self.block_indent += TAB_SPACES;
                visit::walk_item(self, item);
                self.block_indent -= TAB_SPACES;
            }
            _ => {
                visit::walk_item(self, item);
            }
        }
    }

    fn visit_trait_item(&mut self, ti: &'v ast::TraitItem) {
        if ti.attrs.iter().any(|a| is_skip(&a.node.value)) {
            return;
        }
        visit::walk_trait_item(self, ti)
    }

    fn visit_impl_item(&mut self, ii: &'v ast::ImplItem) {
        if ii.attrs.iter().any(|a| is_skip(&a.node.value)) {
            return;
        }
        visit::walk_impl_item(self, ii)
    }

    fn visit_mac(&mut self, mac: &'v ast::Mac) {
        visit::walk_mac(self, mac)
    }

    fn visit_mod(&mut self, m: &'v ast::Mod, s: Span, _: ast::NodeId) {
        // Only visit inline mods here.
        if self.codemap.lookup_char_pos(s.lo).file.name !=
           self.codemap.lookup_char_pos(m.inner.lo).file.name {
            return;
        }
        visit::walk_mod(self, m);
    }
}

impl<'a> FmtVisitor<'a> {
    pub fn from_codemap<'b>(codemap: &'b CodeMap) -> FmtVisitor<'b> {
        FmtVisitor {
            codemap: codemap,
            changes: ChangeSet::from_codemap(codemap),
            last_pos: BytePos(0),
            block_indent: 0,
        }
    }

    pub fn snippet(&self, span: Span) -> String {
        match self.codemap.span_to_snippet(span) {
            Ok(s) => s,
            Err(_) => {
                println!("Couldn't make snippet for span {:?}->{:?}",
                         self.codemap.lookup_char_pos(span.lo),
                         self.codemap.lookup_char_pos(span.hi));
                "".to_string()
            }
        }
    }
}

fn is_skip(meta_item: &ast::MetaItem) -> bool {
    match meta_item.node {
        ast::MetaItem_::MetaWord(ref s) => *s == SKIP_ANNOTATION,
        _ => false,
    }
}
