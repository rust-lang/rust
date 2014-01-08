// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This implements the dead-code warning pass. It follows middle::reachable
// closely. The idea is that all reachable symbols are live, codes called
// from live codes are live, and everything else is dead.

use middle::ty;
use middle::typeck;
use middle::privacy;
use middle::lint::dead_code;

use std::hashmap::HashSet;
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::{local_def, def_id_of_def, is_local};
use syntax::codemap;
use syntax::parse::token;
use syntax::visit::Visitor;
use syntax::visit;

// Any local node that may call something in its body block should be
// explored. For example, if it's a live node_item that is a
// function, then we should explore its block to check for codes that
// may need to be marked as live.
fn should_explore(tcx: ty::ctxt, def_id: ast::DefId) -> bool {
    if !is_local(def_id) {
        return false;
    }

    let items = tcx.items.borrow();
    match items.get().find(&def_id.node) {
        Some(&ast_map::node_item(..))
        | Some(&ast_map::node_method(..))
        | Some(&ast_map::node_foreign_item(..))
        | Some(&ast_map::node_trait_method(..)) => true,
        _ => false
    }
}

struct MarkSymbolVisitor {
    worklist: ~[ast::NodeId],
    method_map: typeck::method_map,
    tcx: ty::ctxt,
    live_symbols: ~HashSet<ast::NodeId>,
}

impl MarkSymbolVisitor {
    fn new(tcx: ty::ctxt,
           method_map: typeck::method_map,
           worklist: ~[ast::NodeId]) -> MarkSymbolVisitor {
        MarkSymbolVisitor {
            worklist: worklist,
            method_map: method_map,
            tcx: tcx,
            live_symbols: ~HashSet::new(),
        }
    }

    fn check_def_id(&mut self, def_id: ast::DefId) {
        if should_explore(self.tcx, def_id) {
            self.worklist.push(def_id.node);
        }
        self.live_symbols.insert(def_id.node);
    }

    fn lookup_and_handle_definition(&mut self, id: &ast::NodeId) {
        let def_map = self.tcx.def_map.borrow();
        let def = match def_map.get().find(id) {
            Some(&def) => def,
            None => return
        };
        let def_id = match def {
            ast::DefVariant(enum_id, _, _) => Some(enum_id),
            ast::DefPrimTy(_) => None,
            _ => Some(def_id_of_def(def)),
        };
        match def_id {
            Some(def_id) => self.check_def_id(def_id),
            None => (),
        }
    }

    fn lookup_and_handle_method(&mut self, id: &ast::NodeId,
                                span: codemap::Span) {
        let method_map = self.method_map.borrow();
        match method_map.get().find(id) {
            Some(&typeck::method_map_entry { origin, .. }) => {
                match origin {
                    typeck::method_static(def_id) => {
                        match ty::provided_source(self.tcx, def_id) {
                            Some(p_did) => self.check_def_id(p_did),
                            None => self.check_def_id(def_id)
                        }
                    }
                    typeck::method_param(typeck::method_param {
                        trait_id: trait_id,
                        method_num: index,
                        ..
                    })
                    | typeck::method_object(typeck::method_object {
                        trait_id: trait_id,
                        method_num: index,
                        ..
                    }) => {
                        let def_id = ty::trait_method(self.tcx,
                                                      trait_id, index).def_id;
                        self.check_def_id(def_id);
                    }
                }
            }
            None => {
                self.tcx.sess.span_bug(span,
                                       "method call expression not \
                                        in method map?!")
            }
        }
    }

    fn mark_live_symbols(&mut self) {
        let mut scanned = HashSet::new();
        while self.worklist.len() > 0 {
            let id = self.worklist.pop();
            if scanned.contains(&id) {
                continue
            }
            scanned.insert(id);

            let items = self.tcx.items.borrow();
            match items.get().find(&id) {
                Some(node) => {
                    self.live_symbols.insert(id);
                    self.visit_node(node);
                }
                None => (),
            }
        }
    }

    fn visit_node(&mut self, node: &ast_map::ast_node) {
        match *node {
            ast_map::node_item(item, _) => {
                match item.node {
                    ast::item_fn(..)
                    | ast::item_ty(..)
                    | ast::item_enum(..)
                    | ast::item_struct(..)
                    | ast::item_static(..) => {
                        visit::walk_item(self, item, ());
                    }
                    _ => ()
                }
            }
            ast_map::node_trait_method(trait_method, _, _) => {
                visit::walk_trait_method(self, trait_method, ());
            }
            ast_map::node_method(method, _, _) => {
                visit::walk_block(self, method.body, ());
            }
            ast_map::node_foreign_item(foreign_item, _, _, _) => {
                visit::walk_foreign_item(self, foreign_item, ());
            }
            _ => ()
        }
    }
}

impl Visitor<()> for MarkSymbolVisitor {

    fn visit_expr(&mut self, expr: &ast::Expr, _: ()) {
        match expr.node {
            ast::ExprMethodCall(..) => {
                self.lookup_and_handle_method(&expr.id, expr.span);
            }
            _ => ()
        }

        visit::walk_expr(self, expr, ())
    }

    fn visit_path(&mut self, path: &ast::Path, id: ast::NodeId, _: ()) {
        self.lookup_and_handle_definition(&id);
        visit::walk_path(self, path, ());
    }

    fn visit_item(&mut self, _item: &ast::item, _: ()) {
        // Do not recurse into items. These items will be added to the
        // worklist and recursed into manually if necessary.
    }
}

// This visitor is used to mark the implemented methods of a trait. Since we
// can not be sure if such methods are live or dead, we simply mark them
// as live.
struct TraitMethodSeeder {
    worklist: ~[ast::NodeId],
}

impl Visitor<()> for TraitMethodSeeder {
    fn visit_item(&mut self, item: &ast::item, _: ()) {
        match item.node {
            ast::item_impl(_, Some(ref _trait_ref), _, ref methods) => {
                for method in methods.iter() {
                    self.worklist.push(method.id);
                }
            }
            ast::item_mod(..) | ast::item_fn(..) => {
                visit::walk_item(self, item, ());
            }
            _ => ()
        }
    }
}

fn create_and_seed_worklist(tcx: ty::ctxt,
                            exported_items: &privacy::ExportedItems,
                            reachable_symbols: &HashSet<ast::NodeId>,
                            crate: &ast::Crate) -> ~[ast::NodeId] {
    let mut worklist = ~[];

    // Preferably, we would only need to seed the worklist with reachable
    // symbols. However, since the set of reachable symbols differs
    // depending on whether a crate is built as bin or lib, and we want
    // the warning to be consistent, we also seed the worklist with
    // exported symbols.
    for &id in exported_items.iter() {
        worklist.push(id);
    }
    for &id in reachable_symbols.iter() {
        worklist.push(id);
    }

    // Seed entry point
    match tcx.sess.entry_fn.get() {
        Some((id, _)) => worklist.push(id),
        None => ()
    }

    // Seed implemeneted trait methods
    let mut trait_method_seeder = TraitMethodSeeder {
        worklist: worklist
    };
    visit::walk_crate(&mut trait_method_seeder, crate, ());

    return trait_method_seeder.worklist;
}

fn find_live(tcx: ty::ctxt,
             method_map: typeck::method_map,
             exported_items: &privacy::ExportedItems,
             reachable_symbols: &HashSet<ast::NodeId>,
             crate: &ast::Crate)
             -> ~HashSet<ast::NodeId> {
    let worklist = create_and_seed_worklist(tcx, exported_items,
                                            reachable_symbols, crate);
    let mut symbol_visitor = MarkSymbolVisitor::new(tcx, method_map, worklist);
    symbol_visitor.mark_live_symbols();
    symbol_visitor.live_symbols
}

fn should_warn(item: &ast::item) -> bool {
    match item.node {
        ast::item_static(..)
        | ast::item_fn(..)
        | ast::item_enum(..)
        | ast::item_struct(..) => true,
        _ => false
    }
}

fn get_struct_ctor_id(item: &ast::item) -> Option<ast::NodeId> {
    match item.node {
        ast::item_struct(struct_def, _) => struct_def.ctor_id,
        _ => None
    }
}

struct DeadVisitor {
    tcx: ty::ctxt,
    live_symbols: ~HashSet<ast::NodeId>,
}

impl DeadVisitor {
    // id := node id of an item's definition.
    // ctor_id := `Some` if the item is a struct_ctor (tuple struct),
    //            `None` otherwise.
    // If the item is a struct_ctor, then either its `id` or
    // `ctor_id` (unwrapped) is in the live_symbols set. More specifically,
    // DefMap maps the ExprPath of a struct_ctor to the node referred by
    // `ctor_id`. On the other hand, in a statement like
    // `type <ident> <generics> = <ty>;` where <ty> refers to a struct_ctor,
    // DefMap maps <ty> to `id` instead.
    fn symbol_is_live(&mut self, id: ast::NodeId,
                      ctor_id: Option<ast::NodeId>) -> bool {
        if self.live_symbols.contains(&id)
           || ctor_id.map_or(false,
                             |ctor| self.live_symbols.contains(&ctor)) {
            return true;
        }
        // If it's a type whose methods are live, then it's live, too.
        // This is done to handle the case where, for example, the static
        // method of a private type is used, but the type itself is never
        // called directly.
        let def_id = local_def(id);
        let inherent_impls = self.tcx.inherent_impls.borrow();
        match inherent_impls.get().find(&def_id) {
            None => (),
            Some(ref impl_list) => {
                let impl_list = impl_list.borrow();
                for impl_ in impl_list.get().iter() {
                    for method in impl_.methods.iter() {
                        if self.live_symbols.contains(&method.def_id.node) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn warn_dead_code(&mut self, id: ast::NodeId,
                      span: codemap::Span, ident: &ast::Ident) {
        self.tcx.sess.add_lint(dead_code, id, span,
                               format!("code is never used: `{}`",
                                       token::ident_to_str(ident)));
    }
}

impl Visitor<()> for DeadVisitor {
    fn visit_item(&mut self, item: &ast::item, _: ()) {
        let ctor_id = get_struct_ctor_id(item);
        if !self.symbol_is_live(item.id, ctor_id) && should_warn(item) {
            self.warn_dead_code(item.id, item.span, &item.ident);
        }
        visit::walk_item(self, item, ());
    }

    fn visit_foreign_item(&mut self, fi: &ast::foreign_item, _: ()) {
        if !self.symbol_is_live(fi.id, None) {
            self.warn_dead_code(fi.id, fi.span, &fi.ident);
        }
        visit::walk_foreign_item(self, fi, ());
    }

    fn visit_fn(&mut self, fk: &visit::fn_kind,
                _: &ast::fn_decl, block: &ast::Block,
                span: codemap::Span, id: ast::NodeId, _: ()) {
        // Have to warn method here because methods are not ast::item
        match *fk {
            visit::fk_method(..) => {
                let ident = visit::name_of_fn(fk);
                if !self.symbol_is_live(id, None) {
                    self.warn_dead_code(id, span, &ident);
                }
            }
            _ => ()
        }
        visit::walk_block(self, block, ());
    }

    // Overwrite so that we don't warn the trait method itself.
    fn visit_trait_method(&mut self, trait_method: &ast::trait_method, _: ()) {
        match *trait_method {
            ast::provided(method) => visit::walk_block(self, method.body, ()),
            ast::required(_) => ()
        }
    }
}

pub fn check_crate(tcx: ty::ctxt,
                   method_map: typeck::method_map,
                   exported_items: &privacy::ExportedItems,
                   reachable_symbols: &HashSet<ast::NodeId>,
                   crate: &ast::Crate) {
    let live_symbols = find_live(tcx, method_map, exported_items,
                                 reachable_symbols, crate);
    let mut visitor = DeadVisitor { tcx: tcx, live_symbols: live_symbols };
    visit::walk_crate(&mut visitor, crate, ());
}
