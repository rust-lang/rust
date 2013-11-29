// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Finds items that are externally reachable, to determine which items
// need to have their metadata (and possibly their AST) serialized.
// All items that can be referred to through an exported name are
// reachable, and when a reachable thing is inline or generic, it
// makes all other generics or inline functions that it references
// reachable as well.

use middle::ty;
use middle::typeck;
use middle::privacy;

use std::hashmap::HashSet;
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::{def_id_of_def, is_local};
use syntax::attr;
use syntax::parse::token;
use syntax::visit::Visitor;
use syntax::visit;

// Returns true if the given set of attributes contains the `#[inline]`
// attribute.
fn attributes_specify_inlining(attrs: &[ast::Attribute]) -> bool {
    attr::contains_name(attrs, "inline")
}

// Returns true if the given set of generics implies that the item it's
// associated with must be inlined.
fn generics_require_inlining(generics: &ast::Generics) -> bool {
    !generics.ty_params.is_empty()
}

// Returns true if the given item must be inlined because it may be
// monomorphized or it was marked with `#[inline]`. This will only return
// true for functions.
fn item_might_be_inlined(item: @ast::item) -> bool {
    if attributes_specify_inlining(item.attrs) {
        return true
    }

    match item.node {
        ast::item_impl(ref generics, _, _, _) |
        ast::item_fn(_, _, _, ref generics, _) => {
            generics_require_inlining(generics)
        }
        _ => false,
    }
}

// Returns true if the given type method must be inlined because it may be
// monomorphized or it was marked with `#[inline]`.
fn ty_method_might_be_inlined(ty_method: &ast::TypeMethod) -> bool {
    attributes_specify_inlining(ty_method.attrs) ||
        generics_require_inlining(&ty_method.generics)
}

fn method_might_be_inlined(tcx: ty::ctxt, method: &ast::method,
                           impl_src: ast::DefId) -> bool {
    if attributes_specify_inlining(method.attrs) ||
        generics_require_inlining(&method.generics) {
        return true
    }
    if is_local(impl_src) {
        match tcx.items.find(&impl_src.node) {
            Some(&ast_map::node_item(item, _)) => item_might_be_inlined(item),
            Some(..) | None => {
                tcx.sess.span_bug(method.span, "impl did is not an item")
            }
        }
    } else {
        tcx.sess.span_bug(method.span, "found a foreign impl as a parent of a \
                                        local method")
    }
}

// Returns true if the given trait method must be inlined because it may be
// monomorphized or it was marked with `#[inline]`.
fn trait_method_might_be_inlined(trait_method: &ast::trait_method) -> bool {
    match *trait_method {
        ast::required(ref ty_method) => ty_method_might_be_inlined(ty_method),
        ast::provided(_) => true
    }
}

// Information needed while computing reachability.
struct ReachableContext {
    // The type context.
    tcx: ty::ctxt,
    // The method map, which links node IDs of method call expressions to the
    // methods they've been resolved to.
    method_map: typeck::method_map,
    // The set of items which must be exported in the linkage sense.
    reachable_symbols: @mut HashSet<ast::NodeId>,
    // A worklist of item IDs. Each item ID in this worklist will be inlined
    // and will be scanned for further references.
    worklist: @mut ~[ast::NodeId],
}

struct MarkSymbolVisitor {
    worklist: @mut ~[ast::NodeId],
    method_map: typeck::method_map,
    tcx: ty::ctxt,
    reachable_symbols: @mut HashSet<ast::NodeId>,
}

impl Visitor<()> for MarkSymbolVisitor {

    fn visit_expr(&mut self, expr:@ast::Expr, _:()) {

        match expr.node {
            ast::ExprPath(_) => {
                let def = match self.tcx.def_map.find(&expr.id) {
                    Some(&def) => def,
                    None => {
                        self.tcx.sess.span_bug(expr.span,
                                               "def ID not in def map?!")
                    }
                };

                let def_id = def_id_of_def(def);
                if ReachableContext::
                    def_id_represents_local_inlined_item(self.tcx, def_id) {
                        self.worklist.push(def_id.node)
                    }
                self.reachable_symbols.insert(def_id.node);
            }
            ast::ExprMethodCall(..) => {
                match self.method_map.find(&expr.id) {
                    Some(&typeck::method_map_entry {
                        origin: typeck::method_static(def_id),
                        ..
                    }) => {
                        if ReachableContext::
                            def_id_represents_local_inlined_item(
                                self.tcx,
                                def_id) {
                                self.worklist.push(def_id.node)
                            }
                        self.reachable_symbols.insert(def_id.node);
                    }
                    Some(_) => {}
                    None => {
                        self.tcx.sess.span_bug(expr.span,
                        "method call expression \
                                                   not in method map?!")
                    }
                }
            }
            _ => {}
        }

        visit::walk_expr(self, expr, ())
    }

    fn visit_item(&mut self, _item: @ast::item, _: ()) {
        // Do not recurse into items. These items will be added to the worklist
        // and recursed into manually if necessary.
    }
}

impl ReachableContext {
    // Creates a new reachability computation context.
    fn new(tcx: ty::ctxt, method_map: typeck::method_map) -> ReachableContext {
        ReachableContext {
            tcx: tcx,
            method_map: method_map,
            reachable_symbols: @mut HashSet::new(),
            worklist: @mut ~[],
        }
    }

    // Returns true if the given def ID represents a local item that is
    // eligible for inlining and false otherwise.
    fn def_id_represents_local_inlined_item(tcx: ty::ctxt, def_id: ast::DefId)
                                            -> bool {
        if def_id.crate != ast::LOCAL_CRATE {
            return false
        }

        let node_id = def_id.node;
        match tcx.items.find(&node_id) {
            Some(&ast_map::node_item(item, _)) => {
                match item.node {
                    ast::item_fn(..) => item_might_be_inlined(item),
                    _ => false,
                }
            }
            Some(&ast_map::node_trait_method(trait_method, _, _)) => {
                match *trait_method {
                    ast::required(_) => false,
                    ast::provided(_) => true,
                }
            }
            Some(&ast_map::node_method(method, impl_did, _)) => {
                if generics_require_inlining(&method.generics) ||
                        attributes_specify_inlining(method.attrs) {
                    true
                } else {
                    // Check the impl. If the generics on the self type of the
                    // impl require inlining, this method does too.
                    assert!(impl_did.crate == ast::LOCAL_CRATE);
                    match tcx.items.find(&impl_did.node) {
                        Some(&ast_map::node_item(item, _)) => {
                            match item.node {
                                ast::item_impl(ref generics, _, _, _) => {
                                    generics_require_inlining(generics)
                                }
                                _ => false
                            }
                        }
                        Some(_) => {
                            tcx.sess.span_bug(method.span,
                                              "method is not inside an \
                                               impl?!")
                        }
                        None => {
                            tcx.sess.span_bug(method.span,
                                              "the impl that this method is \
                                               supposedly inside of doesn't \
                                               exist in the AST map?!")
                        }
                    }
                }
            }
            Some(_) => false,
            None => false   // This will happen for default methods.
        }
    }

    // Helper function to set up a visitor for `propagate()` below.
    fn init_visitor(&self) -> MarkSymbolVisitor {
        let (worklist, method_map) = (self.worklist, self.method_map);
        let (tcx, reachable_symbols) = (self.tcx, self.reachable_symbols);

        MarkSymbolVisitor {
            worklist: worklist,
            method_map: method_map,
            tcx: tcx,
            reachable_symbols: reachable_symbols,
        }
    }

    // Step 2: Mark all symbols that the symbols on the worklist touch.
    fn propagate(&self) {
        let mut visitor = self.init_visitor();
        let mut scanned = HashSet::new();
        while self.worklist.len() > 0 {
            let search_item = self.worklist.pop();
            if scanned.contains(&search_item) {
                continue
            }
            scanned.insert(search_item);
            match self.tcx.items.find(&search_item) {
                Some(item) => self.propagate_node(item, search_item,
                                                  &mut visitor),
                None if search_item == ast::CRATE_NODE_ID => {}
                None => {
                    self.tcx.sess.bug(format!("found unmapped ID in worklist: \
                                               {}",
                                              search_item))
                }
            }
        }
    }

    fn propagate_node(&self, node: &ast_map::ast_node,
                      search_item: ast::NodeId,
                      visitor: &mut MarkSymbolVisitor) {
        if !*self.tcx.sess.building_library {
            // If we are building an executable, then there's no need to flag
            // anything as external except for `extern fn` types. These
            // functions may still participate in some form of native interface,
            // but all other rust-only interfaces can be private (they will not
            // participate in linkage after this product is produced)
            match *node {
                ast_map::node_item(item, _) => {
                    match item.node {
                        ast::item_fn(_, ast::extern_fn, _, _, _) => {
                            self.reachable_symbols.insert(search_item);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        } else {
            // If we are building a library, then reachable symbols will
            // continue to participate in linkage after this product is
            // produced. In this case, we traverse the ast node, recursing on
            // all reachable nodes from this one.
            self.reachable_symbols.insert(search_item);
        }

        match *node {
            ast_map::node_item(item, _) => {
                match item.node {
                    ast::item_fn(_, _, _, _, ref search_block) => {
                        if item_might_be_inlined(item) {
                            visit::walk_block(visitor, search_block, ())
                        }
                    }

                    // These are normal, nothing reachable about these
                    // inherently and their children are already in the
                    // worklist, as determined by the privacy pass
                    ast::item_static(..) | ast::item_ty(..) |
                    ast::item_mod(..) | ast::item_foreign_mod(..) |
                    ast::item_impl(..) | ast::item_trait(..) |
                    ast::item_struct(..) | ast::item_enum(..) => {}

                    _ => {
                        self.tcx.sess.span_bug(item.span,
                                               "found non-function item \
                                                in worklist?!")
                    }
                }
            }
            ast_map::node_trait_method(trait_method, _, _) => {
                match *trait_method {
                    ast::required(..) => {
                        // Keep going, nothing to get exported
                    }
                    ast::provided(ref method) => {
                        visit::walk_block(visitor, &method.body, ())
                    }
                }
            }
            ast_map::node_method(method, did, _) => {
                if method_might_be_inlined(self.tcx, method, did) {
                    visit::walk_block(visitor, &method.body, ())
                }
            }
            // Nothing to recurse on for these
            ast_map::node_foreign_item(..) |
            ast_map::node_variant(..) |
            ast_map::node_struct_ctor(..) => {}
            _ => {
                let ident_interner = token::get_ident_interner();
                let desc = ast_map::node_id_to_str(self.tcx.items,
                                                   search_item,
                                                   ident_interner);
                self.tcx.sess.bug(format!("found unexpected thingy in \
                                           worklist: {}",
                                           desc))
            }
        }
    }

    // Step 3: Mark all destructors as reachable.
    //
    // XXX(pcwalton): This is a conservative overapproximation, but fixing
    // this properly would result in the necessity of computing *type*
    // reachability, which might result in a compile time loss.
    fn mark_destructors_reachable(&self) {
        for (_, destructor_def_id) in self.tcx.destructor_for_type.iter() {
            if destructor_def_id.crate == ast::LOCAL_CRATE {
                self.reachable_symbols.insert(destructor_def_id.node);
            }
        }
    }
}

pub fn find_reachable(tcx: ty::ctxt,
                      method_map: typeck::method_map,
                      exported_items: &privacy::ExportedItems)
                      -> @mut HashSet<ast::NodeId> {
    let reachable_context = ReachableContext::new(tcx, method_map);

    // Step 1: Seed the worklist with all nodes which were found to be public as
    //         a result of the privacy pass
    for &id in exported_items.iter() {
        reachable_context.worklist.push(id);
    }

    // Step 2: Mark all symbols that the symbols on the worklist touch.
    reachable_context.propagate();

    // Step 3: Mark all destructors as reachable.
    reachable_context.mark_destructors_reachable();

    // Return the set of reachable symbols.
    reachable_context.reachable_symbols
}
