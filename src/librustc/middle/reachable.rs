// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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

use std::iterator::IteratorUtil;

use middle::ty;
use middle::typeck;

use std::hashmap::HashSet;
use syntax::ast::*;
use syntax::ast_map;
use syntax::ast_util::def_id_of_def;
use syntax::attr;
use syntax::parse::token;
use syntax::visit::Visitor;
use syntax::visit;

// Returns true if the given set of attributes contains the `#[inline]`
// attribute.
fn attributes_specify_inlining(attrs: &[attribute]) -> bool {
    attr::attrs_contains_name(attrs, "inline")
}

// Returns true if the given set of generics implies that the item it's
// associated with must be inlined.
fn generics_require_inlining(generics: &Generics) -> bool {
    !generics.ty_params.is_empty()
}

// Returns true if the given item must be inlined because it may be
// monomorphized or it was marked with `#[inline]`. This will only return
// true for functions.
fn item_might_be_inlined(item: @item) -> bool {
    if attributes_specify_inlining(item.attrs) {
        return true
    }

    match item.node {
        item_fn(_, _, _, ref generics, _) => {
            generics_require_inlining(generics)
        }
        _ => false,
    }
}

// Returns true if the given type method must be inlined because it may be
// monomorphized or it was marked with `#[inline]`.
fn ty_method_might_be_inlined(ty_method: &ty_method) -> bool {
    attributes_specify_inlining(ty_method.attrs) ||
        generics_require_inlining(&ty_method.generics)
}

// Returns true if the given trait method must be inlined because it may be
// monomorphized or it was marked with `#[inline]`.
fn trait_method_might_be_inlined(trait_method: &trait_method) -> bool {
    match *trait_method {
        required(ref ty_method) => ty_method_might_be_inlined(ty_method),
        provided(_) => true
    }
}

// The context we're in. If we're in a public context, then public symbols are
// marked reachable. If we're in a private context, then only trait
// implementations are marked reachable.
#[deriving(Eq)]
enum PrivacyContext {
    PublicContext,
    PrivateContext,
}

// Information needed while computing reachability.
struct ReachableContext {
    // The type context.
    tcx: ty::ctxt,
    // The method map, which links node IDs of method call expressions to the
    // methods they've been resolved to.
    method_map: typeck::method_map,
    // The set of items which must be exported in the linkage sense.
    reachable_symbols: @mut HashSet<node_id>,
    // A worklist of item IDs. Each item ID in this worklist will be inlined
    // and will be scanned for further references.
    worklist: @mut ~[node_id],
}

impl ReachableContext {
    // Creates a new reachability computation context.
    fn new(tcx: ty::ctxt, method_map: typeck::method_map)
           -> ReachableContext {
        ReachableContext {
            tcx: tcx,
            method_map: method_map,
            reachable_symbols: @mut HashSet::new(),
            worklist: @mut ~[],
        }
    }

    // Step 1: Mark all public symbols, and add all public symbols that might
    // be inlined to a worklist.
    fn mark_public_symbols(&self, crate: @crate) {
        let reachable_symbols = self.reachable_symbols;
        let worklist = self.worklist;
        let visitor = visit::mk_vt(@Visitor {
            visit_item: |item, (privacy_context, visitor):
                    (PrivacyContext, visit::vt<PrivacyContext>)| {
                match item.node {
                    item_fn(*) => {
                        if privacy_context == PublicContext {
                            reachable_symbols.insert(item.id);
                        }
                        if item_might_be_inlined(item) {
                            worklist.push(item.id)
                        }
                    }
                    item_struct(ref struct_def, _) => {
                        match struct_def.ctor_id {
                            Some(ctor_id) if
                                    privacy_context == PublicContext => {
                                reachable_symbols.insert(ctor_id);
                            }
                            Some(_) | None => {}
                        }
                    }
                    item_enum(ref enum_def, _) => {
                        if privacy_context == PublicContext {
                            for enum_def.variants.iter().advance |variant| {
                                reachable_symbols.insert(variant.node.id);
                            }
                        }
                    }
                    item_impl(ref generics, ref trait_ref, _, ref methods) => {
                        // XXX(pcwalton): We conservatively assume any methods
                        // on a trait implementation are reachable, when this
                        // is not the case. We could be more precise by only
                        // treating implementations of reachable or cross-
                        // crate traits as reachable.

                        let should_be_considered_public = |method: @method| {
                            (method.vis == public &&
                                    privacy_context == PublicContext) ||
                                    trait_ref.is_some()
                        };

                        // Mark all public methods as reachable.
                        for methods.iter().advance |&method| {
                            if should_be_considered_public(method) {
                                reachable_symbols.insert(method.id);
                            }
                        }

                        if generics_require_inlining(generics) {
                            // If the impl itself has generics, add all public
                            // symbols to the worklist.
                            for methods.iter().advance |&method| {
                                if should_be_considered_public(method) {
                                    worklist.push(method.id)
                                }
                            }
                        } else {
                            // Otherwise, add only public methods that have
                            // generics to the worklist.
                            for methods.iter().advance |method| {
                                let generics = &method.generics;
                                let attrs = &method.attrs;
                                if generics_require_inlining(generics) ||
                                        attributes_specify_inlining(*attrs) ||
                                        should_be_considered_public(*method) {
                                    worklist.push(method.id)
                                }
                            }
                        }
                    }
                    item_trait(_, _, ref trait_methods) => {
                        // Mark all provided methods as reachable.
                        if privacy_context == PublicContext {
                            for trait_methods.iter().advance |trait_method| {
                                match *trait_method {
                                    provided(method) => {
                                        reachable_symbols.insert(method.id);
                                        worklist.push(method.id)
                                    }
                                    required(_) => {}
                                }
                            }
                        }
                    }
                    _ => {}
                }

                if item.vis == public && privacy_context == PublicContext {
                    visit::visit_item(item, (PublicContext, visitor))
                } else {
                    visit::visit_item(item, (PrivateContext, visitor))
                }
            },
            .. *visit::default_visitor()
        });

        visit::visit_crate(crate, (PublicContext, visitor))
    }

    // Returns true if the given def ID represents a local item that is
    // eligible for inlining and false otherwise.
    fn def_id_represents_local_inlined_item(tcx: ty::ctxt, def_id: def_id)
                                            -> bool {
        if def_id.crate != local_crate {
            return false
        }

        let node_id = def_id.node;
        match tcx.items.find(&node_id) {
            Some(&ast_map::node_item(item, _)) => {
                match item.node {
                    item_fn(*) => item_might_be_inlined(item),
                    _ => false,
                }
            }
            Some(&ast_map::node_trait_method(trait_method, _, _)) => {
                match *trait_method {
                    required(_) => false,
                    provided(_) => true,
                }
            }
            Some(&ast_map::node_method(method, impl_did, _)) => {
                if generics_require_inlining(&method.generics) ||
                        attributes_specify_inlining(method.attrs) {
                    true
                } else {
                    // Check the impl. If the generics on the self type of the
                    // impl require inlining, this method does too.
                    assert!(impl_did.crate == local_crate);
                    match tcx.items.find(&impl_did.node) {
                        Some(&ast_map::node_item(item, _)) => {
                            match item.node {
                                item_impl(ref generics, _, _, _) => {
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
    fn init_visitor(&self) -> visit::vt<()> {
        let (worklist, method_map) = (self.worklist, self.method_map);
        let (tcx, reachable_symbols) = (self.tcx, self.reachable_symbols);
        visit::mk_vt(@visit::Visitor {
            visit_expr: |expr, (_, visitor)| {
                match expr.node {
                    expr_path(_) => {
                        let def = match tcx.def_map.find(&expr.id) {
                            Some(&def) => def,
                            None => {
                                tcx.sess.span_bug(expr.span,
                                                  "def ID not in def map?!")
                            }
                        };

                        let def_id = def_id_of_def(def);
                        if ReachableContext::
                                def_id_represents_local_inlined_item(tcx,
                                                                     def_id) {
                            worklist.push(def_id.node)
                        }
                        reachable_symbols.insert(def_id.node);
                    }
                    expr_method_call(*) => {
                        match method_map.find(&expr.id) {
                            Some(&typeck::method_map_entry {
                                origin: typeck::method_static(def_id),
                                _
                            }) => {
                                if ReachableContext::
                                    def_id_represents_local_inlined_item(
                                        tcx,
                                        def_id) {
                                    worklist.push(def_id.node)
                                }
                                reachable_symbols.insert(def_id.node);
                            }
                            Some(_) => {}
                            None => {
                                tcx.sess.span_bug(expr.span,
                                                  "method call expression \
                                                   not in method map?!")
                            }
                        }
                    }
                    _ => {}
                }

                visit::visit_expr(expr, ((), visitor))
            },
            ..*visit::default_visitor()
        })
    }

    // Step 2: Mark all symbols that the symbols on the worklist touch.
    fn propagate(&self) {
        let visitor = self.init_visitor();
        let mut scanned = HashSet::new();
        while self.worklist.len() > 0 {
            let search_item = self.worklist.pop();
            if scanned.contains(&search_item) {
                loop
            }
            scanned.insert(search_item);
            self.reachable_symbols.insert(search_item);

            // Find the AST block corresponding to the item and visit it,
            // marking all path expressions that resolve to something
            // interesting.
            match self.tcx.items.find(&search_item) {
                Some(&ast_map::node_item(item, _)) => {
                    match item.node {
                        item_fn(_, _, _, _, ref search_block) => {
                            visit::visit_block(search_block, ((), visitor))
                        }
                        _ => {
                            self.tcx.sess.span_bug(item.span,
                                                   "found non-function item \
                                                    in worklist?!")
                        }
                    }
                }
                Some(&ast_map::node_trait_method(trait_method, _, _)) => {
                    match *trait_method {
                        required(ref ty_method) => {
                            self.tcx.sess.span_bug(ty_method.span,
                                                   "found required method in \
                                                    worklist?!")
                        }
                        provided(ref method) => {
                            visit::visit_block(&method.body, ((), visitor))
                        }
                    }
                }
                Some(&ast_map::node_method(ref method, _, _)) => {
                    visit::visit_block(&method.body, ((), visitor))
                }
                Some(_) => {
                    let ident_interner = token::get_ident_interner();
                    let desc = ast_map::node_id_to_str(self.tcx.items,
                                                       search_item,
                                                       ident_interner);
                    self.tcx.sess.bug(fmt!("found unexpected thingy in \
                                            worklist: %s",
                                            desc))
                }
                None => {
                    self.tcx.sess.bug(fmt!("found unmapped ID in worklist: \
                                            %d",
                                           search_item))
                }
            }
        }
    }

    // Step 3: Mark all destructors as reachable.
    //
    // XXX(pcwalton): This is a conservative overapproximation, but fixing
    // this properly would result in the necessity of computing *type*
    // reachability, which might result in a compile time loss.
    fn mark_destructors_reachable(&self) {
        for self.tcx.destructor_for_type.iter().advance
                |(_, destructor_def_id)| {
            if destructor_def_id.crate == local_crate {
                self.reachable_symbols.insert(destructor_def_id.node);
            }
        }
    }
}

pub fn find_reachable(tcx: ty::ctxt,
                      method_map: typeck::method_map,
                      crate: @crate)
                      -> @mut HashSet<node_id> {
    // XXX(pcwalton): We only need to mark symbols that are exported. But this
    // is more complicated than just looking at whether the symbol is `pub`,
    // because it might be the target of a `pub use` somewhere. For now, I
    // think we are fine, because you can't `pub use` something that wasn't
    // exported due to the bug whereby `use` only looks through public
    // modules even if you're inside the module the `use` appears in. When
    // this bug is fixed, however, this code will need to be updated. Probably
    // the easiest way to fix this (although a conservative overapproximation)
    // is to have the name resolution pass mark all targets of a `pub use` as
    // "must be reachable".

    let reachable_context = ReachableContext::new(tcx, method_map);

    // Step 1: Mark all public symbols, and add all public symbols that might
    // be inlined to a worklist.
    reachable_context.mark_public_symbols(crate);

    // Step 2: Mark all symbols that the symbols on the worklist touch.
    reachable_context.propagate();

    // Step 3: Mark all destructors as reachable.
    reachable_context.mark_destructors_reachable();

    // Return the set of reachable symbols.
    reachable_context.reachable_symbols
}

