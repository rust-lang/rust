// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This compiler pass detects constants that refer to themselves
// recursively.

use rustc::dep_graph::DepNode;
use rustc::hir::map as ast_map;
use rustc::session::{CompileResult, Session};
use rustc::hir::def::{Def, CtorKind};
use rustc::util::nodemap::{NodeMap, NodeSet};

use syntax::ast;
use syntax::feature_gate::{GateIssue, emit_feature_err};
use syntax_pos::Span;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir;

struct CheckCrateVisitor<'a, 'ast: 'a> {
    sess: &'a Session,
    ast_map: &'a ast_map::Map<'ast>,
    // `discriminant_map` is a cache that associates the `NodeId`s of local
    // variant definitions with the discriminant expression that applies to
    // each one. If the variant uses the default values (starting from `0`),
    // then `None` is stored.
    discriminant_map: NodeMap<Option<hir::BodyId>>,
    detected_recursive_ids: NodeSet,
}

impl<'a, 'ast: 'a> Visitor<'ast> for CheckCrateVisitor<'a, 'ast> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'ast> {
        NestedVisitorMap::None
    }

    fn visit_item(&mut self, it: &'ast hir::Item) {
        match it.node {
            hir::ItemStatic(..) |
            hir::ItemConst(..) => {
                let mut recursion_visitor = CheckItemRecursionVisitor::new(self, &it.span);
                recursion_visitor.visit_item(it);
            }
            hir::ItemEnum(ref enum_def, ref generics) => {
                // We could process the whole enum, but handling the variants
                // with discriminant expressions one by one gives more specific,
                // less redundant output.
                for variant in &enum_def.variants {
                    if let Some(_) = variant.node.disr_expr {
                        let mut recursion_visitor = CheckItemRecursionVisitor::new(self,
                                                                                   &variant.span);
                        recursion_visitor.populate_enum_discriminants(enum_def);
                        recursion_visitor.visit_variant(variant, generics, it.id);
                    }
                }
            }
            _ => {}
        }
        intravisit::walk_item(self, it)
    }

    fn visit_trait_item(&mut self, ti: &'ast hir::TraitItem) {
        match ti.node {
            hir::TraitItemKind::Const(_, ref default) => {
                if let Some(_) = *default {
                    let mut recursion_visitor = CheckItemRecursionVisitor::new(self, &ti.span);
                    recursion_visitor.visit_trait_item(ti);
                }
            }
            _ => {}
        }
        intravisit::walk_trait_item(self, ti)
    }

    fn visit_impl_item(&mut self, ii: &'ast hir::ImplItem) {
        match ii.node {
            hir::ImplItemKind::Const(..) => {
                let mut recursion_visitor = CheckItemRecursionVisitor::new(self, &ii.span);
                recursion_visitor.visit_impl_item(ii);
            }
            _ => {}
        }
        intravisit::walk_impl_item(self, ii)
    }
}

pub fn check_crate<'ast>(sess: &Session, ast_map: &ast_map::Map<'ast>) -> CompileResult {
    let _task = ast_map.dep_graph.in_task(DepNode::CheckStaticRecursion);

    let mut visitor = CheckCrateVisitor {
        sess: sess,
        ast_map: ast_map,
        discriminant_map: NodeMap(),
        detected_recursive_ids: NodeSet(),
    };
    sess.track_errors(|| {
        // FIXME(#37712) could use ItemLikeVisitor if trait items were item-like
        ast_map.krate().visit_all_item_likes(&mut visitor.as_deep_visitor());
    })
}

struct CheckItemRecursionVisitor<'a, 'b: 'a, 'ast: 'b> {
    root_span: &'b Span,
    sess: &'b Session,
    ast_map: &'b ast_map::Map<'ast>,
    discriminant_map: &'a mut NodeMap<Option<hir::BodyId>>,
    idstack: Vec<ast::NodeId>,
    detected_recursive_ids: &'a mut NodeSet,
}

impl<'a, 'b: 'a, 'ast: 'b> CheckItemRecursionVisitor<'a, 'b, 'ast> {
    fn new(v: &'a mut CheckCrateVisitor<'b, 'ast>, span: &'b Span) -> Self {
        CheckItemRecursionVisitor {
            root_span: span,
            sess: v.sess,
            ast_map: v.ast_map,
            discriminant_map: &mut v.discriminant_map,
            idstack: Vec::new(),
            detected_recursive_ids: &mut v.detected_recursive_ids,
        }
    }
    fn with_item_id_pushed<F>(&mut self, id: ast::NodeId, f: F, span: Span)
        where F: Fn(&mut Self)
    {
        if self.idstack.iter().any(|&x| x == id) {
            if self.detected_recursive_ids.contains(&id) {
                return;
            }
            self.detected_recursive_ids.insert(id);
            let any_static = self.idstack.iter().any(|&x| {
                if let ast_map::NodeItem(item) = self.ast_map.get(x) {
                    if let hir::ItemStatic(..) = item.node {
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            });
            if any_static {
                if !self.sess.features.borrow().static_recursion {
                    emit_feature_err(&self.sess.parse_sess,
                                     "static_recursion",
                                     *self.root_span,
                                     GateIssue::Language,
                                     "recursive static");
                }
            } else {
                struct_span_err!(self.sess, span, E0265, "recursive constant")
                    .span_label(span, &format!("recursion not allowed in constant"))
                    .emit();
            }
            return;
        }
        self.idstack.push(id);
        f(self);
        self.idstack.pop();
    }
    // If a variant has an expression specifying its discriminant, then it needs
    // to be checked just like a static or constant. However, if there are more
    // variants with no explicitly specified discriminant, those variants will
    // increment the same expression to get their values.
    //
    // So for every variant, we need to track whether there is an expression
    // somewhere in the enum definition that controls its discriminant. We do
    // this by starting from the end and searching backward.
    fn populate_enum_discriminants(&mut self, enum_definition: &'ast hir::EnumDef) {
        // Get the map, and return if we already processed this enum or if it
        // has no variants.
        match enum_definition.variants.first() {
            None => {
                return;
            }
            Some(variant) if self.discriminant_map.contains_key(&variant.node.data.id()) => {
                return;
            }
            _ => {}
        }

        // Go through all the variants.
        let mut variant_stack: Vec<ast::NodeId> = Vec::new();
        for variant in enum_definition.variants.iter().rev() {
            variant_stack.push(variant.node.data.id());
            // When we find an expression, every variant currently on the stack
            // is affected by that expression.
            if let Some(expr) = variant.node.disr_expr {
                for id in &variant_stack {
                    self.discriminant_map.insert(*id, Some(expr));
                }
                variant_stack.clear()
            }
        }
        // If we are at the top, that always starts at 0, so any variant on the
        // stack has a default value and does not need to be checked.
        for id in &variant_stack {
            self.discriminant_map.insert(*id, None);
        }
    }
}

impl<'a, 'b: 'a, 'ast: 'b> Visitor<'ast> for CheckItemRecursionVisitor<'a, 'b, 'ast> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'ast> {
        NestedVisitorMap::OnlyBodies(&self.ast_map)
    }
    fn visit_item(&mut self, it: &'ast hir::Item) {
        self.with_item_id_pushed(it.id, |v| intravisit::walk_item(v, it), it.span);
    }

    fn visit_enum_def(&mut self,
                      enum_definition: &'ast hir::EnumDef,
                      generics: &'ast hir::Generics,
                      item_id: ast::NodeId,
                      _: Span) {
        self.populate_enum_discriminants(enum_definition);
        intravisit::walk_enum_def(self, enum_definition, generics, item_id);
    }

    fn visit_variant(&mut self,
                     variant: &'ast hir::Variant,
                     _: &'ast hir::Generics,
                     _: ast::NodeId) {
        let variant_id = variant.node.data.id();
        let maybe_expr = *self.discriminant_map.get(&variant_id).unwrap_or_else(|| {
            span_bug!(variant.span,
                      "`check_static_recursion` attempted to visit \
                      variant with unknown discriminant")
        });
        // If `maybe_expr` is `None`, that's because no discriminant is
        // specified that affects this variant. Thus, no risk of recursion.
        if let Some(expr) = maybe_expr {
            let expr = &self.ast_map.body(expr).value;
            self.with_item_id_pushed(expr.id, |v| intravisit::walk_expr(v, expr), expr.span);
        }
    }

    fn visit_trait_item(&mut self, ti: &'ast hir::TraitItem) {
        self.with_item_id_pushed(ti.id, |v| intravisit::walk_trait_item(v, ti), ti.span);
    }

    fn visit_impl_item(&mut self, ii: &'ast hir::ImplItem) {
        self.with_item_id_pushed(ii.id, |v| intravisit::walk_impl_item(v, ii), ii.span);
    }

    fn visit_path(&mut self, path: &'ast hir::Path, _: ast::NodeId) {
        match path.def {
            Def::Static(def_id, _) |
            Def::AssociatedConst(def_id) |
            Def::Const(def_id) => {
                if let Some(node_id) = self.ast_map.as_local_node_id(def_id) {
                    match self.ast_map.get(node_id) {
                        ast_map::NodeItem(item) => self.visit_item(item),
                        ast_map::NodeTraitItem(item) => self.visit_trait_item(item),
                        ast_map::NodeImplItem(item) => self.visit_impl_item(item),
                        ast_map::NodeForeignItem(_) => {}
                        _ => {
                            span_bug!(path.span,
                                      "expected item, found {}",
                                      self.ast_map.node_to_string(node_id));
                        }
                    }
                }
            }
            // For variants, we only want to check expressions that
            // affect the specific variant used, but we need to check
            // the whole enum definition to see what expression that
            // might be (if any).
            Def::VariantCtor(variant_id, CtorKind::Const) => {
                if let Some(variant_id) = self.ast_map.as_local_node_id(variant_id) {
                    let variant = self.ast_map.expect_variant(variant_id);
                    let enum_id = self.ast_map.get_parent(variant_id);
                    let enum_item = self.ast_map.expect_item(enum_id);
                    if let hir::ItemEnum(ref enum_def, ref generics) = enum_item.node {
                        self.populate_enum_discriminants(enum_def);
                        self.visit_variant(variant, generics, enum_id);
                    } else {
                        span_bug!(path.span,
                                  "`check_static_recursion` found \
                                    non-enum in Def::VariantCtor");
                    }
                }
            }
            _ => (),
        }
        intravisit::walk_path(self, path);
    }
}
