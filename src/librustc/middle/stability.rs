// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A pass that annotates every item and method with its stability level,
//! propagating default levels lexically from parent to children ast nodes.

use session::Session;
use lint;
use middle::ty;
use metadata::csearch;
use syntax::parse::token::InternedString;
use syntax::codemap::{Span, DUMMY_SP};
use syntax::{attr, visit};
use syntax::ast;
use syntax::ast::{Attribute, Block, Crate, DefId, FnDecl, NodeId, Variant};
use syntax::ast::{Item, RequiredMethod, ProvidedMethod, TraitItem};
use syntax::ast::{TypeMethod, Method, Generics, StructField, TypeTraitItem};
use syntax::ast_util::is_local;
use syntax::attr::{Stability, AttrMetaMethods};
use syntax::visit::{FnKind, FkMethod, Visitor};
use syntax::feature_gate::emit_feature_warn;
use util::nodemap::{NodeMap, DefIdMap, FnvHashSet, FnvHashMap};
use util::ppaux::Repr;

use std::mem::replace;

/// A stability index, giving the stability level for items and methods.
pub struct Index {
    // Indicates whether this crate has #![feature(staged_api)]
    staged_api: bool,
    // stability for crate-local items; unmarked stability == no entry
    local: NodeMap<Stability>,
    // cache for extern-crate items; unmarked stability == entry with None
    extern_cache: DefIdMap<Option<Stability>>
}

// A private tree-walker for producing an Index.
struct Annotator<'a> {
    sess: &'a Session,
    index: Index,
    parent: Option<Stability>
}

impl<'a> Annotator<'a> {
    // Determine the stability for a node based on its attributes and inherited
    // stability. The stability is recorded in the index and used as the parent.
    fn annotate<F>(&mut self, id: NodeId, use_parent: bool,
                   attrs: &Vec<Attribute>, item_sp: Span, f: F) where
        F: FnOnce(&mut Annotator),
    {
        match attr::find_stability(self.sess.diagnostic(), attrs.as_slice(), item_sp) {
            Some(stab) => {
                self.index.local.insert(id, stab.clone());

                // Don't inherit #[stable(feature = "rust1", since = "1.0.0")]
                if stab.level != attr::Stable {
                    let parent = replace(&mut self.parent, Some(stab));
                    f(self);
                    self.parent = parent;
                } else {
                    f(self);
                }
            }
            None => {
                if use_parent {
                    self.parent.clone().map(|stab| self.index.local.insert(id, stab));
                }
                f(self);
            }
        }
    }
}

impl<'a, 'v> Visitor<'v> for Annotator<'a> {
    fn visit_item(&mut self, i: &Item) {
        // FIXME (#18969): the following is a hack around the fact
        // that we cannot currently annotate the stability of
        // `deriving`.  Basically, we do *not* allow stability
        // inheritance on trait implementations, so that derived
        // implementations appear to be unannotated. This then allows
        // derived implementations to be automatically tagged with the
        // stability of the trait. This is WRONG, but expedient to get
        // libstd stabilized for the 1.0 release.
        let use_parent = match i.node {
            ast::ItemImpl(_, _, _, Some(_), _, _) => false,
            _ => true,
        };

        self.annotate(i.id, use_parent, &i.attrs, i.span, |v| visit::walk_item(v, i));

        if let ast::ItemStruct(ref sd, _) = i.node {
            sd.ctor_id.map(|id| {
                self.annotate(id, true, &i.attrs, i.span, |_| {})
            });
        }
    }

    fn visit_fn(&mut self, fk: FnKind<'v>, _: &'v FnDecl,
                _: &'v Block, sp: Span, _: NodeId) {
        if let FkMethod(_, _, meth) = fk {
            // Methods are not already annotated, so we annotate it
            self.annotate(meth.id, true, &meth.attrs, sp, |_| {});
        }
        // Items defined in a function body have no reason to have
        // a stability attribute, so we don't recurse.
    }

    fn visit_trait_item(&mut self, t: &TraitItem) {
        let (id, attrs, sp) = match *t {
            RequiredMethod(TypeMethod {id, ref attrs, span, ..}) => (id, attrs, span),

            // work around lack of pattern matching for @ types
            ProvidedMethod(ref method) => {
                match **method {
                    Method {ref attrs, id, span, ..} => (id, attrs, span),
                }
            }

            TypeTraitItem(ref typedef) => (typedef.ty_param.id, &typedef.attrs,
                                           typedef.ty_param.span),
        };
        self.annotate(id, true, attrs, sp, |v| visit::walk_trait_item(v, t));
    }

    fn visit_variant(&mut self, var: &Variant, g: &'v Generics) {
        self.annotate(var.node.id, true, &var.node.attrs, var.span,
                      |v| visit::walk_variant(v, var, g))
    }

    fn visit_struct_field(&mut self, s: &StructField) {
        self.annotate(s.node.id, true, &s.node.attrs, s.span,
                      |v| visit::walk_struct_field(v, s));
    }

    fn visit_foreign_item(&mut self, i: &ast::ForeignItem) {
        self.annotate(i.id, true, &i.attrs, i.span, |_| {});
    }
}

impl Index {
    /// Construct the stability index for a crate being compiled.
    pub fn build(sess: &Session, krate: &Crate) -> Index {
        let mut staged_api = false;
        for attr in &krate.attrs {
            if attr.name().get() == "staged_api" {
                match attr.node.value.node {
                    ast::MetaWord(_) => {
                        attr::mark_used(attr);
                        staged_api = true;
                    }
                    _ => (/*pass*/)
                }
            }
        }
        let index = Index {
            staged_api: staged_api,
            local: NodeMap(),
            extern_cache: DefIdMap()
        };
        if !staged_api {
            return index;
        }
        let mut annotator = Annotator {
            sess: sess,
            index: index,
            parent: None
        };
        annotator.annotate(ast::CRATE_NODE_ID, true, &krate.attrs, krate.span,
                           |v| visit::walk_crate(v, krate));
        annotator.index
    }
}

/// Cross-references the feature names of unstable APIs with enabled
/// features and possibly prints errors. Returns a list of all
/// features used.
pub fn check_unstable_api_usage(tcx: &ty::ctxt) -> FnvHashSet<InternedString> {
    let ref active_lib_features = tcx.sess.features.borrow().lib_features;

    // Put the active features into a map for quick lookup
    let active_features = active_lib_features.iter().map(|&(ref s, _)| s.clone()).collect();

    let mut checker = Checker {
        tcx: tcx,
        active_features: active_features,
        used_features: FnvHashSet()
    };

    let krate = tcx.map.krate();
    visit::walk_crate(&mut checker, krate);

    let used_features = checker.used_features;
    return used_features;
}

struct Checker<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    active_features: FnvHashSet<InternedString>,
    used_features: FnvHashSet<InternedString>
}

impl<'a, 'tcx> Checker<'a, 'tcx> {
    fn check(&mut self, id: ast::DefId, span: Span, stab: &Option<Stability>) {
        // Only the cross-crate scenario matters when checking unstable APIs
        let cross_crate = !is_local(id);
        if !cross_crate { return }

        match *stab {
            Some(Stability { level: attr::Unstable, ref feature, ref reason, .. }) => {
                self.used_features.insert(feature.clone());

                if !self.active_features.contains(feature) {
                    let msg = match *reason {
                        Some(ref r) => format!("use of unstable library feature '{}': {}",
                                               feature.get(), r.get()),
                        None => format!("use of unstable library feature '{}'", feature.get())
                    };

                    emit_feature_warn(&self.tcx.sess.parse_sess.span_diagnostic,
                                      feature.get(), span, &msg[]);
                }
            }
            Some(..) => {
                // Stable APIs are always ok to call and deprecated APIs are
                // handled by a lint.
            }
            None => {
                // This is an 'unmarked' API, which should not exist
                // in the standard library.
                self.tcx.sess.span_err(span, "use of unmarked library feature");
                self.tcx.sess.span_note(span, "this is either a bug in the library you are \
                                               using or a bug in the compiler - there is \
                                               no way to use this feature");
            }
        }
    }
}

impl<'a, 'v, 'tcx> Visitor<'v> for Checker<'a, 'tcx> {
    fn visit_item(&mut self, item: &ast::Item) {
        check_item(self.tcx, item,
                   &mut |id, sp, stab| self.check(id, sp, stab));
        visit::walk_item(self, item);
    }

    fn visit_expr(&mut self, ex: &ast::Expr) {
        check_expr(self.tcx, ex,
                   &mut |id, sp, stab| self.check(id, sp, stab));
        visit::walk_expr(self, ex);
    }
}

/// Helper for discovering nodes to check for stability
pub fn check_item(tcx: &ty::ctxt, item: &ast::Item,
                  cb: &mut FnMut(ast::DefId, Span, &Option<Stability>)) {
    match item.node {
        ast::ItemExternCrate(_) => {
            // compiler-generated `extern crate` items have a dummy span.
            if item.span == DUMMY_SP { return }

            let cnum = match tcx.sess.cstore.find_extern_mod_stmt_cnum(item.id) {
                Some(cnum) => cnum,
                None => return,
            };
            let id = ast::DefId { krate: cnum, node: ast::CRATE_NODE_ID };
            maybe_do_stability_check(tcx, id, item.span, cb);
        }
        ast::ItemTrait(_, _, ref supertraits, _) => {
            for t in &**supertraits {
                if let ast::TraitTyParamBound(ref t, _) = *t {
                    let id = ty::trait_ref_to_def_id(tcx, &t.trait_ref);
                    maybe_do_stability_check(tcx, id, t.trait_ref.path.span, cb);
                }
            }
        }
        ast::ItemImpl(_, _, _, Some(ref t), _, _) => {
            let id = ty::trait_ref_to_def_id(tcx, t);
            maybe_do_stability_check(tcx, id, t.path.span, cb);
        }
        _ => (/* pass */)
    }
}

/// Helper for discovering nodes to check for stability
pub fn check_expr(tcx: &ty::ctxt, e: &ast::Expr,
                  cb: &mut FnMut(ast::DefId, Span, &Option<Stability>)) {
    if is_internal(tcx, e.span) { return; }

    let mut span = e.span;

    let id = match e.node {
        ast::ExprPath(..) | ast::ExprQPath(..) | ast::ExprStruct(..) => {
            match tcx.def_map.borrow().get(&e.id) {
                Some(&def) => def.def_id(),
                None => return
            }
        }
        ast::ExprMethodCall(i, _, _) => {
            span = i.span;
            let method_call = ty::MethodCall::expr(e.id);
            match tcx.method_map.borrow().get(&method_call) {
                Some(method) => {
                    match method.origin {
                        ty::MethodStatic(def_id) => {
                            def_id
                        }
                        ty::MethodStaticClosure(def_id) => {
                            def_id
                        }
                        ty::MethodTypeParam(ty::MethodParam {
                            ref trait_ref,
                            method_num: index,
                            ..
                        }) |
                        ty::MethodTraitObject(ty::MethodObject {
                            ref trait_ref,
                            method_num: index,
                            ..
                        }) => {
                            ty::trait_item(tcx, trait_ref.def_id, index).def_id()
                        }
                    }
                }
                None => return
            }
        }
        _ => return
    };

    maybe_do_stability_check(tcx, id, span, cb);
}

fn maybe_do_stability_check(tcx: &ty::ctxt, id: ast::DefId, span: Span,
                            cb: &mut FnMut(ast::DefId, Span, &Option<Stability>)) {
    if !is_staged_api(tcx, id) { return  }
    let ref stability = lookup(tcx, id);
    cb(id, span, stability);
}

fn is_internal(tcx: &ty::ctxt, span: Span) -> bool {
    tcx.sess.codemap().span_is_internal(span)
}

fn is_staged_api(tcx: &ty::ctxt, id: DefId) -> bool {
    match ty::trait_item_of_item(tcx, id) {
        Some(ty::MethodTraitItemId(trait_method_id))
            if trait_method_id != id => {
                is_staged_api(tcx, trait_method_id)
            }
        _ if is_local(id) => {
            tcx.stability.borrow().staged_api
        }
        _ => {
            csearch::is_staged_api(&tcx.sess.cstore, id)
        }
    }
}

/// Lookup the stability for a node, loading external crate
/// metadata as necessary.
pub fn lookup(tcx: &ty::ctxt, id: DefId) -> Option<Stability> {
    debug!("lookup(id={})",
           id.repr(tcx));

    // is this definition the implementation of a trait method?
    match ty::trait_item_of_item(tcx, id) {
        Some(ty::MethodTraitItemId(trait_method_id)) if trait_method_id != id => {
            debug!("lookup: trait_method_id={:?}", trait_method_id);
            return lookup(tcx, trait_method_id)
        }
        _ => {}
    }

    let item_stab = if is_local(id) {
        tcx.stability.borrow().local.get(&id.node).cloned()
    } else {
        let stab = csearch::get_stability(&tcx.sess.cstore, id);
        let mut index = tcx.stability.borrow_mut();
        (*index).extern_cache.insert(id, stab.clone());
        stab
    };

    item_stab.or_else(|| {
        if let Some(trait_id) = ty::trait_id_of_impl(tcx, id) {
            // FIXME (#18969): for the time being, simply use the
            // stability of the trait to determine the stability of any
            // unmarked impls for it. See FIXME above for more details.

            debug!("lookup: trait_id={:?}", trait_id);
            lookup(tcx, trait_id)
        } else {
            None
        }
    })
}

/// Given the list of enabled features that were not language features (i.e. that
/// were expected to be library features), and the list of features used from
/// libraries, identify activated features that don't exist and error about them.
pub fn check_unused_features(sess: &Session,
                             used_lib_features: &FnvHashSet<InternedString>) {
    let ref lib_features = sess.features.borrow().lib_features;
    let mut active_lib_features: FnvHashMap<InternedString, Span>
        = lib_features.clone().into_iter().collect();

    for used_feature in used_lib_features {
        active_lib_features.remove(used_feature);
    }

    for (_, &span) in &active_lib_features {
        sess.add_lint(lint::builtin::UNUSED_FEATURES,
                      ast::CRATE_NODE_ID,
                      span,
                      "unused or unknown feature".to_string());
    }
}
