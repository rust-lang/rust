// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Support for inlining external documentation into the current AST.

use std::collections::HashSet;

use syntax::ast;
use syntax::attr::AttrMetaMethods;
use rustc_front::hir;

use rustc::middle::cstore::{self, CrateStore};
use rustc::middle::def;
use rustc::middle::def_id::DefId;
use rustc::middle::ty;
use rustc::middle::subst;
use rustc::middle::stability;
use rustc::middle::const_eval;

use core::DocContext;
use doctree;
use clean;

use super::{Clean, ToSource};

/// Attempt to inline the definition of a local node id into this AST.
///
/// This function will fetch the definition of the id specified, and if it is
/// from another crate it will attempt to inline the documentation from the
/// other crate into this crate.
///
/// This is primarily used for `pub use` statements which are, in general,
/// implementation details. Inlining the documentation should help provide a
/// better experience when reading the documentation in this use case.
///
/// The returned value is `None` if the `id` could not be inlined, and `Some`
/// of a vector of items if it was successfully expanded.
pub fn try_inline(cx: &DocContext, id: ast::NodeId, into: Option<ast::Name>)
                  -> Option<Vec<clean::Item>> {
    let tcx = match cx.tcx_opt() {
        Some(tcx) => tcx,
        None => return None,
    };
    let def = match tcx.def_map.borrow().get(&id) {
        Some(d) => d.full_def(),
        None => return None,
    };
    let did = def.def_id();
    if did.is_local() { return None }
    try_inline_def(cx, tcx, def).map(|vec| {
        vec.into_iter().map(|mut item| {
            match into {
                Some(into) if item.name.is_some() => {
                    item.name = Some(into.clean(cx));
                }
                _ => {}
            }
            item
        }).collect()
    })
}

fn try_inline_def(cx: &DocContext, tcx: &ty::ctxt,
                  def: def::Def) -> Option<Vec<clean::Item>> {
    let mut ret = Vec::new();
    let did = def.def_id();
    let inner = match def {
        def::DefTrait(did) => {
            record_extern_fqn(cx, did, clean::TypeTrait);
            clean::TraitItem(build_external_trait(cx, tcx, did))
        }
        def::DefFn(did, false) => {
            // If this function is a tuple struct constructor, we just skip it
            record_extern_fqn(cx, did, clean::TypeFunction);
            clean::FunctionItem(build_external_function(cx, tcx, did))
        }
        def::DefStruct(did) => {
            record_extern_fqn(cx, did, clean::TypeStruct);
            ret.extend(build_impls(cx, tcx, did));
            clean::StructItem(build_struct(cx, tcx, did))
        }
        def::DefTy(did, false) => {
            record_extern_fqn(cx, did, clean::TypeTypedef);
            ret.extend(build_impls(cx, tcx, did));
            build_type(cx, tcx, did)
        }
        def::DefTy(did, true) => {
            record_extern_fqn(cx, did, clean::TypeEnum);
            ret.extend(build_impls(cx, tcx, did));
            build_type(cx, tcx, did)
        }
        // Assume that the enum type is reexported next to the variant, and
        // variants don't show up in documentation specially.
        def::DefVariant(..) => return Some(Vec::new()),
        def::DefMod(did) => {
            record_extern_fqn(cx, did, clean::TypeModule);
            clean::ModuleItem(build_module(cx, tcx, did))
        }
        def::DefStatic(did, mtbl) => {
            record_extern_fqn(cx, did, clean::TypeStatic);
            clean::StaticItem(build_static(cx, tcx, did, mtbl))
        }
        def::DefConst(did) | def::DefAssociatedConst(did) => {
            record_extern_fqn(cx, did, clean::TypeConst);
            clean::ConstantItem(build_const(cx, tcx, did))
        }
        _ => return None,
    };
    cx.inlined.borrow_mut().as_mut().unwrap().insert(did);
    ret.push(clean::Item {
        source: clean::Span::empty(),
        name: Some(tcx.item_name(did).to_string()),
        attrs: load_attrs(cx, tcx, did),
        inner: inner,
        visibility: Some(hir::Public),
        stability: stability::lookup_stability(tcx, did).clean(cx),
        def_id: did,
    });
    Some(ret)
}

pub fn load_attrs(cx: &DocContext, tcx: &ty::ctxt,
                  did: DefId) -> Vec<clean::Attribute> {
    tcx.get_attrs(did).iter().map(|a| a.clean(cx)).collect()
}

/// Record an external fully qualified name in the external_paths cache.
///
/// These names are used later on by HTML rendering to generate things like
/// source links back to the original item.
pub fn record_extern_fqn(cx: &DocContext, did: DefId, kind: clean::TypeKind) {
    match cx.tcx_opt() {
        Some(tcx) => {
            let fqn = tcx.sess.cstore.extern_item_path(did);
            let fqn = fqn.into_iter().map(|i| i.to_string()).collect();
            cx.external_paths.borrow_mut().as_mut().unwrap().insert(did, (fqn, kind));
        }
        None => {}
    }
}

pub fn build_external_trait(cx: &DocContext, tcx: &ty::ctxt,
                            did: DefId) -> clean::Trait {
    let def = tcx.lookup_trait_def(did);
    let trait_items = tcx.trait_items(did).clean(cx);
    let predicates = tcx.lookup_predicates(did);
    let generics = (&def.generics, &predicates, subst::TypeSpace).clean(cx);
    let generics = filter_non_trait_generics(did, generics);
    let (generics, supertrait_bounds) = separate_supertrait_bounds(generics);
    clean::Trait {
        unsafety: def.unsafety,
        generics: generics,
        items: trait_items,
        bounds: supertrait_bounds,
    }
}

fn build_external_function(cx: &DocContext, tcx: &ty::ctxt, did: DefId) -> clean::Function {
    let t = tcx.lookup_item_type(did);
    let (decl, style, abi) = match t.ty.sty {
        ty::TyBareFn(_, ref f) => ((did, &f.sig).clean(cx), f.unsafety, f.abi),
        _ => panic!("bad function"),
    };

    let constness = if tcx.sess.cstore.is_const_fn(did) {
        hir::Constness::Const
    } else {
        hir::Constness::NotConst
    };

    let predicates = tcx.lookup_predicates(did);
    clean::Function {
        decl: decl,
        generics: (&t.generics, &predicates, subst::FnSpace).clean(cx),
        unsafety: style,
        constness: constness,
        abi: abi,
    }
}

fn build_struct(cx: &DocContext, tcx: &ty::ctxt, did: DefId) -> clean::Struct {
    use syntax::parse::token::special_idents::unnamed_field;

    let t = tcx.lookup_item_type(did);
    let predicates = tcx.lookup_predicates(did);
    let variant = tcx.lookup_adt_def(did).struct_variant();

    clean::Struct {
        struct_type: match &*variant.fields {
            [] => doctree::Unit,
            [ref f] if f.name == unnamed_field.name => doctree::Newtype,
            [ref f, ..] if f.name == unnamed_field.name => doctree::Tuple,
            _ => doctree::Plain,
        },
        generics: (&t.generics, &predicates, subst::TypeSpace).clean(cx),
        fields: variant.fields.clean(cx),
        fields_stripped: false,
    }
}

fn build_type(cx: &DocContext, tcx: &ty::ctxt, did: DefId) -> clean::ItemEnum {
    let t = tcx.lookup_item_type(did);
    let predicates = tcx.lookup_predicates(did);
    match t.ty.sty {
        ty::TyEnum(edef, _) if !tcx.sess.cstore.is_typedef(did) => {
            return clean::EnumItem(clean::Enum {
                generics: (&t.generics, &predicates, subst::TypeSpace).clean(cx),
                variants_stripped: false,
                variants: edef.variants.clean(cx),
            })
        }
        _ => {}
    }

    clean::TypedefItem(clean::Typedef {
        type_: t.ty.clean(cx),
        generics: (&t.generics, &predicates, subst::TypeSpace).clean(cx),
    }, false)
}

pub fn build_impls(cx: &DocContext, tcx: &ty::ctxt,
                   did: DefId) -> Vec<clean::Item> {
    tcx.populate_inherent_implementations_for_type_if_necessary(did);
    let mut impls = Vec::new();

    match tcx.inherent_impls.borrow().get(&did) {
        None => {}
        Some(i) => {
            for &did in i.iter() {
                build_impl(cx, tcx, did, &mut impls);
            }
        }
    }

    // If this is the first time we've inlined something from this crate, then
    // we inline *all* impls from the crate into this crate. Note that there's
    // currently no way for us to filter this based on type, and we likely need
    // many impls for a variety of reasons.
    //
    // Primarily, the impls will be used to populate the documentation for this
    // type being inlined, but impls can also be used when generating
    // documentation for primitives (no way to find those specifically).
    if cx.populated_crate_impls.borrow_mut().insert(did.krate) {
        for item in tcx.sess.cstore.crate_top_level_items(did.krate) {
            populate_impls(cx, tcx, item.def, &mut impls);
        }

        fn populate_impls(cx: &DocContext, tcx: &ty::ctxt,
                          def: cstore::DefLike,
                          impls: &mut Vec<clean::Item>) {
            match def {
                cstore::DlImpl(did) => build_impl(cx, tcx, did, impls),
                cstore::DlDef(def::DefMod(did)) => {
                    for item in tcx.sess.cstore.item_children(did) {
                        populate_impls(cx, tcx, item.def, impls)
                    }
                }
                _ => {}
            }
        }
    }

    return impls;
}

pub fn build_impl(cx: &DocContext,
                  tcx: &ty::ctxt,
                  did: DefId,
                  ret: &mut Vec<clean::Item>) {
    if !cx.inlined.borrow_mut().as_mut().unwrap().insert(did) {
        return
    }

    let attrs = load_attrs(cx, tcx, did);
    let associated_trait = tcx.impl_trait_ref(did);
    if let Some(ref t) = associated_trait {
        // If this is an impl for a #[doc(hidden)] trait, be sure to not inline
        let trait_attrs = load_attrs(cx, tcx, t.def_id);
        if trait_attrs.iter().any(|a| is_doc_hidden(a)) {
            return
        }
    }

    // If this is a defaulted impl, then bail out early here
    if tcx.sess.cstore.is_default_impl(did) {
        return ret.push(clean::Item {
            inner: clean::DefaultImplItem(clean::DefaultImpl {
                // FIXME: this should be decoded
                unsafety: hir::Unsafety::Normal,
                trait_: match associated_trait.as_ref().unwrap().clean(cx) {
                    clean::TraitBound(polyt, _) => polyt.trait_,
                    clean::RegionBound(..) => unreachable!(),
                },
            }),
            source: clean::Span::empty(),
            name: None,
            attrs: attrs,
            visibility: Some(hir::Inherited),
            stability: stability::lookup_stability(tcx, did).clean(cx),
            def_id: did,
        });
    }

    let predicates = tcx.lookup_predicates(did);
    let trait_items = tcx.sess.cstore.impl_items(did)
            .iter()
            .filter_map(|did| {
        let did = did.def_id();
        let impl_item = tcx.impl_or_trait_item(did);
        match impl_item {
            ty::ConstTraitItem(ref assoc_const) => {
                let did = assoc_const.def_id;
                let type_scheme = tcx.lookup_item_type(did);
                let default = if assoc_const.has_value {
                    Some(const_eval::lookup_const_by_id(tcx, did, None)
                         .unwrap().span.to_src(cx))
                } else {
                    None
                };
                Some(clean::Item {
                    name: Some(assoc_const.name.clean(cx)),
                    inner: clean::AssociatedConstItem(
                        type_scheme.ty.clean(cx),
                        default,
                    ),
                    source: clean::Span::empty(),
                    attrs: vec![],
                    visibility: None,
                    stability: stability::lookup_stability(tcx, did).clean(cx),
                    def_id: did
                })
            }
            ty::MethodTraitItem(method) => {
                if method.vis != hir::Public && associated_trait.is_none() {
                    return None
                }
                let mut item = method.clean(cx);
                item.inner = match item.inner.clone() {
                    clean::TyMethodItem(clean::TyMethod {
                        unsafety, decl, self_, generics, abi
                    }) => {
                        let constness = if tcx.sess.cstore.is_const_fn(did) {
                            hir::Constness::Const
                        } else {
                            hir::Constness::NotConst
                        };

                        clean::MethodItem(clean::Method {
                            unsafety: unsafety,
                            constness: constness,
                            decl: decl,
                            self_: self_,
                            generics: generics,
                            abi: abi
                        })
                    }
                    _ => panic!("not a tymethod"),
                };
                Some(item)
            }
            ty::TypeTraitItem(ref assoc_ty) => {
                let did = assoc_ty.def_id;
                let type_scheme = ty::TypeScheme {
                    ty: assoc_ty.ty.unwrap(),
                    generics: ty::Generics::empty()
                };
                // Not sure the choice of ParamSpace actually matters here,
                // because an associated type won't have generics on the LHS
                let typedef = (type_scheme, ty::GenericPredicates::empty(),
                               subst::ParamSpace::TypeSpace).clean(cx);
                Some(clean::Item {
                    name: Some(assoc_ty.name.clean(cx)),
                    inner: clean::TypedefItem(typedef, true),
                    source: clean::Span::empty(),
                    attrs: vec![],
                    visibility: None,
                    stability: stability::lookup_stability(tcx, did).clean(cx),
                    def_id: did
                })
            }
        }
    }).collect::<Vec<_>>();
    let polarity = tcx.trait_impl_polarity(did);
    let ty = tcx.lookup_item_type(did);
    let trait_ = associated_trait.clean(cx).map(|bound| {
        match bound {
            clean::TraitBound(polyt, _) => polyt.trait_,
            clean::RegionBound(..) => unreachable!(),
        }
    });
    if let Some(clean::ResolvedPath { did, .. }) = trait_ {
        if Some(did) == cx.deref_trait_did.get() {
            super::build_deref_target_impls(cx, &trait_items, ret);
        }
    }
    ret.push(clean::Item {
        inner: clean::ImplItem(clean::Impl {
            unsafety: hir::Unsafety::Normal, // FIXME: this should be decoded
            derived: clean::detect_derived(&attrs),
            trait_: trait_,
            for_: ty.ty.clean(cx),
            generics: (&ty.generics, &predicates, subst::TypeSpace).clean(cx),
            items: trait_items,
            polarity: polarity.map(|p| { p.clean(cx) }),
        }),
        source: clean::Span::empty(),
        name: None,
        attrs: attrs,
        visibility: Some(hir::Inherited),
        stability: stability::lookup_stability(tcx, did).clean(cx),
        def_id: did,
    });

    fn is_doc_hidden(a: &clean::Attribute) -> bool {
        match *a {
            clean::List(ref name, ref inner) if *name == "doc" => {
                inner.iter().any(|a| {
                    match *a {
                        clean::Word(ref s) => *s == "hidden",
                        _ => false,
                    }
                })
            }
            _ => false
        }
    }
}

fn build_module(cx: &DocContext, tcx: &ty::ctxt,
                did: DefId) -> clean::Module {
    let mut items = Vec::new();
    fill_in(cx, tcx, did, &mut items);
    return clean::Module {
        items: items,
        is_crate: false,
    };

    fn fill_in(cx: &DocContext, tcx: &ty::ctxt, did: DefId,
               items: &mut Vec<clean::Item>) {
        // If we're reexporting a reexport it may actually reexport something in
        // two namespaces, so the target may be listed twice. Make sure we only
        // visit each node at most once.
        let mut visited = HashSet::new();
        for item in tcx.sess.cstore.item_children(did) {
            match item.def {
                cstore::DlDef(def::DefForeignMod(did)) => {
                    fill_in(cx, tcx, did, items);
                }
                cstore::DlDef(def) if item.vis == hir::Public => {
                    if !visited.insert(def) { return }
                    match try_inline_def(cx, tcx, def) {
                        Some(i) => items.extend(i),
                        None => {}
                    }
                }
                cstore::DlDef(..) => {}
                // All impls were inlined above
                cstore::DlImpl(..) => {}
                cstore::DlField => panic!("unimplemented field"),
            }
        }
    }
}

fn build_const(cx: &DocContext, tcx: &ty::ctxt,
               did: DefId) -> clean::Constant {
    use rustc::middle::const_eval;
    use rustc_front::print::pprust;

    let expr = const_eval::lookup_const_by_id(tcx, did, None).unwrap_or_else(|| {
        panic!("expected lookup_const_by_id to succeed for {:?}", did);
    });
    debug!("converting constant expr {:?} to snippet", expr);
    let sn = pprust::expr_to_string(expr);
    debug!("got snippet {}", sn);

    clean::Constant {
        type_: tcx.lookup_item_type(did).ty.clean(cx),
        expr: sn
    }
}

fn build_static(cx: &DocContext, tcx: &ty::ctxt,
                did: DefId,
                mutable: bool) -> clean::Static {
    clean::Static {
        type_: tcx.lookup_item_type(did).ty.clean(cx),
        mutability: if mutable {clean::Mutable} else {clean::Immutable},
        expr: "\n\n\n".to_string(), // trigger the "[definition]" links
    }
}

/// A trait's generics clause actually contains all of the predicates for all of
/// its associated types as well. We specifically move these clauses to the
/// associated types instead when displaying, so when we're genering the
/// generics for the trait itself we need to be sure to remove them.
///
/// The inverse of this filtering logic can be found in the `Clean`
/// implementation for `AssociatedType`
fn filter_non_trait_generics(trait_did: DefId, mut g: clean::Generics)
                             -> clean::Generics {
    g.where_predicates.retain(|pred| {
        match *pred {
            clean::WherePredicate::BoundPredicate {
                ty: clean::QPath {
                    self_type: box clean::Generic(ref s),
                    trait_: box clean::ResolvedPath { did, .. },
                    name: ref _name,
                }, ..
            } => *s != "Self" || did != trait_did,
            _ => true,
        }
    });
    return g;
}

/// Supertrait bounds for a trait are also listed in the generics coming from
/// the metadata for a crate, so we want to separate those out and create a new
/// list of explicit supertrait bounds to render nicely.
fn separate_supertrait_bounds(mut g: clean::Generics)
                              -> (clean::Generics, Vec<clean::TyParamBound>) {
    let mut ty_bounds = Vec::new();
    g.where_predicates.retain(|pred| {
        match *pred {
            clean::WherePredicate::BoundPredicate {
                ty: clean::Generic(ref s),
                ref bounds
            } if *s == "Self" => {
                ty_bounds.extend(bounds.iter().cloned());
                false
            }
            _ => true,
        }
    });
    (g, ty_bounds)
}
