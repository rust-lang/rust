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

use syntax::ast;
use syntax::ast_util;
use syntax::attr::AttrMetaMethods;

use rustc::metadata::csearch;
use rustc::metadata::decoder;
use rustc::middle::def;
use rustc::middle::ty;
use rustc::middle::subst;
use rustc::middle::stability;

use core::DocContext;
use doctree;
use clean;

use super::Clean;

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
pub fn try_inline(cx: &DocContext, id: ast::NodeId, into: Option<ast::Ident>)
                  -> Option<Vec<clean::Item>> {
    let tcx = match cx.tcx_opt() {
        Some(tcx) => tcx,
        None => return None,
    };
    let def = match tcx.def_map.borrow().find(&id) {
        Some(def) => *def,
        None => return None,
    };
    let did = def.def_id();
    if ast_util::is_local(did) { return None }
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
        def::DefFn(did, style) => {
            // If this function is a tuple struct constructor, we just skip it
            if csearch::get_tuple_struct_definition_if_ctor(&tcx.sess.cstore,
                                                            did).is_some() {
                return None
            }
            record_extern_fqn(cx, did, clean::TypeFunction);
            clean::FunctionItem(build_external_function(cx, tcx, did, style))
        }
        def::DefStruct(did) => {
            record_extern_fqn(cx, did, clean::TypeStruct);
            ret.extend(build_impls(cx, tcx, did).into_iter());
            clean::StructItem(build_struct(cx, tcx, did))
        }
        def::DefTy(did, false) => {
            record_extern_fqn(cx, did, clean::TypeTypedef);
            ret.extend(build_impls(cx, tcx, did).into_iter());
            build_type(cx, tcx, did)
        }
        def::DefTy(did, true) => {
            record_extern_fqn(cx, did, clean::TypeEnum);
            ret.extend(build_impls(cx, tcx, did).into_iter());
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
        _ => return None,
    };
    let fqn = csearch::get_item_path(tcx, did);
    cx.inlined.borrow_mut().as_mut().unwrap().insert(did);
    ret.push(clean::Item {
        source: clean::Span::empty(),
        name: Some(fqn.last().unwrap().to_string()),
        attrs: load_attrs(cx, tcx, did),
        inner: inner,
        visibility: Some(ast::Public),
        stability: stability::lookup(tcx, did).clean(cx),
        def_id: did,
    });
    Some(ret)
}

pub fn load_attrs(cx: &DocContext, tcx: &ty::ctxt,
                  did: ast::DefId) -> Vec<clean::Attribute> {
    let mut attrs = Vec::new();
    csearch::get_item_attrs(&tcx.sess.cstore, did, |v| {
        attrs.extend(v.into_iter().map(|a| {
            a.clean(cx)
        }));
    });
    attrs
}

/// Record an external fully qualified name in the external_paths cache.
///
/// These names are used later on by HTML rendering to generate things like
/// source links back to the original item.
pub fn record_extern_fqn(cx: &DocContext, did: ast::DefId, kind: clean::TypeKind) {
    match cx.tcx_opt() {
        Some(tcx) => {
            let fqn = csearch::get_item_path(tcx, did);
            let fqn = fqn.into_iter().map(|i| i.to_string()).collect();
            cx.external_paths.borrow_mut().as_mut().unwrap().insert(did, (fqn, kind));
        }
        None => {}
    }
}

pub fn build_external_trait(cx: &DocContext, tcx: &ty::ctxt,
                            did: ast::DefId) -> clean::Trait {
    let def = ty::lookup_trait_def(tcx, did);
    let trait_items = ty::trait_items(tcx, did).clean(cx);
    let provided = ty::provided_trait_methods(tcx, did);
    let mut items = trait_items.into_iter().map(|trait_item| {
        if provided.iter().any(|a| a.def_id == trait_item.def_id) {
            clean::ProvidedMethod(trait_item)
        } else {
            clean::RequiredMethod(trait_item)
        }
    });
    let trait_def = ty::lookup_trait_def(tcx, did);
    let bounds = trait_def.bounds.clean(cx);
    clean::Trait {
        generics: (&def.generics, subst::TypeSpace).clean(cx),
        items: items.collect(),
        bounds: bounds,
    }
}

fn build_external_function(cx: &DocContext, tcx: &ty::ctxt,
                           did: ast::DefId,
                           style: ast::FnStyle) -> clean::Function {
    let t = ty::lookup_item_type(tcx, did);
    clean::Function {
        decl: match ty::get(t.ty).sty {
            ty::ty_bare_fn(ref f) => (did, &f.sig).clean(cx),
            _ => fail!("bad function"),
        },
        generics: (&t.generics, subst::FnSpace).clean(cx),
        fn_style: style,
    }
}

fn build_struct(cx: &DocContext, tcx: &ty::ctxt, did: ast::DefId) -> clean::Struct {
    use syntax::parse::token::special_idents::unnamed_field;

    let t = ty::lookup_item_type(tcx, did);
    let fields = ty::lookup_struct_fields(tcx, did);

    clean::Struct {
        struct_type: match fields.as_slice() {
            [] => doctree::Unit,
            [ref f] if f.name == unnamed_field.name => doctree::Newtype,
            [ref f, ..] if f.name == unnamed_field.name => doctree::Tuple,
            _ => doctree::Plain,
        },
        generics: (&t.generics, subst::TypeSpace).clean(cx),
        fields: fields.clean(cx),
        fields_stripped: false,
    }
}

fn build_type(cx: &DocContext, tcx: &ty::ctxt, did: ast::DefId) -> clean::ItemEnum {
    let t = ty::lookup_item_type(tcx, did);
    match ty::get(t.ty).sty {
        ty::ty_enum(edid, _) if !csearch::is_typedef(&tcx.sess.cstore, did) => {
            return clean::EnumItem(clean::Enum {
                generics: (&t.generics, subst::TypeSpace).clean(cx),
                variants_stripped: false,
                variants: ty::enum_variants(tcx, edid).clean(cx),
            })
        }
        _ => {}
    }

    clean::TypedefItem(clean::Typedef {
        type_: t.ty.clean(cx),
        generics: (&t.generics, subst::TypeSpace).clean(cx),
    })
}

fn build_impls(cx: &DocContext, tcx: &ty::ctxt,
               did: ast::DefId) -> Vec<clean::Item> {
    ty::populate_implementations_for_type_if_necessary(tcx, did);
    let mut impls = Vec::new();

    match tcx.inherent_impls.borrow().find(&did) {
        None => {}
        Some(i) => {
            impls.extend(i.iter().map(|&did| { build_impl(cx, tcx, did) }));
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
        csearch::each_top_level_item_of_crate(&tcx.sess.cstore,
                                              did.krate,
                                              |def, _, _| {
            populate_impls(cx, tcx, def, &mut impls)
        });

        fn populate_impls(cx: &DocContext, tcx: &ty::ctxt,
                          def: decoder::DefLike,
                          impls: &mut Vec<Option<clean::Item>>) {
            match def {
                decoder::DlImpl(did) => impls.push(build_impl(cx, tcx, did)),
                decoder::DlDef(def::DefMod(did)) => {
                    csearch::each_child_of_item(&tcx.sess.cstore,
                                                did,
                                                |def, _, _| {
                        populate_impls(cx, tcx, def, impls)
                    })
                }
                _ => {}
            }
        }
    }

    impls.into_iter().filter_map(|a| a).collect()
}

fn build_impl(cx: &DocContext, tcx: &ty::ctxt,
              did: ast::DefId) -> Option<clean::Item> {
    if !cx.inlined.borrow_mut().as_mut().unwrap().insert(did) {
        return None
    }

    let associated_trait = csearch::get_impl_trait(tcx, did);
    // If this is an impl for a #[doc(hidden)] trait, be sure to not inline it.
    match associated_trait {
        Some(ref t) => {
            let trait_attrs = load_attrs(cx, tcx, t.def_id);
            if trait_attrs.iter().any(|a| is_doc_hidden(a)) {
                return None
            }
        }
        None => {}
    }

    let attrs = load_attrs(cx, tcx, did);
    let ty = ty::lookup_item_type(tcx, did);
    let trait_items = csearch::get_impl_items(&tcx.sess.cstore, did)
            .iter()
            .filter_map(|did| {
        let did = did.def_id();
        let impl_item = ty::impl_or_trait_item(tcx, did);
        match impl_item {
            ty::MethodTraitItem(method) => {
                if method.vis != ast::Public && associated_trait.is_none() {
                    return None
                }
                let mut item = method.clean(cx);
                item.inner = match item.inner.clone() {
                    clean::TyMethodItem(clean::TyMethod {
                        fn_style, decl, self_, generics
                    }) => {
                        clean::MethodItem(clean::Method {
                            fn_style: fn_style,
                            decl: decl,
                            self_: self_,
                            generics: generics,
                        })
                    }
                    _ => fail!("not a tymethod"),
                };
                Some(item)
            }
        }
    }).collect();
    return Some(clean::Item {
        inner: clean::ImplItem(clean::Impl {
            derived: clean::detect_derived(attrs.as_slice()),
            trait_: associated_trait.clean(cx).map(|bound| {
                match bound {
                    clean::TraitBound(ty) => ty,
                    clean::RegionBound => unreachable!(),
                }
            }),
            for_: ty.ty.clean(cx),
            generics: (&ty.generics, subst::TypeSpace).clean(cx),
            items: trait_items,
        }),
        source: clean::Span::empty(),
        name: None,
        attrs: attrs,
        visibility: Some(ast::Inherited),
        stability: stability::lookup(tcx, did).clean(cx),
        def_id: did,
    });

    fn is_doc_hidden(a: &clean::Attribute) -> bool {
        match *a {
            clean::List(ref name, ref inner) if name.as_slice() == "doc" => {
                inner.iter().any(|a| {
                    match *a {
                        clean::Word(ref s) => s.as_slice() == "hidden",
                        _ => false,
                    }
                })
            }
            _ => false
        }
    }
}

fn build_module(cx: &DocContext, tcx: &ty::ctxt,
                did: ast::DefId) -> clean::Module {
    let mut items = Vec::new();
    fill_in(cx, tcx, did, &mut items);
    return clean::Module {
        items: items,
        is_crate: false,
    };

    // FIXME: this doesn't handle reexports inside the module itself.
    //        Should they be handled?
    fn fill_in(cx: &DocContext, tcx: &ty::ctxt, did: ast::DefId,
               items: &mut Vec<clean::Item>) {
        csearch::each_child_of_item(&tcx.sess.cstore, did, |def, _, vis| {
            match def {
                decoder::DlDef(def::DefForeignMod(did)) => {
                    fill_in(cx, tcx, did, items);
                }
                decoder::DlDef(def) if vis == ast::Public => {
                    match try_inline_def(cx, tcx, def) {
                        Some(i) => items.extend(i.into_iter()),
                        None => {}
                    }
                }
                decoder::DlDef(..) => {}
                // All impls were inlined above
                decoder::DlImpl(..) => {}
                decoder::DlField => fail!("unimplemented field"),
            }
        });
    }
}

fn build_static(cx: &DocContext, tcx: &ty::ctxt,
                did: ast::DefId,
                mutable: bool) -> clean::Static {
    clean::Static {
        type_: ty::lookup_item_type(tcx, did).ty.clean(cx),
        mutability: if mutable {clean::Mutable} else {clean::Immutable},
        expr: "\n\n\n".to_string(), // trigger the "[definition]" links
    }
}
