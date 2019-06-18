//! Support for inlining external documentation into the current AST.

use std::iter::once;

use syntax::ast;
use syntax::ext::base::MacroKind;
use syntax::symbol::sym;
use syntax_pos::Span;

use rustc::hir;
use rustc::hir::def::{Res, DefKind, CtorKind};
use rustc::hir::def_id::DefId;
use rustc_metadata::cstore::LoadedMacro;
use rustc::ty;
use rustc::util::nodemap::FxHashSet;

use crate::core::{DocContext, DocAccessLevels};
use crate::doctree;
use crate::clean::{
    self,
    GetDefId,
    ToSource,
};

use super::Clean;

/// Attempt to inline a definition into this AST.
///
/// This function will fetch the definition specified, and if it is
/// from another crate it will attempt to inline the documentation
/// from the other crate into this crate.
///
/// This is primarily used for `pub use` statements which are, in general,
/// implementation details. Inlining the documentation should help provide a
/// better experience when reading the documentation in this use case.
///
/// The returned value is `None` if the definition could not be inlined,
/// and `Some` of a vector of items if it was successfully expanded.
pub fn try_inline(
    cx: &DocContext<'_>,
    res: Res,
    name: ast::Name,
    visited: &mut FxHashSet<DefId>
) -> Option<Vec<clean::Item>> {
    let did = if let Some(did) = res.opt_def_id() {
        did
    } else {
        return None;
    };
    if did.is_local() { return None }
    let mut ret = Vec::new();
    let inner = match res {
        Res::Def(DefKind::Trait, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Trait);
            ret.extend(build_impls(cx, did));
            clean::TraitItem(build_external_trait(cx, did))
        }
        Res::Def(DefKind::Fn, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Function);
            clean::FunctionItem(build_external_function(cx, did))
        }
        Res::Def(DefKind::Struct, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Struct);
            ret.extend(build_impls(cx, did));
            clean::StructItem(build_struct(cx, did))
        }
        Res::Def(DefKind::Union, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Union);
            ret.extend(build_impls(cx, did));
            clean::UnionItem(build_union(cx, did))
        }
        Res::Def(DefKind::TyAlias, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Typedef);
            ret.extend(build_impls(cx, did));
            clean::TypedefItem(build_type_alias(cx, did), false)
        }
        Res::Def(DefKind::Enum, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Enum);
            ret.extend(build_impls(cx, did));
            clean::EnumItem(build_enum(cx, did))
        }
        Res::Def(DefKind::ForeignTy, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Foreign);
            ret.extend(build_impls(cx, did));
            clean::ForeignTypeItem
        }
        // Never inline enum variants but leave them shown as re-exports.
        Res::Def(DefKind::Variant, _) => return None,
        // Assume that enum variants and struct types are re-exported next to
        // their constructors.
        Res::Def(DefKind::Ctor(..), _) | Res::SelfCtor(..) => return Some(Vec::new()),
        Res::Def(DefKind::Mod, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Module);
            clean::ModuleItem(build_module(cx, did, visited))
        }
        Res::Def(DefKind::Static, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Static);
            clean::StaticItem(build_static(cx, did, cx.tcx.is_mutable_static(did)))
        }
        Res::Def(DefKind::Const, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Const);
            clean::ConstantItem(build_const(cx, did))
        }
        // FIXME: proc-macros don't propagate attributes or spans across crates, so they look empty
        Res::Def(DefKind::Macro(MacroKind::Bang), did) => {
            let mac = build_macro(cx, did, name);
            if let clean::MacroItem(..) = mac {
                record_extern_fqn(cx, did, clean::TypeKind::Macro);
                mac
            } else {
                return None;
            }
        }
        _ => return None,
    };
    cx.renderinfo.borrow_mut().inlined.insert(did);
    ret.push(clean::Item {
        source: cx.tcx.def_span(did).clean(cx),
        name: Some(name.clean(cx)),
        attrs: load_attrs(cx, did),
        inner,
        visibility: Some(clean::Public),
        stability: cx.tcx.lookup_stability(did).clean(cx),
        deprecation: cx.tcx.lookup_deprecation(did).clean(cx),
        def_id: did,
    });
    Some(ret)
}

pub fn try_inline_glob(cx: &DocContext<'_>, res: Res, visited: &mut FxHashSet<DefId>)
    -> Option<Vec<clean::Item>>
{
    if res == Res::Err { return None }
    let did = res.def_id();
    if did.is_local() { return None }

    match res {
        Res::Def(DefKind::Mod, did) => {
            let m = build_module(cx, did, visited);
            Some(m.items)
        }
        // glob imports on things like enums aren't inlined even for local exports, so just bail
        _ => None,
    }
}

pub fn load_attrs(cx: &DocContext<'_>, did: DefId) -> clean::Attributes {
    cx.tcx.get_attrs(did).clean(cx)
}

/// Record an external fully qualified name in the external_paths cache.
///
/// These names are used later on by HTML rendering to generate things like
/// source links back to the original item.
pub fn record_extern_fqn(cx: &DocContext<'_>, did: DefId, kind: clean::TypeKind) {
    let mut crate_name = cx.tcx.crate_name(did.krate).to_string();
    if did.is_local() {
        crate_name = cx.crate_name.clone().unwrap_or(crate_name);
    }

    let relative = cx.tcx.def_path(did).data.into_iter().filter_map(|elem| {
        // extern blocks have an empty name
        let s = elem.data.to_string();
        if !s.is_empty() {
            Some(s)
        } else {
            None
        }
    });
    let fqn = if let clean::TypeKind::Macro = kind {
        vec![crate_name, relative.last().expect("relative was empty")]
    } else {
        once(crate_name).chain(relative).collect()
    };

    if did.is_local() {
        cx.renderinfo.borrow_mut().exact_paths.insert(did, fqn);
    } else {
        cx.renderinfo.borrow_mut().external_paths.insert(did, (fqn, kind));
    }
}

pub fn build_external_trait(cx: &DocContext<'_>, did: DefId) -> clean::Trait {
    let auto_trait = cx.tcx.trait_def(did).has_auto_impl;
    let trait_items = cx.tcx.associated_items(did).map(|item| item.clean(cx)).collect();
    let predicates = cx.tcx.predicates_of(did);
    let generics = (cx.tcx.generics_of(did), &predicates).clean(cx);
    let generics = filter_non_trait_generics(did, generics);
    let (generics, supertrait_bounds) = separate_supertrait_bounds(generics);
    let is_spotlight = load_attrs(cx, did).has_doc_flag(sym::spotlight);
    let is_auto = cx.tcx.trait_is_auto(did);
    clean::Trait {
        auto: auto_trait,
        unsafety: cx.tcx.trait_def(did).unsafety,
        generics,
        items: trait_items,
        bounds: supertrait_bounds,
        is_spotlight,
        is_auto,
    }
}

fn build_external_function(cx: &DocContext<'_>, did: DefId) -> clean::Function {
    let sig = cx.tcx.fn_sig(did);

    let constness = if cx.tcx.is_min_const_fn(did) {
        hir::Constness::Const
    } else {
        hir::Constness::NotConst
    };

    let predicates = cx.tcx.predicates_of(did);
    let generics = (cx.tcx.generics_of(did), &predicates).clean(cx);
    let decl = (did, sig).clean(cx);
    let (all_types, ret_types) = clean::get_all_types(&generics, &decl, cx);
    clean::Function {
        decl,
        generics,
        header: hir::FnHeader {
            unsafety: sig.unsafety(),
            abi: sig.abi(),
            constness,
            asyncness: hir::IsAsync::NotAsync,
        },
        all_types,
        ret_types,
    }
}

fn build_enum(cx: &DocContext<'_>, did: DefId) -> clean::Enum {
    let predicates = cx.tcx.explicit_predicates_of(did);

    clean::Enum {
        generics: (cx.tcx.generics_of(did), &predicates).clean(cx),
        variants_stripped: false,
        variants: cx.tcx.adt_def(did).variants.clean(cx),
    }
}

fn build_struct(cx: &DocContext<'_>, did: DefId) -> clean::Struct {
    let predicates = cx.tcx.explicit_predicates_of(did);
    let variant = cx.tcx.adt_def(did).non_enum_variant();

    clean::Struct {
        struct_type: match variant.ctor_kind {
            CtorKind::Fictive => doctree::Plain,
            CtorKind::Fn => doctree::Tuple,
            CtorKind::Const => doctree::Unit,
        },
        generics: (cx.tcx.generics_of(did), &predicates).clean(cx),
        fields: variant.fields.clean(cx),
        fields_stripped: false,
    }
}

fn build_union(cx: &DocContext<'_>, did: DefId) -> clean::Union {
    let predicates = cx.tcx.explicit_predicates_of(did);
    let variant = cx.tcx.adt_def(did).non_enum_variant();

    clean::Union {
        struct_type: doctree::Plain,
        generics: (cx.tcx.generics_of(did), &predicates).clean(cx),
        fields: variant.fields.clean(cx),
        fields_stripped: false,
    }
}

fn build_type_alias(cx: &DocContext<'_>, did: DefId) -> clean::Typedef {
    let predicates = cx.tcx.explicit_predicates_of(did);

    clean::Typedef {
        type_: cx.tcx.type_of(did).clean(cx),
        generics: (cx.tcx.generics_of(did), &predicates).clean(cx),
    }
}

pub fn build_impls(cx: &DocContext<'_>, did: DefId) -> Vec<clean::Item> {
    let tcx = cx.tcx;
    let mut impls = Vec::new();

    for &did in tcx.inherent_impls(did).iter() {
        build_impl(cx, did, &mut impls);
    }

    impls
}

pub fn build_impl(cx: &DocContext<'_>, did: DefId, ret: &mut Vec<clean::Item>) {
    if !cx.renderinfo.borrow_mut().inlined.insert(did) {
        return
    }

    let attrs = load_attrs(cx, did);
    let tcx = cx.tcx;
    let associated_trait = tcx.impl_trait_ref(did);

    // Only inline impl if the implemented trait is
    // reachable in rustdoc generated documentation
    if !did.is_local() {
        if let Some(traitref) = associated_trait {
            if !cx.renderinfo.borrow().access_levels.is_doc_reachable(traitref.def_id) {
                return
            }
        }
    }

    let for_ = if let Some(hir_id) = tcx.hir().as_local_hir_id(did) {
        match tcx.hir().expect_item(hir_id).node {
            hir::ItemKind::Impl(.., ref t, _) => {
                t.clean(cx)
            }
            _ => panic!("did given to build_impl was not an impl"),
        }
    } else {
        tcx.type_of(did).clean(cx)
    };

    // Only inline impl if the implementing type is
    // reachable in rustdoc generated documentation
    if !did.is_local() {
        if let Some(did) = for_.def_id() {
            if !cx.renderinfo.borrow().access_levels.is_doc_reachable(did) {
                return
            }
        }
    }

    let predicates = tcx.explicit_predicates_of(did);
    let (trait_items, generics) = if let Some(hir_id) = tcx.hir().as_local_hir_id(did) {
        match tcx.hir().expect_item(hir_id).node {
            hir::ItemKind::Impl(.., ref gen, _, _, ref item_ids) => {
                (
                    item_ids.iter()
                            .map(|ii| tcx.hir().impl_item(ii.id).clean(cx))
                            .collect::<Vec<_>>(),
                    gen.clean(cx),
                )
            }
            _ => panic!("did given to build_impl was not an impl"),
        }
    } else {
        (
            tcx.associated_items(did).filter_map(|item| {
                if associated_trait.is_some() || item.vis == ty::Visibility::Public {
                    Some(item.clean(cx))
                } else {
                    None
                }
            }).collect::<Vec<_>>(),
            (tcx.generics_of(did), &predicates).clean(cx),
        )
    };
    let polarity = tcx.impl_polarity(did);
    let trait_ = associated_trait.clean(cx).map(|bound| {
        match bound {
            clean::GenericBound::TraitBound(polyt, _) => polyt.trait_,
            clean::GenericBound::Outlives(..) => unreachable!(),
        }
    });
    if trait_.def_id() == tcx.lang_items().deref_trait() {
        super::build_deref_target_impls(cx, &trait_items, ret);
    }
    if let Some(trait_did) = trait_.def_id() {
        record_extern_trait(cx, trait_did);
    }

    let provided = trait_.def_id().map(|did| {
        tcx.provided_trait_methods(did)
           .into_iter()
           .map(|meth| meth.ident.to_string())
           .collect()
    }).unwrap_or_default();

    debug!("build_impl: impl {:?} for {:?}", trait_.def_id(), for_.def_id());

    ret.push(clean::Item {
        inner: clean::ImplItem(clean::Impl {
            unsafety: hir::Unsafety::Normal,
            generics,
            provided_trait_methods: provided,
            trait_,
            for_,
            items: trait_items,
            polarity: Some(polarity.clean(cx)),
            synthetic: false,
            blanket_impl: None,
        }),
        source: tcx.def_span(did).clean(cx),
        name: None,
        attrs,
        visibility: Some(clean::Inherited),
        stability: tcx.lookup_stability(did).clean(cx),
        deprecation: tcx.lookup_deprecation(did).clean(cx),
        def_id: did,
    });
}

fn build_module(
    cx: &DocContext<'_>,
    did: DefId,
    visited: &mut FxHashSet<DefId>
) -> clean::Module {
    let mut items = Vec::new();
    fill_in(cx, did, &mut items, visited);
    return clean::Module {
        items,
        is_crate: false,
    };

    fn fill_in(cx: &DocContext<'_>, did: DefId, items: &mut Vec<clean::Item>,
               visited: &mut FxHashSet<DefId>) {
        // If we're re-exporting a re-export it may actually re-export something in
        // two namespaces, so the target may be listed twice. Make sure we only
        // visit each node at most once.
        for &item in cx.tcx.item_children(did).iter() {
            let def_id = item.res.def_id();
            if item.vis == ty::Visibility::Public {
                if did == def_id || !visited.insert(def_id) { continue }
                if let Some(i) = try_inline(cx, item.res, item.ident.name, visited) {
                    items.extend(i)
                }
            }
        }
    }
}

pub fn print_inlined_const(cx: &DocContext<'_>, did: DefId) -> String {
    if let Some(node_id) = cx.tcx.hir().as_local_hir_id(did) {
        cx.tcx.hir().hir_to_pretty_string(node_id)
    } else {
        cx.tcx.rendered_const(did)
    }
}

fn build_const(cx: &DocContext<'_>, did: DefId) -> clean::Constant {
    clean::Constant {
        type_: cx.tcx.type_of(did).clean(cx),
        expr: print_inlined_const(cx, did)
    }
}

fn build_static(cx: &DocContext<'_>, did: DefId, mutable: bool) -> clean::Static {
    clean::Static {
        type_: cx.tcx.type_of(did).clean(cx),
        mutability: if mutable {clean::Mutable} else {clean::Immutable},
        expr: "\n\n\n".to_string(), // trigger the "[definition]" links
    }
}

fn build_macro(cx: &DocContext<'_>, did: DefId, name: ast::Name) -> clean::ItemEnum {
    let imported_from = cx.tcx.original_crate_name(did.krate);
    match cx.cstore.load_macro_untracked(did, cx.sess()) {
        LoadedMacro::MacroDef(def) => {
            let matchers: hir::HirVec<Span> = if let ast::ItemKind::MacroDef(ref def) = def.node {
                let tts: Vec<_> = def.stream().into_trees().collect();
                tts.chunks(4).map(|arm| arm[0].span()).collect()
            } else {
                unreachable!()
            };

            let source = format!("macro_rules! {} {{\n{}}}",
                                 name.clean(cx),
                                 matchers.iter().map(|span| {
                                     format!("    {} => {{ ... }};\n", span.to_src(cx))
                                 }).collect::<String>());

            clean::MacroItem(clean::Macro {
                source,
                imported_from: Some(imported_from).clean(cx),
            })
        }
        LoadedMacro::ProcMacro(ext) => {
            clean::ProcMacroItem(clean::ProcMacro {
                kind: ext.macro_kind(),
                helpers: ext.helper_attrs.clean(cx),
            })
        }
    }
}

/// A trait's generics clause actually contains all of the predicates for all of
/// its associated types as well. We specifically move these clauses to the
/// associated types instead when displaying, so when we're generating the
/// generics for the trait itself we need to be sure to remove them.
/// We also need to remove the implied "recursive" Self: Trait bound.
///
/// The inverse of this filtering logic can be found in the `Clean`
/// implementation for `AssociatedType`
fn filter_non_trait_generics(trait_did: DefId, mut g: clean::Generics) -> clean::Generics {
    for pred in &mut g.where_predicates {
        match *pred {
            clean::WherePredicate::BoundPredicate {
                ty: clean::Generic(ref s),
                ref mut bounds
            } if *s == "Self" => {
                bounds.retain(|bound| {
                    match *bound {
                        clean::GenericBound::TraitBound(clean::PolyTrait {
                            trait_: clean::ResolvedPath { did, .. },
                            ..
                        }, _) => did != trait_did,
                        _ => true
                    }
                });
            }
            _ => {}
        }
    }

    g.where_predicates.retain(|pred| {
        match *pred {
            clean::WherePredicate::BoundPredicate {
                ty: clean::QPath {
                    self_type: box clean::Generic(ref s),
                    trait_: box clean::ResolvedPath { did, .. },
                    name: ref _name,
                }, ref bounds
            } => !(*s == "Self" && did == trait_did) && !bounds.is_empty(),
            _ => true,
        }
    });
    g
}

/// Supertrait bounds for a trait are also listed in the generics coming from
/// the metadata for a crate, so we want to separate those out and create a new
/// list of explicit supertrait bounds to render nicely.
fn separate_supertrait_bounds(mut g: clean::Generics)
                              -> (clean::Generics, Vec<clean::GenericBound>) {
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

pub fn record_extern_trait(cx: &DocContext<'_>, did: DefId) {
    if did.is_local() {
        return;
    }

    {
        let external_traits = cx.external_traits.lock();
        if external_traits.borrow().contains_key(&did) ||
            cx.active_extern_traits.borrow().contains(&did)
        {
            return;
        }
    }

    cx.active_extern_traits.borrow_mut().push(did);

    debug!("record_extern_trait: {:?}", did);
    let trait_ = build_external_trait(cx, did);

    {
        let external_traits = cx.external_traits.lock();
        external_traits.borrow_mut().insert(did, trait_);
    }
    cx.active_extern_traits.borrow_mut().remove_item(&did);
}
