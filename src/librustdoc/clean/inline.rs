//! Support for inlining external documentation into the current AST.

use std::iter::once;

use rustc_ast::ast;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::Mutability;
use rustc_metadata::creader::LoadedMacro;
use rustc_middle::ty;
use rustc_mir::const_eval::is_min_const_fn;
use rustc_span::hygiene::MacroKind;
use rustc_span::Span;

use crate::clean::{self, GetDefId, ToSource, TypeKind};
use crate::core::DocContext;
use crate::doctree;

use super::Clean;

type Attrs<'hir> = rustc_middle::ty::Attributes<'hir>;

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
    attrs: Option<Attrs<'_>>,
    visited: &mut FxHashSet<DefId>,
) -> Option<Vec<clean::Item>> {
    let did = res.opt_def_id()?;
    if did.is_local() {
        return None;
    }
    let mut ret = Vec::new();

    let attrs_clone = attrs.clone();

    let inner = match res {
        Res::Def(DefKind::Trait, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Trait);
            ret.extend(build_impls(cx, did, attrs));
            clean::TraitItem(build_external_trait(cx, did))
        }
        Res::Def(DefKind::Fn, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Function);
            clean::FunctionItem(build_external_function(cx, did))
        }
        Res::Def(DefKind::Struct, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Struct);
            ret.extend(build_impls(cx, did, attrs));
            clean::StructItem(build_struct(cx, did))
        }
        Res::Def(DefKind::Union, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Union);
            ret.extend(build_impls(cx, did, attrs));
            clean::UnionItem(build_union(cx, did))
        }
        Res::Def(DefKind::TyAlias, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Typedef);
            ret.extend(build_impls(cx, did, attrs));
            clean::TypedefItem(build_type_alias(cx, did), false)
        }
        Res::Def(DefKind::Enum, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Enum);
            ret.extend(build_impls(cx, did, attrs));
            clean::EnumItem(build_enum(cx, did))
        }
        Res::Def(DefKind::ForeignTy, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Foreign);
            ret.extend(build_impls(cx, did, attrs));
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
        Res::Def(DefKind::Macro(kind), did) => {
            let mac = build_macro(cx, did, name);

            let type_kind = match kind {
                MacroKind::Bang => TypeKind::Macro,
                MacroKind::Attr => TypeKind::Attr,
                MacroKind::Derive => TypeKind::Derive,
            };
            record_extern_fqn(cx, did, type_kind);
            mac
        }
        _ => return None,
    };

    let target_attrs = load_attrs(cx, did);
    let attrs = merge_attrs(cx, target_attrs, attrs_clone);

    cx.renderinfo.borrow_mut().inlined.insert(did);
    ret.push(clean::Item {
        source: cx.tcx.def_span(did).clean(cx),
        name: Some(name.clean(cx)),
        attrs,
        inner,
        visibility: clean::Public,
        stability: cx.tcx.lookup_stability(did).clean(cx),
        deprecation: cx.tcx.lookup_deprecation(did).clean(cx),
        def_id: did,
    });
    Some(ret)
}

pub fn try_inline_glob(
    cx: &DocContext<'_>,
    res: Res,
    visited: &mut FxHashSet<DefId>,
) -> Option<Vec<clean::Item>> {
    if res == Res::Err {
        return None;
    }
    let did = res.def_id();
    if did.is_local() {
        return None;
    }

    match res {
        Res::Def(DefKind::Mod, did) => {
            let m = build_module(cx, did, visited);
            Some(m.items)
        }
        // glob imports on things like enums aren't inlined even for local exports, so just bail
        _ => None,
    }
}

pub fn load_attrs<'hir>(cx: &DocContext<'hir>, did: DefId) -> Attrs<'hir> {
    cx.tcx.get_attrs(did)
}

/// Record an external fully qualified name in the external_paths cache.
///
/// These names are used later on by HTML rendering to generate things like
/// source links back to the original item.
pub fn record_extern_fqn(cx: &DocContext<'_>, did: DefId, kind: clean::TypeKind) {
    let crate_name = cx.tcx.crate_name(did.krate).to_string();

    let relative = cx.tcx.def_path(did).data.into_iter().filter_map(|elem| {
        // extern blocks have an empty name
        let s = elem.data.to_string();
        if !s.is_empty() { Some(s) } else { None }
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
    let trait_items =
        cx.tcx.associated_items(did).in_definition_order().map(|item| item.clean(cx)).collect();

    let auto_trait = cx.tcx.trait_def(did).has_auto_impl;
    let predicates = cx.tcx.predicates_of(did);
    let generics = (cx.tcx.generics_of(did), predicates).clean(cx);
    let generics = filter_non_trait_generics(did, generics);
    let (generics, supertrait_bounds) = separate_supertrait_bounds(generics);
    let is_auto = cx.tcx.trait_is_auto(did);
    clean::Trait {
        auto: auto_trait,
        unsafety: cx.tcx.trait_def(did).unsafety,
        generics,
        items: trait_items,
        bounds: supertrait_bounds,
        is_auto,
    }
}

fn build_external_function(cx: &DocContext<'_>, did: DefId) -> clean::Function {
    let sig = cx.tcx.fn_sig(did);

    let constness =
        if is_min_const_fn(cx.tcx, did) { hir::Constness::Const } else { hir::Constness::NotConst };
    let asyncness = cx.tcx.asyncness(did);
    let predicates = cx.tcx.predicates_of(did);
    let (generics, decl) = clean::enter_impl_trait(cx, || {
        ((cx.tcx.generics_of(did), predicates).clean(cx), (did, sig).clean(cx))
    });
    let (all_types, ret_types) = clean::get_all_types(&generics, &decl, cx);
    clean::Function {
        decl,
        generics,
        header: hir::FnHeader { unsafety: sig.unsafety(), abi: sig.abi(), constness, asyncness },
        all_types,
        ret_types,
    }
}

fn build_enum(cx: &DocContext<'_>, did: DefId) -> clean::Enum {
    let predicates = cx.tcx.explicit_predicates_of(did);

    clean::Enum {
        generics: (cx.tcx.generics_of(did), predicates).clean(cx),
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
        generics: (cx.tcx.generics_of(did), predicates).clean(cx),
        fields: variant.fields.clean(cx),
        fields_stripped: false,
    }
}

fn build_union(cx: &DocContext<'_>, did: DefId) -> clean::Union {
    let predicates = cx.tcx.explicit_predicates_of(did);
    let variant = cx.tcx.adt_def(did).non_enum_variant();

    clean::Union {
        struct_type: doctree::Plain,
        generics: (cx.tcx.generics_of(did), predicates).clean(cx),
        fields: variant.fields.clean(cx),
        fields_stripped: false,
    }
}

fn build_type_alias(cx: &DocContext<'_>, did: DefId) -> clean::Typedef {
    let predicates = cx.tcx.explicit_predicates_of(did);

    clean::Typedef {
        type_: cx.tcx.type_of(did).clean(cx),
        generics: (cx.tcx.generics_of(did), predicates).clean(cx),
        item_type: build_type_alias_type(cx, did),
    }
}

fn build_type_alias_type(cx: &DocContext<'_>, did: DefId) -> Option<clean::Type> {
    let type_ = cx.tcx.type_of(did).clean(cx);
    type_.def_id().and_then(|did| build_ty(cx, did))
}

pub fn build_ty(cx: &DocContext, did: DefId) -> Option<clean::Type> {
    match cx.tcx.def_kind(did)? {
        DefKind::Struct | DefKind::Union | DefKind::Enum | DefKind::Const | DefKind::Static => {
            Some(cx.tcx.type_of(did).clean(cx))
        }
        DefKind::TyAlias => build_type_alias_type(cx, did),
        _ => None,
    }
}

pub fn build_impls(cx: &DocContext<'_>, did: DefId, attrs: Option<Attrs<'_>>) -> Vec<clean::Item> {
    let tcx = cx.tcx;
    let mut impls = Vec::new();

    for &did in tcx.inherent_impls(did).iter() {
        build_impl(cx, did, attrs.clone(), &mut impls);
    }

    impls
}

fn merge_attrs(
    cx: &DocContext<'_>,
    attrs: Attrs<'_>,
    other_attrs: Option<Attrs<'_>>,
) -> clean::Attributes {
    let mut merged_attrs: Vec<ast::Attribute> = Vec::with_capacity(attrs.len());
    // If we have additional attributes (from a re-export),
    // always insert them first. This ensure that re-export
    // doc comments show up before the original doc comments
    // when we render them.
    if let Some(a) = other_attrs {
        merged_attrs.extend(a.iter().cloned());
    }
    merged_attrs.extend(attrs.to_vec());
    merged_attrs.clean(cx)
}

pub fn build_impl(
    cx: &DocContext<'_>,
    did: DefId,
    attrs: Option<Attrs<'_>>,
    ret: &mut Vec<clean::Item>,
) {
    if !cx.renderinfo.borrow_mut().inlined.insert(did) {
        return;
    }

    let attrs = merge_attrs(cx, load_attrs(cx, did), attrs);

    let tcx = cx.tcx;
    let associated_trait = tcx.impl_trait_ref(did);

    // Only inline impl if the implemented trait is
    // reachable in rustdoc generated documentation
    if !did.is_local() {
        if let Some(traitref) = associated_trait {
            if !cx.renderinfo.borrow().access_levels.is_public(traitref.def_id) {
                return;
            }
        }
    }

    let for_ = if let Some(hir_id) = tcx.hir().as_local_hir_id(did) {
        match tcx.hir().expect_item(hir_id).kind {
            hir::ItemKind::Impl { self_ty, .. } => self_ty.clean(cx),
            _ => panic!("did given to build_impl was not an impl"),
        }
    } else {
        tcx.type_of(did).clean(cx)
    };

    // Only inline impl if the implementing type is
    // reachable in rustdoc generated documentation
    if !did.is_local() {
        if let Some(did) = for_.def_id() {
            if !cx.renderinfo.borrow().access_levels.is_public(did) {
                return;
            }
        }
    }

    let predicates = tcx.explicit_predicates_of(did);
    let (trait_items, generics) = if let Some(hir_id) = tcx.hir().as_local_hir_id(did) {
        match tcx.hir().expect_item(hir_id).kind {
            hir::ItemKind::Impl { ref generics, ref items, .. } => (
                items.iter().map(|item| tcx.hir().impl_item(item.id).clean(cx)).collect::<Vec<_>>(),
                generics.clean(cx),
            ),
            _ => panic!("did given to build_impl was not an impl"),
        }
    } else {
        (
            tcx.associated_items(did)
                .in_definition_order()
                .filter_map(|item| {
                    if associated_trait.is_some() || item.vis == ty::Visibility::Public {
                        Some(item.clean(cx))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>(),
            clean::enter_impl_trait(cx, || (tcx.generics_of(did), predicates).clean(cx)),
        )
    };
    let polarity = tcx.impl_polarity(did);
    let trait_ = associated_trait.clean(cx).map(|bound| match bound {
        clean::GenericBound::TraitBound(polyt, _) => polyt.trait_,
        clean::GenericBound::Outlives(..) => unreachable!(),
    });
    if trait_.def_id() == tcx.lang_items().deref_trait() {
        super::build_deref_target_impls(cx, &trait_items, ret);
    }
    if let Some(trait_did) = trait_.def_id() {
        record_extern_trait(cx, trait_did);
    }

    let provided = trait_
        .def_id()
        .map(|did| tcx.provided_trait_methods(did).map(|meth| meth.ident.to_string()).collect())
        .unwrap_or_default();

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
        visibility: clean::Inherited,
        stability: tcx.lookup_stability(did).clean(cx),
        deprecation: tcx.lookup_deprecation(did).clean(cx),
        def_id: did,
    });
}

fn build_module(cx: &DocContext<'_>, did: DefId, visited: &mut FxHashSet<DefId>) -> clean::Module {
    let mut items = Vec::new();
    fill_in(cx, did, &mut items, visited);
    return clean::Module { items, is_crate: false };

    fn fill_in(
        cx: &DocContext<'_>,
        did: DefId,
        items: &mut Vec<clean::Item>,
        visited: &mut FxHashSet<DefId>,
    ) {
        // If we're re-exporting a re-export it may actually re-export something in
        // two namespaces, so the target may be listed twice. Make sure we only
        // visit each node at most once.
        for &item in cx.tcx.item_children(did).iter() {
            if item.vis == ty::Visibility::Public {
                if let Some(def_id) = item.res.mod_def_id() {
                    if did == def_id || !visited.insert(def_id) {
                        continue;
                    }
                }
                if let Res::PrimTy(p) = item.res {
                    // Primitive types can't be inlined so generate an import instead.
                    items.push(clean::Item {
                        name: None,
                        attrs: clean::Attributes::default(),
                        source: clean::Span::empty(),
                        def_id: cx
                            .tcx
                            .hir()
                            .local_def_id_from_node_id(ast::CRATE_NODE_ID)
                            .to_def_id(),
                        visibility: clean::Public,
                        stability: None,
                        deprecation: None,
                        inner: clean::ImportItem(clean::Import::Simple(
                            item.ident.to_string(),
                            clean::ImportSource {
                                path: clean::Path {
                                    global: false,
                                    res: item.res,
                                    segments: vec![clean::PathSegment {
                                        name: clean::PrimitiveType::from(p).as_str().to_string(),
                                        args: clean::GenericArgs::AngleBracketed {
                                            args: Vec::new(),
                                            bindings: Vec::new(),
                                        },
                                    }],
                                },
                                did: None,
                            },
                        )),
                    });
                } else if let Some(i) = try_inline(cx, item.res, item.ident.name, None, visited) {
                    items.extend(i)
                }
            }
        }
    }
}

pub fn print_inlined_const(cx: &DocContext<'_>, did: DefId) -> String {
    if let Some(hir_id) = cx.tcx.hir().as_local_hir_id(did) {
        rustc_hir_pretty::id_to_string(&cx.tcx.hir(), hir_id)
    } else {
        cx.tcx.rendered_const(did)
    }
}

fn build_const(cx: &DocContext<'_>, did: DefId) -> clean::Constant {
    clean::Constant {
        type_: cx.tcx.type_of(did).clean(cx),
        expr: print_inlined_const(cx, did),
        value: clean::utils::print_evaluated_const(cx, did),
        is_literal: cx
            .tcx
            .hir()
            .as_local_hir_id(did)
            .map_or(false, |hir_id| clean::utils::is_literal_expr(cx, hir_id)),
    }
}

fn build_static(cx: &DocContext<'_>, did: DefId, mutable: bool) -> clean::Static {
    clean::Static {
        type_: cx.tcx.type_of(did).clean(cx),
        mutability: if mutable { Mutability::Mut } else { Mutability::Not },
        expr: "\n\n\n".to_string(), // trigger the "[definition]" links
    }
}

fn build_macro(cx: &DocContext<'_>, did: DefId, name: ast::Name) -> clean::ItemEnum {
    let imported_from = cx.tcx.original_crate_name(did.krate);
    match cx.enter_resolver(|r| r.cstore().load_macro_untracked(did, cx.sess())) {
        LoadedMacro::MacroDef(def, _) => {
            let matchers: Vec<Span> = if let ast::ItemKind::MacroDef(ref def) = def.kind {
                let tts: Vec<_> = def.body.inner_tokens().into_trees().collect();
                tts.chunks(4).map(|arm| arm[0].span()).collect()
            } else {
                unreachable!()
            };

            let source = format!(
                "macro_rules! {} {{\n{}}}",
                name.clean(cx),
                matchers
                    .iter()
                    .map(|span| { format!("    {} => {{ ... }};\n", span.to_src(cx)) })
                    .collect::<String>()
            );

            clean::MacroItem(clean::Macro { source, imported_from: Some(imported_from).clean(cx) })
        }
        LoadedMacro::ProcMacro(ext) => clean::ProcMacroItem(clean::ProcMacro {
            kind: ext.macro_kind(),
            helpers: ext.helper_attrs.clean(cx),
        }),
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
            clean::WherePredicate::BoundPredicate { ty: clean::Generic(ref s), ref mut bounds }
                if *s == "Self" =>
            {
                bounds.retain(|bound| match *bound {
                    clean::GenericBound::TraitBound(
                        clean::PolyTrait { trait_: clean::ResolvedPath { did, .. }, .. },
                        _,
                    ) => did != trait_did,
                    _ => true,
                });
            }
            _ => {}
        }
    }

    g.where_predicates.retain(|pred| match *pred {
        clean::WherePredicate::BoundPredicate {
            ty:
                clean::QPath {
                    self_type: box clean::Generic(ref s),
                    trait_: box clean::ResolvedPath { did, .. },
                    name: ref _name,
                },
            ref bounds,
        } => !(bounds.is_empty() || *s == "Self" && did == trait_did),
        _ => true,
    });
    g
}

/// Supertrait bounds for a trait are also listed in the generics coming from
/// the metadata for a crate, so we want to separate those out and create a new
/// list of explicit supertrait bounds to render nicely.
fn separate_supertrait_bounds(
    mut g: clean::Generics,
) -> (clean::Generics, Vec<clean::GenericBound>) {
    let mut ty_bounds = Vec::new();
    g.where_predicates.retain(|pred| match *pred {
        clean::WherePredicate::BoundPredicate { ty: clean::Generic(ref s), ref bounds }
            if *s == "Self" =>
        {
            ty_bounds.extend(bounds.iter().cloned());
            false
        }
        _ => true,
    });
    (g, ty_bounds)
}

pub fn record_extern_trait(cx: &DocContext<'_>, did: DefId) {
    if did.is_local() {
        return;
    }

    {
        if cx.external_traits.borrow().contains_key(&did)
            || cx.active_extern_traits.borrow().contains(&did)
        {
            return;
        }
    }

    cx.active_extern_traits.borrow_mut().insert(did);

    debug!("record_extern_trait: {:?}", did);
    let trait_ = build_external_trait(cx, did);

    cx.external_traits.borrow_mut().insert(did, trait_);
    cx.active_extern_traits.borrow_mut().remove(&did);
}
