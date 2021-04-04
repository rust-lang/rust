//! Support for inlining external documentation into the current AST.

use std::iter::once;

use rustc_ast as ast;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, CRATE_DEF_INDEX};
use rustc_hir::Mutability;
use rustc_metadata::creader::LoadedMacro;
use rustc_middle::ty::{self, TyCtxt};
use rustc_mir::const_eval::is_min_const_fn;
use rustc_span::hygiene::MacroKind;
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;

use crate::clean::{self, Attributes, GetDefId, ToSource, TypeKind};
use crate::core::DocContext;
use crate::formats::item_type::ItemType;

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
///
/// `parent_module` refers to the parent of the *re-export*, not the original item.
crate fn try_inline(
    cx: &mut DocContext<'_>,
    parent_module: DefId,
    res: Res,
    name: Symbol,
    attrs: Option<Attrs<'_>>,
    visited: &mut FxHashSet<DefId>,
) -> Option<Vec<clean::Item>> {
    let did = res.opt_def_id()?;
    if did.is_local() {
        return None;
    }
    let mut ret = Vec::new();

    debug!("attrs={:?}", attrs);
    let attrs_clone = attrs;

    let kind = match res {
        Res::Def(DefKind::Trait, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Trait);
            build_impls(cx, Some(parent_module), did, attrs, &mut ret);
            clean::TraitItem(build_external_trait(cx, did))
        }
        Res::Def(DefKind::Fn, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Function);
            clean::FunctionItem(build_external_function(cx, did))
        }
        Res::Def(DefKind::Struct, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Struct);
            build_impls(cx, Some(parent_module), did, attrs, &mut ret);
            clean::StructItem(build_struct(cx, did))
        }
        Res::Def(DefKind::Union, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Union);
            build_impls(cx, Some(parent_module), did, attrs, &mut ret);
            clean::UnionItem(build_union(cx, did))
        }
        Res::Def(DefKind::TyAlias, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Typedef);
            build_impls(cx, Some(parent_module), did, attrs, &mut ret);
            clean::TypedefItem(build_type_alias(cx, did), false)
        }
        Res::Def(DefKind::Enum, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Enum);
            build_impls(cx, Some(parent_module), did, attrs, &mut ret);
            clean::EnumItem(build_enum(cx, did))
        }
        Res::Def(DefKind::ForeignTy, did) => {
            record_extern_fqn(cx, did, clean::TypeKind::Foreign);
            build_impls(cx, Some(parent_module), did, attrs, &mut ret);
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
    let attrs = box merge_attrs(cx, Some(parent_module), target_attrs, attrs_clone);

    cx.inlined.insert(did);
    let what_rustc_thinks = clean::Item::from_def_id_and_parts(did, Some(name), kind, cx);
    ret.push(clean::Item { attrs, ..what_rustc_thinks });
    Some(ret)
}

crate fn try_inline_glob(
    cx: &mut DocContext<'_>,
    res: Res,
    visited: &mut FxHashSet<DefId>,
) -> Option<Vec<clean::Item>> {
    let did = res.opt_def_id()?;
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

crate fn load_attrs<'hir>(cx: &DocContext<'hir>, did: DefId) -> Attrs<'hir> {
    cx.tcx.get_attrs(did)
}

/// Record an external fully qualified name in the external_paths cache.
///
/// These names are used later on by HTML rendering to generate things like
/// source links back to the original item.
crate fn record_extern_fqn(cx: &mut DocContext<'_>, did: DefId, kind: clean::TypeKind) {
    let crate_name = cx.tcx.crate_name(did.krate).to_string();

    let relative = cx.tcx.def_path(did).data.into_iter().filter_map(|elem| {
        // extern blocks have an empty name
        let s = elem.data.to_string();
        if !s.is_empty() { Some(s) } else { None }
    });
    let fqn = if let clean::TypeKind::Macro = kind {
        // Check to see if it is a macro 2.0 or built-in macro
        if matches!(
            cx.enter_resolver(|r| r.cstore().load_macro_untracked(did, cx.sess())),
            LoadedMacro::MacroDef(def, _)
                if matches!(&def.kind, ast::ItemKind::MacroDef(ast_def)
                    if !ast_def.macro_rules)
        ) {
            once(crate_name).chain(relative).collect()
        } else {
            vec![crate_name, relative.last().expect("relative was empty")]
        }
    } else {
        once(crate_name).chain(relative).collect()
    };

    if did.is_local() {
        cx.cache.exact_paths.insert(did, fqn);
    } else {
        cx.cache.external_paths.insert(did, (fqn, ItemType::from(kind)));
    }
}

crate fn build_external_trait(cx: &mut DocContext<'_>, did: DefId) -> clean::Trait {
    let trait_items =
        cx.tcx.associated_items(did).in_definition_order().map(|item| item.clean(cx)).collect();

    let predicates = cx.tcx.predicates_of(did);
    let generics = (cx.tcx.generics_of(did), predicates).clean(cx);
    let generics = filter_non_trait_generics(did, generics);
    let (generics, supertrait_bounds) = separate_supertrait_bounds(generics);
    let is_auto = cx.tcx.trait_is_auto(did);
    clean::Trait {
        unsafety: cx.tcx.trait_def(did).unsafety,
        generics,
        items: trait_items,
        bounds: supertrait_bounds,
        is_auto,
    }
}

fn build_external_function(cx: &mut DocContext<'_>, did: DefId) -> clean::Function {
    let sig = cx.tcx.fn_sig(did);

    let constness =
        if is_min_const_fn(cx.tcx, did) { hir::Constness::Const } else { hir::Constness::NotConst };
    let asyncness = cx.tcx.asyncness(did);
    let predicates = cx.tcx.predicates_of(did);
    let (generics, decl) = clean::enter_impl_trait(cx, |cx| {
        ((cx.tcx.generics_of(did), predicates).clean(cx), (did, sig).clean(cx))
    });
    clean::Function {
        decl,
        generics,
        header: hir::FnHeader { unsafety: sig.unsafety(), abi: sig.abi(), constness, asyncness },
    }
}

fn build_enum(cx: &mut DocContext<'_>, did: DefId) -> clean::Enum {
    let predicates = cx.tcx.explicit_predicates_of(did);

    clean::Enum {
        generics: (cx.tcx.generics_of(did), predicates).clean(cx),
        variants_stripped: false,
        variants: cx.tcx.adt_def(did).variants.clean(cx),
    }
}

fn build_struct(cx: &mut DocContext<'_>, did: DefId) -> clean::Struct {
    let predicates = cx.tcx.explicit_predicates_of(did);
    let variant = cx.tcx.adt_def(did).non_enum_variant();

    clean::Struct {
        struct_type: variant.ctor_kind,
        generics: (cx.tcx.generics_of(did), predicates).clean(cx),
        fields: variant.fields.clean(cx),
        fields_stripped: false,
    }
}

fn build_union(cx: &mut DocContext<'_>, did: DefId) -> clean::Union {
    let predicates = cx.tcx.explicit_predicates_of(did);
    let variant = cx.tcx.adt_def(did).non_enum_variant();

    clean::Union {
        generics: (cx.tcx.generics_of(did), predicates).clean(cx),
        fields: variant.fields.clean(cx),
        fields_stripped: false,
    }
}

fn build_type_alias(cx: &mut DocContext<'_>, did: DefId) -> clean::Typedef {
    let predicates = cx.tcx.explicit_predicates_of(did);
    let type_ = cx.tcx.type_of(did).clean(cx);

    clean::Typedef {
        type_,
        generics: (cx.tcx.generics_of(did), predicates).clean(cx),
        item_type: None,
    }
}

/// Builds all inherent implementations of an ADT (struct/union/enum) or Trait item/path/reexport.
crate fn build_impls(
    cx: &mut DocContext<'_>,
    parent_module: Option<DefId>,
    did: DefId,
    attrs: Option<Attrs<'_>>,
    ret: &mut Vec<clean::Item>,
) {
    let tcx = cx.tcx;

    // for each implementation of an item represented by `did`, build the clean::Item for that impl
    for &did in tcx.inherent_impls(did).iter() {
        build_impl(cx, parent_module, did, attrs, ret);
    }
}

/// `parent_module` refers to the parent of the re-export, not the original item
fn merge_attrs(
    cx: &mut DocContext<'_>,
    parent_module: Option<DefId>,
    old_attrs: Attrs<'_>,
    new_attrs: Option<Attrs<'_>>,
) -> clean::Attributes {
    // NOTE: If we have additional attributes (from a re-export),
    // always insert them first. This ensure that re-export
    // doc comments show up before the original doc comments
    // when we render them.
    if let Some(inner) = new_attrs {
        if let Some(new_id) = parent_module {
            let diag = cx.sess().diagnostic();
            Attributes::from_ast(diag, old_attrs, Some((inner, new_id)))
        } else {
            let mut both = inner.to_vec();
            both.extend_from_slice(old_attrs);
            both.clean(cx)
        }
    } else {
        old_attrs.clean(cx)
    }
}

/// Builds a specific implementation of a type. The `did` could be a type method or trait method.
crate fn build_impl(
    cx: &mut DocContext<'_>,
    parent_module: impl Into<Option<DefId>>,
    did: DefId,
    attrs: Option<Attrs<'_>>,
    ret: &mut Vec<clean::Item>,
) {
    if !cx.inlined.insert(did) {
        return;
    }

    let tcx = cx.tcx;
    let associated_trait = tcx.impl_trait_ref(did);

    // Only inline impl if the implemented trait is
    // reachable in rustdoc generated documentation
    if !did.is_local() {
        if let Some(traitref) = associated_trait {
            let did = traitref.def_id;
            if !cx.cache.access_levels.is_public(did) {
                return;
            }

            if let Some(stab) = tcx.lookup_stability(did) {
                if stab.level.is_unstable() && stab.feature == sym::rustc_private {
                    return;
                }
            }
        }
    }

    let impl_item = match did.as_local() {
        Some(did) => {
            let hir_id = tcx.hir().local_def_id_to_hir_id(did);
            match &tcx.hir().expect_item(hir_id).kind {
                hir::ItemKind::Impl(impl_) => Some(impl_),
                _ => panic!("`DefID` passed to `build_impl` is not an `impl"),
            }
        }
        None => None,
    };

    let for_ = match &impl_item {
        Some(impl_) => impl_.self_ty.clean(cx),
        None => tcx.type_of(did).clean(cx),
    };

    // Only inline impl if the implementing type is
    // reachable in rustdoc generated documentation
    if !did.is_local() {
        if let Some(did) = for_.def_id() {
            if !cx.cache.access_levels.is_public(did) {
                return;
            }

            if let Some(stab) = tcx.lookup_stability(did) {
                if stab.level.is_unstable() && stab.feature == sym::rustc_private {
                    return;
                }
            }
        }
    }

    let predicates = tcx.explicit_predicates_of(did);
    let (trait_items, generics) = match impl_item {
        Some(impl_) => (
            impl_
                .items
                .iter()
                .map(|item| tcx.hir().impl_item(item.id).clean(cx))
                .collect::<Vec<_>>(),
            impl_.generics.clean(cx),
        ),
        None => (
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
            clean::enter_impl_trait(cx, |cx| (tcx.generics_of(did), predicates).clean(cx)),
        ),
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
        .map(|did| tcx.provided_trait_methods(did).map(|meth| meth.ident.name).collect())
        .unwrap_or_default();

    debug!("build_impl: impl {:?} for {:?}", trait_.def_id(), for_.def_id());

    let attrs = box merge_attrs(cx, parent_module.into(), load_attrs(cx, did), attrs);
    debug!("merged_attrs={:?}", attrs);

    ret.push(clean::Item::from_def_id_and_attrs_and_parts(
        did,
        None,
        clean::ImplItem(clean::Impl {
            unsafety: hir::Unsafety::Normal,
            generics,
            provided_trait_methods: provided,
            trait_,
            for_,
            items: trait_items,
            negative_polarity: polarity.clean(cx),
            synthetic: false,
            blanket_impl: None,
        }),
        attrs,
        cx,
    ));
}

fn build_module(
    cx: &mut DocContext<'_>,
    did: DefId,
    visited: &mut FxHashSet<DefId>,
) -> clean::Module {
    let mut items = Vec::new();

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
                    attrs: box clean::Attributes::default(),
                    span: clean::Span::dummy(),
                    def_id: DefId::local(CRATE_DEF_INDEX),
                    visibility: clean::Public,
                    kind: box clean::ImportItem(clean::Import::new_simple(
                        item.ident.name,
                        clean::ImportSource {
                            path: clean::Path {
                                global: false,
                                res: item.res,
                                segments: vec![clean::PathSegment {
                                    name: clean::PrimitiveType::from(p).as_sym(),
                                    args: clean::GenericArgs::AngleBracketed {
                                        args: Vec::new(),
                                        bindings: Vec::new(),
                                    },
                                }],
                            },
                            did: None,
                        },
                        true,
                    )),
                });
            } else if let Some(i) = try_inline(cx, did, item.res, item.ident.name, None, visited) {
                items.extend(i)
            }
        }
    }

    clean::Module { items, is_crate: false }
}

crate fn print_inlined_const(tcx: TyCtxt<'_>, did: DefId) -> String {
    if let Some(did) = did.as_local() {
        let hir_id = tcx.hir().local_def_id_to_hir_id(did);
        rustc_hir_pretty::id_to_string(&tcx.hir(), hir_id)
    } else {
        tcx.rendered_const(did)
    }
}

fn build_const(cx: &mut DocContext<'_>, def_id: DefId) -> clean::Constant {
    clean::Constant {
        type_: cx.tcx.type_of(def_id).clean(cx),
        kind: clean::ConstantKind::Extern { def_id },
    }
}

fn build_static(cx: &mut DocContext<'_>, did: DefId, mutable: bool) -> clean::Static {
    clean::Static {
        type_: cx.tcx.type_of(did).clean(cx),
        mutability: if mutable { Mutability::Mut } else { Mutability::Not },
        expr: None,
    }
}

fn build_macro(cx: &mut DocContext<'_>, did: DefId, name: Symbol) -> clean::ItemKind {
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

            clean::MacroItem(clean::Macro { source, imported_from: Some(imported_from) })
        }
        LoadedMacro::ProcMacro(ext) => clean::ProcMacroItem(clean::ProcMacro {
            kind: ext.macro_kind(),
            helpers: ext.helper_attrs,
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
                if *s == kw::SelfUpper =>
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
        } => !(bounds.is_empty() || *s == kw::SelfUpper && did == trait_did),
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
            if *s == kw::SelfUpper =>
        {
            ty_bounds.extend(bounds.iter().cloned());
            false
        }
        _ => true,
    });
    (g, ty_bounds)
}

crate fn record_extern_trait(cx: &mut DocContext<'_>, did: DefId) {
    if did.is_local() {
        return;
    }

    {
        if cx.external_traits.borrow().contains_key(&did) || cx.active_extern_traits.contains(&did)
        {
            return;
        }
    }

    {
        cx.active_extern_traits.insert(did);
    }

    debug!("record_extern_trait: {:?}", did);
    let trait_ = build_external_trait(cx, did);

    let trait_ = clean::TraitWithExtraInfo {
        trait_,
        is_notable: clean::utils::has_doc_flag(cx.tcx.get_attrs(did), sym::notable_trait),
    };
    cx.external_traits.borrow_mut().insert(did, trait_);
    cx.active_extern_traits.remove(&did);
}
