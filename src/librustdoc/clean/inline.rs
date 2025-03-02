//! Support for inlining external documentation into the current AST.

use std::iter::once;
use std::sync::Arc;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::Mutability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, DefIdSet, LocalDefId, LocalModDefId};
use rustc_metadata::creader::{CStore, LoadedMacro};
use rustc_middle::ty::fast_reject::SimplifiedType;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::def_id::LOCAL_CRATE;
use rustc_span::hygiene::MacroKind;
use rustc_span::symbol::{Symbol, sym};
use thin_vec::{ThinVec, thin_vec};
use tracing::{debug, trace};

use super::{Item, extract_cfg_from_attrs};
use crate::clean::{
    self, Attributes, ImplKind, ItemId, Type, clean_bound_vars, clean_generics, clean_impl_item,
    clean_middle_assoc_item, clean_middle_field, clean_middle_ty, clean_poly_fn_sig,
    clean_trait_ref_with_constraints, clean_ty, clean_ty_alias_inner_type, clean_ty_generics,
    clean_variant_def, utils,
};
use crate::core::DocContext;
use crate::formats::item_type::ItemType;

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
pub(crate) fn try_inline(
    cx: &mut DocContext<'_>,
    res: Res,
    name: Symbol,
    attrs: Option<(&[hir::Attribute], Option<LocalDefId>)>,
    visited: &mut DefIdSet,
) -> Option<Vec<clean::Item>> {
    let did = res.opt_def_id()?;
    if did.is_local() {
        return None;
    }
    let mut ret = Vec::new();

    debug!("attrs={attrs:?}");

    let attrs_without_docs = attrs.map(|(attrs, def_id)| {
        (attrs.iter().filter(|a| a.doc_str().is_none()).cloned().collect::<Vec<_>>(), def_id)
    });
    let attrs_without_docs =
        attrs_without_docs.as_ref().map(|(attrs, def_id)| (&attrs[..], *def_id));

    let import_def_id = attrs.and_then(|(_, def_id)| def_id);

    let kind = match res {
        Res::Def(DefKind::Trait, did) => {
            record_extern_fqn(cx, did, ItemType::Trait);
            cx.with_param_env(did, |cx| {
                build_impls(cx, did, attrs_without_docs, &mut ret);
                clean::TraitItem(Box::new(build_trait(cx, did)))
            })
        }
        Res::Def(DefKind::TraitAlias, did) => {
            record_extern_fqn(cx, did, ItemType::TraitAlias);
            cx.with_param_env(did, |cx| clean::TraitAliasItem(build_trait_alias(cx, did)))
        }
        Res::Def(DefKind::Fn, did) => {
            record_extern_fqn(cx, did, ItemType::Function);
            cx.with_param_env(did, |cx| {
                clean::enter_impl_trait(cx, |cx| clean::FunctionItem(build_function(cx, did)))
            })
        }
        Res::Def(DefKind::Struct, did) => {
            record_extern_fqn(cx, did, ItemType::Struct);
            cx.with_param_env(did, |cx| {
                build_impls(cx, did, attrs_without_docs, &mut ret);
                clean::StructItem(build_struct(cx, did))
            })
        }
        Res::Def(DefKind::Union, did) => {
            record_extern_fqn(cx, did, ItemType::Union);
            cx.with_param_env(did, |cx| {
                build_impls(cx, did, attrs_without_docs, &mut ret);
                clean::UnionItem(build_union(cx, did))
            })
        }
        Res::Def(DefKind::TyAlias, did) => {
            record_extern_fqn(cx, did, ItemType::TypeAlias);
            cx.with_param_env(did, |cx| {
                build_impls(cx, did, attrs_without_docs, &mut ret);
                clean::TypeAliasItem(build_type_alias(cx, did, &mut ret))
            })
        }
        Res::Def(DefKind::Enum, did) => {
            record_extern_fqn(cx, did, ItemType::Enum);
            cx.with_param_env(did, |cx| {
                build_impls(cx, did, attrs_without_docs, &mut ret);
                clean::EnumItem(build_enum(cx, did))
            })
        }
        Res::Def(DefKind::ForeignTy, did) => {
            record_extern_fqn(cx, did, ItemType::ForeignType);
            cx.with_param_env(did, |cx| {
                build_impls(cx, did, attrs_without_docs, &mut ret);
                clean::ForeignTypeItem
            })
        }
        // Never inline enum variants but leave them shown as re-exports.
        Res::Def(DefKind::Variant, _) => return None,
        // Assume that enum variants and struct types are re-exported next to
        // their constructors.
        Res::Def(DefKind::Ctor(..), _) | Res::SelfCtor(..) => return Some(Vec::new()),
        Res::Def(DefKind::Mod, did) => {
            record_extern_fqn(cx, did, ItemType::Module);
            clean::ModuleItem(build_module(cx, did, visited))
        }
        Res::Def(DefKind::Static { .. }, did) => {
            record_extern_fqn(cx, did, ItemType::Static);
            cx.with_param_env(did, |cx| {
                clean::StaticItem(build_static(cx, did, cx.tcx.is_mutable_static(did)))
            })
        }
        Res::Def(DefKind::Const, did) => {
            record_extern_fqn(cx, did, ItemType::Constant);
            cx.with_param_env(did, |cx| {
                let ct = build_const_item(cx, did);
                clean::ConstantItem(Box::new(ct))
            })
        }
        Res::Def(DefKind::Macro(kind), did) => {
            let mac = build_macro(cx, did, name, kind);

            let type_kind = match kind {
                MacroKind::Bang => ItemType::Macro,
                MacroKind::Attr => ItemType::ProcAttribute,
                MacroKind::Derive => ItemType::ProcDerive,
            };
            record_extern_fqn(cx, did, type_kind);
            mac
        }
        _ => return None,
    };

    cx.inlined.insert(did.into());
    let mut item =
        crate::clean::generate_item_with_correct_attrs(cx, kind, did, name, import_def_id, None);
    // The visibility needs to reflect the one from the reexport and not from the "source" DefId.
    item.inner.inline_stmt_id = import_def_id;
    ret.push(item);
    Some(ret)
}

pub(crate) fn try_inline_glob(
    cx: &mut DocContext<'_>,
    res: Res,
    current_mod: LocalModDefId,
    visited: &mut DefIdSet,
    inlined_names: &mut FxHashSet<(ItemType, Symbol)>,
    import: &hir::Item<'_>,
) -> Option<Vec<clean::Item>> {
    let did = res.opt_def_id()?;
    if did.is_local() {
        return None;
    }

    match res {
        Res::Def(DefKind::Mod, did) => {
            // Use the set of module reexports to filter away names that are not actually
            // reexported by the glob, e.g. because they are shadowed by something else.
            let reexports = cx
                .tcx
                .module_children_local(current_mod.to_local_def_id())
                .iter()
                .filter(|child| !child.reexport_chain.is_empty())
                .filter_map(|child| child.res.opt_def_id())
                .filter(|def_id| !cx.tcx.is_doc_hidden(def_id))
                .collect();
            let attrs = cx.tcx.hir_attrs(import.hir_id());
            let mut items = build_module_items(
                cx,
                did,
                visited,
                inlined_names,
                Some(&reexports),
                Some((attrs, Some(import.owner_id.def_id))),
            );
            items.retain(|item| {
                if let Some(name) = item.name {
                    // If an item with the same type and name already exists,
                    // it takes priority over the inlined stuff.
                    inlined_names.insert((item.type_(), name))
                } else {
                    true
                }
            });
            Some(items)
        }
        // glob imports on things like enums aren't inlined even for local exports, so just bail
        _ => None,
    }
}

pub(crate) fn load_attrs<'hir>(cx: &DocContext<'hir>, did: DefId) -> &'hir [hir::Attribute] {
    cx.tcx.get_attrs_unchecked(did)
}

pub(crate) fn item_relative_path(tcx: TyCtxt<'_>, def_id: DefId) -> Vec<Symbol> {
    tcx.def_path(def_id).data.into_iter().filter_map(|elem| elem.data.get_opt_name()).collect()
}

/// Record an external fully qualified name in the external_paths cache.
///
/// These names are used later on by HTML rendering to generate things like
/// source links back to the original item.
pub(crate) fn record_extern_fqn(cx: &mut DocContext<'_>, did: DefId, kind: ItemType) {
    if did.is_local() {
        if cx.cache.exact_paths.contains_key(&did) {
            return;
        }
    } else if cx.cache.external_paths.contains_key(&did) {
        return;
    }

    let crate_name = cx.tcx.crate_name(did.krate);

    let relative = item_relative_path(cx.tcx, did);
    let fqn = if let ItemType::Macro = kind {
        // Check to see if it is a macro 2.0 or built-in macro
        if matches!(
            CStore::from_tcx(cx.tcx).load_macro_untracked(did, cx.tcx),
            LoadedMacro::MacroDef { def, .. } if !def.macro_rules
        ) {
            once(crate_name).chain(relative).collect()
        } else {
            vec![crate_name, *relative.last().expect("relative was empty")]
        }
    } else {
        once(crate_name).chain(relative).collect()
    };

    if did.is_local() {
        cx.cache.exact_paths.insert(did, fqn);
    } else {
        cx.cache.external_paths.insert(did, (fqn, kind));
    }
}

pub(crate) fn build_trait(cx: &mut DocContext<'_>, did: DefId) -> clean::Trait {
    let trait_items = cx
        .tcx
        .associated_items(did)
        .in_definition_order()
        .filter(|item| !item.is_impl_trait_in_trait())
        .map(|item| clean_middle_assoc_item(item, cx))
        .collect();

    let generics = clean_ty_generics(cx, did);
    let (generics, mut supertrait_bounds) = separate_self_bounds(generics);

    supertrait_bounds.retain(|b| {
        // FIXME(sized-hierarchy): Always skip `MetaSized` bounds so that only `?Sized`
        // is shown and none of the new sizedness traits leak into documentation.
        !b.is_meta_sized_bound(cx)
    });

    clean::Trait { def_id: did, generics, items: trait_items, bounds: supertrait_bounds }
}

fn build_trait_alias(cx: &mut DocContext<'_>, did: DefId) -> clean::TraitAlias {
    let generics = clean_ty_generics(cx, did);
    let (generics, mut bounds) = separate_self_bounds(generics);

    bounds.retain(|b| {
        // FIXME(sized-hierarchy): Always skip `MetaSized` bounds so that only `?Sized`
        // is shown and none of the new sizedness traits leak into documentation.
        !b.is_meta_sized_bound(cx)
    });

    clean::TraitAlias { generics, bounds }
}

pub(super) fn build_function(cx: &mut DocContext<'_>, def_id: DefId) -> Box<clean::Function> {
    let sig = cx.tcx.fn_sig(def_id).instantiate_identity();
    // The generics need to be cleaned before the signature.
    let mut generics = clean_ty_generics(cx, def_id);
    let bound_vars = clean_bound_vars(sig.bound_vars());

    // At the time of writing early & late-bound params are stored separately in rustc,
    // namely in `generics.params` and `bound_vars` respectively.
    //
    // To reestablish the original source code order of the generic parameters, we
    // need to manually sort them by their definition span after concatenation.
    //
    // See also:
    // * https://rustc-dev-guide.rust-lang.org/bound-vars-and-params.html
    // * https://rustc-dev-guide.rust-lang.org/what-does-early-late-bound-mean.html
    let has_early_bound_params = !generics.params.is_empty();
    let has_late_bound_params = !bound_vars.is_empty();
    generics.params.extend(bound_vars);
    if has_early_bound_params && has_late_bound_params {
        // If this ever becomes a performances bottleneck either due to the sorting
        // or due to the query calls, consider inserting the late-bound lifetime params
        // right after the last early-bound lifetime param followed by only sorting
        // the slice of lifetime params.
        generics.params.sort_by_key(|param| cx.tcx.def_ident_span(param.def_id).unwrap());
    }

    let decl = clean_poly_fn_sig(cx, Some(def_id), sig);

    Box::new(clean::Function { decl, generics })
}

fn build_enum(cx: &mut DocContext<'_>, did: DefId) -> clean::Enum {
    clean::Enum {
        generics: clean_ty_generics(cx, did),
        variants: cx.tcx.adt_def(did).variants().iter().map(|v| clean_variant_def(v, cx)).collect(),
    }
}

fn build_struct(cx: &mut DocContext<'_>, did: DefId) -> clean::Struct {
    let variant = cx.tcx.adt_def(did).non_enum_variant();

    clean::Struct {
        ctor_kind: variant.ctor_kind(),
        generics: clean_ty_generics(cx, did),
        fields: variant.fields.iter().map(|x| clean_middle_field(x, cx)).collect(),
    }
}

fn build_union(cx: &mut DocContext<'_>, did: DefId) -> clean::Union {
    let variant = cx.tcx.adt_def(did).non_enum_variant();

    let generics = clean_ty_generics(cx, did);
    let fields = variant.fields.iter().map(|x| clean_middle_field(x, cx)).collect();
    clean::Union { generics, fields }
}

fn build_type_alias(
    cx: &mut DocContext<'_>,
    did: DefId,
    ret: &mut Vec<Item>,
) -> Box<clean::TypeAlias> {
    let ty = cx.tcx.type_of(did).instantiate_identity();
    let type_ = clean_middle_ty(ty::Binder::dummy(ty), cx, Some(did), None);
    let inner_type = clean_ty_alias_inner_type(ty, cx, ret);

    Box::new(clean::TypeAlias {
        type_,
        generics: clean_ty_generics(cx, did),
        inner_type,
        item_type: None,
    })
}

/// Builds all inherent implementations of an ADT (struct/union/enum) or Trait item/path/reexport.
pub(crate) fn build_impls(
    cx: &mut DocContext<'_>,
    did: DefId,
    attrs: Option<(&[hir::Attribute], Option<LocalDefId>)>,
    ret: &mut Vec<clean::Item>,
) {
    let _prof_timer = cx.tcx.sess.prof.generic_activity("build_inherent_impls");
    let tcx = cx.tcx;

    // for each implementation of an item represented by `did`, build the clean::Item for that impl
    for &did in tcx.inherent_impls(did).iter() {
        cx.with_param_env(did, |cx| {
            build_impl(cx, did, attrs, ret);
        });
    }

    // This pretty much exists expressly for `dyn Error` traits that exist in the `alloc` crate.
    // See also:
    //
    // * https://github.com/rust-lang/rust/issues/103170 — where it didn't used to get documented
    // * https://github.com/rust-lang/rust/pull/99917 — where the feature got used
    // * https://github.com/rust-lang/rust/issues/53487 — overall tracking issue for Error
    if tcx.has_attr(did, sym::rustc_has_incoherent_inherent_impls) {
        let type_ =
            if tcx.is_trait(did) { SimplifiedType::Trait(did) } else { SimplifiedType::Adt(did) };
        for &did in tcx.incoherent_impls(type_).iter() {
            cx.with_param_env(did, |cx| {
                build_impl(cx, did, attrs, ret);
            });
        }
    }
}

pub(crate) fn merge_attrs(
    cx: &mut DocContext<'_>,
    old_attrs: &[hir::Attribute],
    new_attrs: Option<(&[hir::Attribute], Option<LocalDefId>)>,
) -> (clean::Attributes, Option<Arc<clean::cfg::Cfg>>) {
    // NOTE: If we have additional attributes (from a re-export),
    // always insert them first. This ensure that re-export
    // doc comments show up before the original doc comments
    // when we render them.
    if let Some((inner, item_id)) = new_attrs {
        let mut both = inner.to_vec();
        both.extend_from_slice(old_attrs);
        (
            if let Some(item_id) = item_id {
                Attributes::from_hir_with_additional(old_attrs, (inner, item_id.to_def_id()))
            } else {
                Attributes::from_hir(&both)
            },
            extract_cfg_from_attrs(both.iter(), cx.tcx, &cx.cache.hidden_cfg),
        )
    } else {
        (
            Attributes::from_hir(old_attrs),
            extract_cfg_from_attrs(old_attrs.iter(), cx.tcx, &cx.cache.hidden_cfg),
        )
    }
}

/// Inline an `impl`, inherent or of a trait. The `did` must be for an `impl`.
pub(crate) fn build_impl(
    cx: &mut DocContext<'_>,
    did: DefId,
    attrs: Option<(&[hir::Attribute], Option<LocalDefId>)>,
    ret: &mut Vec<clean::Item>,
) {
    if !cx.inlined.insert(did.into()) {
        return;
    }

    let tcx = cx.tcx;
    let _prof_timer = tcx.sess.prof.generic_activity("build_impl");

    let associated_trait = tcx.impl_trait_ref(did).map(ty::EarlyBinder::skip_binder);

    // Do not inline compiler-internal items unless we're a compiler-internal crate.
    let is_compiler_internal = |did| {
        tcx.lookup_stability(did)
            .is_some_and(|stab| stab.is_unstable() && stab.feature == sym::rustc_private)
    };
    let document_compiler_internal = is_compiler_internal(LOCAL_CRATE.as_def_id());
    let is_directly_public = |cx: &mut DocContext<'_>, did| {
        cx.cache.effective_visibilities.is_directly_public(tcx, did)
            && (document_compiler_internal || !is_compiler_internal(did))
    };

    // Only inline impl if the implemented trait is
    // reachable in rustdoc generated documentation
    if !did.is_local()
        && let Some(traitref) = associated_trait
        && !is_directly_public(cx, traitref.def_id)
    {
        return;
    }

    let impl_item = match did.as_local() {
        Some(did) => match &tcx.hir_expect_item(did).kind {
            hir::ItemKind::Impl(impl_) => Some(impl_),
            _ => panic!("`DefID` passed to `build_impl` is not an `impl"),
        },
        None => None,
    };

    let for_ = match &impl_item {
        Some(impl_) => clean_ty(impl_.self_ty, cx),
        None => clean_middle_ty(
            ty::Binder::dummy(tcx.type_of(did).instantiate_identity()),
            cx,
            Some(did),
            None,
        ),
    };

    // Only inline impl if the implementing type is
    // reachable in rustdoc generated documentation
    if !did.is_local()
        && let Some(did) = for_.def_id(&cx.cache)
        && !is_directly_public(cx, did)
    {
        return;
    }

    let document_hidden = cx.render_options.document_hidden;
    let (trait_items, generics) = match impl_item {
        Some(impl_) => (
            impl_
                .items
                .iter()
                .map(|item| tcx.hir_impl_item(item.id))
                .filter(|item| {
                    // Filter out impl items whose corresponding trait item has `doc(hidden)`
                    // not to document such impl items.
                    // For inherent impls, we don't do any filtering, because that's already done in strip_hidden.rs.

                    // When `--document-hidden-items` is passed, we don't
                    // do any filtering, too.
                    if document_hidden {
                        return true;
                    }
                    if let Some(associated_trait) = associated_trait {
                        let assoc_tag = match item.kind {
                            hir::ImplItemKind::Const(..) => ty::AssocTag::Const,
                            hir::ImplItemKind::Fn(..) => ty::AssocTag::Fn,
                            hir::ImplItemKind::Type(..) => ty::AssocTag::Type,
                        };
                        let trait_item = tcx
                            .associated_items(associated_trait.def_id)
                            .find_by_ident_and_kind(
                                tcx,
                                item.ident,
                                assoc_tag,
                                associated_trait.def_id,
                            )
                            .unwrap(); // SAFETY: For all impl items there exists trait item that has the same name.
                        !tcx.is_doc_hidden(trait_item.def_id)
                    } else {
                        true
                    }
                })
                .map(|item| clean_impl_item(item, cx))
                .collect::<Vec<_>>(),
            clean_generics(impl_.generics, cx),
        ),
        None => (
            tcx.associated_items(did)
                .in_definition_order()
                .filter(|item| !item.is_impl_trait_in_trait())
                .filter(|item| {
                    // If this is a trait impl, filter out associated items whose corresponding item
                    // in the associated trait is marked `doc(hidden)`.
                    // If this is an inherent impl, filter out private associated items.
                    if let Some(associated_trait) = associated_trait {
                        let trait_item = tcx
                            .associated_items(associated_trait.def_id)
                            .find_by_ident_and_kind(
                                tcx,
                                item.ident(tcx),
                                item.as_tag(),
                                associated_trait.def_id,
                            )
                            .unwrap(); // corresponding associated item has to exist
                        document_hidden || !tcx.is_doc_hidden(trait_item.def_id)
                    } else {
                        item.visibility(tcx).is_public()
                    }
                })
                .map(|item| clean_middle_assoc_item(item, cx))
                .collect::<Vec<_>>(),
            clean::enter_impl_trait(cx, |cx| clean_ty_generics(cx, did)),
        ),
    };
    let polarity = tcx.impl_polarity(did);
    let trait_ = associated_trait
        .map(|t| clean_trait_ref_with_constraints(cx, ty::Binder::dummy(t), ThinVec::new()));
    if trait_.as_ref().map(|t| t.def_id()) == tcx.lang_items().deref_trait() {
        super::build_deref_target_impls(cx, &trait_items, ret);
    }

    // Return if the trait itself or any types of the generic parameters are doc(hidden).
    let mut stack: Vec<&Type> = vec![&for_];

    if let Some(did) = trait_.as_ref().map(|t| t.def_id())
        && !document_hidden
        && tcx.is_doc_hidden(did)
    {
        return;
    }

    if let Some(generics) = trait_.as_ref().and_then(|t| t.generics()) {
        stack.extend(generics);
    }

    while let Some(ty) = stack.pop() {
        if let Some(did) = ty.def_id(&cx.cache)
            && !document_hidden
            && tcx.is_doc_hidden(did)
        {
            return;
        }
        if let Some(generics) = ty.generics() {
            stack.extend(generics);
        }
    }

    if let Some(did) = trait_.as_ref().map(|t| t.def_id()) {
        cx.with_param_env(did, |cx| {
            record_extern_trait(cx, did);
        });
    }

    let (merged_attrs, cfg) = merge_attrs(cx, load_attrs(cx, did), attrs);
    trace!("merged_attrs={merged_attrs:?}");

    trace!(
        "build_impl: impl {:?} for {:?}",
        trait_.as_ref().map(|t| t.def_id()),
        for_.def_id(&cx.cache)
    );
    ret.push(clean::Item::from_def_id_and_attrs_and_parts(
        did,
        None,
        clean::ImplItem(Box::new(clean::Impl {
            safety: hir::Safety::Safe,
            generics,
            trait_,
            for_,
            items: trait_items,
            polarity,
            kind: if utils::has_doc_flag(tcx, did, sym::fake_variadic) {
                ImplKind::FakeVariadic
            } else {
                ImplKind::Normal
            },
        })),
        merged_attrs,
        cfg,
    ));
}

fn build_module(cx: &mut DocContext<'_>, did: DefId, visited: &mut DefIdSet) -> clean::Module {
    let items = build_module_items(cx, did, visited, &mut FxHashSet::default(), None, None);

    let span = clean::Span::new(cx.tcx.def_span(did));
    clean::Module { items, span }
}

fn build_module_items(
    cx: &mut DocContext<'_>,
    did: DefId,
    visited: &mut DefIdSet,
    inlined_names: &mut FxHashSet<(ItemType, Symbol)>,
    allowed_def_ids: Option<&DefIdSet>,
    attrs: Option<(&[hir::Attribute], Option<LocalDefId>)>,
) -> Vec<clean::Item> {
    let mut items = Vec::new();

    // If we're re-exporting a re-export it may actually re-export something in
    // two namespaces, so the target may be listed twice. Make sure we only
    // visit each node at most once.
    for item in cx.tcx.module_children(did).iter() {
        if item.vis.is_public() {
            let res = item.res.expect_non_local();
            if let Some(def_id) = res.opt_def_id()
                && let Some(allowed_def_ids) = allowed_def_ids
                && !allowed_def_ids.contains(&def_id)
            {
                continue;
            }
            if let Some(def_id) = res.mod_def_id() {
                // If we're inlining a glob import, it's possible to have
                // two distinct modules with the same name. We don't want to
                // inline it, or mark any of its contents as visited.
                if did == def_id
                    || inlined_names.contains(&(ItemType::Module, item.ident.name))
                    || !visited.insert(def_id)
                {
                    continue;
                }
            }
            if let Res::PrimTy(p) = res {
                // Primitive types can't be inlined so generate an import instead.
                let prim_ty = clean::PrimitiveType::from(p);
                items.push(clean::Item {
                    inner: Box::new(clean::ItemInner {
                        name: None,
                        // We can use the item's `DefId` directly since the only information ever
                        // used from it is `DefId.krate`.
                        item_id: ItemId::DefId(did),
                        attrs: Default::default(),
                        stability: None,
                        kind: clean::ImportItem(clean::Import::new_simple(
                            item.ident.name,
                            clean::ImportSource {
                                path: clean::Path {
                                    res,
                                    segments: thin_vec![clean::PathSegment {
                                        name: prim_ty.as_sym(),
                                        args: clean::GenericArgs::AngleBracketed {
                                            args: Default::default(),
                                            constraints: ThinVec::new(),
                                        },
                                    }],
                                },
                                did: None,
                            },
                            true,
                        )),
                        cfg: None,
                        inline_stmt_id: None,
                    }),
                });
            } else if let Some(i) = try_inline(cx, res, item.ident.name, attrs, visited) {
                items.extend(i)
            }
        }
    }

    items
}

pub(crate) fn print_inlined_const(tcx: TyCtxt<'_>, did: DefId) -> String {
    if let Some(did) = did.as_local() {
        let hir_id = tcx.local_def_id_to_hir_id(did);
        rustc_hir_pretty::id_to_string(&tcx, hir_id)
    } else {
        tcx.rendered_const(did).clone()
    }
}

fn build_const_item(cx: &mut DocContext<'_>, def_id: DefId) -> clean::Constant {
    let mut generics = clean_ty_generics(cx, def_id);
    clean::simplify::move_bounds_to_generic_parameters(&mut generics);
    let ty = clean_middle_ty(
        ty::Binder::dummy(cx.tcx.type_of(def_id).instantiate_identity()),
        cx,
        None,
        None,
    );
    clean::Constant { generics, type_: ty, kind: clean::ConstantKind::Extern { def_id } }
}

fn build_static(cx: &mut DocContext<'_>, did: DefId, mutable: bool) -> clean::Static {
    clean::Static {
        type_: Box::new(clean_middle_ty(
            ty::Binder::dummy(cx.tcx.type_of(did).instantiate_identity()),
            cx,
            Some(did),
            None,
        )),
        mutability: if mutable { Mutability::Mut } else { Mutability::Not },
        expr: None,
    }
}

fn build_macro(
    cx: &mut DocContext<'_>,
    def_id: DefId,
    name: Symbol,
    macro_kind: MacroKind,
) -> clean::ItemKind {
    match CStore::from_tcx(cx.tcx).load_macro_untracked(def_id, cx.tcx) {
        LoadedMacro::MacroDef { def, .. } => match macro_kind {
            MacroKind::Bang => clean::MacroItem(clean::Macro {
                source: utils::display_macro_source(cx, name, &def),
                macro_rules: def.macro_rules,
            }),
            MacroKind::Derive | MacroKind::Attr => {
                clean::ProcMacroItem(clean::ProcMacro { kind: macro_kind, helpers: Vec::new() })
            }
        },
        LoadedMacro::ProcMacro(ext) => clean::ProcMacroItem(clean::ProcMacro {
            kind: ext.macro_kind(),
            helpers: ext.helper_attrs,
        }),
    }
}

fn separate_self_bounds(mut g: clean::Generics) -> (clean::Generics, Vec<clean::GenericBound>) {
    let mut ty_bounds = Vec::new();
    g.where_predicates.retain(|pred| match *pred {
        clean::WherePredicate::BoundPredicate { ty: clean::SelfTy, ref bounds, .. } => {
            ty_bounds.extend(bounds.iter().cloned());
            false
        }
        _ => true,
    });
    (g, ty_bounds)
}

pub(crate) fn record_extern_trait(cx: &mut DocContext<'_>, did: DefId) {
    if did.is_local()
        || cx.external_traits.contains_key(&did)
        || cx.active_extern_traits.contains(&did)
    {
        return;
    }

    cx.active_extern_traits.insert(did);

    debug!("record_extern_trait: {did:?}");
    let trait_ = build_trait(cx, did);

    cx.external_traits.insert(did, trait_);
    cx.active_extern_traits.remove(&did);
}
