use rustc_ast as ast;
use rustc_data_structures::stable_set::FxHashSet;
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_expand::base::SyntaxExtensionKind;
use rustc_feature::UnstableFeatures;
use rustc_hir as hir;
use rustc_hir::def::{
    DefKind,
    Namespace::{self, *},
    PerNS, Res,
};
use rustc_hir::def_id::DefId;
use rustc_middle::ty;
use rustc_resolve::ParentScope;
use rustc_session::lint;
use rustc_span::hygiene::MacroKind;
use rustc_span::symbol::Ident;
use rustc_span::symbol::Symbol;
use rustc_span::DUMMY_SP;
use smallvec::SmallVec;

use std::cell::Cell;
use std::ops::Range;

use crate::clean::*;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::html::markdown::markdown_links;
use crate::passes::Pass;

use super::span_of_attrs;

pub const COLLECT_INTRA_DOC_LINKS: Pass = Pass {
    name: "collect-intra-doc-links",
    run: collect_intra_doc_links,
    description: "reads a crate's documentation to resolve intra-doc-links",
};

pub fn collect_intra_doc_links(krate: Crate, cx: &DocContext<'_>) -> Crate {
    if !UnstableFeatures::from_environment().is_nightly_build() {
        krate
    } else {
        let mut coll = LinkCollector::new(cx);

        coll.fold_crate(krate)
    }
}

enum ErrorKind {
    ResolutionFailure,
    AnchorFailure(AnchorFailure),
}

enum AnchorFailure {
    MultipleAnchors,
    Primitive,
    Variant,
    AssocConstant,
    AssocType,
    Field,
    Method,
}

struct LinkCollector<'a, 'tcx> {
    cx: &'a DocContext<'tcx>,
    // NOTE: this may not necessarily be a module in the current crate
    mod_ids: Vec<DefId>,
    /// This is used to store the kind of associated items,
    /// because `clean` and the disambiguator code expect them to be different.
    /// See the code for associated items on inherent impls for details.
    kind_side_channel: Cell<Option<DefKind>>,
}

impl<'a, 'tcx> LinkCollector<'a, 'tcx> {
    fn new(cx: &'a DocContext<'tcx>) -> Self {
        LinkCollector { cx, mod_ids: Vec::new(), kind_side_channel: Cell::new(None) }
    }

    fn variant_field(
        &self,
        path_str: &str,
        current_item: &Option<String>,
        module_id: DefId,
    ) -> Result<(Res, Option<String>), ErrorKind> {
        let cx = self.cx;

        let mut split = path_str.rsplitn(3, "::");
        let variant_field_name =
            split.next().map(|f| Symbol::intern(f)).ok_or(ErrorKind::ResolutionFailure)?;
        let variant_name =
            split.next().map(|f| Symbol::intern(f)).ok_or(ErrorKind::ResolutionFailure)?;
        let path = split
            .next()
            .map(|f| {
                if f == "self" || f == "Self" {
                    if let Some(name) = current_item.as_ref() {
                        return name.clone();
                    }
                }
                f.to_owned()
            })
            .ok_or(ErrorKind::ResolutionFailure)?;
        let (_, ty_res) = cx
            .enter_resolver(|resolver| {
                resolver.resolve_str_path_error(DUMMY_SP, &path, TypeNS, module_id)
            })
            .map_err(|_| ErrorKind::ResolutionFailure)?;
        if let Res::Err = ty_res {
            return Err(ErrorKind::ResolutionFailure);
        }
        let ty_res = ty_res.map_id(|_| panic!("unexpected node_id"));
        match ty_res {
            Res::Def(DefKind::Enum, did) => {
                if cx
                    .tcx
                    .inherent_impls(did)
                    .iter()
                    .flat_map(|imp| cx.tcx.associated_items(*imp).in_definition_order())
                    .any(|item| item.ident.name == variant_name)
                {
                    return Err(ErrorKind::ResolutionFailure);
                }
                match cx.tcx.type_of(did).kind {
                    ty::Adt(def, _) if def.is_enum() => {
                        if def.all_fields().any(|item| item.ident.name == variant_field_name) {
                            Ok((
                                ty_res,
                                Some(format!(
                                    "variant.{}.field.{}",
                                    variant_name, variant_field_name
                                )),
                            ))
                        } else {
                            Err(ErrorKind::ResolutionFailure)
                        }
                    }
                    _ => Err(ErrorKind::ResolutionFailure),
                }
            }
            _ => Err(ErrorKind::ResolutionFailure),
        }
    }

    /// Resolves a string as a macro.
    fn macro_resolve(&self, path_str: &str, parent_id: Option<DefId>) -> Option<Res> {
        let cx = self.cx;
        let path = ast::Path::from_ident(Ident::from_str(path_str));
        cx.enter_resolver(|resolver| {
            if let Ok((Some(ext), res)) = resolver.resolve_macro_path(
                &path,
                None,
                &ParentScope::module(resolver.graph_root()),
                false,
                false,
            ) {
                if let SyntaxExtensionKind::LegacyBang { .. } = ext.kind {
                    return Some(res.map_id(|_| panic!("unexpected id")));
                }
            }
            if let Some(res) = resolver.all_macros().get(&Symbol::intern(path_str)) {
                return Some(res.map_id(|_| panic!("unexpected id")));
            }
            if let Some(module_id) = parent_id {
                if let Ok((_, res)) =
                    resolver.resolve_str_path_error(DUMMY_SP, path_str, MacroNS, module_id)
                {
                    // don't resolve builtins like `#[derive]`
                    if let Res::Def(..) = res {
                        let res = res.map_id(|_| panic!("unexpected node_id"));
                        return Some(res);
                    }
                }
            } else {
                debug!("attempting to resolve item without parent module: {}", path_str);
            }
            None
        })
    }
    /// Resolves a string as a path within a particular namespace. Also returns an optional
    /// URL fragment in the case of variants and methods.
    fn resolve(
        &self,
        path_str: &str,
        disambiguator: Option<Disambiguator>,
        ns: Namespace,
        current_item: &Option<String>,
        parent_id: Option<DefId>,
        extra_fragment: &Option<String>,
    ) -> Result<(Res, Option<String>), ErrorKind> {
        let cx = self.cx;

        // In case we're in a module, try to resolve the relative path.
        if let Some(module_id) = parent_id {
            let result = cx.enter_resolver(|resolver| {
                resolver.resolve_str_path_error(DUMMY_SP, &path_str, ns, module_id)
            });
            debug!("{} resolved to {:?} in namespace {:?}", path_str, result, ns);
            let result = match result {
                Ok((_, Res::Err)) => Err(ErrorKind::ResolutionFailure),
                _ => result.map_err(|_| ErrorKind::ResolutionFailure),
            };

            if let Ok((_, res)) = result {
                let res = res.map_id(|_| panic!("unexpected node_id"));
                // In case this is a trait item, skip the
                // early return and try looking for the trait.
                let value = match res {
                    Res::Def(DefKind::AssocFn | DefKind::AssocConst, _) => true,
                    Res::Def(DefKind::AssocTy, _) => false,
                    Res::Def(DefKind::Variant, _) => {
                        return handle_variant(cx, res, extra_fragment);
                    }
                    // Not a trait item; just return what we found.
                    Res::PrimTy(..) => {
                        if extra_fragment.is_some() {
                            return Err(ErrorKind::AnchorFailure(AnchorFailure::Primitive));
                        }
                        return Ok((res, Some(path_str.to_owned())));
                    }
                    Res::Def(DefKind::Mod, _) => {
                        // This resolved to a module, but we want primitive types to take precedence instead.
                        if matches!(
                            disambiguator,
                            None | Some(Disambiguator::Namespace(Namespace::TypeNS))
                        ) {
                            if let Some((path, prim)) = is_primitive(path_str, ns) {
                                if extra_fragment.is_some() {
                                    return Err(ErrorKind::AnchorFailure(AnchorFailure::Primitive));
                                }
                                return Ok((prim, Some(path.to_owned())));
                            }
                        }
                        return Ok((res, extra_fragment.clone()));
                    }
                    _ => {
                        return Ok((res, extra_fragment.clone()));
                    }
                };

                if value != (ns == ValueNS) {
                    return Err(ErrorKind::ResolutionFailure);
                }
            } else if let Some((path, prim)) = is_primitive(path_str, ns) {
                if extra_fragment.is_some() {
                    return Err(ErrorKind::AnchorFailure(AnchorFailure::Primitive));
                }
                return Ok((prim, Some(path.to_owned())));
            }

            // Try looking for methods and associated items.
            let mut split = path_str.rsplitn(2, "::");
            let item_name =
                split.next().map(|f| Symbol::intern(f)).ok_or(ErrorKind::ResolutionFailure)?;
            let path = split
                .next()
                .map(|f| {
                    if f == "self" || f == "Self" {
                        if let Some(name) = current_item.as_ref() {
                            return name.clone();
                        }
                    }
                    f.to_owned()
                })
                .ok_or(ErrorKind::ResolutionFailure)?;

            if let Some((path, prim)) = is_primitive(&path, TypeNS) {
                for &impl_ in primitive_impl(cx, &path).ok_or(ErrorKind::ResolutionFailure)? {
                    let link = cx
                        .tcx
                        .associated_items(impl_)
                        .find_by_name_and_namespace(
                            cx.tcx,
                            Ident::with_dummy_span(item_name),
                            ns,
                            impl_,
                        )
                        .and_then(|item| match item.kind {
                            ty::AssocKind::Fn => Some("method"),
                            _ => None,
                        })
                        .map(|out| (prim, Some(format!("{}#{}.{}", path, out, item_name))));
                    if let Some(link) = link {
                        return Ok(link);
                    }
                }
                return Err(ErrorKind::ResolutionFailure);
            }

            let (_, ty_res) = cx
                .enter_resolver(|resolver| {
                    resolver.resolve_str_path_error(DUMMY_SP, &path, TypeNS, module_id)
                })
                .map_err(|_| ErrorKind::ResolutionFailure)?;
            if let Res::Err = ty_res {
                return if ns == Namespace::ValueNS {
                    self.variant_field(path_str, current_item, module_id)
                } else {
                    Err(ErrorKind::ResolutionFailure)
                };
            }
            let ty_res = ty_res.map_id(|_| panic!("unexpected node_id"));
            let res = match ty_res {
                Res::Def(
                    DefKind::Struct | DefKind::Union | DefKind::Enum | DefKind::TyAlias,
                    did,
                ) => {
                    debug!("looking for associated item named {} for item {:?}", item_name, did);
                    // Checks if item_name belongs to `impl SomeItem`
                    let kind = cx
                        .tcx
                        .inherent_impls(did)
                        .iter()
                        .flat_map(|&imp| {
                            cx.tcx.associated_items(imp).find_by_name_and_namespace(
                                cx.tcx,
                                Ident::with_dummy_span(item_name),
                                ns,
                                imp,
                            )
                        })
                        .map(|item| item.kind)
                        // There should only ever be one associated item that matches from any inherent impl
                        .next()
                        // Check if item_name belongs to `impl SomeTrait for SomeItem`
                        // This gives precedence to `impl SomeItem`:
                        // Although having both would be ambiguous, use impl version for compat. sake.
                        // To handle that properly resolve() would have to support
                        // something like [`ambi_fn`](<SomeStruct as SomeTrait>::ambi_fn)
                        .or_else(|| {
                            let kind = resolve_associated_trait_item(
                                did, module_id, item_name, ns, &self.cx,
                            );
                            debug!("got associated item kind {:?}", kind);
                            kind
                        });

                    if let Some(kind) = kind {
                        let out = match kind {
                            ty::AssocKind::Fn => "method",
                            ty::AssocKind::Const => "associatedconstant",
                            ty::AssocKind::Type => "associatedtype",
                        };
                        Some(if extra_fragment.is_some() {
                            Err(ErrorKind::AnchorFailure(if kind == ty::AssocKind::Fn {
                                AnchorFailure::Method
                            } else {
                                AnchorFailure::AssocConstant
                            }))
                        } else {
                            // HACK(jynelson): `clean` expects the type, not the associated item.
                            // but the disambiguator logic expects the associated item.
                            // Store the kind in a side channel so that only the disambiguator logic looks at it.
                            self.kind_side_channel.set(Some(kind.as_def_kind()));
                            Ok((ty_res, Some(format!("{}.{}", out, item_name))))
                        })
                    } else if ns == Namespace::ValueNS {
                        match cx.tcx.type_of(did).kind {
                            ty::Adt(def, _) => {
                                let field = if def.is_enum() {
                                    def.all_fields().find(|item| item.ident.name == item_name)
                                } else {
                                    def.non_enum_variant()
                                        .fields
                                        .iter()
                                        .find(|item| item.ident.name == item_name)
                                };
                                field.map(|item| {
                                    if extra_fragment.is_some() {
                                        Err(ErrorKind::AnchorFailure(if def.is_enum() {
                                            AnchorFailure::Variant
                                        } else {
                                            AnchorFailure::Field
                                        }))
                                    } else {
                                        Ok((
                                            ty_res,
                                            Some(format!(
                                                "{}.{}",
                                                if def.is_enum() {
                                                    "variant"
                                                } else {
                                                    "structfield"
                                                },
                                                item.ident
                                            )),
                                        ))
                                    }
                                })
                            }
                            _ => None,
                        }
                    } else {
                        // We already know this isn't in ValueNS, so no need to check variant_field
                        return Err(ErrorKind::ResolutionFailure);
                    }
                }
                Res::Def(DefKind::Trait, did) => cx
                    .tcx
                    .associated_items(did)
                    .find_by_name_and_namespace(cx.tcx, Ident::with_dummy_span(item_name), ns, did)
                    .map(|item| {
                        let kind = match item.kind {
                            ty::AssocKind::Const => "associatedconstant",
                            ty::AssocKind::Type => "associatedtype",
                            ty::AssocKind::Fn => {
                                if item.defaultness.has_value() {
                                    "method"
                                } else {
                                    "tymethod"
                                }
                            }
                        };

                        if extra_fragment.is_some() {
                            Err(ErrorKind::AnchorFailure(if item.kind == ty::AssocKind::Const {
                                AnchorFailure::AssocConstant
                            } else if item.kind == ty::AssocKind::Type {
                                AnchorFailure::AssocType
                            } else {
                                AnchorFailure::Method
                            }))
                        } else {
                            let res = Res::Def(item.kind.as_def_kind(), item.def_id);
                            Ok((res, Some(format!("{}.{}", kind, item_name))))
                        }
                    }),
                _ => None,
            };
            res.unwrap_or_else(|| {
                if ns == Namespace::ValueNS {
                    self.variant_field(path_str, current_item, module_id)
                } else {
                    Err(ErrorKind::ResolutionFailure)
                }
            })
        } else {
            debug!("attempting to resolve item without parent module: {}", path_str);
            Err(ErrorKind::ResolutionFailure)
        }
    }
}

fn resolve_associated_trait_item(
    did: DefId,
    module: DefId,
    item_name: Symbol,
    ns: Namespace,
    cx: &DocContext<'_>,
) -> Option<ty::AssocKind> {
    let ty = cx.tcx.type_of(did);
    // First consider automatic impls: `impl From<T> for T`
    let implicit_impls = crate::clean::get_auto_trait_and_blanket_impls(cx, ty, did);
    let mut candidates: Vec<_> = implicit_impls
        .flat_map(|impl_outer| {
            match impl_outer.inner {
                ImplItem(impl_) => {
                    debug!("considering auto or blanket impl for trait {:?}", impl_.trait_);
                    // Give precedence to methods that were overridden
                    if !impl_.provided_trait_methods.contains(&*item_name.as_str()) {
                        let mut items = impl_.items.into_iter().filter_map(|assoc| {
                            if assoc.name.as_deref() != Some(&*item_name.as_str()) {
                                return None;
                            }
                            let kind = assoc
                                .inner
                                .as_assoc_kind()
                                .expect("inner items for a trait should be associated items");
                            if kind.namespace() != ns {
                                return None;
                            }

                            trace!("considering associated item {:?}", assoc.inner);
                            // We have a slight issue: normal methods come from `clean` types,
                            // but provided methods come directly from `tcx`.
                            // Fortunately, we don't need the whole method, we just need to know
                            // what kind of associated item it is.
                            Some((assoc.def_id, kind))
                        });
                        let assoc = items.next();
                        debug_assert_eq!(items.count(), 0);
                        assoc
                    } else {
                        // These are provided methods or default types:
                        // ```
                        // trait T {
                        //   type A = usize;
                        //   fn has_default() -> A { 0 }
                        // }
                        // ```
                        let trait_ = impl_.trait_.unwrap().def_id().unwrap();
                        cx.tcx
                            .associated_items(trait_)
                            .find_by_name_and_namespace(
                                cx.tcx,
                                Ident::with_dummy_span(item_name),
                                ns,
                                trait_,
                            )
                            .map(|assoc| (assoc.def_id, assoc.kind))
                    }
                }
                _ => panic!("get_impls returned something that wasn't an impl"),
            }
        })
        .collect();

    // Next consider explicit impls: `impl MyTrait for MyType`
    // Give precedence to inherent impls.
    if candidates.is_empty() {
        let traits = traits_implemented_by(cx, did, module);
        debug!("considering traits {:?}", traits);
        candidates.extend(traits.iter().filter_map(|&trait_| {
            cx.tcx
                .associated_items(trait_)
                .find_by_name_and_namespace(cx.tcx, Ident::with_dummy_span(item_name), ns, trait_)
                .map(|assoc| (assoc.def_id, assoc.kind))
        }));
    }
    // FIXME: warn about ambiguity
    debug!("the candidates were {:?}", candidates);
    candidates.pop().map(|(_, kind)| kind)
}

/// Given a type, return all traits in scope in `module` implemented by that type.
///
/// NOTE: this cannot be a query because more traits could be available when more crates are compiled!
/// So it is not stable to serialize cross-crate.
fn traits_implemented_by(cx: &DocContext<'_>, type_: DefId, module: DefId) -> FxHashSet<DefId> {
    let mut cache = cx.module_trait_cache.borrow_mut();
    let in_scope_traits = cache.entry(module).or_insert_with(|| {
        cx.enter_resolver(|resolver| {
            resolver.traits_in_scope(module).into_iter().map(|candidate| candidate.def_id).collect()
        })
    });

    let ty = cx.tcx.type_of(type_);
    let iter = in_scope_traits.iter().flat_map(|&trait_| {
        trace!("considering explicit impl for trait {:?}", trait_);
        let mut saw_impl = false;
        // Look at each trait implementation to see if it's an impl for `did`
        cx.tcx.for_each_relevant_impl(trait_, ty, |impl_| {
            // FIXME: this is inefficient, find a way to short-circuit for_each_* so this doesn't take as long
            if saw_impl {
                return;
            }

            let trait_ref = cx.tcx.impl_trait_ref(impl_).expect("this is not an inherent impl");
            // Check if these are the same type.
            let impl_type = trait_ref.self_ty();
            debug!(
                "comparing type {} with kind {:?} against type {:?}",
                impl_type, impl_type.kind, type_
            );
            // Fast path: if this is a primitive simple `==` will work
            saw_impl = impl_type == ty
                || match impl_type.kind {
                    // Check if these are the same def_id
                    ty::Adt(def, _) => {
                        debug!("adt def_id: {:?}", def.did);
                        def.did == type_
                    }
                    ty::Foreign(def_id) => def_id == type_,
                    _ => false,
                };
        });
        if saw_impl { Some(trait_) } else { None }
    });
    iter.collect()
}

/// Check for resolve collisions between a trait and its derive
///
/// These are common and we should just resolve to the trait in that case
fn is_derive_trait_collision<T>(ns: &PerNS<Option<(Res, T)>>) -> bool {
    if let PerNS {
        type_ns: Some((Res::Def(DefKind::Trait, _), _)),
        macro_ns: Some((Res::Def(DefKind::Macro(MacroKind::Derive), _), _)),
        ..
    } = *ns
    {
        true
    } else {
        false
    }
}

impl<'a, 'tcx> DocFolder for LinkCollector<'a, 'tcx> {
    fn fold_item(&mut self, mut item: Item) -> Option<Item> {
        use rustc_middle::ty::DefIdTree;

        let parent_node = if item.is_fake() {
            // FIXME: is this correct?
            None
        } else {
            let mut current = item.def_id;
            // The immediate parent might not always be a module.
            // Find the first parent which is.
            loop {
                if let Some(parent) = self.cx.tcx.parent(current) {
                    if self.cx.tcx.def_kind(parent) == DefKind::Mod {
                        break Some(parent);
                    }
                    current = parent;
                } else {
                    break None;
                }
            }
        };

        if parent_node.is_some() {
            trace!("got parent node for {:?} {:?}, id {:?}", item.type_(), item.name, item.def_id);
        }

        let current_item = match item.inner {
            ModuleItem(..) => {
                if item.attrs.inner_docs {
                    if item.def_id.is_top_level_module() { item.name.clone() } else { None }
                } else {
                    match parent_node.or(self.mod_ids.last().copied()) {
                        Some(parent) if !parent.is_top_level_module() => {
                            // FIXME: can we pull the parent module's name from elsewhere?
                            Some(self.cx.tcx.item_name(parent).to_string())
                        }
                        _ => None,
                    }
                }
            }
            ImplItem(Impl { ref for_, .. }) => {
                for_.def_id().map(|did| self.cx.tcx.item_name(did).to_string())
            }
            // we don't display docs on `extern crate` items anyway, so don't process them.
            ExternCrateItem(..) => {
                debug!("ignoring extern crate item {:?}", item.def_id);
                return self.fold_item_recur(item);
            }
            ImportItem(Import::Simple(ref name, ..)) => Some(name.clone()),
            MacroItem(..) => None,
            _ => item.name.clone(),
        };

        if item.is_mod() && item.attrs.inner_docs {
            self.mod_ids.push(item.def_id);
        }

        let cx = self.cx;
        let dox = item.attrs.collapsed_doc_value().unwrap_or_else(String::new);
        trace!("got documentation '{}'", dox);

        // find item's parent to resolve `Self` in item's docs below
        let parent_name = self.cx.as_local_hir_id(item.def_id).and_then(|item_hir| {
            let parent_hir = self.cx.tcx.hir().get_parent_item(item_hir);
            let item_parent = self.cx.tcx.hir().find(parent_hir);
            match item_parent {
                Some(hir::Node::Item(hir::Item {
                    kind:
                        hir::ItemKind::Impl {
                            self_ty:
                                hir::Ty {
                                    kind:
                                        hir::TyKind::Path(hir::QPath::Resolved(
                                            _,
                                            hir::Path { segments, .. },
                                        )),
                                    ..
                                },
                            ..
                        },
                    ..
                })) => segments.first().map(|seg| seg.ident.to_string()),
                Some(hir::Node::Item(hir::Item {
                    ident, kind: hir::ItemKind::Enum(..), ..
                }))
                | Some(hir::Node::Item(hir::Item {
                    ident, kind: hir::ItemKind::Struct(..), ..
                }))
                | Some(hir::Node::Item(hir::Item {
                    ident, kind: hir::ItemKind::Union(..), ..
                }))
                | Some(hir::Node::Item(hir::Item {
                    ident, kind: hir::ItemKind::Trait(..), ..
                })) => Some(ident.to_string()),
                _ => None,
            }
        });

        for (ori_link, link_range) in markdown_links(&dox) {
            trace!("considering link '{}'", ori_link);

            // Bail early for real links.
            if ori_link.contains('/') {
                continue;
            }

            // [] is mostly likely not supposed to be a link
            if ori_link.is_empty() {
                continue;
            }

            let link = ori_link.replace("`", "");
            let parts = link.split('#').collect::<Vec<_>>();
            let (link, extra_fragment) = if parts.len() > 2 {
                anchor_failure(cx, &item, &link, &dox, link_range, AnchorFailure::MultipleAnchors);
                continue;
            } else if parts.len() == 2 {
                if parts[0].trim().is_empty() {
                    // This is an anchor to an element of the current page, nothing to do in here!
                    continue;
                }
                (parts[0].to_owned(), Some(parts[1].to_owned()))
            } else {
                (parts[0].to_owned(), None)
            };
            let resolved_self;
            let mut path_str;
            let disambiguator;
            let (res, fragment) = {
                path_str = if let Ok((d, path)) = Disambiguator::from_str(&link) {
                    disambiguator = Some(d);
                    path
                } else {
                    disambiguator = None;
                    &link
                }
                .trim();

                if path_str.contains(|ch: char| !(ch.is_alphanumeric() || ch == ':' || ch == '_')) {
                    continue;
                }

                // In order to correctly resolve intra-doc-links we need to
                // pick a base AST node to work from.  If the documentation for
                // this module came from an inner comment (//!) then we anchor
                // our name resolution *inside* the module.  If, on the other
                // hand it was an outer comment (///) then we anchor the name
                // resolution in the parent module on the basis that the names
                // used are more likely to be intended to be parent names.  For
                // this, we set base_node to None for inner comments since
                // we've already pushed this node onto the resolution stack but
                // for outer comments we explicitly try and resolve against the
                // parent_node first.
                let base_node = if item.is_mod() && item.attrs.inner_docs {
                    self.mod_ids.last().copied()
                } else {
                    parent_node
                };

                // replace `Self` with suitable item's parent name
                if path_str.starts_with("Self::") {
                    if let Some(ref name) = parent_name {
                        resolved_self = format!("{}::{}", name, &path_str[6..]);
                        path_str = &resolved_self;
                    }
                }

                match disambiguator.map(Disambiguator::ns) {
                    Some(ns @ (ValueNS | TypeNS)) => {
                        match self.resolve(
                            path_str,
                            disambiguator,
                            ns,
                            &current_item,
                            base_node,
                            &extra_fragment,
                        ) {
                            Ok(res) => res,
                            Err(ErrorKind::ResolutionFailure) => {
                                resolution_failure(cx, &item, path_str, &dox, link_range);
                                // This could just be a normal link or a broken link
                                // we could potentially check if something is
                                // "intra-doc-link-like" and warn in that case.
                                continue;
                            }
                            Err(ErrorKind::AnchorFailure(msg)) => {
                                anchor_failure(cx, &item, &ori_link, &dox, link_range, msg);
                                continue;
                            }
                        }
                    }
                    None => {
                        // Try everything!
                        let mut candidates = PerNS {
                            macro_ns: self
                                .macro_resolve(path_str, base_node)
                                .map(|res| (res, extra_fragment.clone())),
                            type_ns: match self.resolve(
                                path_str,
                                disambiguator,
                                TypeNS,
                                &current_item,
                                base_node,
                                &extra_fragment,
                            ) {
                                Ok(res) => {
                                    debug!("got res in TypeNS: {:?}", res);
                                    Some(res)
                                }
                                Err(ErrorKind::AnchorFailure(msg)) => {
                                    anchor_failure(cx, &item, &ori_link, &dox, link_range, msg);
                                    continue;
                                }
                                Err(ErrorKind::ResolutionFailure) => None,
                            },
                            value_ns: match self.resolve(
                                path_str,
                                disambiguator,
                                ValueNS,
                                &current_item,
                                base_node,
                                &extra_fragment,
                            ) {
                                Ok(res) => Some(res),
                                Err(ErrorKind::AnchorFailure(msg)) => {
                                    anchor_failure(cx, &item, &ori_link, &dox, link_range, msg);
                                    continue;
                                }
                                Err(ErrorKind::ResolutionFailure) => None,
                            }
                            .and_then(|(res, fragment)| {
                                // Constructors are picked up in the type namespace.
                                match res {
                                    Res::Def(DefKind::Ctor(..), _) | Res::SelfCtor(..) => None,
                                    _ => match (fragment, extra_fragment) {
                                        (Some(fragment), Some(_)) => {
                                            // Shouldn't happen but who knows?
                                            Some((res, Some(fragment)))
                                        }
                                        (fragment, None) | (None, fragment) => {
                                            Some((res, fragment))
                                        }
                                    },
                                }
                            }),
                        };

                        if candidates.is_empty() {
                            resolution_failure(cx, &item, path_str, &dox, link_range);
                            // this could just be a normal link
                            continue;
                        }

                        let len = candidates.clone().present_items().count();

                        if len == 1 {
                            candidates.present_items().next().unwrap()
                        } else if len == 2 && is_derive_trait_collision(&candidates) {
                            candidates.type_ns.unwrap()
                        } else {
                            if is_derive_trait_collision(&candidates) {
                                candidates.macro_ns = None;
                            }
                            ambiguity_error(
                                cx,
                                &item,
                                path_str,
                                &dox,
                                link_range,
                                candidates.map(|candidate| candidate.map(|(res, _)| res)),
                            );
                            continue;
                        }
                    }
                    Some(MacroNS) => {
                        if let Some(res) = self.macro_resolve(path_str, base_node) {
                            (res, extra_fragment)
                        } else {
                            resolution_failure(cx, &item, path_str, &dox, link_range);
                            continue;
                        }
                    }
                }
            };

            if let Res::PrimTy(_) = res {
                item.attrs.links.push((ori_link, None, fragment));
            } else {
                debug!("intra-doc link to {} resolved to {:?}", path_str, res);

                // Disallow e.g. linking to enums with `struct@`
                if let Res::Def(kind, id) = res {
                    debug!("saw kind {:?} with disambiguator {:?}", kind, disambiguator);
                    match (self.kind_side_channel.take().unwrap_or(kind), disambiguator) {
                        | (DefKind::Const | DefKind::ConstParam | DefKind::AssocConst | DefKind::AnonConst, Some(Disambiguator::Kind(DefKind::Const)))
                        // NOTE: this allows 'method' to mean both normal functions and associated functions
                        // This can't cause ambiguity because both are in the same namespace.
                        | (DefKind::Fn | DefKind::AssocFn, Some(Disambiguator::Kind(DefKind::Fn)))
                        // These are namespaces; allow anything in the namespace to match
                        | (_, Some(Disambiguator::Namespace(_)))
                        // If no disambiguator given, allow anything
                        | (_, None)
                        // All of these are valid, so do nothing
                        => {}
                        (actual, Some(Disambiguator::Kind(expected))) if actual == expected => {}
                        (_, Some(Disambiguator::Kind(expected))) => {
                            // The resolved item did not match the disambiguator; give a better error than 'not found'
                            let msg = format!("incompatible link kind for `{}`", path_str);
                            report_diagnostic(cx, &msg, &item, &dox, link_range, |diag, sp| {
                                // HACK(jynelson): by looking at the source I saw the DefId we pass
                                // for `expected.descr()` doesn't matter, since it's not a crate
                                let note = format!("this link resolved to {} {}, which is not {} {}", kind.article(), kind.descr(id), expected.article(), expected.descr(id));
                                let suggestion = Disambiguator::display_for(kind, path_str);
                                let help_msg = format!("to link to the {}, use its disambiguator", kind.descr(id));
                                diag.note(&note);
                                if let Some(sp) = sp {
                                    diag.span_suggestion(sp, &help_msg, suggestion, Applicability::MaybeIncorrect);
                                } else {
                                    diag.help(&format!("{}: {}", help_msg, suggestion));
                                }
                            });
                            continue;
                        }
                    }
                }

                // item can be non-local e.g. when using #[doc(primitive = "pointer")]
                if let Some((src_id, dst_id)) = res
                    .opt_def_id()
                    .and_then(|def_id| def_id.as_local())
                    .and_then(|dst_id| item.def_id.as_local().map(|src_id| (src_id, dst_id)))
                {
                    use rustc_hir::def_id::LOCAL_CRATE;

                    let hir_src = self.cx.tcx.hir().local_def_id_to_hir_id(src_id);
                    let hir_dst = self.cx.tcx.hir().local_def_id_to_hir_id(dst_id);

                    if self.cx.tcx.privacy_access_levels(LOCAL_CRATE).is_exported(hir_src)
                        && !self.cx.tcx.privacy_access_levels(LOCAL_CRATE).is_exported(hir_dst)
                    {
                        privacy_error(cx, &item, &path_str, &dox, link_range);
                        continue;
                    }
                }
                let id = register_res(cx, res);
                item.attrs.links.push((ori_link, Some(id), fragment));
            }
        }

        if item.is_mod() && !item.attrs.inner_docs {
            self.mod_ids.push(item.def_id);
        }

        if item.is_mod() {
            let ret = self.fold_item_recur(item);

            self.mod_ids.pop();

            ret
        } else {
            self.fold_item_recur(item)
        }
    }

    // FIXME: if we can resolve intra-doc links from other crates, we can use the stock
    // `fold_crate`, but until then we should avoid scanning `krate.external_traits` since those
    // will never resolve properly
    fn fold_crate(&mut self, mut c: Crate) -> Crate {
        c.module = c.module.take().and_then(|module| self.fold_item(module));

        c
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Disambiguator {
    Kind(DefKind),
    Namespace(Namespace),
}

impl Disambiguator {
    /// (disambiguator, path_str)
    fn from_str(link: &str) -> Result<(Self, &str), ()> {
        use Disambiguator::{Kind, Namespace as NS};

        let find_suffix = || {
            let suffixes = [
                ("!()", DefKind::Macro(MacroKind::Bang)),
                ("()", DefKind::Fn),
                ("!", DefKind::Macro(MacroKind::Bang)),
            ];
            for &(suffix, kind) in &suffixes {
                if link.ends_with(suffix) {
                    return Ok((Kind(kind), link.trim_end_matches(suffix)));
                }
            }
            Err(())
        };

        if let Some(idx) = link.find('@') {
            let (prefix, rest) = link.split_at(idx);
            let d = match prefix {
                "struct" => Kind(DefKind::Struct),
                "enum" => Kind(DefKind::Enum),
                "trait" => Kind(DefKind::Trait),
                "union" => Kind(DefKind::Union),
                "module" | "mod" => Kind(DefKind::Mod),
                "const" | "constant" => Kind(DefKind::Const),
                "static" => Kind(DefKind::Static),
                "function" | "fn" | "method" => Kind(DefKind::Fn),
                "derive" => Kind(DefKind::Macro(MacroKind::Derive)),
                "type" => NS(Namespace::TypeNS),
                "value" => NS(Namespace::ValueNS),
                "macro" => NS(Namespace::MacroNS),
                _ => return find_suffix(),
            };
            Ok((d, &rest[1..]))
        } else {
            find_suffix()
        }
    }

    fn display_for(kind: DefKind, path_str: &str) -> String {
        if kind == DefKind::Macro(MacroKind::Bang) {
            return format!("{}!", path_str);
        } else if kind == DefKind::Fn || kind == DefKind::AssocFn {
            return format!("{}()", path_str);
        }
        let prefix = match kind {
            DefKind::Struct => "struct",
            DefKind::Enum => "enum",
            DefKind::Trait => "trait",
            DefKind::Union => "union",
            DefKind::Mod => "mod",
            DefKind::Const | DefKind::ConstParam | DefKind::AssocConst | DefKind::AnonConst => {
                "const"
            }
            DefKind::Static => "static",
            DefKind::Macro(MacroKind::Derive) => "derive",
            // Now handle things that don't have a specific disambiguator
            _ => match kind
                .ns()
                .expect("tried to calculate a disambiguator for a def without a namespace?")
            {
                Namespace::TypeNS => "type",
                Namespace::ValueNS => "value",
                Namespace::MacroNS => "macro",
            },
        };
        format!("{}@{}", prefix, path_str)
    }

    fn ns(self) -> Namespace {
        match self {
            Self::Namespace(n) => n,
            Self::Kind(k) => {
                k.ns().expect("only DefKinds with a valid namespace can be disambiguators")
            }
        }
    }
}

/// Reports a diagnostic for an intra-doc link.
///
/// If no link range is provided, or the source span of the link cannot be determined, the span of
/// the entire documentation block is used for the lint. If a range is provided but the span
/// calculation fails, a note is added to the diagnostic pointing to the link in the markdown.
///
/// The `decorate` callback is invoked in all cases to allow further customization of the
/// diagnostic before emission. If the span of the link was able to be determined, the second
/// parameter of the callback will contain it, and the primary span of the diagnostic will be set
/// to it.
fn report_diagnostic(
    cx: &DocContext<'_>,
    msg: &str,
    item: &Item,
    dox: &str,
    link_range: Option<Range<usize>>,
    decorate: impl FnOnce(&mut DiagnosticBuilder<'_>, Option<rustc_span::Span>),
) {
    let hir_id = match cx.as_local_hir_id(item.def_id) {
        Some(hir_id) => hir_id,
        None => {
            // If non-local, no need to check anything.
            info!("ignoring warning from parent crate: {}", msg);
            return;
        }
    };

    let attrs = &item.attrs;
    let sp = span_of_attrs(attrs).unwrap_or(item.source.span());

    cx.tcx.struct_span_lint_hir(lint::builtin::BROKEN_INTRA_DOC_LINKS, hir_id, sp, |lint| {
        let mut diag = lint.build(msg);

        let span = link_range
            .as_ref()
            .and_then(|range| super::source_span_for_markdown_range(cx, dox, range, attrs));

        if let Some(link_range) = link_range {
            if let Some(sp) = span {
                diag.set_span(sp);
            } else {
                // blah blah blah\nblah\nblah [blah] blah blah\nblah blah
                //                       ^     ~~~~
                //                       |     link_range
                //                       last_new_line_offset
                let last_new_line_offset = dox[..link_range.start].rfind('\n').map_or(0, |n| n + 1);
                let line = dox[last_new_line_offset..].lines().next().unwrap_or("");

                // Print the line containing the `link_range` and manually mark it with '^'s.
                diag.note(&format!(
                    "the link appears in this line:\n\n{line}\n\
                         {indicator: <before$}{indicator:^<found$}",
                    line = line,
                    indicator = "",
                    before = link_range.start - last_new_line_offset,
                    found = link_range.len(),
                ));
            }
        }

        decorate(&mut diag, span);

        diag.emit();
    });
}

fn resolution_failure(
    cx: &DocContext<'_>,
    item: &Item,
    path_str: &str,
    dox: &str,
    link_range: Option<Range<usize>>,
) {
    report_diagnostic(
        cx,
        &format!("unresolved link to `{}`", path_str),
        item,
        dox,
        link_range,
        |diag, sp| {
            if let Some(sp) = sp {
                diag.span_label(sp, "unresolved link");
            }

            diag.help(r#"to escape `[` and `]` characters, add '\' before them like `\[` or `\]`"#);
        },
    );
}

fn anchor_failure(
    cx: &DocContext<'_>,
    item: &Item,
    path_str: &str,
    dox: &str,
    link_range: Option<Range<usize>>,
    failure: AnchorFailure,
) {
    let msg = match failure {
        AnchorFailure::MultipleAnchors => format!("`{}` contains multiple anchors", path_str),
        AnchorFailure::Primitive
        | AnchorFailure::Variant
        | AnchorFailure::AssocConstant
        | AnchorFailure::AssocType
        | AnchorFailure::Field
        | AnchorFailure::Method => {
            let kind = match failure {
                AnchorFailure::Primitive => "primitive type",
                AnchorFailure::Variant => "enum variant",
                AnchorFailure::AssocConstant => "associated constant",
                AnchorFailure::AssocType => "associated type",
                AnchorFailure::Field => "struct field",
                AnchorFailure::Method => "method",
                AnchorFailure::MultipleAnchors => unreachable!("should be handled already"),
            };

            format!(
                "`{}` contains an anchor, but links to {kind}s are already anchored",
                path_str,
                kind = kind
            )
        }
    };

    report_diagnostic(cx, &msg, item, dox, link_range, |diag, sp| {
        if let Some(sp) = sp {
            diag.span_label(sp, "contains invalid anchor");
        }
    });
}

fn ambiguity_error(
    cx: &DocContext<'_>,
    item: &Item,
    path_str: &str,
    dox: &str,
    link_range: Option<Range<usize>>,
    candidates: PerNS<Option<Res>>,
) {
    let mut msg = format!("`{}` is ", path_str);

    let candidates = [TypeNS, ValueNS, MacroNS]
        .iter()
        .filter_map(|&ns| candidates[ns].map(|res| (res, ns)))
        .collect::<Vec<_>>();
    match candidates.as_slice() {
        [(first_def, _), (second_def, _)] => {
            msg += &format!(
                "both {} {} and {} {}",
                first_def.article(),
                first_def.descr(),
                second_def.article(),
                second_def.descr(),
            );
        }
        _ => {
            let mut candidates = candidates.iter().peekable();
            while let Some((res, _)) = candidates.next() {
                if candidates.peek().is_some() {
                    msg += &format!("{} {}, ", res.article(), res.descr());
                } else {
                    msg += &format!("and {} {}", res.article(), res.descr());
                }
            }
        }
    }

    report_diagnostic(cx, &msg, item, dox, link_range.clone(), |diag, sp| {
        if let Some(sp) = sp {
            diag.span_label(sp, "ambiguous link");

            let link_range = link_range.expect("must have a link range if we have a span");

            for (res, ns) in candidates {
                let (action, mut suggestion) = match res {
                    Res::Def(DefKind::AssocFn | DefKind::Fn, _) => {
                        ("add parentheses", format!("{}()", path_str))
                    }
                    Res::Def(DefKind::Macro(MacroKind::Bang), _) => {
                        ("add an exclamation mark", format!("{}!", path_str))
                    }
                    _ => {
                        let type_ = match (res, ns) {
                            (Res::Def(DefKind::Const, _), _) => "const",
                            (Res::Def(DefKind::Static, _), _) => "static",
                            (Res::Def(DefKind::Struct, _), _) => "struct",
                            (Res::Def(DefKind::Enum, _), _) => "enum",
                            (Res::Def(DefKind::Union, _), _) => "union",
                            (Res::Def(DefKind::Trait, _), _) => "trait",
                            (Res::Def(DefKind::Mod, _), _) => "module",
                            (_, TypeNS) => "type",
                            (_, ValueNS) => "value",
                            (Res::Def(DefKind::Macro(MacroKind::Derive), _), MacroNS) => "derive",
                            (_, MacroNS) => "macro",
                        };

                        // FIXME: if this is an implied shortcut link, it's bad style to suggest `@`
                        ("prefix with the item type", format!("{}@{}", type_, path_str))
                    }
                };

                if dox.bytes().nth(link_range.start) == Some(b'`') {
                    suggestion = format!("`{}`", suggestion);
                }

                // FIXME: Create a version of this suggestion for when we don't have the span.
                diag.span_suggestion(
                    sp,
                    &format!("to link to the {}, {}", res.descr(), action),
                    suggestion,
                    Applicability::MaybeIncorrect,
                );
            }
        }
    });
}

fn privacy_error(
    cx: &DocContext<'_>,
    item: &Item,
    path_str: &str,
    dox: &str,
    link_range: Option<Range<usize>>,
) {
    let item_name = item.name.as_deref().unwrap_or("<unknown>");
    let msg =
        format!("public documentation for `{}` links to private item `{}`", item_name, path_str);

    report_diagnostic(cx, &msg, item, dox, link_range, |diag, sp| {
        if let Some(sp) = sp {
            diag.span_label(sp, "this item is private");
        }

        let note_msg = if cx.render_options.document_private {
            "this link resolves only because you passed `--document-private-items`, but will break without"
        } else {
            "this link will resolve properly if you pass `--document-private-items`"
        };
        diag.note(note_msg);
    });
}

/// Given an enum variant's res, return the res of its enum and the associated fragment.
fn handle_variant(
    cx: &DocContext<'_>,
    res: Res,
    extra_fragment: &Option<String>,
) -> Result<(Res, Option<String>), ErrorKind> {
    use rustc_middle::ty::DefIdTree;

    if extra_fragment.is_some() {
        return Err(ErrorKind::AnchorFailure(AnchorFailure::Variant));
    }
    let parent = if let Some(parent) = cx.tcx.parent(res.def_id()) {
        parent
    } else {
        return Err(ErrorKind::ResolutionFailure);
    };
    let parent_def = Res::Def(DefKind::Enum, parent);
    let variant = cx.tcx.expect_variant_res(res);
    Ok((parent_def, Some(format!("variant.{}", variant.ident.name))))
}

const PRIMITIVES: &[(&str, Res)] = &[
    ("u8", Res::PrimTy(hir::PrimTy::Uint(rustc_ast::UintTy::U8))),
    ("u16", Res::PrimTy(hir::PrimTy::Uint(rustc_ast::UintTy::U16))),
    ("u32", Res::PrimTy(hir::PrimTy::Uint(rustc_ast::UintTy::U32))),
    ("u64", Res::PrimTy(hir::PrimTy::Uint(rustc_ast::UintTy::U64))),
    ("u128", Res::PrimTy(hir::PrimTy::Uint(rustc_ast::UintTy::U128))),
    ("usize", Res::PrimTy(hir::PrimTy::Uint(rustc_ast::UintTy::Usize))),
    ("i8", Res::PrimTy(hir::PrimTy::Int(rustc_ast::IntTy::I8))),
    ("i16", Res::PrimTy(hir::PrimTy::Int(rustc_ast::IntTy::I16))),
    ("i32", Res::PrimTy(hir::PrimTy::Int(rustc_ast::IntTy::I32))),
    ("i64", Res::PrimTy(hir::PrimTy::Int(rustc_ast::IntTy::I64))),
    ("i128", Res::PrimTy(hir::PrimTy::Int(rustc_ast::IntTy::I128))),
    ("isize", Res::PrimTy(hir::PrimTy::Int(rustc_ast::IntTy::Isize))),
    ("f32", Res::PrimTy(hir::PrimTy::Float(rustc_ast::FloatTy::F32))),
    ("f64", Res::PrimTy(hir::PrimTy::Float(rustc_ast::FloatTy::F64))),
    ("str", Res::PrimTy(hir::PrimTy::Str)),
    ("bool", Res::PrimTy(hir::PrimTy::Bool)),
    ("true", Res::PrimTy(hir::PrimTy::Bool)),
    ("false", Res::PrimTy(hir::PrimTy::Bool)),
    ("char", Res::PrimTy(hir::PrimTy::Char)),
];

fn is_primitive(path_str: &str, ns: Namespace) -> Option<(&'static str, Res)> {
    if ns == TypeNS {
        PRIMITIVES
            .iter()
            .filter(|x| x.0 == path_str)
            .copied()
            .map(|x| if x.0 == "true" || x.0 == "false" { ("bool", x.1) } else { x })
            .next()
    } else {
        None
    }
}

fn primitive_impl(cx: &DocContext<'_>, path_str: &str) -> Option<&'static SmallVec<[DefId; 4]>> {
    Some(PrimitiveType::from_symbol(Symbol::intern(path_str))?.impls(cx.tcx))
}
