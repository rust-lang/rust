//! This module implements [RFC 1946]: Intra-rustdoc-links
//!
//! [RFC 1946]: https://github.com/rust-lang/rfcs/blob/master/text/1946-intra-rustdoc-links.md

use pulldown_cmark::LinkType;
use rustc_ast::util::comments::may_have_doc_links;
use rustc_data_structures::{
    fx::{FxHashMap, FxHashSet},
    intern::Interned,
};
use rustc_errors::{Applicability, Diagnostic, DiagnosticMessage};
use rustc_hir::def::Namespace::*;
use rustc_hir::def::{DefKind, Namespace, PerNS};
use rustc_hir::def_id::{DefId, CRATE_DEF_ID};
use rustc_hir::Mutability;
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_middle::{bug, ty};
use rustc_resolve::rustdoc::{has_primitive_or_keyword_docs, prepare_to_doc_link_resolution};
use rustc_resolve::rustdoc::{strip_generics_from_path, MalformedGenerics};
use rustc_session::lint::Lint;
use rustc_span::hygiene::MacroKind;
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::BytePos;
use smallvec::{smallvec, SmallVec};

use std::borrow::Cow;
use std::fmt::Display;
use std::mem;
use std::ops::Range;

use crate::clean::{self, utils::find_nearest_parent_module};
use crate::clean::{Crate, Item, ItemLink, PrimitiveType};
use crate::core::DocContext;
use crate::html::markdown::{markdown_links, MarkdownLink, MarkdownLinkRange};
use crate::lint::{BROKEN_INTRA_DOC_LINKS, PRIVATE_INTRA_DOC_LINKS};
use crate::passes::Pass;
use crate::visit::DocVisitor;

pub(crate) const COLLECT_INTRA_DOC_LINKS: Pass = Pass {
    name: "collect-intra-doc-links",
    run: collect_intra_doc_links,
    description: "resolves intra-doc links",
};

fn collect_intra_doc_links(krate: Crate, cx: &mut DocContext<'_>) -> Crate {
    let mut collector = LinkCollector { cx, visited_links: FxHashMap::default() };
    collector.visit_crate(&krate);
    krate
}

fn filter_assoc_items_by_name_and_namespace<'a>(
    tcx: TyCtxt<'a>,
    assoc_items_of: DefId,
    ident: Ident,
    ns: Namespace,
) -> impl Iterator<Item = &ty::AssocItem> + 'a {
    tcx.associated_items(assoc_items_of).filter_by_name_unhygienic(ident.name).filter(move |item| {
        item.kind.namespace() == ns && tcx.hygienic_eq(ident, item.ident(tcx), assoc_items_of)
    })
}

#[derive(Copy, Clone, Debug, Hash, PartialEq)]
enum Res {
    Def(DefKind, DefId),
    Primitive(PrimitiveType),
}

type ResolveRes = rustc_hir::def::Res<rustc_ast::NodeId>;

impl Res {
    fn descr(self) -> &'static str {
        match self {
            Res::Def(kind, id) => ResolveRes::Def(kind, id).descr(),
            Res::Primitive(_) => "primitive type",
        }
    }

    fn article(self) -> &'static str {
        match self {
            Res::Def(kind, id) => ResolveRes::Def(kind, id).article(),
            Res::Primitive(_) => "a",
        }
    }

    fn name(self, tcx: TyCtxt<'_>) -> Symbol {
        match self {
            Res::Def(_, id) => tcx.item_name(id),
            Res::Primitive(prim) => prim.as_sym(),
        }
    }

    fn def_id(self, tcx: TyCtxt<'_>) -> Option<DefId> {
        match self {
            Res::Def(_, id) => Some(id),
            Res::Primitive(prim) => PrimitiveType::primitive_locations(tcx).get(&prim).copied(),
        }
    }

    fn from_def_id(tcx: TyCtxt<'_>, def_id: DefId) -> Res {
        Res::Def(tcx.def_kind(def_id), def_id)
    }

    /// Used for error reporting.
    fn disambiguator_suggestion(self) -> Suggestion {
        let kind = match self {
            Res::Primitive(_) => return Suggestion::Prefix("prim"),
            Res::Def(kind, _) => kind,
        };
        if kind == DefKind::Macro(MacroKind::Bang) {
            return Suggestion::Macro;
        } else if kind == DefKind::Fn || kind == DefKind::AssocFn {
            return Suggestion::Function;
        } else if kind == DefKind::Field {
            return Suggestion::RemoveDisambiguator;
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
            DefKind::Static(_) => "static",
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

        Suggestion::Prefix(prefix)
    }
}

impl TryFrom<ResolveRes> for Res {
    type Error = ();

    fn try_from(res: ResolveRes) -> Result<Self, ()> {
        use rustc_hir::def::Res::*;
        match res {
            Def(kind, id) => Ok(Res::Def(kind, id)),
            PrimTy(prim) => Ok(Res::Primitive(PrimitiveType::from_hir(prim))),
            // e.g. `#[derive]`
            ToolMod | NonMacroAttr(..) | Err => Result::Err(()),
            other => bug!("unrecognized res {:?}", other),
        }
    }
}

/// The link failed to resolve. [`resolution_failure`] should look to see if there's
/// a more helpful error that can be given.
#[derive(Debug)]
struct UnresolvedPath<'a> {
    /// Item on which the link is resolved, used for resolving `Self`.
    item_id: DefId,
    /// The scope the link was resolved in.
    module_id: DefId,
    /// If part of the link resolved, this has the `Res`.
    ///
    /// In `[std::io::Error::x]`, `std::io::Error` would be a partial resolution.
    partial_res: Option<Res>,
    /// The remaining unresolved path segments.
    ///
    /// In `[std::io::Error::x]`, `x` would be unresolved.
    unresolved: Cow<'a, str>,
}

#[derive(Debug)]
enum ResolutionFailure<'a> {
    /// This resolved, but with the wrong namespace.
    WrongNamespace {
        /// What the link resolved to.
        res: Res,
        /// The expected namespace for the resolution, determined from the link's disambiguator.
        ///
        /// E.g., for `[fn@Result]` this is [`Namespace::ValueNS`],
        /// even though `Result`'s actual namespace is [`Namespace::TypeNS`].
        expected_ns: Namespace,
    },
    NotResolved(UnresolvedPath<'a>),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) enum UrlFragment {
    Item(DefId),
    /// A part of a page that isn't a rust item.
    ///
    /// Eg: `[Vector Examples](std::vec::Vec#examples)`
    UserWritten(String),
}

impl UrlFragment {
    /// Render the fragment, including the leading `#`.
    pub(crate) fn render(&self, s: &mut String, tcx: TyCtxt<'_>) {
        s.push('#');
        match self {
            &UrlFragment::Item(def_id) => {
                let kind = match tcx.def_kind(def_id) {
                    DefKind::AssocFn => {
                        if tcx.defaultness(def_id).has_value() {
                            "method."
                        } else {
                            "tymethod."
                        }
                    }
                    DefKind::AssocConst => "associatedconstant.",
                    DefKind::AssocTy => "associatedtype.",
                    DefKind::Variant => "variant.",
                    DefKind::Field => {
                        let parent_id = tcx.parent(def_id);
                        if tcx.def_kind(parent_id) == DefKind::Variant {
                            s.push_str("variant.");
                            s.push_str(tcx.item_name(parent_id).as_str());
                            ".field."
                        } else {
                            "structfield."
                        }
                    }
                    kind => bug!("unexpected associated item kind: {:?}", kind),
                };
                s.push_str(kind);
                s.push_str(tcx.item_name(def_id).as_str());
            }
            UrlFragment::UserWritten(raw) => s.push_str(&raw),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ResolutionInfo {
    item_id: DefId,
    module_id: DefId,
    dis: Option<Disambiguator>,
    path_str: Box<str>,
    extra_fragment: Option<String>,
}

#[derive(Clone)]
struct DiagnosticInfo<'a> {
    item: &'a Item,
    dox: &'a str,
    ori_link: &'a str,
    link_range: MarkdownLinkRange,
}

struct LinkCollector<'a, 'tcx> {
    cx: &'a mut DocContext<'tcx>,
    /// Cache the resolved links so we can avoid resolving (and emitting errors for) the same link.
    /// The link will be `None` if it could not be resolved (i.e. the error was cached).
    visited_links: FxHashMap<ResolutionInfo, Option<(Res, Option<UrlFragment>)>>,
}

impl<'a, 'tcx> LinkCollector<'a, 'tcx> {
    /// Given a full link, parse it as an [enum struct variant].
    ///
    /// In particular, this will return an error whenever there aren't three
    /// full path segments left in the link.
    ///
    /// [enum struct variant]: rustc_hir::VariantData::Struct
    fn variant_field<'path>(
        &self,
        path_str: &'path str,
        item_id: DefId,
        module_id: DefId,
    ) -> Result<(Res, DefId), UnresolvedPath<'path>> {
        let tcx = self.cx.tcx;
        let no_res = || UnresolvedPath {
            item_id,
            module_id,
            partial_res: None,
            unresolved: path_str.into(),
        };

        debug!("looking for enum variant {}", path_str);
        let mut split = path_str.rsplitn(3, "::");
        let variant_field_name = split
            .next()
            .map(|f| Symbol::intern(f))
            .expect("fold_item should ensure link is non-empty");
        let variant_name =
            // we're not sure this is a variant at all, so use the full string
            // If there's no second component, the link looks like `[path]`.
            // So there's no partial res and we should say the whole link failed to resolve.
            split.next().map(|f|  Symbol::intern(f)).ok_or_else(no_res)?;
        let path = split
            .next()
            // If there's no third component, we saw `[a::b]` before and it failed to resolve.
            // So there's no partial res.
            .ok_or_else(no_res)?;
        let ty_res = self.resolve_path(&path, TypeNS, item_id, module_id).ok_or_else(no_res)?;

        match ty_res {
            Res::Def(DefKind::Enum, did) => match tcx.type_of(did).subst_identity().kind() {
                ty::Adt(def, _) if def.is_enum() => {
                    if let Some(variant) = def.variants().iter().find(|v| v.name == variant_name)
                        && let Some(field) = variant.fields.iter().find(|f| f.name == variant_field_name) {
                        Ok((ty_res, field.did))
                    } else {
                        Err(UnresolvedPath {
                            item_id,
                            module_id,
                            partial_res: Some(Res::Def(DefKind::Enum, def.did())),
                            unresolved: variant_field_name.to_string().into(),
                        })
                    }
                }
                _ => unreachable!(),
            },
            _ => Err(UnresolvedPath {
                item_id,
                module_id,
                partial_res: Some(ty_res),
                unresolved: variant_name.to_string().into(),
            }),
        }
    }

    /// Given a primitive type, try to resolve an associated item.
    fn resolve_primitive_associated_item(
        &self,
        prim_ty: PrimitiveType,
        ns: Namespace,
        item_name: Symbol,
    ) -> Vec<(Res, DefId)> {
        let tcx = self.cx.tcx;

        prim_ty
            .impls(tcx)
            .flat_map(|impl_| {
                filter_assoc_items_by_name_and_namespace(
                    tcx,
                    impl_,
                    Ident::with_dummy_span(item_name),
                    ns,
                )
                .map(|item| (Res::Primitive(prim_ty), item.def_id))
            })
            .collect::<Vec<_>>()
    }

    fn resolve_self_ty(&self, path_str: &str, ns: Namespace, item_id: DefId) -> Option<Res> {
        if ns != TypeNS || path_str != "Self" {
            return None;
        }

        let tcx = self.cx.tcx;
        let self_id = match tcx.def_kind(item_id) {
            def_kind @ (DefKind::AssocFn
            | DefKind::AssocConst
            | DefKind::AssocTy
            | DefKind::Variant
            | DefKind::Field) => {
                let parent_def_id = tcx.parent(item_id);
                if def_kind == DefKind::Field && tcx.def_kind(parent_def_id) == DefKind::Variant {
                    tcx.parent(parent_def_id)
                } else {
                    parent_def_id
                }
            }
            _ => item_id,
        };

        match tcx.def_kind(self_id) {
            DefKind::Impl { .. } => self.def_id_to_res(self_id),
            DefKind::Use => None,
            def_kind => Some(Res::Def(def_kind, self_id)),
        }
    }

    /// Convenience wrapper around `doc_link_resolutions`.
    ///
    /// This also handles resolving `true` and `false` as booleans.
    /// NOTE: `doc_link_resolutions` knows only about paths, not about types.
    /// Associated items will never be resolved by this function.
    fn resolve_path(
        &self,
        path_str: &str,
        ns: Namespace,
        item_id: DefId,
        module_id: DefId,
    ) -> Option<Res> {
        if let res @ Some(..) = self.resolve_self_ty(path_str, ns, item_id) {
            return res;
        }

        // Resolver doesn't know about true, false, and types that aren't paths (e.g. `()`).
        let result = self
            .cx
            .tcx
            .doc_link_resolutions(module_id)
            .get(&(Symbol::intern(path_str), ns))
            .copied()
            .unwrap_or_else(|| panic!("no resolution for {:?} {:?} {:?}", path_str, ns, module_id))
            .and_then(|res| res.try_into().ok())
            .or_else(|| resolve_primitive(path_str, ns));
        debug!("{} resolved to {:?} in namespace {:?}", path_str, result, ns);
        result
    }

    /// Resolves a string as a path within a particular namespace. Returns an
    /// optional URL fragment in the case of variants and methods.
    fn resolve<'path>(
        &mut self,
        path_str: &'path str,
        ns: Namespace,
        item_id: DefId,
        module_id: DefId,
    ) -> Result<Vec<(Res, Option<DefId>)>, UnresolvedPath<'path>> {
        if let Some(res) = self.resolve_path(path_str, ns, item_id, module_id) {
            return Ok(match res {
                Res::Def(
                    DefKind::AssocFn | DefKind::AssocConst | DefKind::AssocTy | DefKind::Variant,
                    def_id,
                ) => {
                    vec![(Res::from_def_id(self.cx.tcx, self.cx.tcx.parent(def_id)), Some(def_id))]
                }
                _ => vec![(res, None)],
            });
        } else if ns == MacroNS {
            return Err(UnresolvedPath {
                item_id,
                module_id,
                partial_res: None,
                unresolved: path_str.into(),
            });
        }

        // Try looking for methods and associated items.
        let mut split = path_str.rsplitn(2, "::");
        // NB: `split`'s first element is always defined, even if the delimiter was not present.
        // NB: `item_str` could be empty when resolving in the root namespace (e.g. `::std`).
        let item_str = split.next().unwrap();
        let item_name = Symbol::intern(item_str);
        let path_root = split
            .next()
            // If there's no `::`, it's not an associated item.
            // So we can be sure that `rustc_resolve` was accurate when it said it wasn't resolved.
            .ok_or_else(|| {
                debug!("found no `::`, assuming {} was correctly not in scope", item_name);
                UnresolvedPath {
                    item_id,
                    module_id,
                    partial_res: None,
                    unresolved: item_str.into(),
                }
            })?;

        // FIXME(#83862): this arbitrarily gives precedence to primitives over modules to support
        // links to primitives when `#[rustc_doc_primitive]` is present. It should give an ambiguity
        // error instead and special case *only* modules with `#[rustc_doc_primitive]`, not all
        // primitives.
        match resolve_primitive(&path_root, TypeNS)
            .or_else(|| self.resolve_path(&path_root, TypeNS, item_id, module_id))
            .and_then(|ty_res| {
                let candidates = self
                    .resolve_associated_item(ty_res, item_name, ns, module_id)
                    .into_iter()
                    .map(|(res, def_id)| (res, Some(def_id)))
                    .collect::<Vec<_>>();
                if !candidates.is_empty() { Some(candidates) } else { None }
            }) {
            Some(r) => Ok(r),
            None => {
                if ns == Namespace::ValueNS {
                    self.variant_field(path_str, item_id, module_id)
                        .map(|(res, def_id)| vec![(res, Some(def_id))])
                } else {
                    Err(UnresolvedPath {
                        item_id,
                        module_id,
                        partial_res: None,
                        unresolved: path_root.into(),
                    })
                }
            }
        }
    }

    /// Convert a DefId to a Res, where possible.
    ///
    /// This is used for resolving type aliases.
    fn def_id_to_res(&self, ty_id: DefId) -> Option<Res> {
        use PrimitiveType::*;
        Some(match *self.cx.tcx.type_of(ty_id).subst_identity().kind() {
            ty::Bool => Res::Primitive(Bool),
            ty::Char => Res::Primitive(Char),
            ty::Int(ity) => Res::Primitive(ity.into()),
            ty::Uint(uty) => Res::Primitive(uty.into()),
            ty::Float(fty) => Res::Primitive(fty.into()),
            ty::Str => Res::Primitive(Str),
            ty::Tuple(tys) if tys.is_empty() => Res::Primitive(Unit),
            ty::Tuple(_) => Res::Primitive(Tuple),
            ty::Array(..) => Res::Primitive(Array),
            ty::Slice(_) => Res::Primitive(Slice),
            ty::RawPtr(_) => Res::Primitive(RawPointer),
            ty::Ref(..) => Res::Primitive(Reference),
            ty::FnDef(..) => panic!("type alias to a function definition"),
            ty::FnPtr(_) => Res::Primitive(Fn),
            ty::Never => Res::Primitive(Never),
            ty::Adt(ty::AdtDef(Interned(&ty::AdtDefData { did, .. }, _)), _) | ty::Foreign(did) => {
                Res::from_def_id(self.cx.tcx, did)
            }
            ty::Alias(..)
            | ty::Closure(..)
            | ty::Generator(..)
            | ty::GeneratorWitness(_)
            | ty::GeneratorWitnessMIR(..)
            | ty::Dynamic(..)
            | ty::Param(_)
            | ty::Bound(..)
            | ty::Placeholder(_)
            | ty::Infer(_)
            | ty::Error(_) => return None,
        })
    }

    /// Convert a PrimitiveType to a Ty, where possible.
    ///
    /// This is used for resolving trait impls for primitives
    fn primitive_type_to_ty(&mut self, prim: PrimitiveType) -> Option<Ty<'tcx>> {
        use PrimitiveType::*;
        let tcx = self.cx.tcx;

        // FIXME: Only simple types are supported here, see if we can support
        // other types such as Tuple, Array, Slice, etc.
        // See https://github.com/rust-lang/rust/issues/90703#issuecomment-1004263455
        Some(match prim {
            Bool => tcx.types.bool,
            Str => tcx.types.str_,
            Char => tcx.types.char,
            Never => tcx.types.never,
            I8 => tcx.types.i8,
            I16 => tcx.types.i16,
            I32 => tcx.types.i32,
            I64 => tcx.types.i64,
            I128 => tcx.types.i128,
            Isize => tcx.types.isize,
            F32 => tcx.types.f32,
            F64 => tcx.types.f64,
            U8 => tcx.types.u8,
            U16 => tcx.types.u16,
            U32 => tcx.types.u32,
            U64 => tcx.types.u64,
            U128 => tcx.types.u128,
            Usize => tcx.types.usize,
            _ => return None,
        })
    }

    /// Resolve an associated item, returning its containing page's `Res`
    /// and the fragment targeting the associated item on its page.
    fn resolve_associated_item(
        &mut self,
        root_res: Res,
        item_name: Symbol,
        ns: Namespace,
        module_id: DefId,
    ) -> Vec<(Res, DefId)> {
        let tcx = self.cx.tcx;

        match root_res {
            Res::Primitive(prim) => {
                let items = self.resolve_primitive_associated_item(prim, ns, item_name);
                if !items.is_empty() {
                    items
                // Inherent associated items take precedence over items that come from trait impls.
                } else {
                    self.primitive_type_to_ty(prim)
                        .map(|ty| {
                            resolve_associated_trait_item(ty, module_id, item_name, ns, self.cx)
                                .iter()
                                .map(|item| (root_res, item.def_id))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or(Vec::new())
                }
            }
            Res::Def(DefKind::TyAlias, did) => {
                // Resolve the link on the type the alias points to.
                // FIXME: if the associated item is defined directly on the type alias,
                // it will show up on its documentation page, we should link there instead.
                let Some(res) = self.def_id_to_res(did) else { return Vec::new() };
                self.resolve_associated_item(res, item_name, ns, module_id)
            }
            Res::Def(
                def_kind @ (DefKind::Struct | DefKind::Union | DefKind::Enum | DefKind::ForeignTy),
                did,
            ) => {
                debug!("looking for associated item named {} for item {:?}", item_name, did);
                // Checks if item_name is a variant of the `SomeItem` enum
                if ns == TypeNS && def_kind == DefKind::Enum {
                    match tcx.type_of(did).subst_identity().kind() {
                        ty::Adt(adt_def, _) => {
                            for variant in adt_def.variants() {
                                if variant.name == item_name {
                                    return vec![(root_res, variant.def_id)];
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                }

                // Checks if item_name belongs to `impl SomeItem`
                let mut assoc_items: Vec<_> = tcx
                    .inherent_impls(did)
                    .iter()
                    .flat_map(|&imp| {
                        filter_assoc_items_by_name_and_namespace(
                            tcx,
                            imp,
                            Ident::with_dummy_span(item_name),
                            ns,
                        )
                    })
                    .map(|item| (root_res, item.def_id))
                    .collect();

                if assoc_items.is_empty() {
                    // Check if item_name belongs to `impl SomeTrait for SomeItem`
                    // FIXME(#74563): This gives precedence to `impl SomeItem`:
                    // Although having both would be ambiguous, use impl version for compatibility's sake.
                    // To handle that properly resolve() would have to support
                    // something like [`ambi_fn`](<SomeStruct as SomeTrait>::ambi_fn)
                    assoc_items = resolve_associated_trait_item(
                        tcx.type_of(did).subst_identity(),
                        module_id,
                        item_name,
                        ns,
                        self.cx,
                    )
                    .into_iter()
                    .map(|item| (root_res, item.def_id))
                    .collect::<Vec<_>>();
                }

                debug!("got associated item {:?}", assoc_items);

                if !assoc_items.is_empty() {
                    return assoc_items;
                }

                if ns != Namespace::ValueNS {
                    return Vec::new();
                }
                debug!("looking for fields named {} for {:?}", item_name, did);
                // FIXME: this doesn't really belong in `associated_item` (maybe `variant_field` is better?)
                // NOTE: it's different from variant_field because it only resolves struct fields,
                // not variant fields (2 path segments, not 3).
                //
                // We need to handle struct (and union) fields in this code because
                // syntactically their paths are identical to associated item paths:
                // `module::Type::field` and `module::Type::Assoc`.
                //
                // On the other hand, variant fields can't be mistaken for associated
                // items because they look like this: `module::Type::Variant::field`.
                //
                // Variants themselves don't need to be handled here, even though
                // they also look like associated items (`module::Type::Variant`),
                // because they are real Rust syntax (unlike the intra-doc links
                // field syntax) and are handled by the compiler's resolver.
                let def = match tcx.type_of(did).subst_identity().kind() {
                    ty::Adt(def, _) if !def.is_enum() => def,
                    _ => return Vec::new(),
                };
                def.non_enum_variant()
                    .fields
                    .iter()
                    .filter(|field| field.name == item_name)
                    .map(|field| (root_res, field.did))
                    .collect::<Vec<_>>()
            }
            Res::Def(DefKind::Trait, did) => filter_assoc_items_by_name_and_namespace(
                tcx,
                did,
                Ident::with_dummy_span(item_name),
                ns,
            )
            .map(|item| {
                let res = Res::Def(item.kind.as_def_kind(), item.def_id);
                (res, item.def_id)
            })
            .collect::<Vec<_>>(),
            _ => Vec::new(),
        }
    }
}

fn full_res(tcx: TyCtxt<'_>, (base, assoc_item): (Res, Option<DefId>)) -> Res {
    assoc_item.map_or(base, |def_id| Res::from_def_id(tcx, def_id))
}

/// Look to see if a resolved item has an associated item named `item_name`.
///
/// Given `[std::io::Error::source]`, where `source` is unresolved, this would
/// find `std::error::Error::source` and return
/// `<io::Error as error::Error>::source`.
fn resolve_associated_trait_item<'a>(
    ty: Ty<'a>,
    module: DefId,
    item_name: Symbol,
    ns: Namespace,
    cx: &mut DocContext<'a>,
) -> Vec<ty::AssocItem> {
    // FIXME: this should also consider blanket impls (`impl<T> X for T`). Unfortunately
    // `get_auto_trait_and_blanket_impls` is broken because the caching behavior is wrong. In the
    // meantime, just don't look for these blanket impls.

    // Next consider explicit impls: `impl MyTrait for MyType`
    // Give precedence to inherent impls.
    let traits = trait_impls_for(cx, ty, module);
    let tcx = cx.tcx;
    debug!("considering traits {:?}", traits);
    let candidates = traits
        .iter()
        .flat_map(|&(impl_, trait_)| {
            filter_assoc_items_by_name_and_namespace(
                tcx,
                trait_,
                Ident::with_dummy_span(item_name),
                ns,
            )
            .map(move |trait_assoc| {
                trait_assoc_to_impl_assoc_item(tcx, impl_, trait_assoc.def_id)
                    .unwrap_or(*trait_assoc)
            })
        })
        .collect::<Vec<_>>();
    // FIXME(#74563): warn about ambiguity
    debug!("the candidates were {:?}", candidates);
    candidates
}

/// Find the associated item in the impl `impl_id` that corresponds to the
/// trait associated item `trait_assoc_id`.
///
/// This function returns `None` if no associated item was found in the impl.
/// This can occur when the trait associated item has a default value that is
/// not overridden in the impl.
///
/// This is just a wrapper around [`TyCtxt::impl_item_implementor_ids()`] and
/// [`TyCtxt::associated_item()`] (with some helpful logging added).
#[instrument(level = "debug", skip(tcx), ret)]
fn trait_assoc_to_impl_assoc_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_id: DefId,
    trait_assoc_id: DefId,
) -> Option<ty::AssocItem> {
    let trait_to_impl_assoc_map = tcx.impl_item_implementor_ids(impl_id);
    debug!(?trait_to_impl_assoc_map);
    let impl_assoc_id = *trait_to_impl_assoc_map.get(&trait_assoc_id)?;
    debug!(?impl_assoc_id);
    Some(tcx.associated_item(impl_assoc_id))
}

/// Given a type, return all trait impls in scope in `module` for that type.
/// Returns a set of pairs of `(impl_id, trait_id)`.
///
/// NOTE: this cannot be a query because more traits could be available when more crates are compiled!
/// So it is not stable to serialize cross-crate.
#[instrument(level = "debug", skip(cx))]
fn trait_impls_for<'a>(
    cx: &mut DocContext<'a>,
    ty: Ty<'a>,
    module: DefId,
) -> FxHashSet<(DefId, DefId)> {
    let tcx = cx.tcx;
    let mut impls = FxHashSet::default();

    for &trait_ in tcx.doc_link_traits_in_scope(module) {
        tcx.for_each_relevant_impl(trait_, ty, |impl_| {
            let trait_ref = tcx.impl_trait_ref(impl_).expect("this is not an inherent impl");
            // Check if these are the same type.
            let impl_type = trait_ref.skip_binder().self_ty();
            trace!(
                "comparing type {} with kind {:?} against type {:?}",
                impl_type,
                impl_type.kind(),
                ty
            );
            // Fast path: if this is a primitive simple `==` will work
            // NOTE: the `match` is necessary; see #92662.
            // this allows us to ignore generics because the user input
            // may not include the generic placeholders
            // e.g. this allows us to match Foo (user comment) with Foo<T> (actual type)
            let saw_impl = impl_type == ty
                || match (impl_type.kind(), ty.kind()) {
                    (ty::Adt(impl_def, _), ty::Adt(ty_def, _)) => {
                        debug!("impl def_id: {:?}, ty def_id: {:?}", impl_def.did(), ty_def.did());
                        impl_def.did() == ty_def.did()
                    }
                    _ => false,
                };

            if saw_impl {
                impls.insert((impl_, trait_));
            }
        });
    }

    impls
}

/// Check for resolve collisions between a trait and its derive.
///
/// These are common and we should just resolve to the trait in that case.
fn is_derive_trait_collision<T>(ns: &PerNS<Result<Vec<(Res, T)>, ResolutionFailure<'_>>>) -> bool {
    if let (Ok(type_ns), Ok(macro_ns)) = (&ns.type_ns, &ns.macro_ns) {
        type_ns.iter().any(|(res, _)| matches!(res, Res::Def(DefKind::Trait, _)))
            && macro_ns
                .iter()
                .any(|(res, _)| matches!(res, Res::Def(DefKind::Macro(MacroKind::Derive), _)))
    } else {
        false
    }
}

impl<'a, 'tcx> DocVisitor for LinkCollector<'a, 'tcx> {
    fn visit_item(&mut self, item: &Item) {
        self.resolve_links(item);
        self.visit_item_recur(item)
    }
}

enum PreprocessingError {
    /// User error: `[std#x#y]` is not valid
    MultipleAnchors,
    Disambiguator(MarkdownLinkRange, String),
    MalformedGenerics(MalformedGenerics, String),
}

impl PreprocessingError {
    fn report(&self, cx: &DocContext<'_>, diag_info: DiagnosticInfo<'_>) {
        match self {
            PreprocessingError::MultipleAnchors => report_multiple_anchors(cx, diag_info),
            PreprocessingError::Disambiguator(range, msg) => {
                disambiguator_error(cx, diag_info, range.clone(), msg.clone())
            }
            PreprocessingError::MalformedGenerics(err, path_str) => {
                report_malformed_generics(cx, diag_info, *err, path_str)
            }
        }
    }
}

#[derive(Clone)]
struct PreprocessingInfo {
    path_str: Box<str>,
    disambiguator: Option<Disambiguator>,
    extra_fragment: Option<String>,
    link_text: Box<str>,
}

// Not a typedef to avoid leaking several private structures from this module.
pub(crate) struct PreprocessedMarkdownLink(
    Result<PreprocessingInfo, PreprocessingError>,
    MarkdownLink,
);

/// Returns:
/// - `None` if the link should be ignored.
/// - `Some(Err)` if the link should emit an error
/// - `Some(Ok)` if the link is valid
///
/// `link_buffer` is needed for lifetime reasons; it will always be overwritten and the contents ignored.
fn preprocess_link(
    ori_link: &MarkdownLink,
    dox: &str,
) -> Option<Result<PreprocessingInfo, PreprocessingError>> {
    // [] is mostly likely not supposed to be a link
    if ori_link.link.is_empty() {
        return None;
    }

    // Bail early for real links.
    if ori_link.link.contains('/') {
        return None;
    }

    let stripped = ori_link.link.replace('`', "");
    let mut parts = stripped.split('#');

    let link = parts.next().unwrap();
    let link = link.trim();
    if link.is_empty() {
        // This is an anchor to an element of the current page, nothing to do in here!
        return None;
    }
    let extra_fragment = parts.next();
    if parts.next().is_some() {
        // A valid link can't have multiple #'s
        return Some(Err(PreprocessingError::MultipleAnchors));
    }

    // Parse and strip the disambiguator from the link, if present.
    let (disambiguator, path_str, link_text) = match Disambiguator::from_str(link) {
        Ok(Some((d, path, link_text))) => (Some(d), path.trim(), link_text.trim()),
        Ok(None) => (None, link, link),
        Err((err_msg, relative_range)) => {
            // Only report error if we would not have ignored this link. See issue #83859.
            if !should_ignore_link_with_disambiguators(link) {
                let disambiguator_range = match range_between_backticks(&ori_link.range, dox) {
                    MarkdownLinkRange::Destination(no_backticks_range) => {
                        MarkdownLinkRange::Destination(
                            (no_backticks_range.start + relative_range.start)
                                ..(no_backticks_range.start + relative_range.end),
                        )
                    }
                    mdlr @ MarkdownLinkRange::WholeLink(_) => mdlr,
                };
                return Some(Err(PreprocessingError::Disambiguator(disambiguator_range, err_msg)));
            } else {
                return None;
            }
        }
    };

    if should_ignore_link(path_str) {
        return None;
    }

    // Strip generics from the path.
    let path_str = match strip_generics_from_path(path_str) {
        Ok(path) => path,
        Err(err) => {
            debug!("link has malformed generics: {}", path_str);
            return Some(Err(PreprocessingError::MalformedGenerics(err, path_str.to_owned())));
        }
    };

    // Sanity check to make sure we don't have any angle brackets after stripping generics.
    assert!(!path_str.contains(['<', '>'].as_slice()));

    // The link is not an intra-doc link if it still contains spaces after stripping generics.
    if path_str.contains(' ') {
        return None;
    }

    Some(Ok(PreprocessingInfo {
        path_str,
        disambiguator,
        extra_fragment: extra_fragment.map(|frag| frag.to_owned()),
        link_text: Box::<str>::from(link_text),
    }))
}

fn preprocessed_markdown_links(s: &str) -> Vec<PreprocessedMarkdownLink> {
    markdown_links(s, |link| {
        preprocess_link(&link, s).map(|pp_link| PreprocessedMarkdownLink(pp_link, link))
    })
}

impl LinkCollector<'_, '_> {
    fn resolve_links(&mut self, item: &Item) {
        if !self.cx.render_options.document_private
            && let Some(def_id) = item.item_id.as_def_id()
            && let Some(def_id) = def_id.as_local()
            && !self.cx.tcx.effective_visibilities(()).is_exported(def_id)
            && !has_primitive_or_keyword_docs(&item.attrs.other_attrs) {
            // Skip link resolution for non-exported items.
            return;
        }

        // We want to resolve in the lexical scope of the documentation.
        // In the presence of re-exports, this is not the same as the module of the item.
        // Rather than merging all documentation into one, resolve it one attribute at a time
        // so we know which module it came from.
        for (item_id, doc) in prepare_to_doc_link_resolution(&item.attrs.doc_strings) {
            if !may_have_doc_links(&doc) {
                continue;
            }
            debug!("combined_docs={}", doc);
            // NOTE: if there are links that start in one crate and end in another, this will not resolve them.
            // This is a degenerate case and it's not supported by rustdoc.
            let item_id = item_id.unwrap_or_else(|| item.item_id.expect_def_id());
            let module_id = match self.cx.tcx.def_kind(item_id) {
                DefKind::Mod if item.inner_docs(self.cx.tcx) => item_id,
                _ => find_nearest_parent_module(self.cx.tcx, item_id).unwrap(),
            };
            for md_link in preprocessed_markdown_links(&doc) {
                let link = self.resolve_link(item, item_id, module_id, &doc, &md_link);
                if let Some(link) = link {
                    self.cx.cache.intra_doc_links.entry(item.item_id).or_default().insert(link);
                }
            }
        }
    }

    /// This is the entry point for resolving an intra-doc link.
    ///
    /// FIXME(jynelson): this is way too many arguments
    fn resolve_link(
        &mut self,
        item: &Item,
        item_id: DefId,
        module_id: DefId,
        dox: &str,
        link: &PreprocessedMarkdownLink,
    ) -> Option<ItemLink> {
        let PreprocessedMarkdownLink(pp_link, ori_link) = link;
        trace!("considering link '{}'", ori_link.link);

        let diag_info = DiagnosticInfo {
            item,
            dox,
            ori_link: &ori_link.link,
            link_range: ori_link.range.clone(),
        };

        let PreprocessingInfo { path_str, disambiguator, extra_fragment, link_text } =
            pp_link.as_ref().map_err(|err| err.report(self.cx, diag_info.clone())).ok()?;
        let disambiguator = *disambiguator;

        let (mut res, fragment) = self.resolve_with_disambiguator_cached(
            ResolutionInfo {
                item_id,
                module_id,
                dis: disambiguator,
                path_str: path_str.clone(),
                extra_fragment: extra_fragment.clone(),
            },
            diag_info.clone(), // this struct should really be Copy, but Range is not :(
            // For reference-style links we want to report only one error so unsuccessful
            // resolutions are cached, for other links we want to report an error every
            // time so they are not cached.
            matches!(ori_link.kind, LinkType::Reference | LinkType::Shortcut),
        )?;

        // Check for a primitive which might conflict with a module
        // Report the ambiguity and require that the user specify which one they meant.
        // FIXME: could there ever be a primitive not in the type namespace?
        if matches!(
            disambiguator,
            None | Some(Disambiguator::Namespace(Namespace::TypeNS) | Disambiguator::Primitive)
        ) && !matches!(res, Res::Primitive(_))
        {
            if let Some(prim) = resolve_primitive(path_str, TypeNS) {
                // `prim@char`
                if matches!(disambiguator, Some(Disambiguator::Primitive)) {
                    res = prim;
                } else {
                    // `[char]` when a `char` module is in scope
                    let candidates = &[(res, res.def_id(self.cx.tcx)), (prim, None)];
                    ambiguity_error(self.cx, &diag_info, path_str, candidates);
                    return None;
                }
            }
        }

        match res {
            Res::Primitive(_) => {
                if let Some(UrlFragment::Item(id)) = fragment {
                    // We're actually resolving an associated item of a primitive, so we need to
                    // verify the disambiguator (if any) matches the type of the associated item.
                    // This case should really follow the same flow as the `Res::Def` branch below,
                    // but attempting to add a call to `clean::register_res` causes an ICE. @jyn514
                    // thinks `register_res` is only needed for cross-crate re-exports, but Rust
                    // doesn't allow statements like `use str::trim;`, making this a (hopefully)
                    // valid omission. See https://github.com/rust-lang/rust/pull/80660#discussion_r551585677
                    // for discussion on the matter.
                    let kind = self.cx.tcx.def_kind(id);
                    self.verify_disambiguator(path_str, kind, id, disambiguator, item, &diag_info)?;
                } else {
                    match disambiguator {
                        Some(Disambiguator::Primitive | Disambiguator::Namespace(_)) | None => {}
                        Some(other) => {
                            self.report_disambiguator_mismatch(path_str, other, res, &diag_info);
                            return None;
                        }
                    }
                }

                res.def_id(self.cx.tcx).map(|page_id| ItemLink {
                    link: Box::<str>::from(&*ori_link.link),
                    link_text: link_text.clone(),
                    page_id,
                    fragment,
                })
            }
            Res::Def(kind, id) => {
                let (kind_for_dis, id_for_dis) = if let Some(UrlFragment::Item(id)) = fragment {
                    (self.cx.tcx.def_kind(id), id)
                } else {
                    (kind, id)
                };
                self.verify_disambiguator(
                    path_str,
                    kind_for_dis,
                    id_for_dis,
                    disambiguator,
                    item,
                    &diag_info,
                )?;

                let page_id = clean::register_res(self.cx, rustc_hir::def::Res::Def(kind, id));
                Some(ItemLink {
                    link: Box::<str>::from(&*ori_link.link),
                    link_text: link_text.clone(),
                    page_id,
                    fragment,
                })
            }
        }
    }

    fn verify_disambiguator(
        &self,
        path_str: &str,
        kind: DefKind,
        id: DefId,
        disambiguator: Option<Disambiguator>,
        item: &Item,
        diag_info: &DiagnosticInfo<'_>,
    ) -> Option<()> {
        debug!("intra-doc link to {} resolved to {:?}", path_str, (kind, id));

        // Disallow e.g. linking to enums with `struct@`
        debug!("saw kind {:?} with disambiguator {:?}", kind, disambiguator);
        match (kind, disambiguator) {
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
                (_, Some(specified @ Disambiguator::Kind(_) | specified @ Disambiguator::Primitive)) => {
                    self.report_disambiguator_mismatch(path_str, specified, Res::Def(kind, id), diag_info);
                    return None;
                }
            }

        // item can be non-local e.g. when using `#[rustc_doc_primitive = "pointer"]`
        if let Some((src_id, dst_id)) = id.as_local().and_then(|dst_id| {
            item.item_id.expect_def_id().as_local().map(|src_id| (src_id, dst_id))
        }) {
            if self.cx.tcx.effective_visibilities(()).is_exported(src_id)
                && !self.cx.tcx.effective_visibilities(()).is_exported(dst_id)
            {
                privacy_error(self.cx, diag_info, path_str);
            }
        }

        Some(())
    }

    fn report_disambiguator_mismatch(
        &self,
        path_str: &str,
        specified: Disambiguator,
        resolved: Res,
        diag_info: &DiagnosticInfo<'_>,
    ) {
        // The resolved item did not match the disambiguator; give a better error than 'not found'
        let msg = format!("incompatible link kind for `{}`", path_str);
        let callback = |diag: &mut Diagnostic, sp: Option<rustc_span::Span>, link_range| {
            let note = format!(
                "this link resolved to {} {}, which is not {} {}",
                resolved.article(),
                resolved.descr(),
                specified.article(),
                specified.descr(),
            );
            if let Some(sp) = sp {
                diag.span_label(sp, note);
            } else {
                diag.note(note);
            }
            suggest_disambiguator(resolved, diag, path_str, link_range, sp, diag_info);
        };
        report_diagnostic(self.cx.tcx, BROKEN_INTRA_DOC_LINKS, msg, diag_info, callback);
    }

    fn report_rawptr_assoc_feature_gate(
        &self,
        dox: &str,
        ori_link: &MarkdownLinkRange,
        item: &Item,
    ) {
        let span = super::source_span_for_markdown_range(
            self.cx.tcx,
            dox,
            ori_link.inner_range(),
            &item.attrs,
        )
        .unwrap_or_else(|| item.attr_span(self.cx.tcx));
        rustc_session::parse::feature_err(
            &self.cx.tcx.sess.parse_sess,
            sym::intra_doc_pointers,
            span,
            "linking to associated items of raw pointers is experimental",
        )
        .note("rustdoc does not allow disambiguating between `*const` and `*mut`, and pointers are unstable until it does")
        .emit();
    }

    fn resolve_with_disambiguator_cached(
        &mut self,
        key: ResolutionInfo,
        diag: DiagnosticInfo<'_>,
        // If errors are cached then they are only reported on first occurrence
        // which we want in some cases but not in others.
        cache_errors: bool,
    ) -> Option<(Res, Option<UrlFragment>)> {
        if let Some(res) = self.visited_links.get(&key) {
            if res.is_some() || cache_errors {
                return res.clone();
            }
        }

        let mut candidates = self.resolve_with_disambiguator(&key, diag.clone());

        // FIXME: it would be nice to check that the feature gate was enabled in the original crate, not just ignore it altogether.
        // However I'm not sure how to check that across crates.
        if let Some(candidate) = candidates.get(0) &&
            candidate.0 == Res::Primitive(PrimitiveType::RawPointer) &&
            key.path_str.contains("::") // We only want to check this if this is an associated item.
        {
            if key.item_id.is_local() && !self.cx.tcx.features().intra_doc_pointers {
                self.report_rawptr_assoc_feature_gate(diag.dox, &diag.link_range, diag.item);
                return None;
            } else {
                candidates = vec![candidates[0]];
            }
        }

        // If there are multiple items with the same "kind" (for example, both "associated types")
        // and after removing duplicated kinds, only one remains, the `ambiguity_error` function
        // won't emit an error. So at this point, we can just take the first candidate as it was
        // the first retrieved and use it to generate the link.
        if candidates.len() > 1 && !ambiguity_error(self.cx, &diag, &key.path_str, &candidates) {
            candidates = vec![candidates[0]];
        }

        if let &[(res, def_id)] = candidates.as_slice() {
            let fragment = match (&key.extra_fragment, def_id) {
                (Some(_), Some(def_id)) => {
                    report_anchor_conflict(self.cx, diag, def_id);
                    return None;
                }
                (Some(u_frag), None) => Some(UrlFragment::UserWritten(u_frag.clone())),
                (None, Some(def_id)) => Some(UrlFragment::Item(def_id)),
                (None, None) => None,
            };
            let r = Some((res, fragment));
            self.visited_links.insert(key, r.clone());
            return r;
        }

        if cache_errors {
            self.visited_links.insert(key, None);
        }
        None
    }

    /// After parsing the disambiguator, resolve the main part of the link.
    // FIXME(jynelson): wow this is just so much
    fn resolve_with_disambiguator(
        &mut self,
        key: &ResolutionInfo,
        diag: DiagnosticInfo<'_>,
    ) -> Vec<(Res, Option<DefId>)> {
        let disambiguator = key.dis;
        let path_str = &key.path_str;
        let item_id = key.item_id;
        let module_id = key.module_id;

        match disambiguator.map(Disambiguator::ns) {
            Some(expected_ns) => {
                match self.resolve(path_str, expected_ns, item_id, module_id) {
                    Ok(candidates) => candidates,
                    Err(err) => {
                        // We only looked in one namespace. Try to give a better error if possible.
                        // FIXME: really it should be `resolution_failure` that does this, not `resolve_with_disambiguator`.
                        // See https://github.com/rust-lang/rust/pull/76955#discussion_r493953382 for a good approach.
                        let mut err = ResolutionFailure::NotResolved(err);
                        for other_ns in [TypeNS, ValueNS, MacroNS] {
                            if other_ns != expected_ns {
                                if let Ok(res) =
                                    self.resolve(path_str, other_ns, item_id, module_id) &&
                                    !res.is_empty()
                                {
                                    err = ResolutionFailure::WrongNamespace {
                                        res: full_res(self.cx.tcx, res[0]),
                                        expected_ns,
                                    };
                                    break;
                                }
                            }
                        }
                        resolution_failure(self, diag, path_str, disambiguator, smallvec![err]);
                        return vec![];
                    }
                }
            }
            None => {
                // Try everything!
                let mut candidate = |ns| {
                    self.resolve(path_str, ns, item_id, module_id)
                        .map_err(ResolutionFailure::NotResolved)
                };

                let candidates = PerNS {
                    macro_ns: candidate(MacroNS),
                    type_ns: candidate(TypeNS),
                    value_ns: candidate(ValueNS).and_then(|v_res| {
                        for (res, _) in v_res.iter() {
                            match res {
                                // Constructors are picked up in the type namespace.
                                Res::Def(DefKind::Ctor(..), _) => {
                                    return Err(ResolutionFailure::WrongNamespace {
                                        res: *res,
                                        expected_ns: TypeNS,
                                    });
                                }
                                _ => {}
                            }
                        }
                        Ok(v_res)
                    }),
                };

                let len = candidates
                    .iter()
                    .fold(0, |acc, res| if let Ok(res) = res { acc + res.len() } else { acc });

                if len == 0 {
                    resolution_failure(
                        self,
                        diag,
                        path_str,
                        disambiguator,
                        candidates.into_iter().filter_map(|res| res.err()).collect(),
                    );
                    return vec![];
                } else if len == 1 {
                    candidates.into_iter().filter_map(|res| res.ok()).flatten().collect::<Vec<_>>()
                } else {
                    let has_derive_trait_collision = is_derive_trait_collision(&candidates);
                    if len == 2 && has_derive_trait_collision {
                        candidates.type_ns.unwrap()
                    } else {
                        // If we're reporting an ambiguity, don't mention the namespaces that failed
                        let mut candidates = candidates.map(|candidate| candidate.ok());
                        // If there a collision between a trait and a derive, we ignore the derive.
                        if has_derive_trait_collision {
                            candidates.macro_ns = None;
                        }
                        candidates.into_iter().flatten().flatten().collect::<Vec<_>>()
                    }
                }
            }
        }
    }
}

/// Get the section of a link between the backticks,
/// or the whole link if there aren't any backticks.
///
/// For example:
///
/// ```text
/// [`Foo`]
///   ^^^
/// ```
///
/// This function does nothing if `ori_link.range` is a `MarkdownLinkRange::WholeLink`.
fn range_between_backticks(ori_link_range: &MarkdownLinkRange, dox: &str) -> MarkdownLinkRange {
    let range = match ori_link_range {
        mdlr @ MarkdownLinkRange::WholeLink(_) => return mdlr.clone(),
        MarkdownLinkRange::Destination(inner) => inner.clone(),
    };
    let ori_link_text = &dox[range.clone()];
    let after_first_backtick_group = ori_link_text.bytes().position(|b| b != b'`').unwrap_or(0);
    let before_second_backtick_group = ori_link_text
        .bytes()
        .skip(after_first_backtick_group)
        .position(|b| b == b'`')
        .unwrap_or(ori_link_text.len());
    MarkdownLinkRange::Destination(
        (range.start + after_first_backtick_group)..(range.start + before_second_backtick_group),
    )
}

/// Returns true if we should ignore `link` due to it being unlikely
/// that it is an intra-doc link. `link` should still have disambiguators
/// if there were any.
///
/// The difference between this and [`should_ignore_link()`] is that this
/// check should only be used on links that still have disambiguators.
fn should_ignore_link_with_disambiguators(link: &str) -> bool {
    link.contains(|ch: char| !(ch.is_alphanumeric() || ":_<>, !*&;@()".contains(ch)))
}

/// Returns true if we should ignore `path_str` due to it being unlikely
/// that it is an intra-doc link.
fn should_ignore_link(path_str: &str) -> bool {
    path_str.contains(|ch: char| !(ch.is_alphanumeric() || ":_<>, !*&;".contains(ch)))
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
/// Disambiguators for a link.
enum Disambiguator {
    /// `prim@`
    ///
    /// This is buggy, see <https://github.com/rust-lang/rust/pull/77875#discussion_r503583103>
    Primitive,
    /// `struct@` or `f()`
    Kind(DefKind),
    /// `type@`
    Namespace(Namespace),
}

impl Disambiguator {
    /// Given a link, parse and return `(disambiguator, path_str, link_text)`.
    ///
    /// This returns `Ok(Some(...))` if a disambiguator was found,
    /// `Ok(None)` if no disambiguator was found, or `Err(...)`
    /// if there was a problem with the disambiguator.
    fn from_str(link: &str) -> Result<Option<(Self, &str, &str)>, (String, Range<usize>)> {
        use Disambiguator::{Kind, Namespace as NS, Primitive};

        if let Some(idx) = link.find('@') {
            let (prefix, rest) = link.split_at(idx);
            let d = match prefix {
                // If you update this list, please also update the relevant rustdoc book section!
                "struct" => Kind(DefKind::Struct),
                "enum" => Kind(DefKind::Enum),
                "trait" => Kind(DefKind::Trait),
                "union" => Kind(DefKind::Union),
                "module" | "mod" => Kind(DefKind::Mod),
                "const" | "constant" => Kind(DefKind::Const),
                "static" => Kind(DefKind::Static(Mutability::Not)),
                "function" | "fn" | "method" => Kind(DefKind::Fn),
                "derive" => Kind(DefKind::Macro(MacroKind::Derive)),
                "type" => NS(Namespace::TypeNS),
                "value" => NS(Namespace::ValueNS),
                "macro" => NS(Namespace::MacroNS),
                "prim" | "primitive" => Primitive,
                _ => return Err((format!("unknown disambiguator `{}`", prefix), 0..idx)),
            };
            Ok(Some((d, &rest[1..], &rest[1..])))
        } else {
            let suffixes = [
                // If you update this list, please also update the relevant rustdoc book section!
                ("!()", DefKind::Macro(MacroKind::Bang)),
                ("!{}", DefKind::Macro(MacroKind::Bang)),
                ("![]", DefKind::Macro(MacroKind::Bang)),
                ("()", DefKind::Fn),
                ("!", DefKind::Macro(MacroKind::Bang)),
            ];
            for (suffix, kind) in suffixes {
                if let Some(path_str) = link.strip_suffix(suffix) {
                    // Avoid turning `!` or `()` into an empty string
                    if !path_str.is_empty() {
                        return Ok(Some((Kind(kind), path_str, link)));
                    }
                }
            }
            Ok(None)
        }
    }

    fn ns(self) -> Namespace {
        match self {
            Self::Namespace(n) => n,
            Self::Kind(k) => {
                k.ns().expect("only DefKinds with a valid namespace can be disambiguators")
            }
            Self::Primitive => TypeNS,
        }
    }

    fn article(self) -> &'static str {
        match self {
            Self::Namespace(_) => panic!("article() doesn't make sense for namespaces"),
            Self::Kind(k) => k.article(),
            Self::Primitive => "a",
        }
    }

    fn descr(self) -> &'static str {
        match self {
            Self::Namespace(n) => n.descr(),
            // HACK(jynelson): the source of `DefKind::descr` only uses the DefId for
            // printing "module" vs "crate" so using the wrong ID is not a huge problem
            Self::Kind(k) => k.descr(CRATE_DEF_ID.to_def_id()),
            Self::Primitive => "builtin type",
        }
    }
}

/// A suggestion to show in a diagnostic.
enum Suggestion {
    /// `struct@`
    Prefix(&'static str),
    /// `f()`
    Function,
    /// `m!`
    Macro,
    /// `foo` without any disambiguator
    RemoveDisambiguator,
}

impl Suggestion {
    fn descr(&self) -> Cow<'static, str> {
        match self {
            Self::Prefix(x) => format!("prefix with `{}@`", x).into(),
            Self::Function => "add parentheses".into(),
            Self::Macro => "add an exclamation mark".into(),
            Self::RemoveDisambiguator => "remove the disambiguator".into(),
        }
    }

    fn as_help(&self, path_str: &str) -> String {
        // FIXME: if this is an implied shortcut link, it's bad style to suggest `@`
        match self {
            Self::Prefix(prefix) => format!("{}@{}", prefix, path_str),
            Self::Function => format!("{}()", path_str),
            Self::Macro => format!("{}!", path_str),
            Self::RemoveDisambiguator => path_str.into(),
        }
    }

    fn as_help_span(
        &self,
        path_str: &str,
        ori_link: &str,
        sp: rustc_span::Span,
    ) -> Vec<(rustc_span::Span, String)> {
        let inner_sp = match ori_link.find('(') {
            Some(index) if index != 0 && ori_link.as_bytes()[index - 1] == b'\\' => {
                sp.with_hi(sp.lo() + BytePos((index - 1) as _))
            }
            Some(index) => sp.with_hi(sp.lo() + BytePos(index as _)),
            None => sp,
        };
        let inner_sp = match ori_link.find('!') {
            Some(index) if index != 0 && ori_link.as_bytes()[index - 1] == b'\\' => {
                sp.with_hi(sp.lo() + BytePos((index - 1) as _))
            }
            Some(index) => inner_sp.with_hi(inner_sp.lo() + BytePos(index as _)),
            None => inner_sp,
        };
        let inner_sp = match ori_link.find('@') {
            Some(index) if index != 0 && ori_link.as_bytes()[index - 1] == b'\\' => {
                sp.with_hi(sp.lo() + BytePos((index - 1) as _))
            }
            Some(index) => inner_sp.with_lo(inner_sp.lo() + BytePos(index as u32 + 1)),
            None => inner_sp,
        };
        match self {
            Self::Prefix(prefix) => {
                // FIXME: if this is an implied shortcut link, it's bad style to suggest `@`
                let mut sugg = vec![(sp.with_hi(inner_sp.lo()), format!("{}@", prefix))];
                if sp.hi() != inner_sp.hi() {
                    sugg.push((inner_sp.shrink_to_hi().with_hi(sp.hi()), String::new()));
                }
                sugg
            }
            Self::Function => {
                let mut sugg = vec![(inner_sp.shrink_to_hi().with_hi(sp.hi()), "()".to_string())];
                if sp.lo() != inner_sp.lo() {
                    sugg.push((inner_sp.shrink_to_lo().with_lo(sp.lo()), String::new()));
                }
                sugg
            }
            Self::Macro => {
                let mut sugg = vec![(inner_sp.shrink_to_hi(), "!".to_string())];
                if sp.lo() != inner_sp.lo() {
                    sugg.push((inner_sp.shrink_to_lo().with_lo(sp.lo()), String::new()));
                }
                sugg
            }
            Self::RemoveDisambiguator => vec![(sp, path_str.into())],
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
    tcx: TyCtxt<'_>,
    lint: &'static Lint,
    msg: impl Into<DiagnosticMessage> + Display,
    DiagnosticInfo { item, ori_link: _, dox, link_range }: &DiagnosticInfo<'_>,
    decorate: impl FnOnce(&mut Diagnostic, Option<rustc_span::Span>, MarkdownLinkRange),
) {
    let Some(hir_id) = DocContext::as_local_hir_id(tcx, item.item_id)
    else {
        // If non-local, no need to check anything.
        info!("ignoring warning from parent crate: {}", msg);
        return;
    };

    let sp = item.attr_span(tcx);

    tcx.struct_span_lint_hir(lint, hir_id, sp, msg, |lint| {
        let (span, link_range) = match link_range {
            MarkdownLinkRange::Destination(md_range) => {
                let mut md_range = md_range.clone();
                let sp = super::source_span_for_markdown_range(tcx, dox, &md_range, &item.attrs)
                    .map(|mut sp| {
                        while dox.as_bytes().get(md_range.start) == Some(&b' ')
                            || dox.as_bytes().get(md_range.start) == Some(&b'`')
                        {
                            md_range.start += 1;
                            sp = sp.with_lo(sp.lo() + BytePos(1));
                        }
                        while dox.as_bytes().get(md_range.end - 1) == Some(&b' ')
                            || dox.as_bytes().get(md_range.end - 1) == Some(&b'`')
                        {
                            md_range.end -= 1;
                            sp = sp.with_hi(sp.hi() - BytePos(1));
                        }
                        sp
                    });
                (sp, MarkdownLinkRange::Destination(md_range))
            }
            MarkdownLinkRange::WholeLink(md_range) => (
                super::source_span_for_markdown_range(tcx, dox, &md_range, &item.attrs),
                link_range.clone(),
            ),
        };

        if let Some(sp) = span {
            lint.set_span(sp);
        } else {
            // blah blah blah\nblah\nblah [blah] blah blah\nblah blah
            //                       ^     ~~~~
            //                       |     link_range
            //                       last_new_line_offset
            let md_range = link_range.inner_range().clone();
            let last_new_line_offset = dox[..md_range.start].rfind('\n').map_or(0, |n| n + 1);
            let line = dox[last_new_line_offset..].lines().next().unwrap_or("");

            // Print the line containing the `md_range` and manually mark it with '^'s.
            lint.note(format!(
                "the link appears in this line:\n\n{line}\n\
                     {indicator: <before$}{indicator:^<found$}",
                line = line,
                indicator = "",
                before = md_range.start - last_new_line_offset,
                found = md_range.len(),
            ));
        }

        decorate(lint, span, link_range);

        lint
    });
}

/// Reports a link that failed to resolve.
///
/// This also tries to resolve any intermediate path segments that weren't
/// handled earlier. For example, if passed `Item::Crate(std)` and `path_str`
/// `std::io::Error::x`, this will resolve `std::io::Error`.
fn resolution_failure(
    collector: &mut LinkCollector<'_, '_>,
    diag_info: DiagnosticInfo<'_>,
    path_str: &str,
    disambiguator: Option<Disambiguator>,
    kinds: SmallVec<[ResolutionFailure<'_>; 3]>,
) {
    let tcx = collector.cx.tcx;
    report_diagnostic(
        tcx,
        BROKEN_INTRA_DOC_LINKS,
        format!("unresolved link to `{}`", path_str),
        &diag_info,
        |diag, sp, link_range| {
            let item = |res: Res| format!("the {} `{}`", res.descr(), res.name(tcx),);
            let assoc_item_not_allowed = |res: Res| {
                let name = res.name(tcx);
                format!(
                    "`{}` is {} {}, not a module or type, and cannot have associated items",
                    name,
                    res.article(),
                    res.descr()
                )
            };
            // ignore duplicates
            let mut variants_seen = SmallVec::<[_; 3]>::new();
            for mut failure in kinds {
                let variant = mem::discriminant(&failure);
                if variants_seen.contains(&variant) {
                    continue;
                }
                variants_seen.push(variant);

                if let ResolutionFailure::NotResolved(UnresolvedPath {
                    item_id,
                    module_id,
                    partial_res,
                    unresolved,
                }) = &mut failure
                {
                    use DefKind::*;

                    let item_id = *item_id;
                    let module_id = *module_id;
                    // FIXME(jynelson): this might conflict with my `Self` fix in #76467
                    // FIXME: maybe use itertools `collect_tuple` instead?
                    fn split(path: &str) -> Option<(&str, &str)> {
                        let mut splitter = path.rsplitn(2, "::");
                        splitter.next().and_then(|right| splitter.next().map(|left| (left, right)))
                    }

                    // Check if _any_ parent of the path gets resolved.
                    // If so, report it and say the first which failed; if not, say the first path segment didn't resolve.
                    let mut name = path_str;
                    'outer: loop {
                        let Some((start, end)) = split(name) else {
                            // avoid bug that marked [Quux::Z] as missing Z, not Quux
                            if partial_res.is_none() {
                                *unresolved = name.into();
                            }
                            break;
                        };
                        name = start;
                        for ns in [TypeNS, ValueNS, MacroNS] {
                            if let Ok(v_res) = collector.resolve(start, ns, item_id, module_id) {
                                debug!("found partial_res={:?}", v_res);
                                if !v_res.is_empty() {
                                    *partial_res = Some(full_res(tcx, v_res[0]));
                                    *unresolved = end.into();
                                    break 'outer;
                                }
                            }
                        }
                        *unresolved = end.into();
                    }

                    let last_found_module = match *partial_res {
                        Some(Res::Def(DefKind::Mod, id)) => Some(id),
                        None => Some(module_id),
                        _ => None,
                    };
                    // See if this was a module: `[path]` or `[std::io::nope]`
                    if let Some(module) = last_found_module {
                        let note = if partial_res.is_some() {
                            // Part of the link resolved; e.g. `std::io::nonexistent`
                            let module_name = tcx.item_name(module);
                            format!("no item named `{}` in module `{}`", unresolved, module_name)
                        } else {
                            // None of the link resolved; e.g. `Notimported`
                            format!("no item named `{}` in scope", unresolved)
                        };
                        if let Some(span) = sp {
                            diag.span_label(span, note);
                        } else {
                            diag.note(note);
                        }

                        if !path_str.contains("::") {
                            if disambiguator.map_or(true, |d| d.ns() == MacroNS)
                                && collector
                                    .cx
                                    .tcx
                                    .resolutions(())
                                    .all_macro_rules
                                    .get(&Symbol::intern(path_str))
                                    .is_some()
                            {
                                diag.note(format!(
                                    "`macro_rules` named `{path_str}` exists in this crate, \
                                     but it is not in scope at this link's location"
                                ));
                            } else {
                                // If the link has `::` in it, assume it was meant to be an
                                // intra-doc link. Otherwise, the `[]` might be unrelated.
                                diag.help(
                                    "to escape `[` and `]` characters, \
                                           add '\\' before them like `\\[` or `\\]`",
                                );
                            }
                        }

                        continue;
                    }

                    // Otherwise, it must be an associated item or variant
                    let res = partial_res.expect("None case was handled by `last_found_module`");
                    let kind_did = match res {
                        Res::Def(kind, did) => Some((kind, did)),
                        Res::Primitive(_) => None,
                    };
                    let is_struct_variant = |did| {
                        if let ty::Adt(def, _) = tcx.type_of(did).subst_identity().kind()
                        && def.is_enum()
                        && let Some(variant) = def.variants().iter().find(|v| v.name == res.name(tcx)) {
                            // ctor is `None` if variant is a struct
                            variant.ctor.is_none()
                        } else {
                            false
                        }
                    };
                    let path_description = if let Some((kind, did)) = kind_did {
                        match kind {
                            Mod | ForeignMod => "inner item",
                            Struct => "field or associated item",
                            Enum | Union => "variant or associated item",
                            Variant if is_struct_variant(did) => {
                                let variant = res.name(tcx);
                                let note = format!("variant `{variant}` has no such field");
                                if let Some(span) = sp {
                                    diag.span_label(span, note);
                                } else {
                                    diag.note(note);
                                }
                                return;
                            }
                            Variant
                            | Field
                            | Closure
                            | Generator
                            | AssocTy
                            | AssocConst
                            | AssocFn
                            | Fn
                            | Macro(_)
                            | Const
                            | ConstParam
                            | ExternCrate
                            | Use
                            | LifetimeParam
                            | Ctor(_, _)
                            | AnonConst
                            | InlineConst => {
                                let note = assoc_item_not_allowed(res);
                                if let Some(span) = sp {
                                    diag.span_label(span, note);
                                } else {
                                    diag.note(note);
                                }
                                return;
                            }
                            Trait | TyAlias | ForeignTy | OpaqueTy | ImplTraitPlaceholder
                            | TraitAlias | TyParam | Static(_) => "associated item",
                            Impl { .. } | GlobalAsm => unreachable!("not a path"),
                        }
                    } else {
                        "associated item"
                    };
                    let name = res.name(tcx);
                    let note = format!(
                        "the {} `{}` has no {} named `{}`",
                        res.descr(),
                        name,
                        disambiguator.map_or(path_description, |d| d.descr()),
                        unresolved,
                    );
                    if let Some(span) = sp {
                        diag.span_label(span, note);
                    } else {
                        diag.note(note);
                    }

                    continue;
                }
                let note = match failure {
                    ResolutionFailure::NotResolved { .. } => unreachable!("handled above"),
                    ResolutionFailure::WrongNamespace { res, expected_ns } => {
                        suggest_disambiguator(
                            res,
                            diag,
                            path_str,
                            link_range.clone(),
                            sp,
                            &diag_info,
                        );

                        format!(
                            "this link resolves to {}, which is not in the {} namespace",
                            item(res),
                            expected_ns.descr()
                        )
                    }
                };
                if let Some(span) = sp {
                    diag.span_label(span, note);
                } else {
                    diag.note(note);
                }
            }
        },
    );
}

fn report_multiple_anchors(cx: &DocContext<'_>, diag_info: DiagnosticInfo<'_>) {
    let msg = format!("`{}` contains multiple anchors", diag_info.ori_link);
    anchor_failure(cx, diag_info, msg, 1)
}

fn report_anchor_conflict(cx: &DocContext<'_>, diag_info: DiagnosticInfo<'_>, def_id: DefId) {
    let (link, kind) = (diag_info.ori_link, Res::from_def_id(cx.tcx, def_id).descr());
    let msg = format!("`{link}` contains an anchor, but links to {kind}s are already anchored");
    anchor_failure(cx, diag_info, msg, 0)
}

/// Report an anchor failure.
fn anchor_failure(
    cx: &DocContext<'_>,
    diag_info: DiagnosticInfo<'_>,
    msg: String,
    anchor_idx: usize,
) {
    report_diagnostic(cx.tcx, BROKEN_INTRA_DOC_LINKS, msg, &diag_info, |diag, sp, _link_range| {
        if let Some(mut sp) = sp {
            if let Some((fragment_offset, _)) =
                diag_info.ori_link.char_indices().filter(|(_, x)| *x == '#').nth(anchor_idx)
            {
                sp = sp.with_lo(sp.lo() + BytePos(fragment_offset as _));
            }
            diag.span_label(sp, "invalid anchor");
        }
    });
}

/// Report an error in the link disambiguator.
fn disambiguator_error(
    cx: &DocContext<'_>,
    mut diag_info: DiagnosticInfo<'_>,
    disambiguator_range: MarkdownLinkRange,
    msg: impl Into<DiagnosticMessage> + Display,
) {
    diag_info.link_range = disambiguator_range;
    report_diagnostic(cx.tcx, BROKEN_INTRA_DOC_LINKS, msg, &diag_info, |diag, _sp, _link_range| {
        let msg = format!(
            "see {}/rustdoc/write-documentation/linking-to-items-by-name.html#namespaces-and-disambiguators for more info about disambiguators",
            crate::DOC_RUST_LANG_ORG_CHANNEL
        );
        diag.note(msg);
    });
}

fn report_malformed_generics(
    cx: &DocContext<'_>,
    diag_info: DiagnosticInfo<'_>,
    err: MalformedGenerics,
    path_str: &str,
) {
    report_diagnostic(
        cx.tcx,
        BROKEN_INTRA_DOC_LINKS,
        format!("unresolved link to `{}`", path_str),
        &diag_info,
        |diag, sp, _link_range| {
            let note = match err {
                MalformedGenerics::UnbalancedAngleBrackets => "unbalanced angle brackets",
                MalformedGenerics::MissingType => "missing type for generic parameters",
                MalformedGenerics::HasFullyQualifiedSyntax => {
                    diag.note(
                        "see https://github.com/rust-lang/rust/issues/74563 for more information",
                    );
                    "fully-qualified syntax is unsupported"
                }
                MalformedGenerics::InvalidPathSeparator => "has invalid path separator",
                MalformedGenerics::TooManyAngleBrackets => "too many angle brackets",
                MalformedGenerics::EmptyAngleBrackets => "empty angle brackets",
            };
            if let Some(span) = sp {
                diag.span_label(span, note);
            } else {
                diag.note(note);
            }
        },
    );
}

/// Report an ambiguity error, where there were multiple possible resolutions.
///
/// If all `candidates` have the same kind, it's not possible to disambiguate so in this case,
/// the function won't emit an error and will return `false`. Otherwise, it'll emit the error and
/// return `true`.
fn ambiguity_error(
    cx: &DocContext<'_>,
    diag_info: &DiagnosticInfo<'_>,
    path_str: &str,
    candidates: &[(Res, Option<DefId>)],
) -> bool {
    let mut descrs = FxHashSet::default();
    let kinds = candidates
        .iter()
        .map(
            |(res, def_id)| {
                if let Some(def_id) = def_id { Res::from_def_id(cx.tcx, *def_id) } else { *res }
            },
        )
        .filter(|res| descrs.insert(res.descr()))
        .collect::<Vec<_>>();
    if descrs.len() == 1 {
        // There is no way for users to disambiguate at this point, so better return the first
        // candidate and not show a warning.
        return false;
    }

    let mut msg = format!("`{}` is ", path_str);
    match kinds.as_slice() {
        [res1, res2] => {
            msg += &format!(
                "both {} {} and {} {}",
                res1.article(),
                res1.descr(),
                res2.article(),
                res2.descr()
            );
        }
        _ => {
            let mut kinds = kinds.iter().peekable();
            while let Some(res) = kinds.next() {
                if kinds.peek().is_some() {
                    msg += &format!("{} {}, ", res.article(), res.descr());
                } else {
                    msg += &format!("and {} {}", res.article(), res.descr());
                }
            }
        }
    }

    report_diagnostic(cx.tcx, BROKEN_INTRA_DOC_LINKS, msg, diag_info, |diag, sp, link_range| {
        if let Some(sp) = sp {
            diag.span_label(sp, "ambiguous link");
        } else {
            diag.note("ambiguous link");
        }

        for res in kinds {
            suggest_disambiguator(res, diag, path_str, link_range.clone(), sp, diag_info);
        }
    });
    true
}

/// In case of an ambiguity or mismatched disambiguator, suggest the correct
/// disambiguator.
fn suggest_disambiguator(
    res: Res,
    diag: &mut Diagnostic,
    path_str: &str,
    link_range: MarkdownLinkRange,
    sp: Option<rustc_span::Span>,
    diag_info: &DiagnosticInfo<'_>,
) {
    let suggestion = res.disambiguator_suggestion();
    let help = format!("to link to the {}, {}", res.descr(), suggestion.descr());

    let ori_link = match link_range {
        MarkdownLinkRange::Destination(range) => Some(&diag_info.dox[range]),
        MarkdownLinkRange::WholeLink(_) => None,
    };

    if let (Some(sp), Some(ori_link)) = (sp, ori_link) {
        let mut spans = suggestion.as_help_span(path_str, ori_link, sp);
        if spans.len() > 1 {
            diag.multipart_suggestion(help, spans, Applicability::MaybeIncorrect);
        } else {
            let (sp, suggestion_text) = spans.pop().unwrap();
            diag.span_suggestion_verbose(sp, help, suggestion_text, Applicability::MaybeIncorrect);
        }
    } else {
        diag.help(format!("{}: {}", help, suggestion.as_help(path_str)));
    }
}

/// Report a link from a public item to a private one.
fn privacy_error(cx: &DocContext<'_>, diag_info: &DiagnosticInfo<'_>, path_str: &str) {
    let sym;
    let item_name = match diag_info.item.name {
        Some(name) => {
            sym = name;
            sym.as_str()
        }
        None => "<unknown>",
    };
    let msg =
        format!("public documentation for `{}` links to private item `{}`", item_name, path_str);

    report_diagnostic(cx.tcx, PRIVATE_INTRA_DOC_LINKS, msg, diag_info, |diag, sp, _link_range| {
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

/// Resolve a primitive type or value.
fn resolve_primitive(path_str: &str, ns: Namespace) -> Option<Res> {
    if ns != TypeNS {
        return None;
    }
    use PrimitiveType::*;
    let prim = match path_str {
        "isize" => Isize,
        "i8" => I8,
        "i16" => I16,
        "i32" => I32,
        "i64" => I64,
        "i128" => I128,
        "usize" => Usize,
        "u8" => U8,
        "u16" => U16,
        "u32" => U32,
        "u64" => U64,
        "u128" => U128,
        "f32" => F32,
        "f64" => F64,
        "char" => Char,
        "bool" | "true" | "false" => Bool,
        "str" | "&str" => Str,
        // See #80181 for why these don't have symbols associated.
        "slice" => Slice,
        "array" => Array,
        "tuple" => Tuple,
        "unit" => Unit,
        "pointer" | "*const" | "*mut" => RawPointer,
        "reference" | "&" | "&mut" => Reference,
        "fn" => Fn,
        "never" | "!" => Never,
        _ => return None,
    };
    debug!("resolved primitives {:?}", prim);
    Some(Res::Primitive(prim))
}
