//! This module implements [RFC 1946]: Intra-rustdoc-links
//!
//! [RFC 1946]: https://github.com/rust-lang/rfcs/blob/master/text/1946-intra-rustdoc-links.md

use rustc_ast as ast;
use rustc_data_structures::{fx::FxHashMap, stable_set::FxHashSet};
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_expand::base::SyntaxExtensionKind;
use rustc_hir as hir;
use rustc_hir::def::{
    DefKind,
    Namespace::{self, *},
    PerNS, Res,
};
use rustc_hir::def_id::{CrateNum, DefId};
use rustc_middle::ty;
use rustc_resolve::ParentScope;
use rustc_session::lint::{
    builtin::{BROKEN_INTRA_DOC_LINKS, PRIVATE_INTRA_DOC_LINKS},
    Lint,
};
use rustc_span::hygiene::MacroKind;
use rustc_span::symbol::sym;
use rustc_span::symbol::Ident;
use rustc_span::symbol::Symbol;
use rustc_span::DUMMY_SP;
use smallvec::{smallvec, SmallVec};

use std::borrow::Cow;
use std::cell::Cell;
use std::mem;
use std::ops::Range;

use crate::clean::{self, Crate, Item, ItemLink, PrimitiveType};
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::html::markdown::markdown_links;
use crate::passes::Pass;

use super::span_of_attrs;

crate const COLLECT_INTRA_DOC_LINKS: Pass = Pass {
    name: "collect-intra-doc-links",
    run: collect_intra_doc_links,
    description: "reads a crate's documentation to resolve intra-doc-links",
};

crate fn collect_intra_doc_links(krate: Crate, cx: &DocContext<'_>) -> Crate {
    LinkCollector::new(cx).fold_crate(krate)
}

/// Top-level errors emitted by this pass.
enum ErrorKind<'a> {
    Resolve(Box<ResolutionFailure<'a>>),
    AnchorFailure(AnchorFailure),
}

impl<'a> From<ResolutionFailure<'a>> for ErrorKind<'a> {
    fn from(err: ResolutionFailure<'a>) -> Self {
        ErrorKind::Resolve(box err)
    }
}

#[derive(Debug)]
/// A link failed to resolve.
enum ResolutionFailure<'a> {
    /// This resolved, but with the wrong namespace.
    ///
    /// `Namespace` is the namespace specified with a disambiguator
    /// (as opposed to the actual namespace of the `Res`).
    WrongNamespace(Res, /* disambiguated */ Namespace),
    /// The link failed to resolve. `resolution_failure` should look to see if there's
    /// a more helpful error that can be given.
    NotResolved {
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
    },
    /// This happens when rustdoc can't determine the parent scope for an item.
    ///
    /// It is always a bug in rustdoc.
    NoParentItem,
    /// This link has malformed generic parameters; e.g., the angle brackets are unbalanced.
    MalformedGenerics(MalformedGenerics),
    /// Used to communicate that this should be ignored, but shouldn't be reported to the user
    ///
    /// This happens when there is no disambiguator and one of the namespaces
    /// failed to resolve.
    Dummy,
}

#[derive(Debug)]
enum MalformedGenerics {
    /// This link has unbalanced angle brackets.
    ///
    /// For example, `Vec<T` should trigger this, as should `Vec<T>>`.
    UnbalancedAngleBrackets,
    /// The generics are not attached to a type.
    ///
    /// For example, `<T>` should trigger this.
    ///
    /// This is detected by checking if the path is empty after the generics are stripped.
    MissingType,
    /// The link uses fully-qualified syntax, which is currently unsupported.
    ///
    /// For example, `<Vec as IntoIterator>::into_iter` should trigger this.
    ///
    /// This is detected by checking if ` as ` (the keyword `as` with spaces around it) is inside
    /// angle brackets.
    HasFullyQualifiedSyntax,
    /// The link has an invalid path separator.
    ///
    /// For example, `Vec:<T>:new()` should trigger this. Note that `Vec:new()` will **not**
    /// trigger this because it has no generics and thus [`strip_generics_from_path`] will not be
    /// called.
    ///
    /// Note that this will also **not** be triggered if the invalid path separator is inside angle
    /// brackets because rustdoc mostly ignores what's inside angle brackets (except for
    /// [`HasFullyQualifiedSyntax`](MalformedGenerics::HasFullyQualifiedSyntax)).
    ///
    /// This is detected by checking if there is a colon followed by a non-colon in the link.
    InvalidPathSeparator,
    /// The link has too many angle brackets.
    ///
    /// For example, `Vec<<T>>` should trigger this.
    TooManyAngleBrackets,
    /// The link has empty angle brackets.
    ///
    /// For example, `Vec<>` should trigger this.
    EmptyAngleBrackets,
}

impl ResolutionFailure<'a> {
    /// This resolved fully (not just partially) but is erroneous for some other reason
    ///
    /// Returns the full resolution of the link, if present.
    fn full_res(&self) -> Option<Res> {
        match self {
            Self::WrongNamespace(res, _) => Some(*res),
            _ => None,
        }
    }
}

enum AnchorFailure {
    /// User error: `[std#x#y]` is not valid
    MultipleAnchors,
    /// The anchor provided by the user conflicts with Rustdoc's generated anchor.
    ///
    /// This is an unfortunate state of affairs. Not every item that can be
    /// linked to has its own page; sometimes it is a subheading within a page,
    /// like for associated items. In those cases, rustdoc uses an anchor to
    /// link to the subheading. Since you can't have two anchors for the same
    /// link, Rustdoc disallows having a user-specified anchor.
    ///
    /// Most of the time this is fine, because you can just link to the page of
    /// the item if you want to provide your own anchor. For primitives, though,
    /// rustdoc uses the anchor as a side channel to know which page to link to;
    /// it doesn't show up in the generated link. Ideally, rustdoc would remove
    /// this limitation, allowing you to link to subheaders on primitives.
    RustdocAnchorConflict(Res),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ResolutionInfo {
    module_id: DefId,
    dis: Option<Disambiguator>,
    path_str: String,
    extra_fragment: Option<String>,
}

struct DiagnosticInfo<'a> {
    item: &'a Item,
    dox: &'a str,
    ori_link: &'a str,
    link_range: Range<usize>,
}

#[derive(Clone, Debug, Hash)]
struct CachedLink {
    pub res: (Res, Option<String>),
    pub side_channel: Option<(DefKind, DefId)>,
}

struct LinkCollector<'a, 'tcx> {
    cx: &'a DocContext<'tcx>,
    /// A stack of modules used to decide what scope to resolve in.
    ///
    /// The last module will be used if the parent scope of the current item is
    /// unknown.
    mod_ids: Vec<DefId>,
    /// This is used to store the kind of associated items,
    /// because `clean` and the disambiguator code expect them to be different.
    /// See the code for associated items on inherent impls for details.
    kind_side_channel: Cell<Option<(DefKind, DefId)>>,
    /// Cache the resolved links so we can avoid resolving (and emitting errors for) the same link
    visited_links: FxHashMap<ResolutionInfo, CachedLink>,
}

impl<'a, 'tcx> LinkCollector<'a, 'tcx> {
    fn new(cx: &'a DocContext<'tcx>) -> Self {
        LinkCollector {
            cx,
            mod_ids: Vec::new(),
            kind_side_channel: Cell::new(None),
            visited_links: FxHashMap::default(),
        }
    }

    /// Given a full link, parse it as an [enum struct variant].
    ///
    /// In particular, this will return an error whenever there aren't three
    /// full path segments left in the link.
    ///
    /// [enum struct variant]: hir::VariantData::Struct
    fn variant_field(
        &self,
        path_str: &'path str,
        module_id: DefId,
    ) -> Result<(Res, Option<String>), ErrorKind<'path>> {
        let cx = self.cx;
        let no_res = || ResolutionFailure::NotResolved {
            module_id,
            partial_res: None,
            unresolved: path_str.into(),
        };

        debug!("looking for enum variant {}", path_str);
        let mut split = path_str.rsplitn(3, "::");
        let (variant_field_str, variant_field_name) = split
            .next()
            .map(|f| (f, Symbol::intern(f)))
            .expect("fold_item should ensure link is non-empty");
        let (variant_str, variant_name) =
            // we're not sure this is a variant at all, so use the full string
            // If there's no second component, the link looks like `[path]`.
            // So there's no partial res and we should say the whole link failed to resolve.
            split.next().map(|f| (f, Symbol::intern(f))).ok_or_else(no_res)?;
        let path = split
            .next()
            .map(|f| f.to_owned())
            // If there's no third component, we saw `[a::b]` before and it failed to resolve.
            // So there's no partial res.
            .ok_or_else(no_res)?;
        let ty_res = cx
            .enter_resolver(|resolver| {
                resolver.resolve_str_path_error(DUMMY_SP, &path, TypeNS, module_id)
            })
            .map(|(_, res)| res)
            .unwrap_or(Res::Err);
        if let Res::Err = ty_res {
            return Err(no_res().into());
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
                    // This is just to let `fold_item` know that this shouldn't be considered;
                    // it's a bug for the error to make it to the user
                    return Err(ResolutionFailure::Dummy.into());
                }
                match cx.tcx.type_of(did).kind() {
                    ty::Adt(def, _) if def.is_enum() => {
                        if def.all_fields().any(|item| item.ident.name == variant_field_name) {
                            Ok((
                                ty_res,
                                Some(format!(
                                    "variant.{}.field.{}",
                                    variant_str, variant_field_name
                                )),
                            ))
                        } else {
                            Err(ResolutionFailure::NotResolved {
                                module_id,
                                partial_res: Some(Res::Def(DefKind::Enum, def.did)),
                                unresolved: variant_field_str.into(),
                            }
                            .into())
                        }
                    }
                    _ => unreachable!(),
                }
            }
            _ => Err(ResolutionFailure::NotResolved {
                module_id,
                partial_res: Some(ty_res),
                unresolved: variant_str.into(),
            }
            .into()),
        }
    }

    /// Given a primitive type, try to resolve an associated item.
    ///
    /// HACK(jynelson): `item_str` is passed in instead of derived from `item_name` so the
    /// lifetimes on `&'path` will work.
    fn resolve_primitive_associated_item(
        &self,
        prim_ty: hir::PrimTy,
        ns: Namespace,
        module_id: DefId,
        item_name: Symbol,
        item_str: &'path str,
    ) -> Result<(Res, Option<String>), ErrorKind<'path>> {
        let cx = self.cx;

        PrimitiveType::from_hir(prim_ty)
            .impls(cx.tcx)
            .into_iter()
            .find_map(|&impl_| {
                cx.tcx
                    .associated_items(impl_)
                    .find_by_name_and_namespace(
                        cx.tcx,
                        Ident::with_dummy_span(item_name),
                        ns,
                        impl_,
                    )
                    .map(|item| match item.kind {
                        ty::AssocKind::Fn => "method",
                        ty::AssocKind::Const => "associatedconstant",
                        ty::AssocKind::Type => "associatedtype",
                    })
                    .map(|out| {
                        (
                            Res::PrimTy(prim_ty),
                            Some(format!("{}#{}.{}", prim_ty.name(), out, item_str)),
                        )
                    })
            })
            .ok_or_else(|| {
                debug!(
                    "returning primitive error for {}::{} in {} namespace",
                    prim_ty.name(),
                    item_name,
                    ns.descr()
                );
                ResolutionFailure::NotResolved {
                    module_id,
                    partial_res: Some(Res::PrimTy(prim_ty)),
                    unresolved: item_str.into(),
                }
                .into()
            })
    }

    /// Resolves a string as a macro.
    ///
    /// FIXME(jynelson): Can this be unified with `resolve()`?
    fn resolve_macro(
        &self,
        path_str: &'a str,
        module_id: DefId,
    ) -> Result<Res, ResolutionFailure<'a>> {
        let cx = self.cx;
        let path = ast::Path::from_ident(Ident::from_str(path_str));
        cx.enter_resolver(|resolver| {
            // FIXME(jynelson): does this really need 3 separate lookups?
            if let Ok((Some(ext), res)) = resolver.resolve_macro_path(
                &path,
                None,
                &ParentScope::module(resolver.graph_root(), resolver),
                false,
                false,
            ) {
                if let SyntaxExtensionKind::LegacyBang { .. } = ext.kind {
                    return Ok(res.map_id(|_| panic!("unexpected id")));
                }
            }
            if let Some(res) = resolver.all_macros().get(&Symbol::intern(path_str)) {
                return Ok(res.map_id(|_| panic!("unexpected id")));
            }
            debug!("resolving {} as a macro in the module {:?}", path_str, module_id);
            if let Ok((_, res)) =
                resolver.resolve_str_path_error(DUMMY_SP, path_str, MacroNS, module_id)
            {
                // don't resolve builtins like `#[derive]`
                if let Res::Def(..) = res {
                    let res = res.map_id(|_| panic!("unexpected node_id"));
                    return Ok(res);
                }
            }
            Err(ResolutionFailure::NotResolved {
                module_id,
                partial_res: None,
                unresolved: path_str.into(),
            })
        })
    }

    /// Convenience wrapper around `resolve_str_path_error`.
    ///
    /// This also handles resolving `true` and `false` as booleans.
    /// NOTE: `resolve_str_path_error` knows only about paths, not about types.
    /// Associated items will never be resolved by this function.
    fn resolve_path(&self, path_str: &str, ns: Namespace, module_id: DefId) -> Option<Res> {
        let result = self.cx.enter_resolver(|resolver| {
            resolver.resolve_str_path_error(DUMMY_SP, &path_str, ns, module_id)
        });
        debug!("{} resolved to {:?} in namespace {:?}", path_str, result, ns);
        match result.map(|(_, res)| res) {
            // resolver doesn't know about true and false so we'll have to resolve them
            // manually as bool
            Ok(Res::Err) | Err(()) => is_bool_value(path_str, ns).map(|(_, res)| res),
            Ok(res) => Some(res.map_id(|_| panic!("unexpected node_id"))),
        }
    }

    /// Resolves a string as a path within a particular namespace. Returns an
    /// optional URL fragment in the case of variants and methods.
    fn resolve<'path>(
        &self,
        path_str: &'path str,
        ns: Namespace,
        module_id: DefId,
        extra_fragment: &Option<String>,
    ) -> Result<(Res, Option<String>), ErrorKind<'path>> {
        let cx = self.cx;

        if let Some(res) = self.resolve_path(path_str, ns, module_id) {
            match res {
                // FIXME(#76467): make this fallthrough to lookup the associated
                // item a separate function.
                Res::Def(DefKind::AssocFn | DefKind::AssocConst, _) => {
                    assert_eq!(ns, ValueNS);
                }
                Res::Def(DefKind::AssocTy, _) => {
                    assert_eq!(ns, TypeNS);
                }
                Res::Def(DefKind::Variant, _) => {
                    return handle_variant(cx, res, extra_fragment);
                }
                // Not a trait item; just return what we found.
                Res::PrimTy(ty) => {
                    if extra_fragment.is_some() {
                        return Err(ErrorKind::AnchorFailure(
                            AnchorFailure::RustdocAnchorConflict(res),
                        ));
                    }
                    return Ok((res, Some(ty.name_str().to_owned())));
                }
                Res::Def(DefKind::Mod, _) => {
                    return Ok((res, extra_fragment.clone()));
                }
                _ => {
                    return Ok((res, extra_fragment.clone()));
                }
            }
        }

        // Try looking for methods and associated items.
        let mut split = path_str.rsplitn(2, "::");
        // NB: `split`'s first element is always defined, even if the delimiter was not present.
        // NB: `item_str` could be empty when resolving in the root namespace (e.g. `::std`).
        let item_str = split.next().unwrap();
        let item_name = Symbol::intern(item_str);
        let path_root = split
            .next()
            .map(|f| f.to_owned())
            // If there's no `::`, it's not an associated item.
            // So we can be sure that `rustc_resolve` was accurate when it said it wasn't resolved.
            .ok_or_else(|| {
                debug!("found no `::`, assumming {} was correctly not in scope", item_name);
                ResolutionFailure::NotResolved {
                    module_id,
                    partial_res: None,
                    unresolved: item_str.into(),
                }
            })?;

        // FIXME: are these both necessary?
        let ty_res = if let Some(ty_res) = resolve_primitive(&path_root, TypeNS)
            .map(|(_, res)| res)
            .or_else(|| self.resolve_path(&path_root, TypeNS, module_id))
        {
            ty_res
        } else {
            // FIXME: this is duplicated on the end of this function.
            return if ns == Namespace::ValueNS {
                self.variant_field(path_str, module_id)
            } else {
                Err(ResolutionFailure::NotResolved {
                    module_id,
                    partial_res: None,
                    unresolved: path_root.into(),
                }
                .into())
            };
        };

        let res = match ty_res {
            Res::PrimTy(prim) => Some(
                self.resolve_primitive_associated_item(prim, ns, module_id, item_name, item_str),
            ),
            Res::Def(
                DefKind::Struct
                | DefKind::Union
                | DefKind::Enum
                | DefKind::TyAlias
                | DefKind::ForeignTy,
                did,
            ) => {
                debug!("looking for associated item named {} for item {:?}", item_name, did);
                // Checks if item_name belongs to `impl SomeItem`
                let assoc_item = cx
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
                    .map(|item| (item.kind, item.def_id))
                    // There should only ever be one associated item that matches from any inherent impl
                    .next()
                    // Check if item_name belongs to `impl SomeTrait for SomeItem`
                    // FIXME(#74563): This gives precedence to `impl SomeItem`:
                    // Although having both would be ambiguous, use impl version for compatibility's sake.
                    // To handle that properly resolve() would have to support
                    // something like [`ambi_fn`](<SomeStruct as SomeTrait>::ambi_fn)
                    .or_else(|| {
                        let kind =
                            resolve_associated_trait_item(did, module_id, item_name, ns, &self.cx);
                        debug!("got associated item kind {:?}", kind);
                        kind
                    });

                if let Some((kind, id)) = assoc_item {
                    let out = match kind {
                        ty::AssocKind::Fn => "method",
                        ty::AssocKind::Const => "associatedconstant",
                        ty::AssocKind::Type => "associatedtype",
                    };
                    Some(if extra_fragment.is_some() {
                        Err(ErrorKind::AnchorFailure(AnchorFailure::RustdocAnchorConflict(ty_res)))
                    } else {
                        // HACK(jynelson): `clean` expects the type, not the associated item
                        // but the disambiguator logic expects the associated item.
                        // Store the kind in a side channel so that only the disambiguator logic looks at it.
                        self.kind_side_channel.set(Some((kind.as_def_kind(), id)));
                        Ok((ty_res, Some(format!("{}.{}", out, item_str))))
                    })
                } else if ns == Namespace::ValueNS {
                    debug!("looking for variants or fields named {} for {:?}", item_name, did);
                    // FIXME(jynelson): why is this different from
                    // `variant_field`?
                    match cx.tcx.type_of(did).kind() {
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
                                    let res = Res::Def(
                                        if def.is_enum() {
                                            DefKind::Variant
                                        } else {
                                            DefKind::Field
                                        },
                                        item.did,
                                    );
                                    Err(ErrorKind::AnchorFailure(
                                        AnchorFailure::RustdocAnchorConflict(res),
                                    ))
                                } else {
                                    Ok((
                                        ty_res,
                                        Some(format!(
                                            "{}.{}",
                                            if def.is_enum() { "variant" } else { "structfield" },
                                            item.ident
                                        )),
                                    ))
                                }
                            })
                        }
                        _ => None,
                    }
                } else {
                    None
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
                        Err(ErrorKind::AnchorFailure(AnchorFailure::RustdocAnchorConflict(ty_res)))
                    } else {
                        let res = Res::Def(item.kind.as_def_kind(), item.def_id);
                        Ok((res, Some(format!("{}.{}", kind, item_str))))
                    }
                }),
            _ => None,
        };
        res.unwrap_or_else(|| {
            if ns == Namespace::ValueNS {
                self.variant_field(path_str, module_id)
            } else {
                Err(ResolutionFailure::NotResolved {
                    module_id,
                    partial_res: Some(ty_res),
                    unresolved: item_str.into(),
                }
                .into())
            }
        })
    }

    /// Used for reporting better errors.
    ///
    /// Returns whether the link resolved 'fully' in another namespace.
    /// 'fully' here means that all parts of the link resolved, not just some path segments.
    /// This returns the `Res` even if it was erroneous for some reason
    /// (such as having invalid URL fragments or being in the wrong namespace).
    fn check_full_res(
        &self,
        ns: Namespace,
        path_str: &str,
        module_id: DefId,
        extra_fragment: &Option<String>,
    ) -> Option<Res> {
        // resolve() can't be used for macro namespace
        let result = match ns {
            Namespace::MacroNS => self.resolve_macro(path_str, module_id).map_err(ErrorKind::from),
            Namespace::TypeNS | Namespace::ValueNS => {
                self.resolve(path_str, ns, module_id, extra_fragment).map(|(res, _)| res)
            }
        };

        let res = match result {
            Ok(res) => Some(res),
            Err(ErrorKind::Resolve(box kind)) => kind.full_res(),
            Err(ErrorKind::AnchorFailure(AnchorFailure::RustdocAnchorConflict(res))) => Some(res),
            Err(ErrorKind::AnchorFailure(AnchorFailure::MultipleAnchors)) => None,
        };
        self.kind_side_channel.take().map(|(kind, id)| Res::Def(kind, id)).or(res)
    }
}

/// Look to see if a resolved item has an associated item named `item_name`.
///
/// Given `[std::io::Error::source]`, where `source` is unresolved, this would
/// find `std::error::Error::source` and return
/// `<io::Error as error::Error>::source`.
fn resolve_associated_trait_item(
    did: DefId,
    module: DefId,
    item_name: Symbol,
    ns: Namespace,
    cx: &DocContext<'_>,
) -> Option<(ty::AssocKind, DefId)> {
    // FIXME: this should also consider blanket impls (`impl<T> X for T`). Unfortunately
    // `get_auto_trait_and_blanket_impls` is broken because the caching behavior is wrong. In the
    // meantime, just don't look for these blanket impls.

    // Next consider explicit impls: `impl MyTrait for MyType`
    // Give precedence to inherent impls.
    let traits = traits_implemented_by(cx, did, module);
    debug!("considering traits {:?}", traits);
    let mut candidates = traits.iter().filter_map(|&trait_| {
        cx.tcx
            .associated_items(trait_)
            .find_by_name_and_namespace(cx.tcx, Ident::with_dummy_span(item_name), ns, trait_)
            .map(|assoc| (assoc.kind, assoc.def_id))
    });
    // FIXME(#74563): warn about ambiguity
    debug!("the candidates were {:?}", candidates.clone().collect::<Vec<_>>());
    candidates.next()
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

        // Look at each trait implementation to see if it's an impl for `did`
        cx.tcx.find_map_relevant_impl(trait_, ty, |impl_| {
            let trait_ref = cx.tcx.impl_trait_ref(impl_).expect("this is not an inherent impl");
            // Check if these are the same type.
            let impl_type = trait_ref.self_ty();
            trace!(
                "comparing type {} with kind {:?} against type {:?}",
                impl_type,
                impl_type.kind(),
                type_
            );
            // Fast path: if this is a primitive simple `==` will work
            let saw_impl = impl_type == ty
                || match impl_type.kind() {
                    // Check if these are the same def_id
                    ty::Adt(def, _) => {
                        debug!("adt def_id: {:?}", def.did);
                        def.did == type_
                    }
                    ty::Foreign(def_id) => *def_id == type_,
                    _ => false,
                };

            if saw_impl { Some(trait_) } else { None }
        })
    });
    iter.collect()
}

/// Check for resolve collisions between a trait and its derive.
///
/// These are common and we should just resolve to the trait in that case.
fn is_derive_trait_collision<T>(ns: &PerNS<Result<(Res, T), ResolutionFailure<'_>>>) -> bool {
    matches!(
        *ns,
        PerNS {
            type_ns: Ok((Res::Def(DefKind::Trait, _), _)),
            macro_ns: Ok((Res::Def(DefKind::Macro(MacroKind::Derive), _), _)),
            ..
        }
    )
}

impl<'a, 'tcx> DocFolder for LinkCollector<'a, 'tcx> {
    fn fold_item(&mut self, mut item: Item) -> Option<Item> {
        use rustc_middle::ty::DefIdTree;

        let parent_node = if item.is_fake() {
            // FIXME: is this correct?
            None
        // If we're documenting the crate root itself, it has no parent. Use the root instead.
        } else if item.def_id.is_top_level_module() {
            Some(item.def_id)
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
                    debug!(
                        "{:?} has no parent (kind={:?}, original was {:?})",
                        current,
                        self.cx.tcx.def_kind(current),
                        item.def_id
                    );
                    break None;
                }
            }
        };

        if parent_node.is_some() {
            trace!("got parent node for {:?} {:?}, id {:?}", item.type_(), item.name, item.def_id);
        }

        // find item's parent to resolve `Self` in item's docs below
        debug!("looking for the `Self` type");
        let self_id = if item.is_fake() {
            None
        } else if matches!(
            self.cx.tcx.def_kind(item.def_id),
            DefKind::AssocConst
                | DefKind::AssocFn
                | DefKind::AssocTy
                | DefKind::Variant
                | DefKind::Field
        ) {
            self.cx.tcx.parent(item.def_id)
        // HACK(jynelson): `clean` marks associated types as `TypedefItem`, not as `AssocTypeItem`.
        // Fixing this breaks `fn render_deref_methods`.
        // As a workaround, see if the parent of the item is an `impl`; if so this must be an associated item,
        // regardless of what rustdoc wants to call it.
        } else if let Some(parent) = self.cx.tcx.parent(item.def_id) {
            let parent_kind = self.cx.tcx.def_kind(parent);
            Some(if parent_kind == DefKind::Impl { parent } else { item.def_id })
        } else {
            Some(item.def_id)
        };

        // FIXME(jynelson): this shouldn't go through stringification, rustdoc should just use the DefId directly
        let self_name = self_id.and_then(|self_id| {
            use ty::TyKind;
            if matches!(self.cx.tcx.def_kind(self_id), DefKind::Impl) {
                // using `ty.to_string()` (or any variant) has issues with raw idents
                let ty = self.cx.tcx.type_of(self_id);
                let name = match ty.kind() {
                    TyKind::Adt(def, _) => Some(self.cx.tcx.item_name(def.did).to_string()),
                    other if other.is_primitive() => Some(ty.to_string()),
                    _ => None,
                };
                debug!("using type_of(): {:?}", name);
                name
            } else {
                let name = self.cx.tcx.opt_item_name(self_id).map(|sym| sym.to_string());
                debug!("using item_name(): {:?}", name);
                name
            }
        });

        if item.is_mod() && item.attrs.inner_docs {
            self.mod_ids.push(item.def_id);
        }

        // We want to resolve in the lexical scope of the documentation.
        // In the presence of re-exports, this is not the same as the module of the item.
        // Rather than merging all documentation into one, resolve it one attribute at a time
        // so we know which module it came from.
        let mut attrs = item.attrs.doc_strings.iter().peekable();
        while let Some(attr) = attrs.next() {
            // `collapse_docs` does not have the behavior we want:
            // we want `///` and `#[doc]` to count as the same attribute,
            // but currently it will treat them as separate.
            // As a workaround, combine all attributes with the same parent module into the same attribute.
            let mut combined_docs = attr.doc.clone();
            loop {
                match attrs.peek() {
                    Some(next) if next.parent_module == attr.parent_module => {
                        combined_docs.push('\n');
                        combined_docs.push_str(&attrs.next().unwrap().doc);
                    }
                    _ => break,
                }
            }
            debug!("combined_docs={}", combined_docs);

            let (krate, parent_node) = if let Some(id) = attr.parent_module {
                trace!("docs {:?} came from {:?}", attr.doc, id);
                (id.krate, Some(id))
            } else {
                trace!("no parent found for {:?}", attr.doc);
                (item.def_id.krate, parent_node)
            };
            // NOTE: if there are links that start in one crate and end in another, this will not resolve them.
            // This is a degenerate case and it's not supported by rustdoc.
            for (ori_link, link_range) in markdown_links(&combined_docs) {
                let link = self.resolve_link(
                    &item,
                    &combined_docs,
                    &self_name,
                    parent_node,
                    krate,
                    ori_link,
                    link_range,
                );
                if let Some(link) = link {
                    item.attrs.links.push(link);
                }
            }
        }

        Some(if item.is_mod() {
            if !item.attrs.inner_docs {
                self.mod_ids.push(item.def_id);
            }

            let ret = self.fold_item_recur(item);
            self.mod_ids.pop();
            ret
        } else {
            self.fold_item_recur(item)
        })
    }
}

impl LinkCollector<'_, '_> {
    /// This is the entry point for resolving an intra-doc link.
    ///
    /// FIXME(jynelson): this is way too many arguments
    fn resolve_link(
        &mut self,
        item: &Item,
        dox: &str,
        self_name: &Option<String>,
        parent_node: Option<DefId>,
        krate: CrateNum,
        ori_link: String,
        link_range: Range<usize>,
    ) -> Option<ItemLink> {
        trace!("considering link '{}'", ori_link);

        // Bail early for real links.
        if ori_link.contains('/') {
            return None;
        }

        // [] is mostly likely not supposed to be a link
        if ori_link.is_empty() {
            return None;
        }

        let cx = self.cx;
        let link = ori_link.replace("`", "");
        let parts = link.split('#').collect::<Vec<_>>();
        let (link, extra_fragment) = if parts.len() > 2 {
            // A valid link can't have multiple #'s
            anchor_failure(cx, &item, &link, dox, link_range, AnchorFailure::MultipleAnchors);
            return None;
        } else if parts.len() == 2 {
            if parts[0].trim().is_empty() {
                // This is an anchor to an element of the current page, nothing to do in here!
                return None;
            }
            (parts[0], Some(parts[1].to_owned()))
        } else {
            (parts[0], None)
        };

        // Parse and strip the disambiguator from the link, if present.
        let (mut path_str, disambiguator) = if let Ok((d, path)) = Disambiguator::from_str(&link) {
            (path.trim(), Some(d))
        } else {
            (link.trim(), None)
        };

        if path_str.contains(|ch: char| !(ch.is_alphanumeric() || ":_<>, !".contains(ch))) {
            return None;
        }

        // We stripped `()` and `!` when parsing the disambiguator.
        // Add them back to be displayed, but not prefix disambiguators.
        let link_text =
            disambiguator.map(|d| d.display_for(path_str)).unwrap_or_else(|| path_str.to_owned());

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

        let mut module_id = if let Some(id) = base_node {
            id
        } else {
            // This is a bug.
            debug!("attempting to resolve item without parent module: {}", path_str);
            resolution_failure(
                self,
                &item,
                path_str,
                disambiguator,
                dox,
                link_range,
                smallvec![ResolutionFailure::NoParentItem],
            );
            return None;
        };

        let resolved_self;
        // replace `Self` with suitable item's parent name
        if path_str.starts_with("Self::") {
            if let Some(ref name) = self_name {
                resolved_self = format!("{}::{}", name, &path_str[6..]);
                path_str = &resolved_self;
            }
        } else if path_str.starts_with("crate::") {
            use rustc_span::def_id::CRATE_DEF_INDEX;

            // HACK(jynelson): rustc_resolve thinks that `crate` is the crate currently being documented.
            // But rustdoc wants it to mean the crate this item was originally present in.
            // To work around this, remove it and resolve relative to the crate root instead.
            // HACK(jynelson)(2): If we just strip `crate::` then suddenly primitives become ambiguous
            // (consider `crate::char`). Instead, change it to `self::`. This works because 'self' is now the crate root.
            // FIXME(#78696): This doesn't always work.
            resolved_self = format!("self::{}", &path_str["crate::".len()..]);
            path_str = &resolved_self;
            module_id = DefId { krate, index: CRATE_DEF_INDEX };
        }

        // Strip generics from the path.
        let stripped_path_string;
        if path_str.contains(['<', '>'].as_slice()) {
            stripped_path_string = match strip_generics_from_path(path_str) {
                Ok(path) => path,
                Err(err_kind) => {
                    debug!("link has malformed generics: {}", path_str);
                    resolution_failure(
                        self,
                        &item,
                        path_str,
                        disambiguator,
                        dox,
                        link_range,
                        smallvec![err_kind],
                    );
                    return None;
                }
            };
            path_str = &stripped_path_string;
        }
        // Sanity check to make sure we don't have any angle brackets after stripping generics.
        assert!(!path_str.contains(['<', '>'].as_slice()));

        // The link is not an intra-doc link if it still contains commas or spaces after
        // stripping generics.
        if path_str.contains([',', ' '].as_slice()) {
            return None;
        }

        let key = ResolutionInfo {
            module_id,
            dis: disambiguator,
            path_str: path_str.to_owned(),
            extra_fragment,
        };
        let diag =
            DiagnosticInfo { item, dox, ori_link: &ori_link, link_range: link_range.clone() };
        let (mut res, mut fragment) = self.resolve_with_disambiguator_cached(key, diag)?;

        // Check for a primitive which might conflict with a module
        // Report the ambiguity and require that the user specify which one they meant.
        // FIXME: could there ever be a primitive not in the type namespace?
        if matches!(
            disambiguator,
            None | Some(Disambiguator::Namespace(Namespace::TypeNS) | Disambiguator::Primitive)
        ) && !matches!(res, Res::PrimTy(_))
        {
            if let Some((path, prim)) = resolve_primitive(path_str, TypeNS) {
                // `prim@char`
                if matches!(disambiguator, Some(Disambiguator::Primitive)) {
                    if fragment.is_some() {
                        anchor_failure(
                            cx,
                            &item,
                            path_str,
                            dox,
                            link_range,
                            AnchorFailure::RustdocAnchorConflict(prim),
                        );
                        return None;
                    }
                    res = prim;
                    fragment = Some(path.as_str().to_string());
                } else {
                    // `[char]` when a `char` module is in scope
                    let candidates = vec![res, prim];
                    ambiguity_error(cx, &item, path_str, dox, link_range, candidates);
                    return None;
                }
            }
        }

        let report_mismatch = |specified: Disambiguator, resolved: Disambiguator| {
            // The resolved item did not match the disambiguator; give a better error than 'not found'
            let msg = format!("incompatible link kind for `{}`", path_str);
            let callback = |diag: &mut DiagnosticBuilder<'_>, sp| {
                let note = format!(
                    "this link resolved to {} {}, which is not {} {}",
                    resolved.article(),
                    resolved.descr(),
                    specified.article(),
                    specified.descr()
                );
                diag.note(&note);
                suggest_disambiguator(resolved, diag, path_str, dox, sp, &link_range);
            };
            report_diagnostic(cx, BROKEN_INTRA_DOC_LINKS, &msg, &item, dox, &link_range, callback);
        };
        if let Res::PrimTy(..) = res {
            match disambiguator {
                Some(Disambiguator::Primitive | Disambiguator::Namespace(_)) | None => {
                    Some(ItemLink { link: ori_link, link_text, did: None, fragment })
                }
                Some(other) => {
                    report_mismatch(other, Disambiguator::Primitive);
                    None
                }
            }
        } else {
            debug!("intra-doc link to {} resolved to {:?}", path_str, res);

            // Disallow e.g. linking to enums with `struct@`
            if let Res::Def(kind, _) = res {
                debug!("saw kind {:?} with disambiguator {:?}", kind, disambiguator);
                match (self.kind_side_channel.take().map(|(kind, _)| kind).unwrap_or(kind), disambiguator) {
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
                        report_mismatch(specified, Disambiguator::Kind(kind));
                        return None;
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
                    privacy_error(cx, &item, &path_str, dox, link_range);
                }
            }
            let id = clean::register_res(cx, res);
            Some(ItemLink { link: ori_link, link_text, did: Some(id), fragment })
        }
    }

    fn resolve_with_disambiguator_cached(
        &mut self,
        key: ResolutionInfo,
        diag: DiagnosticInfo<'_>,
    ) -> Option<(Res, Option<String>)> {
        // Try to look up both the result and the corresponding side channel value
        if let Some(ref cached) = self.visited_links.get(&key) {
            self.kind_side_channel.set(cached.side_channel.clone());
            return Some(cached.res.clone());
        }

        let res = self.resolve_with_disambiguator(&key, diag);

        // Cache only if resolved successfully - don't silence duplicate errors
        if let Some(res) = &res {
            // Store result for the actual namespace
            self.visited_links.insert(
                key,
                CachedLink {
                    res: res.clone(),
                    side_channel: self.kind_side_channel.clone().into_inner(),
                },
            );
        }

        res
    }

    /// After parsing the disambiguator, resolve the main part of the link.
    // FIXME(jynelson): wow this is just so much
    fn resolve_with_disambiguator(
        &self,
        key: &ResolutionInfo,
        diag: DiagnosticInfo<'_>,
    ) -> Option<(Res, Option<String>)> {
        let disambiguator = key.dis;
        let path_str = &key.path_str;
        let base_node = key.module_id;
        let extra_fragment = &key.extra_fragment;

        match disambiguator.map(Disambiguator::ns) {
            Some(ns @ (ValueNS | TypeNS)) => {
                match self.resolve(path_str, ns, base_node, extra_fragment) {
                    Ok(res) => Some(res),
                    Err(ErrorKind::Resolve(box mut kind)) => {
                        // We only looked in one namespace. Try to give a better error if possible.
                        if kind.full_res().is_none() {
                            let other_ns = if ns == ValueNS { TypeNS } else { ValueNS };
                            // FIXME: really it should be `resolution_failure` that does this, not `resolve_with_disambiguator`
                            // See https://github.com/rust-lang/rust/pull/76955#discussion_r493953382 for a good approach
                            for &new_ns in &[other_ns, MacroNS] {
                                if let Some(res) =
                                    self.check_full_res(new_ns, path_str, base_node, extra_fragment)
                                {
                                    kind = ResolutionFailure::WrongNamespace(res, ns);
                                    break;
                                }
                            }
                        }
                        resolution_failure(
                            self,
                            diag.item,
                            path_str,
                            disambiguator,
                            diag.dox,
                            diag.link_range,
                            smallvec![kind],
                        );
                        // This could just be a normal link or a broken link
                        // we could potentially check if something is
                        // "intra-doc-link-like" and warn in that case.
                        return None;
                    }
                    Err(ErrorKind::AnchorFailure(msg)) => {
                        anchor_failure(
                            self.cx,
                            diag.item,
                            diag.ori_link,
                            diag.dox,
                            diag.link_range,
                            msg,
                        );
                        return None;
                    }
                }
            }
            None => {
                // Try everything!
                let mut candidates = PerNS {
                    macro_ns: self
                        .resolve_macro(path_str, base_node)
                        .map(|res| (res, extra_fragment.clone())),
                    type_ns: match self.resolve(path_str, TypeNS, base_node, extra_fragment) {
                        Ok(res) => {
                            debug!("got res in TypeNS: {:?}", res);
                            Ok(res)
                        }
                        Err(ErrorKind::AnchorFailure(msg)) => {
                            anchor_failure(
                                self.cx,
                                diag.item,
                                diag.ori_link,
                                diag.dox,
                                diag.link_range,
                                msg,
                            );
                            return None;
                        }
                        Err(ErrorKind::Resolve(box kind)) => Err(kind),
                    },
                    value_ns: match self.resolve(path_str, ValueNS, base_node, extra_fragment) {
                        Ok(res) => Ok(res),
                        Err(ErrorKind::AnchorFailure(msg)) => {
                            anchor_failure(
                                self.cx,
                                diag.item,
                                diag.ori_link,
                                diag.dox,
                                diag.link_range,
                                msg,
                            );
                            return None;
                        }
                        Err(ErrorKind::Resolve(box kind)) => Err(kind),
                    }
                    .and_then(|(res, fragment)| {
                        // Constructors are picked up in the type namespace.
                        match res {
                            Res::Def(DefKind::Ctor(..), _) | Res::SelfCtor(..) => {
                                Err(ResolutionFailure::WrongNamespace(res, TypeNS))
                            }
                            _ => match (fragment, extra_fragment.clone()) {
                                (Some(fragment), Some(_)) => {
                                    // Shouldn't happen but who knows?
                                    Ok((res, Some(fragment)))
                                }
                                (fragment, None) | (None, fragment) => Ok((res, fragment)),
                            },
                        }
                    }),
                };

                let len = candidates.iter().filter(|res| res.is_ok()).count();

                if len == 0 {
                    resolution_failure(
                        self,
                        diag.item,
                        path_str,
                        disambiguator,
                        diag.dox,
                        diag.link_range,
                        candidates.into_iter().filter_map(|res| res.err()).collect(),
                    );
                    // this could just be a normal link
                    return None;
                }

                if len == 1 {
                    Some(candidates.into_iter().filter_map(|res| res.ok()).next().unwrap())
                } else if len == 2 && is_derive_trait_collision(&candidates) {
                    Some(candidates.type_ns.unwrap())
                } else {
                    if is_derive_trait_collision(&candidates) {
                        candidates.macro_ns = Err(ResolutionFailure::Dummy);
                    }
                    // If we're reporting an ambiguity, don't mention the namespaces that failed
                    let candidates = candidates.map(|candidate| candidate.ok().map(|(res, _)| res));
                    ambiguity_error(
                        self.cx,
                        diag.item,
                        path_str,
                        diag.dox,
                        diag.link_range,
                        candidates.present_items().collect(),
                    );
                    return None;
                }
            }
            Some(MacroNS) => {
                match self.resolve_macro(path_str, base_node) {
                    Ok(res) => Some((res, extra_fragment.clone())),
                    Err(mut kind) => {
                        // `resolve_macro` only looks in the macro namespace. Try to give a better error if possible.
                        for &ns in &[TypeNS, ValueNS] {
                            if let Some(res) =
                                self.check_full_res(ns, path_str, base_node, extra_fragment)
                            {
                                kind = ResolutionFailure::WrongNamespace(res, MacroNS);
                                break;
                            }
                        }
                        resolution_failure(
                            self,
                            diag.item,
                            path_str,
                            disambiguator,
                            diag.dox,
                            diag.link_range,
                            smallvec![kind],
                        );
                        return None;
                    }
                }
            }
        }
    }
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
    /// The text that should be displayed when the path is rendered as HTML.
    ///
    /// NOTE: `path` is not the original link given by the user, but a name suitable for passing to `resolve`.
    fn display_for(&self, path: &str) -> String {
        match self {
            // FIXME: this will have different output if the user had `m!()` originally.
            Self::Kind(DefKind::Macro(MacroKind::Bang)) => format!("{}!", path),
            Self::Kind(DefKind::Fn) => format!("{}()", path),
            _ => path.to_owned(),
        }
    }

    /// Given a link, parse and return `(disambiguator, path_str)`
    fn from_str(link: &str) -> Result<(Self, &str), ()> {
        use Disambiguator::{Kind, Namespace as NS, Primitive};

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
                "prim" | "primitive" => Primitive,
                _ => return find_suffix(),
            };
            Ok((d, &rest[1..]))
        } else {
            find_suffix()
        }
    }

    /// WARNING: panics on `Res::Err`
    fn from_res(res: Res) -> Self {
        match res {
            Res::Def(kind, _) => Disambiguator::Kind(kind),
            Res::PrimTy(_) => Disambiguator::Primitive,
            _ => Disambiguator::Namespace(res.ns().expect("can't call `from_res` on Res::err")),
        }
    }

    /// Used for error reporting.
    fn suggestion(self) -> Suggestion {
        let kind = match self {
            Disambiguator::Primitive => return Suggestion::Prefix("prim"),
            Disambiguator::Kind(kind) => kind,
            Disambiguator::Namespace(_) => panic!("display_for cannot be used on namespaces"),
        };
        if kind == DefKind::Macro(MacroKind::Bang) {
            return Suggestion::Macro;
        } else if kind == DefKind::Fn || kind == DefKind::AssocFn {
            return Suggestion::Function;
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

        Suggestion::Prefix(prefix)
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
            // HACK(jynelson): by looking at the source I saw the DefId we pass
            // for `expected.descr()` doesn't matter, since it's not a crate
            Self::Kind(k) => k.descr(DefId::local(hir::def_id::DefIndex::from_usize(0))),
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
}

impl Suggestion {
    fn descr(&self) -> Cow<'static, str> {
        match self {
            Self::Prefix(x) => format!("prefix with `{}@`", x).into(),
            Self::Function => "add parentheses".into(),
            Self::Macro => "add an exclamation mark".into(),
        }
    }

    fn as_help(&self, path_str: &str) -> String {
        // FIXME: if this is an implied shortcut link, it's bad style to suggest `@`
        match self {
            Self::Prefix(prefix) => format!("{}@{}", prefix, path_str),
            Self::Function => format!("{}()", path_str),
            Self::Macro => format!("{}!", path_str),
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
    lint: &'static Lint,
    msg: &str,
    item: &Item,
    dox: &str,
    link_range: &Range<usize>,
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

    cx.tcx.struct_span_lint_hir(lint, hir_id, sp, |lint| {
        let mut diag = lint.build(msg);

        let span = super::source_span_for_markdown_range(cx, dox, link_range, attrs);
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

        decorate(&mut diag, span);

        diag.emit();
    });
}

/// Reports a link that failed to resolve.
///
/// This also tries to resolve any intermediate path segments that weren't
/// handled earlier. For example, if passed `Item::Crate(std)` and `path_str`
/// `std::io::Error::x`, this will resolve `std::io::Error`.
fn resolution_failure(
    collector: &LinkCollector<'_, '_>,
    item: &Item,
    path_str: &str,
    disambiguator: Option<Disambiguator>,
    dox: &str,
    link_range: Range<usize>,
    kinds: SmallVec<[ResolutionFailure<'_>; 3]>,
) {
    report_diagnostic(
        collector.cx,
        BROKEN_INTRA_DOC_LINKS,
        &format!("unresolved link to `{}`", path_str),
        item,
        dox,
        &link_range,
        |diag, sp| {
            let item = |res: Res| {
                format!(
                    "the {} `{}`",
                    res.descr(),
                    collector.cx.tcx.item_name(res.def_id()).to_string()
                )
            };
            let assoc_item_not_allowed = |res: Res| {
                let def_id = res.def_id();
                let name = collector.cx.tcx.item_name(def_id);
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
                let variant = std::mem::discriminant(&failure);
                if variants_seen.contains(&variant) {
                    continue;
                }
                variants_seen.push(variant);

                if let ResolutionFailure::NotResolved { module_id, partial_res, unresolved } =
                    &mut failure
                {
                    use DefKind::*;

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
                        let (start, end) = if let Some(x) = split(name) {
                            x
                        } else {
                            // avoid bug that marked [Quux::Z] as missing Z, not Quux
                            if partial_res.is_none() {
                                *unresolved = name.into();
                            }
                            break;
                        };
                        name = start;
                        for &ns in &[TypeNS, ValueNS, MacroNS] {
                            if let Some(res) =
                                collector.check_full_res(ns, &start, module_id, &None)
                            {
                                debug!("found partial_res={:?}", res);
                                *partial_res = Some(res);
                                *unresolved = end.into();
                                break 'outer;
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
                            let module_name = collector.cx.tcx.item_name(module);
                            format!("no item named `{}` in module `{}`", unresolved, module_name)
                        } else {
                            // None of the link resolved; e.g. `Notimported`
                            format!("no item named `{}` in scope", unresolved)
                        };
                        if let Some(span) = sp {
                            diag.span_label(span, &note);
                        } else {
                            diag.note(&note);
                        }

                        // If the link has `::` in it, assume it was meant to be an intra-doc link.
                        // Otherwise, the `[]` might be unrelated.
                        // FIXME: don't show this for autolinks (`<>`), `()` style links, or reference links
                        if !path_str.contains("::") {
                            diag.help(r#"to escape `[` and `]` characters, add '\' before them like `\[` or `\]`"#);
                        }

                        continue;
                    }

                    // Otherwise, it must be an associated item or variant
                    let res = partial_res.expect("None case was handled by `last_found_module`");
                    let diagnostic_name;
                    let (kind, name) = match res {
                        Res::Def(kind, def_id) => {
                            diagnostic_name = collector.cx.tcx.item_name(def_id).as_str();
                            (Some(kind), &*diagnostic_name)
                        }
                        Res::PrimTy(ty) => (None, ty.name_str()),
                        _ => unreachable!("only ADTs and primitives are in scope at module level"),
                    };
                    let path_description = if let Some(kind) = kind {
                        match kind {
                            Mod | ForeignMod => "inner item",
                            Struct => "field or associated item",
                            Enum | Union => "variant or associated item",
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
                            | AnonConst => {
                                let note = assoc_item_not_allowed(res);
                                if let Some(span) = sp {
                                    diag.span_label(span, &note);
                                } else {
                                    diag.note(&note);
                                }
                                return;
                            }
                            Trait | TyAlias | ForeignTy | OpaqueTy | TraitAlias | TyParam
                            | Static => "associated item",
                            Impl | GlobalAsm => unreachable!("not a path"),
                        }
                    } else {
                        "associated item"
                    };
                    let note = format!(
                        "the {} `{}` has no {} named `{}`",
                        res.descr(),
                        name,
                        disambiguator.map_or(path_description, |d| d.descr()),
                        unresolved,
                    );
                    if let Some(span) = sp {
                        diag.span_label(span, &note);
                    } else {
                        diag.note(&note);
                    }

                    continue;
                }
                let note = match failure {
                    ResolutionFailure::NotResolved { .. } => unreachable!("handled above"),
                    ResolutionFailure::Dummy => continue,
                    ResolutionFailure::WrongNamespace(res, expected_ns) => {
                        if let Res::Def(kind, _) = res {
                            let disambiguator = Disambiguator::Kind(kind);
                            suggest_disambiguator(
                                disambiguator,
                                diag,
                                path_str,
                                dox,
                                sp,
                                &link_range,
                            )
                        }

                        format!(
                            "this link resolves to {}, which is not in the {} namespace",
                            item(res),
                            expected_ns.descr()
                        )
                    }
                    ResolutionFailure::NoParentItem => {
                        diag.level = rustc_errors::Level::Bug;
                        "all intra doc links should have a parent item".to_owned()
                    }
                    ResolutionFailure::MalformedGenerics(variant) => match variant {
                        MalformedGenerics::UnbalancedAngleBrackets => {
                            String::from("unbalanced angle brackets")
                        }
                        MalformedGenerics::MissingType => {
                            String::from("missing type for generic parameters")
                        }
                        MalformedGenerics::HasFullyQualifiedSyntax => {
                            diag.note("see https://github.com/rust-lang/rust/issues/74563 for more information");
                            String::from("fully-qualified syntax is unsupported")
                        }
                        MalformedGenerics::InvalidPathSeparator => {
                            String::from("has invalid path separator")
                        }
                        MalformedGenerics::TooManyAngleBrackets => {
                            String::from("too many angle brackets")
                        }
                        MalformedGenerics::EmptyAngleBrackets => {
                            String::from("empty angle brackets")
                        }
                    },
                };
                if let Some(span) = sp {
                    diag.span_label(span, &note);
                } else {
                    diag.note(&note);
                }
            }
        },
    );
}

/// Report an anchor failure.
fn anchor_failure(
    cx: &DocContext<'_>,
    item: &Item,
    path_str: &str,
    dox: &str,
    link_range: Range<usize>,
    failure: AnchorFailure,
) {
    let msg = match failure {
        AnchorFailure::MultipleAnchors => format!("`{}` contains multiple anchors", path_str),
        AnchorFailure::RustdocAnchorConflict(res) => format!(
            "`{}` contains an anchor, but links to {kind}s are already anchored",
            path_str,
            kind = res.descr(),
        ),
    };

    report_diagnostic(cx, BROKEN_INTRA_DOC_LINKS, &msg, item, dox, &link_range, |diag, sp| {
        if let Some(sp) = sp {
            diag.span_label(sp, "contains invalid anchor");
        }
    });
}

/// Report an ambiguity error, where there were multiple possible resolutions.
fn ambiguity_error(
    cx: &DocContext<'_>,
    item: &Item,
    path_str: &str,
    dox: &str,
    link_range: Range<usize>,
    candidates: Vec<Res>,
) {
    let mut msg = format!("`{}` is ", path_str);

    match candidates.as_slice() {
        [first_def, second_def] => {
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
            while let Some(res) = candidates.next() {
                if candidates.peek().is_some() {
                    msg += &format!("{} {}, ", res.article(), res.descr());
                } else {
                    msg += &format!("and {} {}", res.article(), res.descr());
                }
            }
        }
    }

    report_diagnostic(cx, BROKEN_INTRA_DOC_LINKS, &msg, item, dox, &link_range, |diag, sp| {
        if let Some(sp) = sp {
            diag.span_label(sp, "ambiguous link");
        } else {
            diag.note("ambiguous link");
        }

        for res in candidates {
            let disambiguator = Disambiguator::from_res(res);
            suggest_disambiguator(disambiguator, diag, path_str, dox, sp, &link_range);
        }
    });
}

/// In case of an ambiguity or mismatched disambiguator, suggest the correct
/// disambiguator.
fn suggest_disambiguator(
    disambiguator: Disambiguator,
    diag: &mut DiagnosticBuilder<'_>,
    path_str: &str,
    dox: &str,
    sp: Option<rustc_span::Span>,
    link_range: &Range<usize>,
) {
    let suggestion = disambiguator.suggestion();
    let help = format!("to link to the {}, {}", disambiguator.descr(), suggestion.descr());

    if let Some(sp) = sp {
        let msg = if dox.bytes().nth(link_range.start) == Some(b'`') {
            format!("`{}`", suggestion.as_help(path_str))
        } else {
            suggestion.as_help(path_str)
        };

        diag.span_suggestion(sp, &help, msg, Applicability::MaybeIncorrect);
    } else {
        diag.help(&format!("{}: {}", help, suggestion.as_help(path_str)));
    }
}

/// Report a link from a public item to a private one.
fn privacy_error(
    cx: &DocContext<'_>,
    item: &Item,
    path_str: &str,
    dox: &str,
    link_range: Range<usize>,
) {
    let sym;
    let item_name = match item.name {
        Some(name) => {
            sym = name.as_str();
            &*sym
        }
        None => "<unknown>",
    };
    let msg =
        format!("public documentation for `{}` links to private item `{}`", item_name, path_str);

    report_diagnostic(cx, PRIVATE_INTRA_DOC_LINKS, &msg, item, dox, &link_range, |diag, sp| {
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
) -> Result<(Res, Option<String>), ErrorKind<'static>> {
    use rustc_middle::ty::DefIdTree;

    if extra_fragment.is_some() {
        return Err(ErrorKind::AnchorFailure(AnchorFailure::RustdocAnchorConflict(res)));
    }
    cx.tcx
        .parent(res.def_id())
        .map(|parent| {
            let parent_def = Res::Def(DefKind::Enum, parent);
            let variant = cx.tcx.expect_variant_res(res);
            (parent_def, Some(format!("variant.{}", variant.ident.name)))
        })
        .ok_or_else(|| ResolutionFailure::NoParentItem.into())
}

// FIXME: At this point, this is basically a copy of the PrimitiveTypeTable
const PRIMITIVES: &[(Symbol, Res)] = &[
    (sym::u8, Res::PrimTy(hir::PrimTy::Uint(rustc_ast::UintTy::U8))),
    (sym::u16, Res::PrimTy(hir::PrimTy::Uint(rustc_ast::UintTy::U16))),
    (sym::u32, Res::PrimTy(hir::PrimTy::Uint(rustc_ast::UintTy::U32))),
    (sym::u64, Res::PrimTy(hir::PrimTy::Uint(rustc_ast::UintTy::U64))),
    (sym::u128, Res::PrimTy(hir::PrimTy::Uint(rustc_ast::UintTy::U128))),
    (sym::usize, Res::PrimTy(hir::PrimTy::Uint(rustc_ast::UintTy::Usize))),
    (sym::i8, Res::PrimTy(hir::PrimTy::Int(rustc_ast::IntTy::I8))),
    (sym::i16, Res::PrimTy(hir::PrimTy::Int(rustc_ast::IntTy::I16))),
    (sym::i32, Res::PrimTy(hir::PrimTy::Int(rustc_ast::IntTy::I32))),
    (sym::i64, Res::PrimTy(hir::PrimTy::Int(rustc_ast::IntTy::I64))),
    (sym::i128, Res::PrimTy(hir::PrimTy::Int(rustc_ast::IntTy::I128))),
    (sym::isize, Res::PrimTy(hir::PrimTy::Int(rustc_ast::IntTy::Isize))),
    (sym::f32, Res::PrimTy(hir::PrimTy::Float(rustc_ast::FloatTy::F32))),
    (sym::f64, Res::PrimTy(hir::PrimTy::Float(rustc_ast::FloatTy::F64))),
    (sym::str, Res::PrimTy(hir::PrimTy::Str)),
    (sym::bool, Res::PrimTy(hir::PrimTy::Bool)),
    (sym::char, Res::PrimTy(hir::PrimTy::Char)),
];

/// Resolve a primitive type or value.
fn resolve_primitive(path_str: &str, ns: Namespace) -> Option<(Symbol, Res)> {
    is_bool_value(path_str, ns).or_else(|| {
        if ns == TypeNS {
            // FIXME: this should be replaced by a lookup in PrimitiveTypeTable
            let maybe_primitive = Symbol::intern(path_str);
            PRIMITIVES.iter().find(|x| x.0 == maybe_primitive).copied()
        } else {
            None
        }
    })
}

/// Resolve a primitive value.
fn is_bool_value(path_str: &str, ns: Namespace) -> Option<(Symbol, Res)> {
    if ns == TypeNS && (path_str == "true" || path_str == "false") {
        Some((sym::bool, Res::PrimTy(hir::PrimTy::Bool)))
    } else {
        None
    }
}

fn strip_generics_from_path(path_str: &str) -> Result<String, ResolutionFailure<'static>> {
    let mut stripped_segments = vec![];
    let mut path = path_str.chars().peekable();
    let mut segment = Vec::new();

    while let Some(chr) = path.next() {
        match chr {
            ':' => {
                if path.next_if_eq(&':').is_some() {
                    let stripped_segment =
                        strip_generics_from_path_segment(mem::take(&mut segment))?;
                    if !stripped_segment.is_empty() {
                        stripped_segments.push(stripped_segment);
                    }
                } else {
                    return Err(ResolutionFailure::MalformedGenerics(
                        MalformedGenerics::InvalidPathSeparator,
                    ));
                }
            }
            '<' => {
                segment.push(chr);

                match path.next() {
                    Some('<') => {
                        return Err(ResolutionFailure::MalformedGenerics(
                            MalformedGenerics::TooManyAngleBrackets,
                        ));
                    }
                    Some('>') => {
                        return Err(ResolutionFailure::MalformedGenerics(
                            MalformedGenerics::EmptyAngleBrackets,
                        ));
                    }
                    Some(chr) => {
                        segment.push(chr);

                        while let Some(chr) = path.next_if(|c| *c != '>') {
                            segment.push(chr);
                        }
                    }
                    None => break,
                }
            }
            _ => segment.push(chr),
        }
        trace!("raw segment: {:?}", segment);
    }

    if !segment.is_empty() {
        let stripped_segment = strip_generics_from_path_segment(segment)?;
        if !stripped_segment.is_empty() {
            stripped_segments.push(stripped_segment);
        }
    }

    debug!("path_str: {:?}\nstripped segments: {:?}", path_str, &stripped_segments);

    let stripped_path = stripped_segments.join("::");

    if !stripped_path.is_empty() {
        Ok(stripped_path)
    } else {
        Err(ResolutionFailure::MalformedGenerics(MalformedGenerics::MissingType))
    }
}

fn strip_generics_from_path_segment(
    segment: Vec<char>,
) -> Result<String, ResolutionFailure<'static>> {
    let mut stripped_segment = String::new();
    let mut param_depth = 0;

    let mut latest_generics_chunk = String::new();

    for c in segment {
        if c == '<' {
            param_depth += 1;
            latest_generics_chunk.clear();
        } else if c == '>' {
            param_depth -= 1;
            if latest_generics_chunk.contains(" as ") {
                // The segment tries to use fully-qualified syntax, which is currently unsupported.
                // Give a helpful error message instead of completely ignoring the angle brackets.
                return Err(ResolutionFailure::MalformedGenerics(
                    MalformedGenerics::HasFullyQualifiedSyntax,
                ));
            }
        } else {
            if param_depth == 0 {
                stripped_segment.push(c);
            } else {
                latest_generics_chunk.push(c);
            }
        }
    }

    if param_depth == 0 {
        Ok(stripped_segment)
    } else {
        // The segment has unbalanced angle brackets, e.g. `Vec<T` or `Vec<T>>`
        Err(ResolutionFailure::MalformedGenerics(MalformedGenerics::UnbalancedAngleBrackets))
    }
}
