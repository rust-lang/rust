use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{
    self as hir, Expr, ExprKind, HirId, LangItem, Pat, PatExpr, PatExprKind, PatKind, Path, PathSegment, QPath, TyKind,
};
use rustc_lint::LateContext;
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_middle::ty::{AdtDef, AdtKind, Binder, EarlyBinder, Ty, TypeckResults};
use rustc_span::{Ident, Symbol};

/// Either a `HirId` or a type which can be identified by one.
pub trait HasHirId: Copy {
    fn hir_id(self) -> HirId;
}
impl HasHirId for HirId {
    #[inline]
    fn hir_id(self) -> HirId {
        self
    }
}
impl HasHirId for &Expr<'_> {
    #[inline]
    fn hir_id(self) -> HirId {
        self.hir_id
    }
}

type DefRes = (DefKind, DefId);

pub trait MaybeTypeckRes<'tcx> {
    /// Gets the contained `TypeckResults`.
    ///
    /// With debug assertions enabled this will always return `Some`. `None` is
    /// only returned so logic errors can be handled by not emitting a lint on
    /// release builds.
    fn typeck_res(&self) -> Option<&TypeckResults<'tcx>>;

    /// Gets the type-dependent resolution of the specified node.
    ///
    /// With debug assertions enabled this will always return `Some`. `None` is
    /// only returned so logic errors can be handled by not emitting a lint on
    /// release builds.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    fn ty_based_def(&self, node: impl HasHirId) -> Option<DefRes> {
        #[inline]
        #[cfg_attr(debug_assertions, track_caller)]
        fn f(typeck: &TypeckResults<'_>, id: HirId) -> Option<DefRes> {
            if typeck.hir_owner == id.owner {
                let def = typeck.type_dependent_def(id);
                debug_assert!(
                    def.is_some(),
                    "attempted type-dependent lookup for a node with no definition\
                        \n  node `{id:?}`",
                );
                def
            } else {
                debug_assert!(
                    false,
                    "attempted type-dependent lookup for a node in the wrong body\
                        \n  in body `{:?}`\
                        \n  expected body `{:?}`",
                    typeck.hir_owner, id.owner,
                );
                None
            }
        }
        self.typeck_res().and_then(|typeck| f(typeck, node.hir_id()))
    }
}
impl<'tcx> MaybeTypeckRes<'tcx> for LateContext<'tcx> {
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    fn typeck_res(&self) -> Option<&TypeckResults<'tcx>> {
        if let Some(typeck) = self.maybe_typeck_results() {
            Some(typeck)
        } else {
            // It's possible to get the `TypeckResults` for any other body, but
            // attempting to lookup the type of something across bodies like this
            // is a good indication of a bug.
            debug_assert!(false, "attempted type-dependent lookup in a non-body context");
            None
        }
    }
}
impl<'tcx> MaybeTypeckRes<'tcx> for TypeckResults<'tcx> {
    #[inline]
    fn typeck_res(&self) -> Option<&TypeckResults<'tcx>> {
        Some(self)
    }
}

/// A `QPath` with the `HirId` of the node containing it.
type QPathId<'tcx> = (&'tcx QPath<'tcx>, HirId);

/// A HIR node which might be a `QPath`.
pub trait MaybeQPath<'a>: Copy {
    /// If this node is a path gets both the contained path and the `HirId` to
    /// use for type dependant lookup.
    fn opt_qpath(self) -> Option<QPathId<'a>>;

    /// If this node is a `QPath::LangItem` gets the item it resolves to.
    #[inline]
    fn opt_lang_path(self) -> Option<LangItem> {
        match self.opt_qpath() {
            Some((&QPath::LangItem(item, _), _)) => Some(item),
            _ => None,
        }
    }

    /// If this is a path gets its resolution. Returns `Res::Err` otherwise.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    fn res<'tcx>(self, typeck: &impl MaybeTypeckRes<'tcx>) -> Res {
        #[cfg_attr(debug_assertions, track_caller)]
        fn f(qpath: &QPath<'_>, id: HirId, typeck: &TypeckResults<'_>) -> Res {
            match *qpath {
                QPath::Resolved(_, p) => p.res,
                QPath::TypeRelative(..) | QPath::LangItem(..) if let Some((kind, id)) = typeck.ty_based_def(id) => {
                    Res::Def(kind, id)
                },
                QPath::TypeRelative(..) | QPath::LangItem(..) => Res::Err,
            }
        }
        match self.opt_qpath() {
            Some((qpath, id)) if let Some(typeck) = typeck.typeck_res() => f(qpath, id, typeck),
            _ => Res::Err,
        }
    }

    /// If this is a path with the specified name as its final segment gets its
    /// resolution. Returns `Res::Err` otherwise.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    fn res_if_named<'tcx>(self, typeck: &impl MaybeTypeckRes<'tcx>, name: Symbol) -> Res {
        #[cfg_attr(debug_assertions, track_caller)]
        fn f(qpath: &QPath<'_>, id: HirId, typeck: &TypeckResults<'_>, name: Symbol) -> Res {
            match *qpath {
                QPath::Resolved(_, p)
                    if let [.., seg] = p.segments
                        && seg.ident.name == name =>
                {
                    p.res
                },
                QPath::TypeRelative(_, seg)
                    if seg.ident.name == name
                        && let Some((kind, id)) = typeck.ty_based_def(id) =>
                {
                    Res::Def(kind, id)
                },
                QPath::Resolved(..) | QPath::TypeRelative(..) | QPath::LangItem(..) => Res::Err,
            }
        }
        match self.opt_qpath() {
            Some((qpath, id)) if let Some(typeck) = typeck.typeck_res() => f(qpath, id, typeck, name),
            _ => Res::Err,
        }
    }

    /// If this is a path gets both its resolution and final segment.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    fn res_with_seg<'tcx>(self, typeck: &impl MaybeTypeckRes<'tcx>) -> (Res, Option<&'a PathSegment<'a>>) {
        #[cfg_attr(debug_assertions, track_caller)]
        fn f<'a>(qpath: &QPath<'a>, id: HirId, typeck: &TypeckResults<'_>) -> (Res, Option<&'a PathSegment<'a>>) {
            match *qpath {
                QPath::Resolved(_, p) if let [.., seg] = p.segments => (p.res, Some(seg)),
                QPath::TypeRelative(_, seg) if let Some((kind, id)) = typeck.ty_based_def(id) => {
                    (Res::Def(kind, id), Some(seg))
                },
                QPath::Resolved(..) | QPath::TypeRelative(..) | QPath::LangItem(..) => (Res::Err, None),
            }
        }
        match self.opt_qpath() {
            Some((qpath, id)) if let Some(typeck) = typeck.typeck_res() => f(qpath, id, typeck),
            _ => (Res::Err, None),
        }
    }

    /// If this is a path without an explicit `Self` type gets its resolution.
    /// Returns `Res::Err` otherwise.
    ///
    /// Only paths to trait items can optionally contain a `Self` type.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    fn typeless_res<'tcx>(self, typeck: &impl MaybeTypeckRes<'tcx>) -> Res {
        #[cfg_attr(debug_assertions, track_caller)]
        fn f(qpath: &QPath<'_>, id: HirId, typeck: &TypeckResults<'_>) -> Res {
            match *qpath {
                QPath::Resolved(
                    None
                    | Some(&hir::Ty {
                        kind: TyKind::Infer(()),
                        ..
                    }),
                    p,
                ) => p.res,
                QPath::TypeRelative(
                    &hir::Ty {
                        kind: TyKind::Infer(()),
                        ..
                    },
                    _,
                ) if let Some((kind, id)) = typeck.ty_based_def(id) => Res::Def(kind, id),
                QPath::Resolved(..) | QPath::TypeRelative(..) | QPath::LangItem(..) => Res::Err,
            }
        }
        match self.opt_qpath() {
            Some((qpath, id)) if let Some(typeck) = typeck.typeck_res() => f(qpath, id, typeck),
            _ => Res::Err,
        }
    }

    /// If this is a path without an explicit `Self` type to an item with the
    /// specified name gets its resolution. Returns `Res::Err` otherwise.
    ///
    /// Only paths to trait items can optionally contain a `Self` type.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    fn typeless_res_if_named<'tcx>(self, typeck: &impl MaybeTypeckRes<'tcx>, name: Symbol) -> Res {
        #[cfg_attr(debug_assertions, track_caller)]
        fn f(qpath: &QPath<'_>, id: HirId, typeck: &TypeckResults<'_>, name: Symbol) -> Res {
            match *qpath {
                QPath::Resolved(
                    None
                    | Some(&hir::Ty {
                        kind: TyKind::Infer(()),
                        ..
                    }),
                    p,
                ) if let [.., seg] = p.segments
                    && seg.ident.name == name =>
                {
                    p.res
                },
                QPath::TypeRelative(
                    &hir::Ty {
                        kind: TyKind::Infer(()),
                        ..
                    },
                    seg,
                ) if seg.ident.name == name
                    && let Some((kind, id)) = typeck.ty_based_def(id) =>
                {
                    Res::Def(kind, id)
                },
                QPath::Resolved(..) | QPath::TypeRelative(..) | QPath::LangItem(..) => Res::Err,
            }
        }
        match self.opt_qpath() {
            Some((qpath, id)) if let Some(typeck) = typeck.typeck_res() => f(qpath, id, typeck, name),
            _ => Res::Err,
        }
    }

    /// If this is a type-relative path gets the definition it resolves to.
    ///
    /// Only inherent associated items require a type-relative path.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    fn ty_rel_def<'tcx>(self, typeck: &impl MaybeTypeckRes<'tcx>) -> Option<DefRes> {
        match self.opt_qpath() {
            Some((QPath::TypeRelative(..), id)) => typeck.ty_based_def(id),
            _ => None,
        }
    }

    /// If this is a type-relative path to an item with the specified name gets
    /// the definition it resolves to.
    ///
    /// Only inherent associated items require a type-relative path.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    fn ty_rel_def_if_named<'tcx>(self, typeck: &impl MaybeTypeckRes<'tcx>, name: Symbol) -> Option<DefRes> {
        match self.opt_qpath() {
            Some((&QPath::TypeRelative(_, seg), id)) if seg.ident.name == name => typeck.ty_based_def(id),
            _ => None,
        }
    }

    /// If this is a type-relative path gets the definition it resolves to and
    /// its final segment.
    ///
    /// Only inherent associated items require a type-relative path.
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    fn ty_rel_def_with_seg<'tcx>(self, typeck: &impl MaybeTypeckRes<'tcx>) -> Option<(DefRes, &'a PathSegment<'a>)> {
        match self.opt_qpath() {
            Some((QPath::TypeRelative(_, seg), id)) if let Some(def) = typeck.ty_based_def(id) => Some((def, seg)),
            _ => None,
        }
    }
}

impl<'tcx> MaybeQPath<'tcx> for QPathId<'tcx> {
    #[inline]
    fn opt_qpath(self) -> Option<QPathId<'tcx>> {
        Some((self.0, self.1))
    }
}
impl<'tcx> MaybeQPath<'tcx> for &'tcx Expr<'_> {
    #[inline]
    fn opt_qpath(self) -> Option<QPathId<'tcx>> {
        match &self.kind {
            ExprKind::Path(qpath) => Some((qpath, self.hir_id)),
            _ => None,
        }
    }
}
impl<'tcx> MaybeQPath<'tcx> for &'tcx PatExpr<'_> {
    #[inline]
    fn opt_qpath(self) -> Option<QPathId<'tcx>> {
        match &self.kind {
            PatExprKind::Path(qpath) => Some((qpath, self.hir_id)),
            _ => None,
        }
    }
}
impl<'tcx, AmbigArg> MaybeQPath<'tcx> for &'tcx hir::Ty<'_, AmbigArg> {
    #[inline]
    fn opt_qpath(self) -> Option<QPathId<'tcx>> {
        match &self.kind {
            TyKind::Path(qpath) => Some((qpath, self.hir_id)),
            _ => None,
        }
    }
}
impl<'tcx> MaybeQPath<'tcx> for &'_ Pat<'tcx> {
    #[inline]
    fn opt_qpath(self) -> Option<QPathId<'tcx>> {
        match self.kind {
            PatKind::Expr(e) => e.opt_qpath(),
            _ => None,
        }
    }
}
impl<'tcx, T: MaybeQPath<'tcx>> MaybeQPath<'tcx> for Option<T> {
    #[inline]
    fn opt_qpath(self) -> Option<QPathId<'tcx>> {
        self.and_then(T::opt_qpath)
    }
}
impl<'tcx, T: Copy + MaybeQPath<'tcx>> MaybeQPath<'tcx> for &Option<T> {
    #[inline]
    fn opt_qpath(self) -> Option<QPathId<'tcx>> {
        self.and_then(T::opt_qpath)
    }
}

/// A resolved path and the explicit `Self` type if there is one.
type OptResPath<'tcx> = (Option<&'tcx hir::Ty<'tcx>>, Option<&'tcx Path<'tcx>>);

/// A HIR node which might be a `QPath::Resolved`.
///
/// The following are resolved paths:
/// * A path to a module or crate item.
/// * A path to a trait item via the trait's name.
/// * A path to a struct or variant constructor via the original type's path.
/// * A local.
///
/// All other paths are `TypeRelative` and require using `PathRes` to lookup the
/// resolution.
pub trait MaybeResPath<'a>: Copy {
    /// If this node is a resolved path gets both the contained path and the
    /// type associated with it.
    fn opt_res_path(self) -> OptResPath<'a>;

    /// If this node is a resolved path gets it's resolution. Returns `Res::Err`
    /// otherwise.
    #[inline]
    fn basic_res(self) -> &'a Res {
        self.opt_res_path().1.map_or(&Res::Err, |p| &p.res)
    }

    /// If this node is a path to a local gets the local's `HirId`.
    #[inline]
    fn res_local_id(self) -> Option<HirId> {
        if let (_, Some(p)) = self.opt_res_path()
            && let Res::Local(id) = p.res
        {
            Some(id)
        } else {
            None
        }
    }

    /// If this node is a path to a local gets the local's `HirId` and identifier.
    fn res_local_id_and_ident(self) -> Option<(HirId, &'a Ident)> {
        if let (_, Some(p)) = self.opt_res_path()
            && let Res::Local(id) = p.res
            && let [seg] = p.segments
        {
            Some((id, &seg.ident))
        } else {
            None
        }
    }
}
impl<'a> MaybeResPath<'a> for &'a Path<'a> {
    #[inline]
    fn opt_res_path(self) -> OptResPath<'a> {
        (None, Some(self))
    }

    #[inline]
    fn basic_res(self) -> &'a Res {
        &self.res
    }
}
impl<'a> MaybeResPath<'a> for &QPath<'a> {
    #[inline]
    fn opt_res_path(self) -> OptResPath<'a> {
        match *self {
            QPath::Resolved(ty, path) => (ty, Some(path)),
            _ => (None, None),
        }
    }
}
impl<'a> MaybeResPath<'a> for &Expr<'a> {
    #[inline]
    fn opt_res_path(self) -> OptResPath<'a> {
        match &self.kind {
            ExprKind::Path(qpath) => qpath.opt_res_path(),
            _ => (None, None),
        }
    }
}
impl<'a> MaybeResPath<'a> for &PatExpr<'a> {
    #[inline]
    fn opt_res_path(self) -> OptResPath<'a> {
        match &self.kind {
            PatExprKind::Path(qpath) => qpath.opt_res_path(),
            _ => (None, None),
        }
    }
}
impl<'a, AmbigArg> MaybeResPath<'a> for &hir::Ty<'a, AmbigArg> {
    #[inline]
    fn opt_res_path(self) -> OptResPath<'a> {
        match &self.kind {
            TyKind::Path(qpath) => qpath.opt_res_path(),
            _ => (None, None),
        }
    }
}
impl<'a> MaybeResPath<'a> for &Pat<'a> {
    #[inline]
    fn opt_res_path(self) -> OptResPath<'a> {
        match self.kind {
            PatKind::Expr(e) => e.opt_res_path(),
            _ => (None, None),
        }
    }
}
impl<'a, T: MaybeResPath<'a>> MaybeResPath<'a> for Option<T> {
    #[inline]
    fn opt_res_path(self) -> OptResPath<'a> {
        match self {
            Some(x) => T::opt_res_path(x),
            None => (None, None),
        }
    }

    #[inline]
    fn basic_res(self) -> &'a Res {
        self.map_or(&Res::Err, T::basic_res)
    }
}

/// A type which may either contain a `DefId` or be referred to by a `DefId`.
pub trait MaybeDef: Copy {
    fn opt_def_id(self) -> Option<DefId>;

    /// Gets this definition's id and kind. This will lookup the kind in the def
    /// tree if needed.
    fn opt_def<'tcx>(self, tcx: &impl HasTyCtxt<'tcx>) -> Option<(DefKind, DefId)>;

    /// Gets the diagnostic name of this definition if it has one.
    #[inline]
    fn opt_diag_name<'tcx>(self, tcx: &impl HasTyCtxt<'tcx>) -> Option<Symbol> {
        self.opt_def_id().and_then(|id| tcx.tcx().get_diagnostic_name(id))
    }

    /// Checks if this definition has the specified diagnostic name.
    #[inline]
    fn is_diag_item<'tcx>(self, tcx: &impl HasTyCtxt<'tcx>, name: Symbol) -> bool {
        self.opt_def_id()
            .is_some_and(|id| tcx.tcx().is_diagnostic_item(name, id))
    }

    /// Checks if this definition is the specified `LangItem`.
    #[inline]
    fn is_lang_item<'tcx>(self, tcx: &impl HasTyCtxt<'tcx>, item: LangItem) -> bool {
        self.opt_def_id()
            .is_some_and(|id| tcx.tcx().lang_items().get(item) == Some(id))
    }

    /// If this definition is an impl block gets its type.
    #[inline]
    fn opt_impl_ty<'tcx>(self, tcx: &impl HasTyCtxt<'tcx>) -> Option<EarlyBinder<'tcx, Ty<'tcx>>> {
        match self.opt_def(tcx) {
            Some((DefKind::Impl { .. }, id)) => Some(tcx.tcx().type_of(id)),
            _ => None,
        }
    }

    /// Gets the parent of this definition if it has one.
    #[inline]
    fn opt_parent<'tcx>(self, tcx: &impl HasTyCtxt<'tcx>) -> Option<DefId> {
        self.opt_def_id().and_then(|id| tcx.tcx().opt_parent(id))
    }

    /// Checks if this definition is an impl block.
    #[inline]
    fn is_impl<'tcx>(self, tcx: &impl HasTyCtxt<'tcx>) -> bool {
        matches!(self.opt_def(tcx), Some((DefKind::Impl { .. }, _)))
    }

    /// If this definition is a constructor gets the `DefId` of it's type or variant.
    #[inline]
    fn ctor_parent<'tcx>(self, tcx: &impl HasTyCtxt<'tcx>) -> Option<DefId> {
        match self.opt_def(tcx) {
            Some((DefKind::Ctor(..), id)) => tcx.tcx().opt_parent(id),
            _ => None,
        }
    }

    /// If this definition is an associated item of an impl or trait gets the
    /// `DefId` of its parent.
    #[inline]
    fn assoc_parent<'tcx>(self, tcx: &impl HasTyCtxt<'tcx>) -> Option<DefId> {
        match self.opt_def(tcx) {
            Some((DefKind::AssocConst | DefKind::AssocFn | DefKind::AssocTy, id)) => tcx.tcx().opt_parent(id),
            _ => None,
        }
    }

    /// If this definition is an associated function of an impl or trait gets the
    /// `DefId` of its parent.
    #[inline]
    fn assoc_fn_parent<'tcx>(self, tcx: &impl HasTyCtxt<'tcx>) -> Option<DefId> {
        match self.opt_def(tcx) {
            Some((DefKind::AssocFn, id)) => tcx.tcx().opt_parent(id),
            _ => None,
        }
    }
}
impl MaybeDef for DefId {
    #[inline]
    fn opt_def_id(self) -> Option<DefId> {
        Some(self)
    }

    #[inline]
    fn opt_def<'tcx>(self, tcx: &impl HasTyCtxt<'tcx>) -> Option<(DefKind, DefId)> {
        self.opt_def_id().map(|id| (tcx.tcx().def_kind(id), id))
    }
}
impl MaybeDef for (DefKind, DefId) {
    #[inline]
    fn opt_def_id(self) -> Option<DefId> {
        Some(self.1)
    }

    #[inline]
    fn opt_def<'tcx>(self, _: &impl HasTyCtxt<'tcx>) -> Option<(DefKind, DefId)> {
        Some(self)
    }
}
impl MaybeDef for AdtDef<'_> {
    #[inline]
    fn opt_def_id(self) -> Option<DefId> {
        Some(self.did())
    }

    #[inline]
    fn opt_def<'tcx>(self, _: &impl HasTyCtxt<'tcx>) -> Option<(DefKind, DefId)> {
        let did = self.did();
        match self.adt_kind() {
            AdtKind::Enum => Some((DefKind::Enum, did)),
            AdtKind::Struct => Some((DefKind::Struct, did)),
            AdtKind::Union => Some((DefKind::Union, did)),
        }
    }
}
impl MaybeDef for Ty<'_> {
    #[inline]
    fn opt_def_id(self) -> Option<DefId> {
        self.ty_adt_def().opt_def_id()
    }

    #[inline]
    fn opt_def<'tcx>(self, tcx: &impl HasTyCtxt<'tcx>) -> Option<(DefKind, DefId)> {
        self.ty_adt_def().opt_def(tcx)
    }
}
impl MaybeDef for Res {
    #[inline]
    fn opt_def_id(self) -> Option<DefId> {
        Res::opt_def_id(&self)
    }

    #[inline]
    fn opt_def<'tcx>(self, _: &impl HasTyCtxt<'tcx>) -> Option<(DefKind, DefId)> {
        match self {
            Res::Def(kind, id) => Some((kind, id)),
            _ => None,
        }
    }
}
impl<T: MaybeDef> MaybeDef for Option<T> {
    #[inline]
    fn opt_def_id(self) -> Option<DefId> {
        self.and_then(T::opt_def_id)
    }

    #[inline]
    fn opt_def<'tcx>(self, tcx: &impl HasTyCtxt<'tcx>) -> Option<(DefKind, DefId)> {
        self.and_then(|x| T::opt_def(x, tcx))
    }
}
impl<T: MaybeDef> MaybeDef for EarlyBinder<'_, T> {
    #[inline]
    fn opt_def_id(self) -> Option<DefId> {
        self.skip_binder().opt_def_id()
    }

    #[inline]
    fn opt_def<'tcx>(self, tcx: &impl HasTyCtxt<'tcx>) -> Option<(DefKind, DefId)> {
        self.skip_binder().opt_def(tcx)
    }
}
impl<T: MaybeDef> MaybeDef for Binder<'_, T> {
    #[inline]
    fn opt_def_id(self) -> Option<DefId> {
        self.skip_binder().opt_def_id()
    }

    #[inline]
    fn opt_def<'tcx>(self, tcx: &impl HasTyCtxt<'tcx>) -> Option<(DefKind, DefId)> {
        self.skip_binder().opt_def(tcx)
    }
}
