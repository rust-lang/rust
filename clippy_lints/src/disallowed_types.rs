use clippy_config::Conf;
use clippy_config::types::{DisallowedPath, create_disallowed_map};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::disallowed_profiles::{ProfileEntry, ProfileResolver};
use clippy_utils::paths::PathNS;
use clippy_utils::sym;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::smallvec::SmallVec;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefIdMap;
use rustc_hir::{AmbigArg, Item, ItemKind, PolyTraitRef, PrimTy, Ty, TyKind, UseKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;
use rustc_span::{Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Denies the configured types in clippy.toml.
    ///
    /// Note: Even though this lint is warn-by-default, it will only trigger if
    /// types are defined in the clippy.toml file.
    ///
    /// ### Why is this bad?
    /// Some types are undesirable in certain contexts.
    ///
    /// ### Example:
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// disallowed-types = [
    ///     # Can use a string as the path of the disallowed type.
    ///     "std::collections::BTreeMap",
    ///     # Can also use an inline table with a `path` key.
    ///     { path = "std::net::TcpListener" },
    ///     # When using an inline table, can add a `reason` for why the type
    ///     # is disallowed.
    ///     { path = "std::net::Ipv4Addr", reason = "no IPv4 allowed" },
    ///     # Can also add a `replacement` that will be offered as a suggestion.
    ///     { path = "std::sync::Mutex", reason = "prefer faster & simpler non-poisonable mutex", replacement = "parking_lot::Mutex" },
    ///     # This would normally error if the path is incorrect, but with `allow-invalid` = `true`,
    ///     # it will be silently ignored
    ///     { path = "std::invalid::Type", reason = "use alternative instead", allow-invalid = true }
    /// ]
    /// ```
    ///
    /// ```rust,ignore
    /// use std::collections::BTreeMap;
    /// // or its use
    /// let x = std::collections::BTreeMap::new();
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// // A similar type that is allowed by the config
    /// use std::collections::HashMap;
    /// ```
    ///
    /// Profiles can scope lists to specific modules:
    /// ```toml
    /// [disallowed-types-profiles.forward_pass]
    /// paths = [
    ///     { path = "crate::buffers::HostBuffer", reason = "Prefer device buffers in forward computations" }
    /// ]
    /// ```
    ///
    /// ```rust,ignore
    /// #[clippy::disallowed_profile("forward_pass")]
    /// fn forward_step(buffer: crate::buffers::DeviceBuffer) { /* ... */ }
    /// ```
    #[clippy::version = "1.55.0"]
    pub DISALLOWED_TYPES,
    style,
    "use of disallowed types"
}

impl_lint_pass!(DisallowedTypes => [DISALLOWED_TYPES]);

struct TypeLookup {
    def_ids: DefIdMap<(&'static str, &'static DisallowedPath)>,
    prim_tys: FxHashMap<PrimTy, (&'static str, &'static DisallowedPath)>,
}

impl TypeLookup {
    fn from_config(tcx: TyCtxt<'_>, paths: &'static [DisallowedPath]) -> Self {
        let (def_ids, prim_tys) = create_disallowed_map(tcx, paths, PathNS::Type, def_kind_predicate, "type", true);
        Self { def_ids, prim_tys }
    }

    fn find(&self, res: &Res) -> Option<(&'static str, &'static DisallowedPath)> {
        match res {
            Res::Def(_, did) => self.def_ids.get(did).copied(),
            Res::PrimTy(prim) => self.prim_tys.get(prim).copied(),
            _ => None,
        }
    }
}

pub struct DisallowedTypes {
    default: TypeLookup,
    profiles: FxHashMap<Symbol, TypeLookup>,
    known_profiles: FxHashSet<Symbol>,
    profile_cache: ProfileResolver,
    warned_unknown_profiles: FxHashSet<Span>,
}

impl DisallowedTypes {
    #[allow(rustc::potential_query_instability)] // Profiles are sorted for deterministic iteration.
    pub fn new(tcx: TyCtxt<'_>, conf: &'static Conf) -> Self {
        let default = TypeLookup::from_config(tcx, &conf.disallowed_types);

        let mut profiles = FxHashMap::default();
        let mut names: Vec<_> = conf.disallowed_types_profiles.keys().collect();
        names.sort();
        for name in names {
            let symbol = Symbol::intern(name.as_str());
            let paths = conf
                .disallowed_types_profiles
                .get(name)
                .expect("profile entry must exist");
            profiles.insert(symbol, TypeLookup::from_config(tcx, paths));
        }

        let mut known_profiles = FxHashSet::default();
        for name in conf
            .disallowed_types_profiles
            .keys()
            .chain(conf.disallowed_methods_profiles.keys())
        {
            known_profiles.insert(Symbol::intern(name.as_str()));
        }

        Self {
            default,
            profiles,
            known_profiles,
            profile_cache: ProfileResolver::default(),
            warned_unknown_profiles: FxHashSet::default(),
        }
    }

    fn warn_unknown_profile(&mut self, cx: &LateContext<'_>, entry: &ProfileEntry) {
        if self.warned_unknown_profiles.insert(entry.span) {
            let attr_name = if entry.attr_name == sym::disallowed_profiles {
                "clippy::disallowed_profiles"
            } else {
                "clippy::disallowed_profile"
            };
            cx.tcx
                .sess
                .dcx()
                .struct_span_warn(
                    entry.span,
                    format!(
                        "`{attr_name}` references unknown profile `{}` for `clippy::disallowed_types`",
                        entry.name
                    ),
                )
                .emit();
        }
    }

    fn check_res_emit(&mut self, cx: &LateContext<'_>, hir_id: rustc_hir::HirId, res: &Res, span: Span) {
        let mut active_profiles = SmallVec::<[Symbol; 2]>::new();
        let mut unknown_profiles = SmallVec::<[ProfileEntry; 2]>::new();
        if let Some(selection) = self.profile_cache.active_profiles(cx, hir_id) {
            for entry in selection.iter() {
                if self.profiles.contains_key(&entry.name) {
                    active_profiles.push(entry.name);
                } else if !self.known_profiles.contains(&entry.name) {
                    unknown_profiles.push(entry.clone());
                }
            }
        }

        for entry in unknown_profiles {
            self.warn_unknown_profile(cx, &entry);
        }

        if let Some((profile, (path, disallowed_path))) = active_profiles.iter().find_map(|symbol| {
            self.profiles
                .get(symbol)
                .and_then(|lookup| lookup.find(res).map(|info| (*symbol, info)))
        }) {
            let diag_amendment = disallowed_path.diag_amendment(span);
            span_lint_and_then(
                cx,
                DISALLOWED_TYPES,
                span,
                format!("use of a disallowed type `{path}` (profile: {profile})"),
                |diag| diag_amendment(diag),
            );
        } else if let Some((path, disallowed_path)) = self.default.find(res) {
            let diag_amendment = disallowed_path.diag_amendment(span);
            span_lint_and_then(
                cx,
                DISALLOWED_TYPES,
                span,
                format!("use of a disallowed type `{path}`"),
                |diag| diag_amendment(diag),
            );
        }
    }
}

pub fn def_kind_predicate(def_kind: DefKind) -> bool {
    matches!(
        def_kind,
        DefKind::Struct
            | DefKind::Union
            | DefKind::Enum
            | DefKind::Trait
            | DefKind::TyAlias
            | DefKind::ForeignTy
            | DefKind::AssocTy
    )
}

impl<'tcx> LateLintPass<'tcx> for DisallowedTypes {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Use(path, UseKind::Single(_)) = &item.kind
            && let Some(res) = path.res.type_ns
        {
            self.check_res_emit(cx, item.hir_id(), &res, item.span);
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx Ty<'tcx, AmbigArg>) {
        if let TyKind::Path(path) = &ty.kind {
            self.check_res_emit(cx, ty.hir_id, &cx.qpath_res(path, ty.hir_id), ty.span);
        }
    }

    fn check_poly_trait_ref(&mut self, cx: &LateContext<'tcx>, poly: &'tcx PolyTraitRef<'tcx>) {
        self.check_res_emit(
            cx,
            poly.trait_ref.hir_ref_id,
            &poly.trait_ref.path.res,
            poly.trait_ref.path.span,
        );
    }
}
