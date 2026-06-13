use clippy_config::Conf;
use clippy_config::types::{DisallowedPath, create_disallowed_map};
use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::disallowed_profiles::{ProfileEntry, ProfileResolver};
use clippy_utils::paths::PathNS;
use clippy_utils::sym;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::smallvec::SmallVec;
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::DefIdMap;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;
use rustc_span::{Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Denies the configured methods and functions in clippy.toml
    ///
    /// Note: Even though this lint is warn-by-default, it will only trigger if
    /// methods are defined in the clippy.toml file.
    ///
    /// ### Why is this bad?
    /// Some methods are undesirable in certain contexts, and it's beneficial to
    /// lint for them as needed.
    ///
    /// ### Example
    /// An example clippy.toml configuration:
    /// ```toml
    /// # clippy.toml
    /// disallowed-methods = [
    ///     # Can use a string as the path of the disallowed method.
    ///     "std::boxed::Box::new",
    ///     # Can also use an inline table with a `path` key.
    ///     { path = "std::time::Instant::now" },
    ///     # When using an inline table, can add a `reason` for why the method
    ///     # is disallowed.
    ///     { path = "std::vec::Vec::leak", reason = "no leaking memory" },
    ///     # Can also add a `replacement` that will be offered as a suggestion.
    ///     { path = "std::sync::Mutex::new", reason = "prefer faster & simpler non-poisonable mutex", replacement = "parking_lot::Mutex::new" },
    ///     # This would normally error if the path is incorrect, but with `allow-invalid` = `true`,
    ///     # it will be silently ignored
    ///     { path = "std::fs::InvalidPath", reason = "use alternative instead", allow-invalid = true },
    /// ]
    /// ```
    ///
    /// ```rust,ignore
    /// let xs = vec![1, 2, 3, 4];
    /// xs.leak(); // Vec::leak is disallowed in the config.
    /// // The diagnostic contains the message "no leaking memory".
    ///
    /// let _now = Instant::now(); // Instant::now is disallowed in the config.
    ///
    /// let _box = Box::new(3); // Box::new is disallowed in the config.
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// let mut xs = Vec::new(); // Vec::new is _not_ disallowed in the config.
    /// xs.push(123); // Vec::push is _not_ disallowed in the config.
    /// ```
    ///
    /// Disallowed profiles allow scoping different disallow lists:
    /// ```toml
    /// [profiles.forward_pass]
    /// disallowed-methods = [{ path = "crate::devices::Buffer::copy_to_host", reason = "Forward code must not touch host buffers" }]
    /// ```
    ///
    /// ```rust,ignore
    /// #[clippy::disallowed_profile("forward_pass")]
    /// fn evaluate() {
    ///     // Method calls in this function use the `forward_pass` profile.
    /// }
    /// ```
    #[clippy::version = "1.49.0"]
    pub DISALLOWED_METHODS,
    style,
    "use of a disallowed method call"
}

impl_lint_pass!(DisallowedMethods => [DISALLOWED_METHODS]);

pub struct DisallowedMethods {
    default: DefIdMap<(&'static str, &'static DisallowedPath)>,
    /// Lookup per profile that declares a non-empty `disallowed_methods` list. Profiles
    /// declared in `[profiles.*]` but without `disallowed_methods` entries are absent here.
    profiles: FxHashMap<Symbol, DefIdMap<(&'static str, &'static DisallowedPath)>>,
    /// Every profile name declared in `[profiles.*]`, regardless of whether it contributes
    /// to this lint. Used to suppress the "unknown profile" warning for profiles that exist
    /// in config but only define entries for other lints (e.g. `disallowed_types`).
    known_profiles: FxHashSet<Symbol>,
    profile_cache: ProfileResolver,
    warned_unknown_profiles: FxHashSet<Span>,
}

impl DisallowedMethods {
    #[allow(rustc::potential_query_instability)] // Profiles are sorted for deterministic iteration.
    pub fn new(tcx: TyCtxt<'_>, conf: &'static Conf) -> Self {
        let (default, _) = create_disallowed_map(
            tcx,
            &conf.disallowed_methods,
            PathNS::Value,
            |def_kind| {
                matches!(
                    def_kind,
                    DefKind::Fn | DefKind::Ctor(_, CtorKind::Fn) | DefKind::AssocFn
                )
            },
            "function",
            false,
        );

        let mut profiles = FxHashMap::default();
        let mut known_profiles = FxHashSet::default();
        let mut profile_entries: Vec<_> = conf.profiles.iter().collect();
        profile_entries.sort_by_key(|(a, _)| *a);
        for (name, profile) in profile_entries {
            let symbol = Symbol::intern(name.as_str());
            known_profiles.insert(symbol);

            let paths = profile.disallowed_methods.as_slice();
            if paths.is_empty() {
                continue;
            }

            let (map, _) = create_disallowed_map(
                tcx,
                paths,
                PathNS::Value,
                |def_kind| {
                    matches!(
                        def_kind,
                        DefKind::Fn | DefKind::Ctor(_, CtorKind::Fn) | DefKind::AssocFn
                    )
                },
                "function",
                false,
            );
            profiles.insert(symbol, map);
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
            span_lint(
                cx,
                DISALLOWED_METHODS,
                entry.span,
                format!(
                    "`{attr_name}` references unknown profile `{}` for `clippy::disallowed_methods`",
                    entry.name
                ),
            );
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for DisallowedMethods {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.desugaring_kind().is_some() {
            return;
        }
        let (id, span) = match &expr.kind {
            ExprKind::Path(path) if let Res::Def(_, id) = cx.qpath_res(path, expr.hir_id) => (id, expr.span),
            ExprKind::MethodCall(name, ..) if let Some(id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) => {
                (id, name.ident.span)
            },
            _ => return,
        };
        let mut active_profiles = SmallVec::<[Symbol; 2]>::new();
        // Copy entries out of the cache before iterating: `warn_unknown_profile` takes
        // `&mut self`, which conflicts with the borrow held by `active_profiles(...)`.
        let entries: SmallVec<[ProfileEntry; 2]> = self
            .profile_cache
            .active_profiles(cx, expr.hir_id)
            .map(|selection| selection.iter().copied().collect())
            .unwrap_or_default();
        for entry in &entries {
            if self.profiles.contains_key(&entry.name) {
                active_profiles.push(entry.name);
            } else if !self.known_profiles.contains(&entry.name) {
                self.warn_unknown_profile(cx, entry);
            }
        }

        if let Some((profile, &(path, disallowed_path))) = active_profiles.iter().find_map(|symbol| {
            self.profiles
                .get(symbol)
                .and_then(|map| map.get(&id).map(|info| (*symbol, info)))
        }) {
            let diag_amendment = disallowed_path.diag_amendment(span);
            span_lint_and_then(
                cx,
                DISALLOWED_METHODS,
                span,
                format!("use of a disallowed method `{path}` (profile: {profile})"),
                |diag| diag_amendment(diag),
            );
        } else if let Some(&(path, disallowed_path)) = self.default.get(&id) {
            let diag_amendment = disallowed_path.diag_amendment(span);
            span_lint_and_then(
                cx,
                DISALLOWED_METHODS,
                span,
                format!("use of a disallowed method `{path}`"),
                |diag| diag_amendment(diag),
            );
        }
    }
}
