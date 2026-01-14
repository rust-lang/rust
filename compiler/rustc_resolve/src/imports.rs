//! A bunch of methods and structures more or less related to resolving imports.

use std::mem;

use rustc_ast::NodeId;
use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_data_structures::intern::Interned;
use rustc_errors::codes::*;
use rustc_errors::{Applicability, MultiSpan, pluralize, struct_span_code_err};
use rustc_hir::def::{self, DefKind, PartialRes};
use rustc_hir::def_id::{DefId, LocalDefIdMap};
use rustc_middle::metadata::{AmbigModChild, ModChild, Reexport};
use rustc_middle::span_bug;
use rustc_middle::ty::Visibility;
use rustc_session::lint::BuiltinLintDiag;
use rustc_session::lint::builtin::{
    AMBIGUOUS_GLOB_REEXPORTS, EXPORTED_PRIVATE_DEPENDENCIES, HIDDEN_GLOB_REEXPORTS,
    PUB_USE_OF_PRIVATE_EXTERN_CRATE, REDUNDANT_IMPORTS, UNUSED_IMPORTS,
};
use rustc_session::parse::feature_err;
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::hygiene::LocalExpnId;
use rustc_span::{Ident, Macros20NormalizedIdent, Span, Symbol, kw, sym};
use tracing::debug;

use crate::Namespace::{self, *};
use crate::diagnostics::{DiagMode, Suggestion, import_candidates};
use crate::errors::{
    CannotBeReexportedCratePublic, CannotBeReexportedCratePublicNS, CannotBeReexportedPrivate,
    CannotBeReexportedPrivateNS, CannotDetermineImportResolution, CannotGlobImportAllCrates,
    ConsiderAddingMacroExport, ConsiderMarkingAsPub, ConsiderMarkingAsPubCrate,
};
use crate::ref_mut::CmCell;
use crate::{
    AmbiguityError, BindingKey, CmResolver, Decl, DeclData, DeclKind, Determinacy, Finalize,
    ImportSuggestion, Module, ModuleOrUniformRoot, ParentScope, PathResult, PerNS, ResolutionError,
    Resolver, ScopeSet, Segment, Used, module_to_string, names_to_string,
};

type Res = def::Res<NodeId>;

/// A potential import declaration in the process of being planted into a module.
/// Also used for lazily planting names from `--extern` flags to extern prelude.
#[derive(Clone, Copy, Default, PartialEq)]
pub(crate) enum PendingDecl<'ra> {
    Ready(Option<Decl<'ra>>),
    #[default]
    Pending,
}

impl<'ra> PendingDecl<'ra> {
    pub(crate) fn decl(self) -> Option<Decl<'ra>> {
        match self {
            PendingDecl::Ready(decl) => decl,
            PendingDecl::Pending => None,
        }
    }
}

/// Contains data for specific kinds of imports.
#[derive(Clone)]
pub(crate) enum ImportKind<'ra> {
    Single {
        /// `source` in `use prefix::source as target`.
        source: Ident,
        /// `target` in `use prefix::source as target`.
        /// It will directly use `source` when the format is `use prefix::source`.
        target: Ident,
        /// Name declarations introduced by the import.
        decls: PerNS<CmCell<PendingDecl<'ra>>>,
        /// `true` for `...::{self [as target]}` imports, `false` otherwise.
        type_ns_only: bool,
        /// Did this import result from a nested import? ie. `use foo::{bar, baz};`
        nested: bool,
        /// The ID of the `UseTree` that imported this `Import`.
        ///
        /// In the case where the `Import` was expanded from a "nested" use tree,
        /// this id is the ID of the leaf tree. For example:
        ///
        /// ```ignore (pacify the merciless tidy)
        /// use foo::bar::{a, b}
        /// ```
        ///
        /// If this is the import for `foo::bar::a`, we would have the ID of the `UseTree`
        /// for `a` in this field.
        id: NodeId,
    },
    Glob {
        // The visibility of the greatest re-export.
        // n.b. `max_vis` is only used in `finalize_import` to check for re-export errors.
        max_vis: CmCell<Option<Visibility>>,
        id: NodeId,
    },
    ExternCrate {
        source: Option<Symbol>,
        target: Ident,
        id: NodeId,
    },
    MacroUse {
        /// A field has been added indicating whether it should be reported as a lint,
        /// addressing issue#119301.
        warn_private: bool,
    },
    MacroExport,
}

/// Manually implement `Debug` for `ImportKind` because the `source/target_bindings`
/// contain `Cell`s which can introduce infinite loops while printing.
impl<'ra> std::fmt::Debug for ImportKind<'ra> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ImportKind::*;
        match self {
            Single { source, target, decls, type_ns_only, nested, id, .. } => f
                .debug_struct("Single")
                .field("source", source)
                .field("target", target)
                // Ignore the nested bindings to avoid an infinite loop while printing.
                .field(
                    "decls",
                    &decls.clone().map(|b| b.into_inner().decl().map(|_| format_args!(".."))),
                )
                .field("type_ns_only", type_ns_only)
                .field("nested", nested)
                .field("id", id)
                .finish(),
            Glob { max_vis, id } => {
                f.debug_struct("Glob").field("max_vis", max_vis).field("id", id).finish()
            }
            ExternCrate { source, target, id } => f
                .debug_struct("ExternCrate")
                .field("source", source)
                .field("target", target)
                .field("id", id)
                .finish(),
            MacroUse { warn_private } => {
                f.debug_struct("MacroUse").field("warn_private", warn_private).finish()
            }
            MacroExport => f.debug_struct("MacroExport").finish(),
        }
    }
}

/// One import.
#[derive(Debug, Clone)]
pub(crate) struct ImportData<'ra> {
    pub kind: ImportKind<'ra>,

    /// Node ID of the "root" use item -- this is always the same as `ImportKind`'s `id`
    /// (if it exists) except in the case of "nested" use trees, in which case
    /// it will be the ID of the root use tree. e.g., in the example
    /// ```ignore (incomplete code)
    /// use foo::bar::{a, b}
    /// ```
    /// this would be the ID of the `use foo::bar` `UseTree` node.
    /// In case of imports without their own node ID it's the closest node that can be used,
    /// for example, for reporting lints.
    pub root_id: NodeId,

    /// Span of the entire use statement.
    pub use_span: Span,

    /// Span of the entire use statement with attributes.
    pub use_span_with_attributes: Span,

    /// Did the use statement have any attributes?
    pub has_attributes: bool,

    /// Span of this use tree.
    pub span: Span,

    /// Span of the *root* use tree (see `root_id`).
    pub root_span: Span,

    pub parent_scope: ParentScope<'ra>,
    pub module_path: Vec<Segment>,
    /// The resolution of `module_path`:
    ///
    /// | `module_path` | `imported_module` | remark |
    /// |-|-|-|
    /// |`use prefix::foo`| `ModuleOrUniformRoot::Module(prefix)`         | - |
    /// |`use ::foo`      | `ModuleOrUniformRoot::ExternPrelude`          | 2018+ editions |
    /// |`use ::foo`      | `ModuleOrUniformRoot::ModuleAndExternPrelude` | a special case in 2015 edition |
    /// |`use foo`        | `ModuleOrUniformRoot::CurrentScope`           | - |
    pub imported_module: CmCell<Option<ModuleOrUniformRoot<'ra>>>,
    pub vis: Visibility,

    /// Span of the visibility.
    pub vis_span: Span,
}

/// All imports are unique and allocated on a same arena,
/// so we can use referential equality to compare them.
pub(crate) type Import<'ra> = Interned<'ra, ImportData<'ra>>;

// Allows us to use Interned without actually enforcing (via Hash/PartialEq/...) uniqueness of the
// contained data.
// FIXME: We may wish to actually have at least debug-level assertions that Interned's guarantees
// are upheld.
impl std::hash::Hash for ImportData<'_> {
    fn hash<H>(&self, _: &mut H)
    where
        H: std::hash::Hasher,
    {
        unreachable!()
    }
}

impl<'ra> ImportData<'ra> {
    pub(crate) fn is_glob(&self) -> bool {
        matches!(self.kind, ImportKind::Glob { .. })
    }

    pub(crate) fn is_nested(&self) -> bool {
        match self.kind {
            ImportKind::Single { nested, .. } => nested,
            _ => false,
        }
    }

    pub(crate) fn id(&self) -> Option<NodeId> {
        match self.kind {
            ImportKind::Single { id, .. }
            | ImportKind::Glob { id, .. }
            | ImportKind::ExternCrate { id, .. } => Some(id),
            ImportKind::MacroUse { .. } | ImportKind::MacroExport => None,
        }
    }

    pub(crate) fn simplify(&self, r: &Resolver<'_, '_>) -> Reexport {
        let to_def_id = |id| r.local_def_id(id).to_def_id();
        match self.kind {
            ImportKind::Single { id, .. } => Reexport::Single(to_def_id(id)),
            ImportKind::Glob { id, .. } => Reexport::Glob(to_def_id(id)),
            ImportKind::ExternCrate { id, .. } => Reexport::ExternCrate(to_def_id(id)),
            ImportKind::MacroUse { .. } => Reexport::MacroUse,
            ImportKind::MacroExport => Reexport::MacroExport,
        }
    }
}

/// Records information about the resolution of a name in a namespace of a module.
#[derive(Clone, Default, Debug)]
pub(crate) struct NameResolution<'ra> {
    /// Single imports that may define the name in the namespace.
    /// Imports are arena-allocated, so it's ok to use pointers as keys.
    pub single_imports: FxIndexSet<Import<'ra>>,
    /// The non-glob declaration for this name, if it is known to exist.
    pub non_glob_decl: Option<Decl<'ra>>,
    /// The glob declaration for this name, if it is known to exist.
    pub glob_decl: Option<Decl<'ra>>,
}

impl<'ra> NameResolution<'ra> {
    /// Returns the binding for the name if it is known or None if it not known.
    pub(crate) fn binding(&self) -> Option<Decl<'ra>> {
        self.best_decl().and_then(|binding| {
            if !binding.is_glob_import() || self.single_imports.is_empty() {
                Some(binding)
            } else {
                None
            }
        })
    }

    pub(crate) fn best_decl(&self) -> Option<Decl<'ra>> {
        self.non_glob_decl.or(self.glob_decl)
    }
}

/// An error that may be transformed into a diagnostic later. Used to combine multiple unresolved
/// import errors within the same use tree into a single diagnostic.
#[derive(Debug, Clone)]
struct UnresolvedImportError {
    span: Span,
    label: Option<String>,
    note: Option<String>,
    suggestion: Option<Suggestion>,
    candidates: Option<Vec<ImportSuggestion>>,
    segment: Option<Symbol>,
    /// comes from `PathRes::Failed { module }`
    module: Option<DefId>,
}

// Reexports of the form `pub use foo as bar;` where `foo` is `extern crate foo;`
// are permitted for backward-compatibility under a deprecation lint.
fn pub_use_of_private_extern_crate_hack(import: Import<'_>, decl: Decl<'_>) -> Option<NodeId> {
    match (&import.kind, &decl.kind) {
        (ImportKind::Single { .. }, DeclKind::Import { import: decl_import, .. })
            if let ImportKind::ExternCrate { id, .. } = decl_import.kind
                && import.vis.is_public() =>
        {
            Some(id)
        }
        _ => None,
    }
}

/// Removes identical import layers from two declarations.
fn remove_same_import<'ra>(d1: Decl<'ra>, d2: Decl<'ra>) -> (Decl<'ra>, Decl<'ra>) {
    if let DeclKind::Import { import: import1, source_decl: d1_next } = d1.kind
        && let DeclKind::Import { import: import2, source_decl: d2_next } = d2.kind
        && import1 == import2
        && d1.warn_ambiguity.get() == d2.warn_ambiguity.get()
    {
        assert_eq!(d1.ambiguity.get(), d2.ambiguity.get());
        assert!(!d1.warn_ambiguity.get());
        assert_eq!(d1.expansion, d2.expansion);
        assert_eq!(d1.span, d2.span);
        assert_eq!(d1.vis(), d2.vis());
        remove_same_import(d1_next, d2_next)
    } else {
        (d1, d2)
    }
}

impl<'ra, 'tcx> Resolver<'ra, 'tcx> {
    /// Given an import and the declaration that it points to,
    /// create the corresponding import declaration.
    pub(crate) fn new_import_decl(&self, decl: Decl<'ra>, import: Import<'ra>) -> Decl<'ra> {
        let import_vis = import.vis.to_def_id();
        let vis = if decl.vis().is_at_least(import_vis, self.tcx)
            || pub_use_of_private_extern_crate_hack(import, decl).is_some()
        {
            import_vis
        } else {
            decl.vis()
        };

        if let ImportKind::Glob { ref max_vis, .. } = import.kind
            && (vis == import_vis
                || max_vis.get().is_none_or(|max_vis| vis.is_at_least(max_vis, self.tcx)))
        {
            max_vis.set_unchecked(Some(vis.expect_local()))
        }

        self.arenas.alloc_decl(DeclData {
            kind: DeclKind::Import { source_decl: decl, import },
            ambiguity: CmCell::new(None),
            warn_ambiguity: CmCell::new(false),
            span: import.span,
            vis: CmCell::new(vis),
            expansion: import.parent_scope.expansion,
            parent_module: Some(import.parent_scope.module),
        })
    }

    /// If `glob_decl` attempts to overwrite `old_glob_decl` in a module,
    /// decide which one to keep.
    fn select_glob_decl(
        &self,
        glob_decl: Decl<'ra>,
        old_glob_decl: Decl<'ra>,
        warn_ambiguity: bool,
    ) -> Decl<'ra> {
        assert!(glob_decl.is_glob_import());
        assert!(old_glob_decl.is_glob_import());
        assert_ne!(glob_decl, old_glob_decl);
        // `best_decl` with a given key in a module may be overwritten in a
        // number of cases (all of them can be seen below in the `match` in `try_define_local`),
        // all these overwrites will be re-fetched by glob imports importing
        // from that module without generating new ambiguities.
        // - A glob decl is overwritten by a non-glob decl arriving later.
        // - A glob decl is overwritten by its clone after setting ambiguity in it.
        //   FIXME: avoid this by removing `warn_ambiguity`, or by triggering glob re-fetch
        //   with the same decl in some way.
        // - A glob decl is overwritten by a glob decl re-fetching an
        //   overwritten decl from other module (the recursive case).
        // Here we are detecting all such re-fetches and overwrite old decls
        // with the re-fetched decls.
        // This is probably incorrect in corner cases, and the outdated decls still get
        // propagated to other places and get stuck there, but that's what we have at the moment.
        let (deep_decl, old_deep_decl) = remove_same_import(glob_decl, old_glob_decl);
        if deep_decl != glob_decl {
            // Some import layers have been removed, need to overwrite.
            assert_ne!(old_deep_decl, old_glob_decl);
            // FIXME: reenable the asserts when `warn_ambiguity` is removed (#149195).
            // assert_ne!(old_deep_decl, deep_decl);
            // assert!(old_deep_decl.is_glob_import());
            assert!(!deep_decl.is_glob_import());
            if glob_decl.is_ambiguity_recursive() {
                glob_decl.warn_ambiguity.set_unchecked(true);
            }
            glob_decl
        } else if glob_decl.res() != old_glob_decl.res() {
            old_glob_decl.ambiguity.set_unchecked(Some(glob_decl));
            old_glob_decl.warn_ambiguity.set_unchecked(warn_ambiguity);
            if warn_ambiguity {
                old_glob_decl
            } else {
                // Need a fresh decl so other glob imports importing it could re-fetch it
                // and set their own `warn_ambiguity` to true.
                // FIXME: remove this when `warn_ambiguity` is removed (#149195).
                self.arenas.alloc_decl((*old_glob_decl).clone())
            }
        } else if !old_glob_decl.vis().is_at_least(glob_decl.vis(), self.tcx) {
            // We are glob-importing the same item but with greater visibility.
            old_glob_decl.vis.set_unchecked(glob_decl.vis());
            old_glob_decl
        } else if glob_decl.is_ambiguity_recursive() && !old_glob_decl.is_ambiguity_recursive() {
            // Overwriting a non-ambiguous glob import with an ambiguous glob import.
            old_glob_decl.ambiguity.set_unchecked(Some(glob_decl));
            old_glob_decl.warn_ambiguity.set_unchecked(true);
            old_glob_decl
        } else {
            old_glob_decl
        }
    }

    /// Attempt to put the declaration with the given name and namespace into the module,
    /// and return existing declaration if there is a collision.
    pub(crate) fn try_plant_decl_into_local_module(
        &mut self,
        ident: Macros20NormalizedIdent,
        ns: Namespace,
        decl: Decl<'ra>,
        warn_ambiguity: bool,
    ) -> Result<(), Decl<'ra>> {
        let module = decl.parent_module.unwrap();
        let res = decl.res();
        self.check_reserved_macro_name(ident.0, res);
        // Even if underscore names cannot be looked up, we still need to add them to modules,
        // because they can be fetched by glob imports from those modules, and bring traits
        // into scope both directly and through glob imports.
        let key = BindingKey::new_disambiguated(ident, ns, || {
            module.underscore_disambiguator.update_unchecked(|d| d + 1);
            module.underscore_disambiguator.get()
        });
        self.update_local_resolution(module, key, warn_ambiguity, |this, resolution| {
            if let Some(old_decl) = resolution.best_decl() {
                assert_ne!(decl, old_decl);
                assert!(!decl.warn_ambiguity.get());
                if res == Res::Err && old_decl.res() != Res::Err {
                    // Do not override real declarations with `Res::Err`s from error recovery.
                    return Ok(());
                }
                match (old_decl.is_glob_import(), decl.is_glob_import()) {
                    (true, true) => {
                        resolution.glob_decl =
                            Some(this.select_glob_decl(decl, old_decl, warn_ambiguity));
                    }
                    (old_glob @ true, false) | (old_glob @ false, true) => {
                        let (glob_decl, non_glob_decl) =
                            if old_glob { (old_decl, decl) } else { (decl, old_decl) };
                        resolution.non_glob_decl = Some(non_glob_decl);
                        if let Some(old_glob_decl) = resolution.glob_decl
                            && old_glob_decl != glob_decl
                        {
                            resolution.glob_decl =
                                Some(this.select_glob_decl(glob_decl, old_glob_decl, false));
                        } else {
                            resolution.glob_decl = Some(glob_decl);
                        }
                    }
                    (false, false) => {
                        return Err(old_decl);
                    }
                }
            } else {
                if decl.is_glob_import() {
                    resolution.glob_decl = Some(decl);
                } else {
                    resolution.non_glob_decl = Some(decl);
                }
            }

            Ok(())
        })
    }

    // Use `f` to mutate the resolution of the name in the module.
    // If the resolution becomes a success, define it in the module's glob importers.
    fn update_local_resolution<T, F>(
        &mut self,
        module: Module<'ra>,
        key: BindingKey,
        warn_ambiguity: bool,
        f: F,
    ) -> T
    where
        F: FnOnce(&Resolver<'ra, 'tcx>, &mut NameResolution<'ra>) -> T,
    {
        // Ensure that `resolution` isn't borrowed when defining in the module's glob importers,
        // during which the resolution might end up getting re-defined via a glob cycle.
        let (binding, t, warn_ambiguity) = {
            let resolution = &mut *self.resolution_or_default(module, key).borrow_mut_unchecked();
            let old_decl = resolution.binding();

            let t = f(self, resolution);

            if let Some(binding) = resolution.binding()
                && old_decl != Some(binding)
            {
                (binding, t, warn_ambiguity || old_decl.is_some())
            } else {
                return t;
            }
        };

        let Ok(glob_importers) = module.glob_importers.try_borrow_mut_unchecked() else {
            return t;
        };

        // Define or update `binding` in `module`s glob importers.
        for import in glob_importers.iter() {
            let mut ident = key.ident;
            let scope = match ident.0.span.reverse_glob_adjust(module.expansion, import.span) {
                Some(Some(def)) => self.expn_def_scope(def),
                Some(None) => import.parent_scope.module,
                None => continue,
            };
            if self.is_accessible_from(binding.vis(), scope) {
                let import_decl = self.new_import_decl(binding, *import);
                let _ = self.try_plant_decl_into_local_module(
                    ident,
                    key.ns,
                    import_decl,
                    warn_ambiguity,
                );
            }
        }

        t
    }

    // Define a dummy resolution containing a `Res::Err` as a placeholder for a failed
    // or indeterminate resolution, also mark such failed imports as used to avoid duplicate diagnostics.
    fn import_dummy_binding(&mut self, import: Import<'ra>, is_indeterminate: bool) {
        if let ImportKind::Single { target, ref decls, .. } = import.kind {
            if !(is_indeterminate || decls.iter().all(|d| d.get().decl().is_none())) {
                return; // Has resolution, do not create the dummy binding
            }
            let dummy_decl = self.dummy_decl;
            let dummy_decl = self.new_import_decl(dummy_decl, import);
            self.per_ns(|this, ns| {
                let module = import.parent_scope.module;
                let ident = Macros20NormalizedIdent::new(target);
                let _ = this.try_plant_decl_into_local_module(ident, ns, dummy_decl, false);
                // Don't remove underscores from `single_imports`, they were never added.
                if target.name != kw::Underscore {
                    let key = BindingKey::new(ident, ns);
                    this.update_local_resolution(module, key, false, |_, resolution| {
                        resolution.single_imports.swap_remove(&import);
                    })
                }
            });
            self.record_use(target, dummy_decl, Used::Other);
        } else if import.imported_module.get().is_none() {
            self.import_use_map.insert(import, Used::Other);
            if let Some(id) = import.id() {
                self.used_imports.insert(id);
            }
        }
    }

    // Import resolution
    //
    // This is a fixed-point algorithm. We resolve imports until our efforts
    // are stymied by an unresolved import; then we bail out of the current
    // module and continue. We terminate successfully once no more imports
    // remain or unsuccessfully when no forward progress in resolving imports
    // is made.

    /// Resolves all imports for the crate. This method performs the fixed-
    /// point iteration.
    pub(crate) fn resolve_imports(&mut self) {
        let mut prev_indeterminate_count = usize::MAX;
        let mut indeterminate_count = self.indeterminate_imports.len() * 3;
        while indeterminate_count < prev_indeterminate_count {
            prev_indeterminate_count = indeterminate_count;
            indeterminate_count = 0;
            self.assert_speculative = true;
            for import in mem::take(&mut self.indeterminate_imports) {
                let import_indeterminate_count = self.cm().resolve_import(import);
                indeterminate_count += import_indeterminate_count;
                match import_indeterminate_count {
                    0 => self.determined_imports.push(import),
                    _ => self.indeterminate_imports.push(import),
                }
            }
            self.assert_speculative = false;
        }
    }

    pub(crate) fn finalize_imports(&mut self) {
        let mut module_children = Default::default();
        let mut ambig_module_children = Default::default();
        for module in &self.local_modules {
            self.finalize_resolutions_in(*module, &mut module_children, &mut ambig_module_children);
        }
        self.module_children = module_children;
        self.ambig_module_children = ambig_module_children;

        let mut seen_spans = FxHashSet::default();
        let mut errors = vec![];
        let mut prev_root_id: NodeId = NodeId::ZERO;
        let determined_imports = mem::take(&mut self.determined_imports);
        let indeterminate_imports = mem::take(&mut self.indeterminate_imports);

        let mut glob_error = false;
        for (is_indeterminate, import) in determined_imports
            .iter()
            .map(|i| (false, i))
            .chain(indeterminate_imports.iter().map(|i| (true, i)))
        {
            let unresolved_import_error = self.finalize_import(*import);
            // If this import is unresolved then create a dummy import
            // resolution for it so that later resolve stages won't complain.
            self.import_dummy_binding(*import, is_indeterminate);

            let Some(err) = unresolved_import_error else { continue };

            glob_error |= import.is_glob();

            if let ImportKind::Single { source, ref decls, .. } = import.kind
                && source.name == kw::SelfLower
                // Silence `unresolved import` error if E0429 is already emitted
                && let PendingDecl::Ready(None) = decls.value_ns.get()
            {
                continue;
            }

            if prev_root_id != NodeId::ZERO && prev_root_id != import.root_id && !errors.is_empty()
            {
                // In the case of a new import line, throw a diagnostic message
                // for the previous line.
                self.throw_unresolved_import_error(errors, glob_error);
                errors = vec![];
            }
            if seen_spans.insert(err.span) {
                errors.push((*import, err));
                prev_root_id = import.root_id;
            }
        }

        if !errors.is_empty() {
            self.throw_unresolved_import_error(errors, glob_error);
            return;
        }

        for import in &indeterminate_imports {
            let path = import_path_to_string(
                &import.module_path.iter().map(|seg| seg.ident).collect::<Vec<_>>(),
                &import.kind,
                import.span,
            );
            // FIXME: there should be a better way of doing this than
            // formatting this as a string then checking for `::`
            if path.contains("::") {
                let err = UnresolvedImportError {
                    span: import.span,
                    label: None,
                    note: None,
                    suggestion: None,
                    candidates: None,
                    segment: None,
                    module: None,
                };
                errors.push((*import, err))
            }
        }

        if !errors.is_empty() {
            self.throw_unresolved_import_error(errors, glob_error);
        }
    }

    pub(crate) fn lint_reexports(&mut self, exported_ambiguities: FxHashSet<Decl<'ra>>) {
        for module in &self.local_modules {
            for (key, resolution) in self.resolutions(*module).borrow().iter() {
                let resolution = resolution.borrow();
                let Some(binding) = resolution.best_decl() else { continue };

                if let DeclKind::Import { import, .. } = binding.kind
                    && let Some(amb_binding) = binding.ambiguity.get()
                    && binding.res() != Res::Err
                    && exported_ambiguities.contains(&binding)
                {
                    self.lint_buffer.buffer_lint(
                        AMBIGUOUS_GLOB_REEXPORTS,
                        import.root_id,
                        import.root_span,
                        BuiltinLintDiag::AmbiguousGlobReexports {
                            name: key.ident.to_string(),
                            namespace: key.ns.descr().to_string(),
                            first_reexport_span: import.root_span,
                            duplicate_reexport_span: amb_binding.span,
                        },
                    );
                }

                if let Some(glob_decl) = resolution.glob_decl
                    && resolution.non_glob_decl.is_some()
                {
                    if binding.res() != Res::Err
                        && glob_decl.res() != Res::Err
                        && let DeclKind::Import { import: glob_import, .. } = glob_decl.kind
                        && let Some(glob_import_id) = glob_import.id()
                        && let glob_import_def_id = self.local_def_id(glob_import_id)
                        && self.effective_visibilities.is_exported(glob_import_def_id)
                        && glob_decl.vis().is_public()
                        && !binding.vis().is_public()
                    {
                        let binding_id = match binding.kind {
                            DeclKind::Def(res) => {
                                Some(self.def_id_to_node_id(res.def_id().expect_local()))
                            }
                            DeclKind::Import { import, .. } => import.id(),
                        };
                        if let Some(binding_id) = binding_id {
                            self.lint_buffer.buffer_lint(
                                HIDDEN_GLOB_REEXPORTS,
                                binding_id,
                                binding.span,
                                BuiltinLintDiag::HiddenGlobReexports {
                                    name: key.ident.name.to_string(),
                                    namespace: key.ns.descr().to_owned(),
                                    glob_reexport_span: glob_decl.span,
                                    private_item_span: binding.span,
                                },
                            );
                        }
                    }
                }

                if let DeclKind::Import { import, .. } = binding.kind
                    && let Some(binding_id) = import.id()
                    && let import_def_id = self.local_def_id(binding_id)
                    && self.effective_visibilities.is_exported(import_def_id)
                    && let Res::Def(reexported_kind, reexported_def_id) = binding.res()
                    && !matches!(reexported_kind, DefKind::Ctor(..))
                    && !reexported_def_id.is_local()
                    && self.tcx.is_private_dep(reexported_def_id.krate)
                {
                    self.lint_buffer.buffer_lint(
                        EXPORTED_PRIVATE_DEPENDENCIES,
                        binding_id,
                        binding.span,
                        crate::errors::ReexportPrivateDependency {
                            name: key.ident.name,
                            kind: binding.res().descr(),
                            krate: self.tcx.crate_name(reexported_def_id.krate),
                        },
                    );
                }
            }
        }
    }

    fn throw_unresolved_import_error(
        &mut self,
        mut errors: Vec<(Import<'_>, UnresolvedImportError)>,
        glob_error: bool,
    ) {
        errors.retain(|(_import, err)| match err.module {
            // Skip `use` errors for `use foo::Bar;` if `foo.rs` has unrecovered parse errors.
            Some(def_id) if self.mods_with_parse_errors.contains(&def_id) => false,
            // If we've encountered something like `use _;`, we've already emitted an error stating
            // that `_` is not a valid identifier, so we ignore that resolve error.
            _ => err.segment != Some(kw::Underscore),
        });
        if errors.is_empty() {
            self.tcx.dcx().delayed_bug("expected a parse or \"`_` can't be an identifier\" error");
            return;
        }

        let span = MultiSpan::from_spans(errors.iter().map(|(_, err)| err.span).collect());

        let paths = errors
            .iter()
            .map(|(import, err)| {
                let path = import_path_to_string(
                    &import.module_path.iter().map(|seg| seg.ident).collect::<Vec<_>>(),
                    &import.kind,
                    err.span,
                );
                format!("`{path}`")
            })
            .collect::<Vec<_>>();
        let msg = format!("unresolved import{} {}", pluralize!(paths.len()), paths.join(", "),);

        let mut diag = struct_span_code_err!(self.dcx(), span, E0432, "{msg}");

        if let Some((_, UnresolvedImportError { note: Some(note), .. })) = errors.iter().last() {
            diag.note(note.clone());
        }

        /// Upper limit on the number of `span_label` messages.
        const MAX_LABEL_COUNT: usize = 10;

        for (import, err) in errors.into_iter().take(MAX_LABEL_COUNT) {
            if let Some(label) = err.label {
                diag.span_label(err.span, label);
            }

            if let Some((suggestions, msg, applicability)) = err.suggestion {
                if suggestions.is_empty() {
                    diag.help(msg);
                    continue;
                }
                diag.multipart_suggestion(msg, suggestions, applicability);
            }

            if let Some(candidates) = &err.candidates {
                match &import.kind {
                    ImportKind::Single { nested: false, source, target, .. } => import_candidates(
                        self.tcx,
                        &mut diag,
                        Some(err.span),
                        candidates,
                        DiagMode::Import { append: false, unresolved_import: true },
                        (source != target)
                            .then(|| format!(" as {target}"))
                            .as_deref()
                            .unwrap_or(""),
                    ),
                    ImportKind::Single { nested: true, source, target, .. } => {
                        import_candidates(
                            self.tcx,
                            &mut diag,
                            None,
                            candidates,
                            DiagMode::Normal,
                            (source != target)
                                .then(|| format!(" as {target}"))
                                .as_deref()
                                .unwrap_or(""),
                        );
                    }
                    _ => {}
                }
            }

            if matches!(import.kind, ImportKind::Single { .. })
                && let Some(segment) = err.segment
                && let Some(module) = err.module
            {
                self.find_cfg_stripped(&mut diag, &segment, module)
            }
        }

        let guar = diag.emit();
        if glob_error {
            self.glob_error = Some(guar);
        }
    }

    /// Attempts to resolve the given import, returning:
    /// - `0` means its resolution is determined.
    /// - Other values mean that indeterminate exists under certain namespaces.
    ///
    /// Meanwhile, if resolve successful, the resolved bindings are written
    /// into the module.
    fn resolve_import<'r>(mut self: CmResolver<'r, 'ra, 'tcx>, import: Import<'ra>) -> usize {
        debug!(
            "(resolving import for module) resolving import `{}::...` in `{}`",
            Segment::names_to_string(&import.module_path),
            module_to_string(import.parent_scope.module).unwrap_or_else(|| "???".to_string()),
        );
        let module = if let Some(module) = import.imported_module.get() {
            module
        } else {
            let path_res = self.reborrow().maybe_resolve_path(
                &import.module_path,
                None,
                &import.parent_scope,
                Some(import),
            );

            match path_res {
                PathResult::Module(module) => module,
                PathResult::Indeterminate => return 3,
                PathResult::NonModule(..) | PathResult::Failed { .. } => return 0,
            }
        };

        import.imported_module.set_unchecked(Some(module));
        let (source, target, bindings, type_ns_only) = match import.kind {
            ImportKind::Single { source, target, ref decls, type_ns_only, .. } => {
                (source, target, decls, type_ns_only)
            }
            ImportKind::Glob { .. } => {
                self.get_mut_unchecked().resolve_glob_import(import);
                return 0;
            }
            _ => unreachable!(),
        };

        let mut indeterminate_count = 0;
        self.per_ns_cm(|this, ns| {
            if !type_ns_only || ns == TypeNS {
                if bindings[ns].get() != PendingDecl::Pending {
                    return;
                };
                let binding_result = this.reborrow().maybe_resolve_ident_in_module(
                    module,
                    source,
                    ns,
                    &import.parent_scope,
                    Some(import),
                );
                let parent = import.parent_scope.module;
                let binding = match binding_result {
                    Ok(binding) => {
                        if binding.is_assoc_item()
                            && !this.tcx.features().import_trait_associated_functions()
                        {
                            feature_err(
                                this.tcx.sess,
                                sym::import_trait_associated_functions,
                                import.span,
                                "`use` associated items of traits is unstable",
                            )
                            .emit();
                        }
                        // We need the `target`, `source` can be extracted.
                        let import_decl = this.new_import_decl(binding, import);
                        this.get_mut_unchecked().plant_decl_into_local_module(
                            Macros20NormalizedIdent::new(target),
                            ns,
                            import_decl,
                        );
                        PendingDecl::Ready(Some(import_decl))
                    }
                    Err(Determinacy::Determined) => {
                        // Don't remove underscores from `single_imports`, they were never added.
                        if target.name != kw::Underscore {
                            let key = BindingKey::new(Macros20NormalizedIdent::new(target), ns);
                            this.get_mut_unchecked().update_local_resolution(
                                parent,
                                key,
                                false,
                                |_, resolution| {
                                    resolution.single_imports.swap_remove(&import);
                                },
                            );
                        }
                        PendingDecl::Ready(None)
                    }
                    Err(Determinacy::Undetermined) => {
                        indeterminate_count += 1;
                        PendingDecl::Pending
                    }
                };
                bindings[ns].set_unchecked(binding);
            }
        });

        indeterminate_count
    }

    /// Performs final import resolution, consistency checks and error reporting.
    ///
    /// Optionally returns an unresolved import error. This error is buffered and used to
    /// consolidate multiple unresolved import errors into a single diagnostic.
    fn finalize_import(&mut self, import: Import<'ra>) -> Option<UnresolvedImportError> {
        let ignore_decl = match &import.kind {
            ImportKind::Single { decls, .. } => decls[TypeNS].get().decl(),
            _ => None,
        };
        let ambiguity_errors_len = |errors: &Vec<AmbiguityError<'_>>| {
            errors.iter().filter(|error| error.warning.is_none()).count()
        };
        let prev_ambiguity_errors_len = ambiguity_errors_len(&self.ambiguity_errors);
        let finalize = Finalize::with_root_span(import.root_id, import.span, import.root_span);

        // We'll provide more context to the privacy errors later, up to `len`.
        let privacy_errors_len = self.privacy_errors.len();

        let path_res = self.cm().resolve_path(
            &import.module_path,
            None,
            &import.parent_scope,
            Some(finalize),
            ignore_decl,
            Some(import),
        );

        let no_ambiguity =
            ambiguity_errors_len(&self.ambiguity_errors) == prev_ambiguity_errors_len;

        let module = match path_res {
            PathResult::Module(module) => {
                // Consistency checks, analogous to `finalize_macro_resolutions`.
                if let Some(initial_module) = import.imported_module.get() {
                    if module != initial_module && no_ambiguity {
                        span_bug!(import.span, "inconsistent resolution for an import");
                    }
                } else if self.privacy_errors.is_empty() {
                    self.dcx()
                        .create_err(CannotDetermineImportResolution { span: import.span })
                        .emit();
                }

                module
            }
            PathResult::Failed {
                is_error_from_last_segment: false,
                span,
                segment_name,
                label,
                suggestion,
                module,
                error_implied_by_parse_error: _,
            } => {
                if no_ambiguity {
                    assert!(import.imported_module.get().is_none());
                    self.report_error(
                        span,
                        ResolutionError::FailedToResolve {
                            segment: Some(segment_name),
                            label,
                            suggestion,
                            module,
                        },
                    );
                }
                return None;
            }
            PathResult::Failed {
                is_error_from_last_segment: true,
                span,
                label,
                suggestion,
                module,
                segment_name,
                ..
            } => {
                if no_ambiguity {
                    assert!(import.imported_module.get().is_none());
                    let module = if let Some(ModuleOrUniformRoot::Module(m)) = module {
                        m.opt_def_id()
                    } else {
                        None
                    };
                    let err = match self
                        .make_path_suggestion(import.module_path.clone(), &import.parent_scope)
                    {
                        Some((suggestion, note)) => UnresolvedImportError {
                            span,
                            label: None,
                            note,
                            suggestion: Some((
                                vec![(span, Segment::names_to_string(&suggestion))],
                                String::from("a similar path exists"),
                                Applicability::MaybeIncorrect,
                            )),
                            candidates: None,
                            segment: Some(segment_name),
                            module,
                        },
                        None => UnresolvedImportError {
                            span,
                            label: Some(label),
                            note: None,
                            suggestion,
                            candidates: None,
                            segment: Some(segment_name),
                            module,
                        },
                    };
                    return Some(err);
                }
                return None;
            }
            PathResult::NonModule(partial_res) => {
                if no_ambiguity && partial_res.full_res() != Some(Res::Err) {
                    // Check if there are no ambiguities and the result is not dummy.
                    assert!(import.imported_module.get().is_none());
                }
                // The error was already reported earlier.
                return None;
            }
            PathResult::Indeterminate => unreachable!(),
        };

        let (ident, target, bindings, type_ns_only, import_id) = match import.kind {
            ImportKind::Single { source, target, ref decls, type_ns_only, id, .. } => {
                (source, target, decls, type_ns_only, id)
            }
            ImportKind::Glob { ref max_vis, id } => {
                if import.module_path.len() <= 1 {
                    // HACK(eddyb) `lint_if_path_starts_with_module` needs at least
                    // 2 segments, so the `resolve_path` above won't trigger it.
                    let mut full_path = import.module_path.clone();
                    full_path.push(Segment::from_ident(Ident::dummy()));
                    self.lint_if_path_starts_with_module(finalize, &full_path, None);
                }

                if let ModuleOrUniformRoot::Module(module) = module
                    && module == import.parent_scope.module
                {
                    // Importing a module into itself is not allowed.
                    return Some(UnresolvedImportError {
                        span: import.span,
                        label: Some(String::from("cannot glob-import a module into itself")),
                        note: None,
                        suggestion: None,
                        candidates: None,
                        segment: None,
                        module: None,
                    });
                }
                if let Some(max_vis) = max_vis.get()
                    && !max_vis.is_at_least(import.vis, self.tcx)
                {
                    let def_id = self.local_def_id(id);
                    self.lint_buffer.buffer_lint(
                        UNUSED_IMPORTS,
                        id,
                        import.span,
                        crate::errors::RedundantImportVisibility {
                            span: import.span,
                            help: (),
                            max_vis: max_vis.to_string(def_id, self.tcx),
                            import_vis: import.vis.to_string(def_id, self.tcx),
                        },
                    );
                }
                return None;
            }
            _ => unreachable!(),
        };

        if self.privacy_errors.len() != privacy_errors_len {
            // Get the Res for the last element, so that we can point to alternative ways of
            // importing it if available.
            let mut path = import.module_path.clone();
            path.push(Segment::from_ident(ident));
            if let PathResult::Module(ModuleOrUniformRoot::Module(module)) = self.cm().resolve_path(
                &path,
                None,
                &import.parent_scope,
                Some(finalize),
                ignore_decl,
                None,
            ) {
                let res = module.res().map(|r| (r, ident));
                for error in &mut self.privacy_errors[privacy_errors_len..] {
                    error.outermost_res = res;
                }
            }
        }

        let mut all_ns_err = true;
        self.per_ns(|this, ns| {
            if !type_ns_only || ns == TypeNS {
                let binding = this.cm().resolve_ident_in_module(
                    module,
                    ident,
                    ns,
                    &import.parent_scope,
                    Some(Finalize { report_private: false, ..finalize }),
                    bindings[ns].get().decl(),
                    Some(import),
                );

                match binding {
                    Ok(binding) => {
                        // Consistency checks, analogous to `finalize_macro_resolutions`.
                        let initial_res = bindings[ns].get().decl().map(|binding| {
                            let initial_binding = binding.import_source();
                            all_ns_err = false;
                            if target.name == kw::Underscore
                                && initial_binding.is_extern_crate()
                                && !initial_binding.is_import()
                            {
                                let used = if import.module_path.is_empty() {
                                    Used::Scope
                                } else {
                                    Used::Other
                                };
                                this.record_use(ident, binding, used);
                            }
                            initial_binding.res()
                        });
                        let res = binding.res();
                        let has_ambiguity_error =
                            this.ambiguity_errors.iter().any(|error| error.warning.is_none());
                        if res == Res::Err || has_ambiguity_error {
                            this.dcx()
                                .span_delayed_bug(import.span, "some error happened for an import");
                            return;
                        }
                        if let Some(initial_res) = initial_res {
                            if res != initial_res && !this.issue_145575_hack_applied {
                                span_bug!(import.span, "inconsistent resolution for an import");
                            }
                        } else if this.privacy_errors.is_empty() {
                            this.dcx()
                                .create_err(CannotDetermineImportResolution { span: import.span })
                                .emit();
                        }
                    }
                    Err(..) => {
                        // FIXME: This assert may fire if public glob is later shadowed by a private
                        // single import (see test `issue-55884-2.rs`). In theory single imports should
                        // always block globs, even if they are not yet resolved, so that this kind of
                        // self-inconsistent resolution never happens.
                        // Re-enable the assert when the issue is fixed.
                        // assert!(result[ns].get().is_err());
                    }
                }
            }
        });

        if all_ns_err {
            let mut all_ns_failed = true;
            self.per_ns(|this, ns| {
                if !type_ns_only || ns == TypeNS {
                    let binding = this.cm().resolve_ident_in_module(
                        module,
                        ident,
                        ns,
                        &import.parent_scope,
                        Some(finalize),
                        None,
                        None,
                    );
                    if binding.is_ok() {
                        all_ns_failed = false;
                    }
                }
            });

            return if all_ns_failed {
                let names = match module {
                    ModuleOrUniformRoot::Module(module) => {
                        self.resolutions(module)
                            .borrow()
                            .iter()
                            .filter_map(|(BindingKey { ident: i, .. }, resolution)| {
                                if i.name == ident.name {
                                    return None;
                                } // Never suggest the same name

                                let resolution = resolution.borrow();
                                if let Some(name_binding) = resolution.best_decl() {
                                    match name_binding.kind {
                                        DeclKind::Import { source_decl, .. } => {
                                            match source_decl.kind {
                                                // Never suggest names that previously could not
                                                // be resolved.
                                                DeclKind::Def(Res::Err) => None,
                                                _ => Some(i.name),
                                            }
                                        }
                                        _ => Some(i.name),
                                    }
                                } else if resolution.single_imports.is_empty() {
                                    None
                                } else {
                                    Some(i.name)
                                }
                            })
                            .collect()
                    }
                    _ => Vec::new(),
                };

                let lev_suggestion =
                    find_best_match_for_name(&names, ident.name, None).map(|suggestion| {
                        (
                            vec![(ident.span, suggestion.to_string())],
                            String::from("a similar name exists in the module"),
                            Applicability::MaybeIncorrect,
                        )
                    });

                let (suggestion, note) =
                    match self.check_for_module_export_macro(import, module, ident) {
                        Some((suggestion, note)) => (suggestion.or(lev_suggestion), note),
                        _ => (lev_suggestion, None),
                    };

                let label = match module {
                    ModuleOrUniformRoot::Module(module) => {
                        let module_str = module_to_string(module);
                        if let Some(module_str) = module_str {
                            format!("no `{ident}` in `{module_str}`")
                        } else {
                            format!("no `{ident}` in the root")
                        }
                    }
                    _ => {
                        if !ident.is_path_segment_keyword() {
                            format!("no external crate `{ident}`")
                        } else {
                            // HACK(eddyb) this shows up for `self` & `super`, which
                            // should work instead - for now keep the same error message.
                            format!("no `{ident}` in the root")
                        }
                    }
                };

                let parent_suggestion =
                    self.lookup_import_candidates(ident, TypeNS, &import.parent_scope, |_| true);

                Some(UnresolvedImportError {
                    span: import.span,
                    label: Some(label),
                    note,
                    suggestion,
                    candidates: if !parent_suggestion.is_empty() {
                        Some(parent_suggestion)
                    } else {
                        None
                    },
                    module: import.imported_module.get().and_then(|module| {
                        if let ModuleOrUniformRoot::Module(m) = module {
                            m.opt_def_id()
                        } else {
                            None
                        }
                    }),
                    segment: Some(ident.name),
                })
            } else {
                // `resolve_ident_in_module` reported a privacy error.
                None
            };
        }

        let mut reexport_error = None;
        let mut any_successful_reexport = false;
        let mut crate_private_reexport = false;
        self.per_ns(|this, ns| {
            let Some(binding) = bindings[ns].get().decl().map(|b| b.import_source()) else {
                return;
            };

            if !binding.vis().is_at_least(import.vis, this.tcx) {
                reexport_error = Some((ns, binding));
                if let Visibility::Restricted(binding_def_id) = binding.vis()
                    && binding_def_id.is_top_level_module()
                {
                    crate_private_reexport = true;
                }
            } else {
                any_successful_reexport = true;
            }
        });

        // All namespaces must be re-exported with extra visibility for an error to occur.
        if !any_successful_reexport {
            let (ns, binding) = reexport_error.unwrap();
            if let Some(extern_crate_id) = pub_use_of_private_extern_crate_hack(import, binding) {
                let extern_crate_sp = self.tcx.source_span(self.local_def_id(extern_crate_id));
                self.lint_buffer.buffer_lint(
                    PUB_USE_OF_PRIVATE_EXTERN_CRATE,
                    import_id,
                    import.span,
                    crate::errors::PrivateExternCrateReexport {
                        ident,
                        sugg: extern_crate_sp.shrink_to_lo(),
                    },
                );
            } else if ns == TypeNS {
                let err = if crate_private_reexport {
                    self.dcx()
                        .create_err(CannotBeReexportedCratePublicNS { span: import.span, ident })
                } else {
                    self.dcx().create_err(CannotBeReexportedPrivateNS { span: import.span, ident })
                };
                err.emit();
            } else {
                let mut err = if crate_private_reexport {
                    self.dcx()
                        .create_err(CannotBeReexportedCratePublic { span: import.span, ident })
                } else {
                    self.dcx().create_err(CannotBeReexportedPrivate { span: import.span, ident })
                };

                match binding.kind {
                        DeclKind::Def(Res::Def(DefKind::Macro(_), def_id))
                            // exclude decl_macro
                            if self.get_macro_by_def_id(def_id).macro_rules =>
                        {
                            err.subdiagnostic( ConsiderAddingMacroExport {
                                span: binding.span,
                            });
                            err.subdiagnostic( ConsiderMarkingAsPubCrate {
                                vis_span: import.vis_span,
                            });
                        }
                        _ => {
                            err.subdiagnostic( ConsiderMarkingAsPub {
                                span: import.span,
                                ident,
                            });
                        }
                    }
                err.emit();
            }
        }

        if import.module_path.len() <= 1 {
            // HACK(eddyb) `lint_if_path_starts_with_module` needs at least
            // 2 segments, so the `resolve_path` above won't trigger it.
            let mut full_path = import.module_path.clone();
            full_path.push(Segment::from_ident(ident));
            self.per_ns(|this, ns| {
                if let Some(binding) = bindings[ns].get().decl().map(|b| b.import_source()) {
                    this.lint_if_path_starts_with_module(finalize, &full_path, Some(binding));
                }
            });
        }

        // Record what this import resolves to for later uses in documentation,
        // this may resolve to either a value or a type, but for documentation
        // purposes it's good enough to just favor one over the other.
        self.per_ns(|this, ns| {
            if let Some(binding) = bindings[ns].get().decl().map(|b| b.import_source()) {
                this.import_res_map.entry(import_id).or_default()[ns] = Some(binding.res());
            }
        });

        debug!("(resolving single import) successfully resolved import");
        None
    }

    pub(crate) fn check_for_redundant_imports(&mut self, import: Import<'ra>) -> bool {
        // This function is only called for single imports.
        let ImportKind::Single { source, target, ref decls, id, .. } = import.kind else {
            unreachable!()
        };

        // Skip if the import is of the form `use source as target` and source != target.
        if source != target {
            return false;
        }

        // Skip if the import was produced by a macro.
        if import.parent_scope.expansion != LocalExpnId::ROOT {
            return false;
        }

        // Skip if we are inside a named module (in contrast to an anonymous
        // module defined by a block).
        // Skip if the import is public or was used through non scope-based resolution,
        // e.g. through a module-relative path.
        if self.import_use_map.get(&import) == Some(&Used::Other)
            || self.effective_visibilities.is_exported(self.local_def_id(id))
        {
            return false;
        }

        let mut is_redundant = true;
        let mut redundant_span = PerNS { value_ns: None, type_ns: None, macro_ns: None };
        self.per_ns(|this, ns| {
            let binding = decls[ns].get().decl().map(|b| b.import_source());
            if is_redundant && let Some(binding) = binding {
                if binding.res() == Res::Err {
                    return;
                }

                match this.cm().resolve_ident_in_scope_set(
                    target,
                    ScopeSet::All(ns),
                    &import.parent_scope,
                    None,
                    false,
                    decls[ns].get().decl(),
                    None,
                ) {
                    Ok(other_binding) => {
                        is_redundant = binding.res() == other_binding.res()
                            && !other_binding.is_ambiguity_recursive();
                        if is_redundant {
                            redundant_span[ns] =
                                Some((other_binding.span, other_binding.is_import()));
                        }
                    }
                    Err(_) => is_redundant = false,
                }
            }
        });

        if is_redundant && !redundant_span.is_empty() {
            let mut redundant_spans: Vec<_> = redundant_span.present_items().collect();
            redundant_spans.sort();
            redundant_spans.dedup();
            self.lint_buffer.buffer_lint(
                REDUNDANT_IMPORTS,
                id,
                import.span,
                BuiltinLintDiag::RedundantImport(redundant_spans, source),
            );
            return true;
        }

        false
    }

    fn resolve_glob_import(&mut self, import: Import<'ra>) {
        // This function is only called for glob imports.
        let ImportKind::Glob { id, .. } = import.kind else { unreachable!() };

        let ModuleOrUniformRoot::Module(module) = import.imported_module.get().unwrap() else {
            self.dcx().emit_err(CannotGlobImportAllCrates { span: import.span });
            return;
        };

        if module.is_trait() && !self.tcx.features().import_trait_associated_functions() {
            feature_err(
                self.tcx.sess,
                sym::import_trait_associated_functions,
                import.span,
                "`use` associated items of traits is unstable",
            )
            .emit();
        }

        if module == import.parent_scope.module {
            return;
        }

        // Add to module's glob_importers
        module.glob_importers.borrow_mut_unchecked().push(import);

        // Ensure that `resolutions` isn't borrowed during `try_define`,
        // since it might get updated via a glob cycle.
        let bindings = self
            .resolutions(module)
            .borrow()
            .iter()
            .filter_map(|(key, resolution)| {
                resolution.borrow().binding().map(|binding| (*key, binding))
            })
            .collect::<Vec<_>>();
        for (mut key, binding) in bindings {
            let scope = match key.ident.0.span.reverse_glob_adjust(module.expansion, import.span) {
                Some(Some(def)) => self.expn_def_scope(def),
                Some(None) => import.parent_scope.module,
                None => continue,
            };
            if self.is_accessible_from(binding.vis(), scope) {
                let import_decl = self.new_import_decl(binding, import);
                let warn_ambiguity = self
                    .resolution(import.parent_scope.module, key)
                    .and_then(|r| r.binding())
                    .is_some_and(|binding| binding.warn_ambiguity_recursive());
                let _ = self.try_plant_decl_into_local_module(
                    key.ident,
                    key.ns,
                    import_decl,
                    warn_ambiguity,
                );
            }
        }

        // Record the destination of this import
        self.record_partial_res(id, PartialRes::new(module.res().unwrap()));
    }

    // Miscellaneous post-processing, including recording re-exports,
    // reporting conflicts, and reporting unresolved imports.
    fn finalize_resolutions_in(
        &self,
        module: Module<'ra>,
        module_children: &mut LocalDefIdMap<Vec<ModChild>>,
        ambig_module_children: &mut LocalDefIdMap<Vec<AmbigModChild>>,
    ) {
        // Since import resolution is finished, globs will not define any more names.
        *module.globs.borrow_mut(self) = Vec::new();

        let Some(def_id) = module.opt_def_id() else { return };

        let mut children = Vec::new();
        let mut ambig_children = Vec::new();

        module.for_each_child(self, |this, ident, _, binding| {
            let res = binding.res().expect_non_local();
            if res != def::Res::Err {
                let child = |reexport_chain| ModChild {
                    ident: ident.0,
                    res,
                    vis: binding.vis(),
                    reexport_chain,
                };
                if let Some((ambig_binding1, ambig_binding2)) = binding.descent_to_ambiguity() {
                    let main = child(ambig_binding1.reexport_chain(this));
                    let second = ModChild {
                        ident: ident.0,
                        res: ambig_binding2.res().expect_non_local(),
                        vis: ambig_binding2.vis(),
                        reexport_chain: ambig_binding2.reexport_chain(this),
                    };
                    ambig_children.push(AmbigModChild { main, second })
                } else {
                    children.push(child(binding.reexport_chain(this)));
                }
            }
        });

        if !children.is_empty() {
            module_children.insert(def_id.expect_local(), children);
        }
        if !ambig_children.is_empty() {
            ambig_module_children.insert(def_id.expect_local(), ambig_children);
        }
    }
}

fn import_path_to_string(names: &[Ident], import_kind: &ImportKind<'_>, span: Span) -> String {
    let pos = names.iter().position(|p| span == p.span && p.name != kw::PathRoot);
    let global = !names.is_empty() && names[0].name == kw::PathRoot;
    if let Some(pos) = pos {
        let names = if global { &names[1..pos + 1] } else { &names[..pos + 1] };
        names_to_string(names.iter().map(|ident| ident.name))
    } else {
        let names = if global { &names[1..] } else { names };
        if names.is_empty() {
            import_kind_to_string(import_kind)
        } else {
            format!(
                "{}::{}",
                names_to_string(names.iter().map(|ident| ident.name)),
                import_kind_to_string(import_kind),
            )
        }
    }
}

fn import_kind_to_string(import_kind: &ImportKind<'_>) -> String {
    match import_kind {
        ImportKind::Single { source, .. } => source.to_string(),
        ImportKind::Glob { .. } => "*".to_string(),
        ImportKind::ExternCrate { .. } => "<extern crate>".to_string(),
        ImportKind::MacroUse { .. } => "#[macro_use]".to_string(),
        ImportKind::MacroExport => "#[macro_export]".to_string(),
    }
}
