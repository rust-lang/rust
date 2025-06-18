//! A bunch of methods and structures more or less related to resolving imports.

use std::cell::Cell;
use std::mem;

use rustc_ast::NodeId;
use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_data_structures::intern::Interned;
use rustc_errors::codes::*;
use rustc_errors::{Applicability, MultiSpan, pluralize, struct_span_code_err};
use rustc_hir::def::{self, DefKind, PartialRes};
use rustc_hir::def_id::DefId;
use rustc_middle::metadata::{ModChild, Reexport};
use rustc_middle::{span_bug, ty};
use rustc_session::lint::BuiltinLintDiag;
use rustc_session::lint::builtin::{
    AMBIGUOUS_GLOB_REEXPORTS, HIDDEN_GLOB_REEXPORTS, PUB_USE_OF_PRIVATE_EXTERN_CRATE,
    REDUNDANT_IMPORTS, UNUSED_IMPORTS,
};
use rustc_session::parse::feature_err;
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::hygiene::LocalExpnId;
use rustc_span::{Ident, Span, Symbol, kw, sym};
use smallvec::SmallVec;
use tracing::debug;

use crate::Determinacy::{self, *};
use crate::Namespace::*;
use crate::diagnostics::{DiagMode, Suggestion, import_candidates};
use crate::errors::{
    CannotBeReexportedCratePublic, CannotBeReexportedCratePublicNS, CannotBeReexportedPrivate,
    CannotBeReexportedPrivateNS, CannotDetermineImportResolution, CannotGlobImportAllCrates,
    ConsiderAddingMacroExport, ConsiderMarkingAsPub,
};
use crate::{
    AmbiguityError, AmbiguityKind, BindingKey, Finalize, ImportSuggestion, Module,
    ModuleOrUniformRoot, NameBinding, NameBindingData, NameBindingKind, ParentScope, PathResult,
    PerNS, ResolutionError, Resolver, ScopeSet, Segment, Used, module_to_string, names_to_string,
};

type Res = def::Res<NodeId>;

/// Contains data for specific kinds of imports.
#[derive(Clone)]
pub(crate) enum ImportKind<'ra> {
    Single {
        /// `source` in `use prefix::source as target`.
        source: Ident,
        /// `target` in `use prefix::source as target`.
        /// It will directly use `source` when the format is `use prefix::source`.
        target: Ident,
        /// Bindings to which `source` refers to.
        source_bindings: PerNS<Cell<Result<NameBinding<'ra>, Determinacy>>>,
        /// Bindings introduced by `target`.
        target_bindings: PerNS<Cell<Option<NameBinding<'ra>>>>,
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
        is_prelude: bool,
        // The visibility of the greatest re-export.
        // n.b. `max_vis` is only used in `finalize_import` to check for re-export errors.
        max_vis: Cell<Option<ty::Visibility>>,
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
            Single {
                source,
                target,
                source_bindings,
                target_bindings,
                type_ns_only,
                nested,
                id,
            } => f
                .debug_struct("Single")
                .field("source", source)
                .field("target", target)
                // Ignore the nested bindings to avoid an infinite loop while printing.
                .field(
                    "source_bindings",
                    &source_bindings.clone().map(|b| b.into_inner().map(|_| format_args!(".."))),
                )
                .field(
                    "target_bindings",
                    &target_bindings.clone().map(|b| b.into_inner().map(|_| format_args!(".."))),
                )
                .field("type_ns_only", type_ns_only)
                .field("nested", nested)
                .field("id", id)
                .finish(),
            Glob { is_prelude, max_vis, id } => f
                .debug_struct("Glob")
                .field("is_prelude", is_prelude)
                .field("max_vis", max_vis)
                .field("id", id)
                .finish(),
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
    /// The resolution of `module_path`.
    pub imported_module: Cell<Option<ModuleOrUniformRoot<'ra>>>,
    pub vis: ty::Visibility,
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

    fn simplify(&self, r: &Resolver<'_, '_>) -> Reexport {
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
    /// The least shadowable known binding for this name, or None if there are no known bindings.
    pub binding: Option<NameBinding<'ra>>,
    pub shadowed_glob: Option<NameBinding<'ra>>,
}

impl<'ra> NameResolution<'ra> {
    /// Returns the binding for the name if it is known or None if it not known.
    pub(crate) fn binding(&self) -> Option<NameBinding<'ra>> {
        self.binding.and_then(|binding| {
            if !binding.is_glob_import() || self.single_imports.is_empty() {
                Some(binding)
            } else {
                None
            }
        })
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
fn pub_use_of_private_extern_crate_hack(
    import: Import<'_>,
    binding: NameBinding<'_>,
) -> Option<NodeId> {
    match (&import.kind, &binding.kind) {
        (ImportKind::Single { .. }, NameBindingKind::Import { import: binding_import, .. })
            if let ImportKind::ExternCrate { id, .. } = binding_import.kind
                && import.vis.is_public() =>
        {
            Some(id)
        }
        _ => None,
    }
}

impl<'ra, 'tcx> Resolver<'ra, 'tcx> {
    /// Given a binding and an import that resolves to it,
    /// return the corresponding binding defined by the import.
    pub(crate) fn import(
        &self,
        binding: NameBinding<'ra>,
        import: Import<'ra>,
    ) -> NameBinding<'ra> {
        let import_vis = import.vis.to_def_id();
        let vis = if binding.vis.is_at_least(import_vis, self.tcx)
            || pub_use_of_private_extern_crate_hack(import, binding).is_some()
        {
            import_vis
        } else {
            binding.vis
        };

        if let ImportKind::Glob { ref max_vis, .. } = import.kind
            && (vis == import_vis
                || max_vis.get().is_none_or(|max_vis| vis.is_at_least(max_vis, self.tcx)))
        {
            max_vis.set(Some(vis.expect_local()))
        }

        self.arenas.alloc_name_binding(NameBindingData {
            kind: NameBindingKind::Import { binding, import },
            ambiguity: None,
            warn_ambiguity: false,
            span: import.span,
            vis,
            expansion: import.parent_scope.expansion,
        })
    }

    /// Define the name or return the existing binding if there is a collision.
    /// `update` indicates if the definition is a redefinition of an existing binding.
    pub(crate) fn try_define(
        &mut self,
        module: Module<'ra>,
        key: BindingKey,
        binding: NameBinding<'ra>,
        warn_ambiguity: bool,
    ) -> Result<(), NameBinding<'ra>> {
        let res = binding.res();
        self.check_reserved_macro_name(key.ident, res);
        self.set_binding_parent_module(binding, module);
        self.update_resolution(module, key, warn_ambiguity, |this, resolution| {
            if let Some(old_binding) = resolution.binding {
                if res == Res::Err && old_binding.res() != Res::Err {
                    // Do not override real bindings with `Res::Err`s from error recovery.
                    return Ok(());
                }
                match (old_binding.is_glob_import(), binding.is_glob_import()) {
                    (true, true) => {
                        // FIXME: remove `!binding.is_ambiguity_recursive()` after delete the warning ambiguity.
                        if !binding.is_ambiguity_recursive()
                            && let NameBindingKind::Import { import: old_import, .. } =
                                old_binding.kind
                            && let NameBindingKind::Import { import, .. } = binding.kind
                            && old_import == import
                        {
                            // We should replace the `old_binding` with `binding` regardless
                            // of whether they has same resolution or not when they are
                            // imported from the same glob-import statement.
                            resolution.binding = Some(binding);
                        } else if res != old_binding.res() {
                            resolution.binding = Some(this.new_ambiguity_binding(
                                AmbiguityKind::GlobVsGlob,
                                old_binding,
                                binding,
                                warn_ambiguity,
                            ));
                        } else if !old_binding.vis.is_at_least(binding.vis, this.tcx) {
                            // We are glob-importing the same item but with greater visibility.
                            resolution.binding = Some(binding);
                        } else if binding.is_ambiguity_recursive() {
                            resolution.binding = Some(this.new_warn_ambiguity_binding(binding));
                        }
                    }
                    (old_glob @ true, false) | (old_glob @ false, true) => {
                        let (glob_binding, nonglob_binding) =
                            if old_glob { (old_binding, binding) } else { (binding, old_binding) };
                        if key.ns == MacroNS
                            && nonglob_binding.expansion != LocalExpnId::ROOT
                            && glob_binding.res() != nonglob_binding.res()
                        {
                            resolution.binding = Some(this.new_ambiguity_binding(
                                AmbiguityKind::GlobVsExpanded,
                                nonglob_binding,
                                glob_binding,
                                false,
                            ));
                        } else {
                            resolution.binding = Some(nonglob_binding);
                        }

                        if let Some(old_shadowed_glob) = resolution.shadowed_glob {
                            assert!(old_shadowed_glob.is_glob_import());
                            if glob_binding.res() != old_shadowed_glob.res() {
                                resolution.shadowed_glob = Some(this.new_ambiguity_binding(
                                    AmbiguityKind::GlobVsGlob,
                                    old_shadowed_glob,
                                    glob_binding,
                                    false,
                                ));
                            } else if !old_shadowed_glob.vis.is_at_least(binding.vis, this.tcx) {
                                resolution.shadowed_glob = Some(glob_binding);
                            }
                        } else {
                            resolution.shadowed_glob = Some(glob_binding);
                        }
                    }
                    (false, false) => {
                        return Err(old_binding);
                    }
                }
            } else {
                resolution.binding = Some(binding);
            }

            Ok(())
        })
    }

    fn new_ambiguity_binding(
        &self,
        ambiguity_kind: AmbiguityKind,
        primary_binding: NameBinding<'ra>,
        secondary_binding: NameBinding<'ra>,
        warn_ambiguity: bool,
    ) -> NameBinding<'ra> {
        let ambiguity = Some((secondary_binding, ambiguity_kind));
        let data = NameBindingData { ambiguity, warn_ambiguity, ..*primary_binding };
        self.arenas.alloc_name_binding(data)
    }

    fn new_warn_ambiguity_binding(&self, binding: NameBinding<'ra>) -> NameBinding<'ra> {
        assert!(binding.is_ambiguity_recursive());
        self.arenas.alloc_name_binding(NameBindingData { warn_ambiguity: true, ..*binding })
    }

    // Use `f` to mutate the resolution of the name in the module.
    // If the resolution becomes a success, define it in the module's glob importers.
    fn update_resolution<T, F>(
        &mut self,
        module: Module<'ra>,
        key: BindingKey,
        warn_ambiguity: bool,
        f: F,
    ) -> T
    where
        F: FnOnce(&mut Resolver<'ra, 'tcx>, &mut NameResolution<'ra>) -> T,
    {
        // Ensure that `resolution` isn't borrowed when defining in the module's glob importers,
        // during which the resolution might end up getting re-defined via a glob cycle.
        let (binding, t, warn_ambiguity) = {
            let resolution = &mut *self.resolution(module, key).borrow_mut();
            let old_binding = resolution.binding();

            let t = f(self, resolution);

            if let Some(binding) = resolution.binding()
                && old_binding != Some(binding)
            {
                (binding, t, warn_ambiguity || old_binding.is_some())
            } else {
                return t;
            }
        };

        let Ok(glob_importers) = module.glob_importers.try_borrow_mut() else {
            return t;
        };

        // Define or update `binding` in `module`s glob importers.
        for import in glob_importers.iter() {
            let mut ident = key.ident;
            let scope = match ident.span.reverse_glob_adjust(module.expansion, import.span) {
                Some(Some(def)) => self.expn_def_scope(def),
                Some(None) => import.parent_scope.module,
                None => continue,
            };
            if self.is_accessible_from(binding.vis, scope) {
                let imported_binding = self.import(binding, *import);
                let key = BindingKey { ident, ..key };
                let _ = self.try_define(
                    import.parent_scope.module,
                    key,
                    imported_binding,
                    warn_ambiguity,
                );
            }
        }

        t
    }

    // Define a dummy resolution containing a `Res::Err` as a placeholder for a failed
    // or indeterminate resolution, also mark such failed imports as used to avoid duplicate diagnostics.
    fn import_dummy_binding(&mut self, import: Import<'ra>, is_indeterminate: bool) {
        if let ImportKind::Single { target, ref target_bindings, .. } = import.kind {
            if !(is_indeterminate || target_bindings.iter().all(|binding| binding.get().is_none()))
            {
                return; // Has resolution, do not create the dummy binding
            }
            let dummy_binding = self.dummy_binding;
            let dummy_binding = self.import(dummy_binding, import);
            self.per_ns(|this, ns| {
                let key = BindingKey::new(target, ns);
                let _ = this.try_define(import.parent_scope.module, key, dummy_binding, false);
                this.update_resolution(import.parent_scope.module, key, false, |_, resolution| {
                    resolution.single_imports.swap_remove(&import);
                })
            });
            self.record_use(target, dummy_binding, Used::Other);
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
            for import in mem::take(&mut self.indeterminate_imports) {
                let import_indeterminate_count = self.resolve_import(import);
                indeterminate_count += import_indeterminate_count;
                match import_indeterminate_count {
                    0 => self.determined_imports.push(import),
                    _ => self.indeterminate_imports.push(import),
                }
            }
        }
    }

    pub(crate) fn finalize_imports(&mut self) {
        for module in self.arenas.local_modules().iter() {
            self.finalize_resolutions_in(*module);
        }

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

            if let ImportKind::Single { source, ref source_bindings, .. } = import.kind
                && source.name == kw::SelfLower
                // Silence `unresolved import` error if E0429 is already emitted
                && let Err(Determined) = source_bindings.value_ns.get()
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

        self.throw_unresolved_import_error(errors, glob_error);
    }

    pub(crate) fn check_hidden_glob_reexports(
        &mut self,
        exported_ambiguities: FxHashSet<NameBinding<'ra>>,
    ) {
        for module in self.arenas.local_modules().iter() {
            for (key, resolution) in self.resolutions(*module).borrow().iter() {
                let resolution = resolution.borrow();

                let Some(binding) = resolution.binding else { continue };

                if let NameBindingKind::Import { import, .. } = binding.kind
                    && let Some((amb_binding, _)) = binding.ambiguity
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

                if let Some(glob_binding) = resolution.shadowed_glob {
                    if binding.res() != Res::Err
                        && glob_binding.res() != Res::Err
                        && let NameBindingKind::Import { import: glob_import, .. } =
                            glob_binding.kind
                        && let Some(glob_import_id) = glob_import.id()
                        && let glob_import_def_id = self.local_def_id(glob_import_id)
                        && self.effective_visibilities.is_exported(glob_import_def_id)
                        && glob_binding.vis.is_public()
                        && !binding.vis.is_public()
                    {
                        let binding_id = match binding.kind {
                            NameBindingKind::Res(res) => {
                                Some(self.def_id_to_node_id(res.def_id().expect_local()))
                            }
                            NameBindingKind::Module(module) => {
                                Some(self.def_id_to_node_id(module.def_id().expect_local()))
                            }
                            NameBindingKind::Import { import, .. } => import.id(),
                        };
                        if let Some(binding_id) = binding_id {
                            self.lint_buffer.buffer_lint(
                                HIDDEN_GLOB_REEXPORTS,
                                binding_id,
                                binding.span,
                                BuiltinLintDiag::HiddenGlobReexports {
                                    name: key.ident.name.to_string(),
                                    namespace: key.ns.descr().to_owned(),
                                    glob_reexport_span: glob_binding.span,
                                    private_item_span: binding.span,
                                },
                            );
                        }
                    }
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
            _ => true,
        });
        if errors.is_empty() {
            return;
        }

        /// Upper limit on the number of `span_label` messages.
        const MAX_LABEL_COUNT: usize = 10;

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
    fn resolve_import(&mut self, import: Import<'ra>) -> usize {
        debug!(
            "(resolving import for module) resolving import `{}::...` in `{}`",
            Segment::names_to_string(&import.module_path),
            module_to_string(import.parent_scope.module).unwrap_or_else(|| "???".to_string()),
        );
        let module = if let Some(module) = import.imported_module.get() {
            module
        } else {
            let path_res = self.maybe_resolve_path(
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

        import.imported_module.set(Some(module));
        let (source, target, source_bindings, target_bindings, type_ns_only) = match import.kind {
            ImportKind::Single {
                source,
                target,
                ref source_bindings,
                ref target_bindings,
                type_ns_only,
                ..
            } => (source, target, source_bindings, target_bindings, type_ns_only),
            ImportKind::Glob { .. } => {
                self.resolve_glob_import(import);
                return 0;
            }
            _ => unreachable!(),
        };

        let mut indeterminate_count = 0;
        self.per_ns(|this, ns| {
            if !type_ns_only || ns == TypeNS {
                if let Err(Undetermined) = source_bindings[ns].get() {
                    let binding = this.maybe_resolve_ident_in_module(
                        module,
                        source,
                        ns,
                        &import.parent_scope,
                        Some(import),
                    );
                    source_bindings[ns].set(binding);
                } else {
                    return;
                };

                let parent = import.parent_scope.module;
                match source_bindings[ns].get() {
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

                        let imported_binding = this.import(binding, import);
                        target_bindings[ns].set(Some(imported_binding));
                        this.define(parent, target, ns, imported_binding);
                    }
                    Err(Determined) => {
                        // Don't update the resolution for underscores, because it was never added.
                        if target.name != kw::Underscore {
                            let key = BindingKey::new(target, ns);
                            this.update_resolution(parent, key, false, |_, resolution| {
                                resolution.single_imports.swap_remove(&import);
                            });
                        }
                    }
                    Err(Undetermined) => indeterminate_count += 1,
                }
            }
        });

        indeterminate_count
    }

    /// Performs final import resolution, consistency checks and error reporting.
    ///
    /// Optionally returns an unresolved import error. This error is buffered and used to
    /// consolidate multiple unresolved import errors into a single diagnostic.
    fn finalize_import(&mut self, import: Import<'ra>) -> Option<UnresolvedImportError> {
        let ignore_binding = match &import.kind {
            ImportKind::Single { target_bindings, .. } => target_bindings[TypeNS].get(),
            _ => None,
        };
        let ambiguity_errors_len =
            |errors: &Vec<AmbiguityError<'_>>| errors.iter().filter(|error| !error.warning).count();
        let prev_ambiguity_errors_len = ambiguity_errors_len(&self.ambiguity_errors);
        let finalize = Finalize::with_root_span(import.root_id, import.span, import.root_span);

        // We'll provide more context to the privacy errors later, up to `len`.
        let privacy_errors_len = self.privacy_errors.len();

        let path_res = self.resolve_path(
            &import.module_path,
            None,
            &import.parent_scope,
            Some(finalize),
            ignore_binding,
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

        let (ident, target, source_bindings, target_bindings, type_ns_only, import_id) =
            match import.kind {
                ImportKind::Single {
                    source,
                    target,
                    ref source_bindings,
                    ref target_bindings,
                    type_ns_only,
                    id,
                    ..
                } => (source, target, source_bindings, target_bindings, type_ns_only, id),
                ImportKind::Glob { is_prelude, ref max_vis, id } => {
                    if import.module_path.len() <= 1 {
                        // HACK(eddyb) `lint_if_path_starts_with_module` needs at least
                        // 2 segments, so the `resolve_path` above won't trigger it.
                        let mut full_path = import.module_path.clone();
                        full_path.push(Segment::from_ident(Ident::dummy()));
                        self.lint_if_path_starts_with_module(Some(finalize), &full_path, None);
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
                    if !is_prelude
                        && let Some(max_vis) = max_vis.get()
                        && !max_vis.is_at_least(import.vis, self.tcx)
                    {
                        let def_id = self.local_def_id(id);
                        self.lint_buffer.buffer_lint(
                            UNUSED_IMPORTS,
                            id,
                            import.span,
                            BuiltinLintDiag::RedundantImportVisibility {
                                max_vis: max_vis.to_string(def_id, self.tcx),
                                import_vis: import.vis.to_string(def_id, self.tcx),
                                span: import.span,
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
            if let PathResult::Module(ModuleOrUniformRoot::Module(module)) = self.resolve_path(
                &path,
                None,
                &import.parent_scope,
                Some(finalize),
                ignore_binding,
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
                let binding = this.resolve_ident_in_module(
                    module,
                    ident,
                    ns,
                    &import.parent_scope,
                    Some(Finalize { report_private: false, ..finalize }),
                    target_bindings[ns].get(),
                    Some(import),
                );

                match binding {
                    Ok(binding) => {
                        // Consistency checks, analogous to `finalize_macro_resolutions`.
                        let initial_res = source_bindings[ns].get().map(|initial_binding| {
                            all_ns_err = false;
                            if let Some(target_binding) = target_bindings[ns].get()
                                && target.name == kw::Underscore
                                && initial_binding.is_extern_crate()
                                && !initial_binding.is_import()
                            {
                                let used = if import.module_path.is_empty() {
                                    Used::Scope
                                } else {
                                    Used::Other
                                };
                                this.record_use(ident, target_binding, used);
                            }
                            initial_binding.res()
                        });
                        let res = binding.res();
                        let has_ambiguity_error =
                            this.ambiguity_errors.iter().any(|error| !error.warning);
                        if res == Res::Err || has_ambiguity_error {
                            this.dcx()
                                .span_delayed_bug(import.span, "some error happened for an import");
                            return;
                        }
                        if let Ok(initial_res) = initial_res {
                            if res != initial_res {
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
                    let binding = this.resolve_ident_in_module(
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
                let resolutions = match module {
                    ModuleOrUniformRoot::Module(module) => Some(self.resolutions(module).borrow()),
                    _ => None,
                };
                let resolutions = resolutions.as_ref().into_iter().flat_map(|r| r.iter());
                let names = resolutions
                    .filter_map(|(BindingKey { ident: i, .. }, resolution)| {
                        if i.name == ident.name {
                            return None;
                        } // Never suggest the same name
                        match *resolution.borrow() {
                            NameResolution { binding: Some(name_binding), .. } => {
                                match name_binding.kind {
                                    NameBindingKind::Import { binding, .. } => {
                                        match binding.kind {
                                            // Never suggest the name that has binding error
                                            // i.e., the name that cannot be previously resolved
                                            NameBindingKind::Res(Res::Err) => None,
                                            _ => Some(i.name),
                                        }
                                    }
                                    _ => Some(i.name),
                                }
                            }
                            NameResolution { ref single_imports, .. }
                                if single_imports.is_empty() =>
                            {
                                None
                            }
                            _ => Some(i.name),
                        }
                    })
                    .collect::<Vec<Symbol>>();

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
            let Ok(binding) = source_bindings[ns].get() else {
                return;
            };

            if !binding.vis.is_at_least(import.vis, this.tcx) {
                reexport_error = Some((ns, binding));
                if let ty::Visibility::Restricted(binding_def_id) = binding.vis
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
                self.lint_buffer.buffer_lint(
                    PUB_USE_OF_PRIVATE_EXTERN_CRATE,
                    import_id,
                    import.span,
                    BuiltinLintDiag::PrivateExternCrateReexport {
                        source: ident,
                        extern_crate_span: self.tcx.source_span(self.local_def_id(extern_crate_id)),
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
                        NameBindingKind::Res(Res::Def(DefKind::Macro(_), def_id))
                            // exclude decl_macro
                            if self.get_macro_by_def_id(def_id).macro_rules =>
                        {
                            err.subdiagnostic( ConsiderAddingMacroExport {
                                span: binding.span,
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
                if let Ok(binding) = source_bindings[ns].get() {
                    this.lint_if_path_starts_with_module(Some(finalize), &full_path, Some(binding));
                }
            });
        }

        // Record what this import resolves to for later uses in documentation,
        // this may resolve to either a value or a type, but for documentation
        // purposes it's good enough to just favor one over the other.
        self.per_ns(|this, ns| {
            if let Ok(binding) = source_bindings[ns].get() {
                this.import_res_map.entry(import_id).or_default()[ns] = Some(binding.res());
            }
        });

        debug!("(resolving single import) successfully resolved import");
        None
    }

    pub(crate) fn check_for_redundant_imports(&mut self, import: Import<'ra>) -> bool {
        // This function is only called for single imports.
        let ImportKind::Single {
            source, target, ref source_bindings, ref target_bindings, id, ..
        } = import.kind
        else {
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
            if is_redundant && let Ok(binding) = source_bindings[ns].get() {
                if binding.res() == Res::Err {
                    return;
                }

                match this.early_resolve_ident_in_lexical_scope(
                    target,
                    ScopeSet::All(ns),
                    &import.parent_scope,
                    None,
                    false,
                    target_bindings[ns].get(),
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
        let ImportKind::Glob { id, is_prelude, .. } = import.kind else { unreachable!() };

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
        } else if is_prelude {
            self.prelude = Some(module);
            return;
        }

        // Add to module's glob_importers
        module.glob_importers.borrow_mut().push(import);

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
            let scope = match key.ident.span.reverse_glob_adjust(module.expansion, import.span) {
                Some(Some(def)) => self.expn_def_scope(def),
                Some(None) => import.parent_scope.module,
                None => continue,
            };
            if self.is_accessible_from(binding.vis, scope) {
                let imported_binding = self.import(binding, import);
                let warn_ambiguity = self
                    .resolution(import.parent_scope.module, key)
                    .borrow()
                    .binding()
                    .is_some_and(|binding| binding.warn_ambiguity_recursive());
                let _ = self.try_define(
                    import.parent_scope.module,
                    key,
                    imported_binding,
                    warn_ambiguity,
                );
            }
        }

        // Record the destination of this import
        self.record_partial_res(id, PartialRes::new(module.res().unwrap()));
    }

    // Miscellaneous post-processing, including recording re-exports,
    // reporting conflicts, and reporting unresolved imports.
    fn finalize_resolutions_in(&mut self, module: Module<'ra>) {
        // Since import resolution is finished, globs will not define any more names.
        *module.globs.borrow_mut() = Vec::new();

        let Some(def_id) = module.opt_def_id() else { return };

        let mut children = Vec::new();

        module.for_each_child(self, |this, ident, _, binding| {
            let res = binding.res().expect_non_local();
            let error_ambiguity = binding.is_ambiguity_recursive() && !binding.warn_ambiguity;
            if res != def::Res::Err && !error_ambiguity {
                let mut reexport_chain = SmallVec::new();
                let mut next_binding = binding;
                while let NameBindingKind::Import { binding, import, .. } = next_binding.kind {
                    reexport_chain.push(import.simplify(this));
                    next_binding = binding;
                }

                children.push(ModChild { ident, res, vis: binding.vis, reexport_chain });
            }
        });

        if !children.is_empty() {
            // Should be fine because this code is only called for local modules.
            self.module_children.insert(def_id.expect_local(), children);
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
