//! A bunch of methods and structures more or less related to resolving imports.

use crate::diagnostics::Suggestion;
use crate::Determinacy::{self, *};
use crate::Namespace::{MacroNS, TypeNS};
use crate::{module_to_string, names_to_string};
use crate::{AmbiguityKind, BindingKey, ModuleKind, ResolutionError, Resolver, Segment};
use crate::{Finalize, Module, ModuleOrUniformRoot, ParentScope, PerNS, ScopeSet};
use crate::{NameBinding, NameBindingKind, PathResult};

use rustc_ast::NodeId;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::intern::Interned;
use rustc_errors::{pluralize, struct_span_err, Applicability, MultiSpan};
use rustc_hir::def::{self, DefKind, PartialRes};
use rustc_middle::metadata::ModChild;
use rustc_middle::span_bug;
use rustc_middle::ty;
use rustc_session::lint::builtin::{PUB_USE_OF_PRIVATE_EXTERN_CRATE, UNUSED_IMPORTS};
use rustc_session::lint::BuiltinLintDiagnostics;
use rustc_span::hygiene::LocalExpnId;
use rustc_span::lev_distance::find_best_match_for_name;
use rustc_span::symbol::{kw, Ident, Symbol};
use rustc_span::Span;

use std::cell::Cell;
use std::{mem, ptr};

type Res = def::Res<NodeId>;

/// Contains data for specific kinds of imports.
#[derive(Clone)]
pub enum ImportKind<'a> {
    Single {
        /// `source` in `use prefix::source as target`.
        source: Ident,
        /// `target` in `use prefix::source as target`.
        target: Ident,
        /// Bindings to which `source` refers to.
        source_bindings: PerNS<Cell<Result<&'a NameBinding<'a>, Determinacy>>>,
        /// Bindings introduced by `target`.
        target_bindings: PerNS<Cell<Option<&'a NameBinding<'a>>>>,
        /// `true` for `...::{self [as target]}` imports, `false` otherwise.
        type_ns_only: bool,
        /// Did this import result from a nested import? ie. `use foo::{bar, baz};`
        nested: bool,
        /// Additional `NodeId`s allocated to a `ast::UseTree` for automatically generated `use` statement
        /// (eg. implicit struct constructors)
        additional_ids: (NodeId, NodeId),
    },
    Glob {
        is_prelude: bool,
        max_vis: Cell<Option<ty::Visibility>>, // The visibility of the greatest re-export.
                                               // n.b. `max_vis` is only used in `finalize_import` to check for re-export errors.
    },
    ExternCrate {
        source: Option<Symbol>,
        target: Ident,
    },
    MacroUse,
}

/// Manually implement `Debug` for `ImportKind` because the `source/target_bindings`
/// contain `Cell`s which can introduce infinite loops while printing.
impl<'a> std::fmt::Debug for ImportKind<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ImportKind::*;
        match self {
            Single {
                ref source,
                ref target,
                ref type_ns_only,
                ref nested,
                ref additional_ids,
                // Ignore the following to avoid an infinite loop while printing.
                source_bindings: _,
                target_bindings: _,
            } => f
                .debug_struct("Single")
                .field("source", source)
                .field("target", target)
                .field("type_ns_only", type_ns_only)
                .field("nested", nested)
                .field("additional_ids", additional_ids)
                .finish_non_exhaustive(),
            Glob { ref is_prelude, ref max_vis } => f
                .debug_struct("Glob")
                .field("is_prelude", is_prelude)
                .field("max_vis", max_vis)
                .finish(),
            ExternCrate { ref source, ref target } => f
                .debug_struct("ExternCrate")
                .field("source", source)
                .field("target", target)
                .finish(),
            MacroUse => f.debug_struct("MacroUse").finish(),
        }
    }
}

/// One import.
#[derive(Debug, Clone)]
pub(crate) struct Import<'a> {
    pub kind: ImportKind<'a>,

    /// The ID of the `extern crate`, `UseTree` etc that imported this `Import`.
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
    pub id: NodeId,

    /// The `id` of the "root" use-kind -- this is always the same as
    /// `id` except in the case of "nested" use trees, in which case
    /// it will be the `id` of the root use tree. e.g., in the example
    /// from `id`, this would be the ID of the `use foo::bar`
    /// `UseTree` node.
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

    pub parent_scope: ParentScope<'a>,
    pub module_path: Vec<Segment>,
    /// The resolution of `module_path`.
    pub imported_module: Cell<Option<ModuleOrUniformRoot<'a>>>,
    pub vis: Cell<Option<ty::Visibility>>,
    pub used: Cell<bool>,
}

impl<'a> Import<'a> {
    pub fn is_glob(&self) -> bool {
        matches!(self.kind, ImportKind::Glob { .. })
    }

    pub fn is_nested(&self) -> bool {
        match self.kind {
            ImportKind::Single { nested, .. } => nested,
            _ => false,
        }
    }

    pub(crate) fn expect_vis(&self) -> ty::Visibility {
        self.vis.get().expect("encountered cleared import visibility")
    }
}

/// Records information about the resolution of a name in a namespace of a module.
#[derive(Clone, Default, Debug)]
pub(crate) struct NameResolution<'a> {
    /// Single imports that may define the name in the namespace.
    /// Imports are arena-allocated, so it's ok to use pointers as keys.
    pub single_imports: FxHashSet<Interned<'a, Import<'a>>>,
    /// The least shadowable known binding for this name, or None if there are no known bindings.
    pub binding: Option<&'a NameBinding<'a>>,
    pub shadowed_glob: Option<&'a NameBinding<'a>>,
}

impl<'a> NameResolution<'a> {
    // Returns the binding for the name if it is known or None if it not known.
    pub(crate) fn binding(&self) -> Option<&'a NameBinding<'a>> {
        self.binding.and_then(|binding| {
            if !binding.is_glob_import() || self.single_imports.is_empty() {
                Some(binding)
            } else {
                None
            }
        })
    }

    pub(crate) fn add_single_import(&mut self, import: &'a Import<'a>) {
        self.single_imports.insert(Interned::new_unchecked(import));
    }
}

// Reexports of the form `pub use foo as bar;` where `foo` is `extern crate foo;`
// are permitted for backward-compatibility under a deprecation lint.
fn pub_use_of_private_extern_crate_hack(import: &Import<'_>, binding: &NameBinding<'_>) -> bool {
    match (&import.kind, &binding.kind) {
        (
            ImportKind::Single { .. },
            NameBindingKind::Import {
                import: Import { kind: ImportKind::ExternCrate { .. }, .. },
                ..
            },
        ) => import.expect_vis().is_public(),
        _ => false,
    }
}

impl<'a> Resolver<'a> {
    // Given a binding and an import that resolves to it,
    // return the corresponding binding defined by the import.
    pub(crate) fn import(
        &self,
        binding: &'a NameBinding<'a>,
        import: &'a Import<'a>,
    ) -> &'a NameBinding<'a> {
        let import_vis = import.expect_vis();
        let vis = if binding.vis.is_at_least(import_vis, self)
            || pub_use_of_private_extern_crate_hack(import, binding)
        {
            import_vis
        } else {
            binding.vis
        };

        if let ImportKind::Glob { ref max_vis, .. } = import.kind {
            if vis == import_vis
                || max_vis.get().map_or(true, |max_vis| vis.is_at_least(max_vis, self))
            {
                max_vis.set(Some(vis))
            }
        }

        self.arenas.alloc_name_binding(NameBinding {
            kind: NameBindingKind::Import { binding, import, used: Cell::new(false) },
            ambiguity: None,
            span: import.span,
            vis,
            expansion: import.parent_scope.expansion,
        })
    }

    // Define the name or return the existing binding if there is a collision.
    pub(crate) fn try_define(
        &mut self,
        module: Module<'a>,
        key: BindingKey,
        binding: &'a NameBinding<'a>,
    ) -> Result<(), &'a NameBinding<'a>> {
        let res = binding.res();
        self.check_reserved_macro_name(key.ident, res);
        self.set_binding_parent_module(binding, module);
        self.update_resolution(module, key, |this, resolution| {
            if let Some(old_binding) = resolution.binding {
                if res == Res::Err {
                    // Do not override real bindings with `Res::Err`s from error recovery.
                    return Ok(());
                }
                match (old_binding.is_glob_import(), binding.is_glob_import()) {
                    (true, true) => {
                        if res != old_binding.res() {
                            resolution.binding = Some(this.ambiguity(
                                AmbiguityKind::GlobVsGlob,
                                old_binding,
                                binding,
                            ));
                        } else if !old_binding.vis.is_at_least(binding.vis, &*this) {
                            // We are glob-importing the same item but with greater visibility.
                            resolution.binding = Some(binding);
                        }
                    }
                    (old_glob @ true, false) | (old_glob @ false, true) => {
                        let (glob_binding, nonglob_binding) =
                            if old_glob { (old_binding, binding) } else { (binding, old_binding) };
                        if glob_binding.res() != nonglob_binding.res()
                            && key.ns == MacroNS
                            && nonglob_binding.expansion != LocalExpnId::ROOT
                        {
                            resolution.binding = Some(this.ambiguity(
                                AmbiguityKind::GlobVsExpanded,
                                nonglob_binding,
                                glob_binding,
                            ));
                        } else {
                            resolution.binding = Some(nonglob_binding);
                        }
                        resolution.shadowed_glob = Some(glob_binding);
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

    fn ambiguity(
        &self,
        kind: AmbiguityKind,
        primary_binding: &'a NameBinding<'a>,
        secondary_binding: &'a NameBinding<'a>,
    ) -> &'a NameBinding<'a> {
        self.arenas.alloc_name_binding(NameBinding {
            ambiguity: Some((secondary_binding, kind)),
            ..primary_binding.clone()
        })
    }

    // Use `f` to mutate the resolution of the name in the module.
    // If the resolution becomes a success, define it in the module's glob importers.
    fn update_resolution<T, F>(&mut self, module: Module<'a>, key: BindingKey, f: F) -> T
    where
        F: FnOnce(&mut Resolver<'a>, &mut NameResolution<'a>) -> T,
    {
        // Ensure that `resolution` isn't borrowed when defining in the module's glob importers,
        // during which the resolution might end up getting re-defined via a glob cycle.
        let (binding, t) = {
            let resolution = &mut *self.resolution(module, key).borrow_mut();
            let old_binding = resolution.binding();

            let t = f(self, resolution);

            match resolution.binding() {
                _ if old_binding.is_some() => return t,
                None => return t,
                Some(binding) => match old_binding {
                    Some(old_binding) if ptr::eq(old_binding, binding) => return t,
                    _ => (binding, t),
                },
            }
        };

        // Define `binding` in `module`s glob importers.
        for import in module.glob_importers.borrow_mut().iter() {
            let mut ident = key.ident;
            let scope = match ident.span.reverse_glob_adjust(module.expansion, import.span) {
                Some(Some(def)) => self.expn_def_scope(def),
                Some(None) => import.parent_scope.module,
                None => continue,
            };
            if self.is_accessible_from(binding.vis, scope) {
                let imported_binding = self.import(binding, import);
                let key = BindingKey { ident, ..key };
                let _ = self.try_define(import.parent_scope.module, key, imported_binding);
            }
        }

        t
    }

    // Define a dummy resolution containing a `Res::Err` as a placeholder for a failed resolution,
    // also mark such failed imports as used to avoid duplicate diagnostics.
    fn import_dummy_binding(&mut self, import: &'a Import<'a>) {
        if let ImportKind::Single { target, ref target_bindings, .. } = import.kind {
            if target_bindings.iter().any(|binding| binding.get().is_some()) {
                return; // Has resolution, do not create the dummy binding
            }
            let dummy_binding = self.dummy_binding;
            let dummy_binding = self.import(dummy_binding, import);
            self.per_ns(|this, ns| {
                let key = this.new_key(target, ns);
                let _ = this.try_define(import.parent_scope.module, key, dummy_binding);
            });
            self.record_use(target, dummy_binding, false);
        } else if import.imported_module.get().is_none() {
            import.used.set(true);
            self.used_imports.insert(import.id);
        }
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
}

pub struct ImportResolver<'a, 'b> {
    pub r: &'a mut Resolver<'b>,
}

impl<'a, 'b> ImportResolver<'a, 'b> {
    // Import resolution
    //
    // This is a fixed-point algorithm. We resolve imports until our efforts
    // are stymied by an unresolved import; then we bail out of the current
    // module and continue. We terminate successfully once no more imports
    // remain or unsuccessfully when no forward progress in resolving imports
    // is made.

    /// Resolves all imports for the crate. This method performs the fixed-
    /// point iteration.
    pub fn resolve_imports(&mut self) {
        let mut prev_num_indeterminates = self.r.indeterminate_imports.len() + 1;
        while self.r.indeterminate_imports.len() < prev_num_indeterminates {
            prev_num_indeterminates = self.r.indeterminate_imports.len();
            for import in mem::take(&mut self.r.indeterminate_imports) {
                match self.resolve_import(&import) {
                    true => self.r.determined_imports.push(import),
                    false => self.r.indeterminate_imports.push(import),
                }
            }
        }
    }

    pub fn finalize_imports(&mut self) {
        for module in self.r.arenas.local_modules().iter() {
            self.finalize_resolutions_in(module);
        }

        let mut seen_spans = FxHashSet::default();
        let mut errors = vec![];
        let mut prev_root_id: NodeId = NodeId::from_u32(0);
        let determined_imports = mem::take(&mut self.r.determined_imports);
        let indeterminate_imports = mem::take(&mut self.r.indeterminate_imports);

        for (is_indeterminate, import) in determined_imports
            .into_iter()
            .map(|i| (false, i))
            .chain(indeterminate_imports.into_iter().map(|i| (true, i)))
        {
            let unresolved_import_error = self.finalize_import(import);

            // If this import is unresolved then create a dummy import
            // resolution for it so that later resolve stages won't complain.
            self.r.import_dummy_binding(import);

            if let Some(err) = unresolved_import_error {
                if let ImportKind::Single { source, ref source_bindings, .. } = import.kind {
                    if source.name == kw::SelfLower {
                        // Silence `unresolved import` error if E0429 is already emitted
                        if let Err(Determined) = source_bindings.value_ns.get() {
                            continue;
                        }
                    }
                }

                if prev_root_id.as_u32() != 0
                    && prev_root_id.as_u32() != import.root_id.as_u32()
                    && !errors.is_empty()
                {
                    // In the case of a new import line, throw a diagnostic message
                    // for the previous line.
                    self.throw_unresolved_import_error(errors, None);
                    errors = vec![];
                }
                if seen_spans.insert(err.span) {
                    let path = import_path_to_string(
                        &import.module_path.iter().map(|seg| seg.ident).collect::<Vec<_>>(),
                        &import.kind,
                        err.span,
                    );
                    errors.push((path, err));
                    prev_root_id = import.root_id;
                }
            } else if is_indeterminate {
                let path = import_path_to_string(
                    &import.module_path.iter().map(|seg| seg.ident).collect::<Vec<_>>(),
                    &import.kind,
                    import.span,
                );
                let err = UnresolvedImportError {
                    span: import.span,
                    label: None,
                    note: None,
                    suggestion: None,
                };
                if path.contains("::") {
                    errors.push((path, err))
                }
            }
        }

        if !errors.is_empty() {
            self.throw_unresolved_import_error(errors, None);
        }
    }

    fn throw_unresolved_import_error(
        &self,
        errors: Vec<(String, UnresolvedImportError)>,
        span: Option<MultiSpan>,
    ) {
        /// Upper limit on the number of `span_label` messages.
        const MAX_LABEL_COUNT: usize = 10;

        let (span, msg) = if errors.is_empty() {
            (span.unwrap(), "unresolved import".to_string())
        } else {
            let span = MultiSpan::from_spans(errors.iter().map(|(_, err)| err.span).collect());

            let paths = errors.iter().map(|(path, _)| format!("`{}`", path)).collect::<Vec<_>>();

            let msg = format!("unresolved import{} {}", pluralize!(paths.len()), paths.join(", "),);

            (span, msg)
        };

        let mut diag = struct_span_err!(self.r.session, span, E0432, "{}", &msg);

        if let Some((_, UnresolvedImportError { note: Some(note), .. })) = errors.iter().last() {
            diag.note(note);
        }

        for (_, err) in errors.into_iter().take(MAX_LABEL_COUNT) {
            if let Some(label) = err.label {
                diag.span_label(err.span, label);
            }

            if let Some((suggestions, msg, applicability)) = err.suggestion {
                if suggestions.is_empty() {
                    diag.help(&msg);
                    continue;
                }
                diag.multipart_suggestion(&msg, suggestions, applicability);
            }
        }

        diag.emit();
    }

    /// Attempts to resolve the given import, returning true if its resolution is determined.
    /// If successful, the resolved bindings are written into the module.
    fn resolve_import(&mut self, import: &'b Import<'b>) -> bool {
        debug!(
            "(resolving import for module) resolving import `{}::...` in `{}`",
            Segment::names_to_string(&import.module_path),
            module_to_string(import.parent_scope.module).unwrap_or_else(|| "???".to_string()),
        );

        let module = if let Some(module) = import.imported_module.get() {
            module
        } else {
            // For better failure detection, pretend that the import will
            // not define any names while resolving its module path.
            let orig_vis = import.vis.take();
            let path_res =
                self.r.maybe_resolve_path(&import.module_path, None, &import.parent_scope);
            import.vis.set(orig_vis);

            match path_res {
                PathResult::Module(module) => module,
                PathResult::Indeterminate => return false,
                PathResult::NonModule(..) | PathResult::Failed { .. } => return true,
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
                return true;
            }
            _ => unreachable!(),
        };

        let mut indeterminate = false;
        self.r.per_ns(|this, ns| {
            if !type_ns_only || ns == TypeNS {
                if let Err(Undetermined) = source_bindings[ns].get() {
                    // For better failure detection, pretend that the import will
                    // not define any names while resolving its module path.
                    let orig_vis = import.vis.take();
                    let binding = this.resolve_ident_in_module(
                        module,
                        source,
                        ns,
                        &import.parent_scope,
                        None,
                        None,
                    );
                    import.vis.set(orig_vis);
                    source_bindings[ns].set(binding);
                } else {
                    return;
                };

                let parent = import.parent_scope.module;
                match source_bindings[ns].get() {
                    Err(Undetermined) => indeterminate = true,
                    // Don't update the resolution, because it was never added.
                    Err(Determined) if target.name == kw::Underscore => {}
                    Ok(binding) if binding.is_importable() => {
                        let imported_binding = this.import(binding, import);
                        target_bindings[ns].set(Some(imported_binding));
                        this.define(parent, target, ns, imported_binding);
                    }
                    source_binding @ (Ok(..) | Err(Determined)) => {
                        if source_binding.is_ok() {
                            let msg = format!("`{}` is not directly importable", target);
                            struct_span_err!(this.session, import.span, E0253, "{}", &msg)
                                .span_label(import.span, "cannot be imported directly")
                                .emit();
                        }
                        let key = this.new_key(target, ns);
                        this.update_resolution(parent, key, |_, resolution| {
                            resolution.single_imports.remove(&Interned::new_unchecked(import));
                        });
                    }
                }
            }
        });

        !indeterminate
    }

    /// Performs final import resolution, consistency checks and error reporting.
    ///
    /// Optionally returns an unresolved import error. This error is buffered and used to
    /// consolidate multiple unresolved import errors into a single diagnostic.
    fn finalize_import(&mut self, import: &'b Import<'b>) -> Option<UnresolvedImportError> {
        let orig_vis = import.vis.take();
        let ignore_binding = match &import.kind {
            ImportKind::Single { target_bindings, .. } => target_bindings[TypeNS].get(),
            _ => None,
        };
        let prev_ambiguity_errors_len = self.r.ambiguity_errors.len();
        let finalize = Finalize::with_root_span(import.root_id, import.span, import.root_span);
        let path_res = self.r.resolve_path(
            &import.module_path,
            None,
            &import.parent_scope,
            Some(finalize),
            ignore_binding,
        );
        let no_ambiguity = self.r.ambiguity_errors.len() == prev_ambiguity_errors_len;
        import.vis.set(orig_vis);
        let module = match path_res {
            PathResult::Module(module) => {
                // Consistency checks, analogous to `finalize_macro_resolutions`.
                if let Some(initial_module) = import.imported_module.get() {
                    if !ModuleOrUniformRoot::same_def(module, initial_module) && no_ambiguity {
                        span_bug!(import.span, "inconsistent resolution for an import");
                    }
                } else if self.r.privacy_errors.is_empty() {
                    let msg = "cannot determine resolution for the import";
                    let msg_note = "import resolution is stuck, try simplifying other imports";
                    self.r.session.struct_span_err(import.span, msg).note(msg_note).emit();
                }

                module
            }
            PathResult::Failed { is_error_from_last_segment: false, span, label, suggestion } => {
                if no_ambiguity {
                    assert!(import.imported_module.get().is_none());
                    self.r
                        .report_error(span, ResolutionError::FailedToResolve { label, suggestion });
                }
                return None;
            }
            PathResult::Failed { is_error_from_last_segment: true, span, label, suggestion } => {
                if no_ambiguity {
                    assert!(import.imported_module.get().is_none());
                    let err = match self.make_path_suggestion(
                        span,
                        import.module_path.clone(),
                        &import.parent_scope,
                    ) {
                        Some((suggestion, note)) => UnresolvedImportError {
                            span,
                            label: None,
                            note,
                            suggestion: Some((
                                vec![(span, Segment::names_to_string(&suggestion))],
                                String::from("a similar path exists"),
                                Applicability::MaybeIncorrect,
                            )),
                        },
                        None => UnresolvedImportError {
                            span,
                            label: Some(label),
                            note: None,
                            suggestion,
                        },
                    };
                    return Some(err);
                }
                return None;
            }
            PathResult::NonModule(_) => {
                if no_ambiguity {
                    assert!(import.imported_module.get().is_none());
                }
                // The error was already reported earlier.
                return None;
            }
            PathResult::Indeterminate => unreachable!(),
        };

        let (ident, target, source_bindings, target_bindings, type_ns_only) = match import.kind {
            ImportKind::Single {
                source,
                target,
                ref source_bindings,
                ref target_bindings,
                type_ns_only,
                ..
            } => (source, target, source_bindings, target_bindings, type_ns_only),
            ImportKind::Glob { is_prelude, ref max_vis } => {
                if import.module_path.len() <= 1 {
                    // HACK(eddyb) `lint_if_path_starts_with_module` needs at least
                    // 2 segments, so the `resolve_path` above won't trigger it.
                    let mut full_path = import.module_path.clone();
                    full_path.push(Segment::from_ident(Ident::empty()));
                    self.r.lint_if_path_starts_with_module(Some(finalize), &full_path, None);
                }

                if let ModuleOrUniformRoot::Module(module) = module {
                    if ptr::eq(module, import.parent_scope.module) {
                        // Importing a module into itself is not allowed.
                        return Some(UnresolvedImportError {
                            span: import.span,
                            label: Some(String::from("cannot glob-import a module into itself")),
                            note: None,
                            suggestion: None,
                        });
                    }
                }
                if !is_prelude
                    && let Some(max_vis) = max_vis.get()
                    && !max_vis.is_at_least(import.expect_vis(), &*self.r)
                {
                    let msg = "glob import doesn't reexport anything because no candidate is public enough";
                    self.r.lint_buffer.buffer_lint(UNUSED_IMPORTS, import.id, import.span, msg);
                }
                return None;
            }
            _ => unreachable!(),
        };

        let mut all_ns_err = true;
        self.r.per_ns(|this, ns| {
            if !type_ns_only || ns == TypeNS {
                let orig_vis = import.vis.take();
                let binding = this.resolve_ident_in_module(
                    module,
                    ident,
                    ns,
                    &import.parent_scope,
                    Some(Finalize { report_private: false, ..finalize }),
                    target_bindings[ns].get(),
                );
                import.vis.set(orig_vis);

                match binding {
                    Ok(binding) => {
                        // Consistency checks, analogous to `finalize_macro_resolutions`.
                        let initial_res = source_bindings[ns].get().map(|initial_binding| {
                            all_ns_err = false;
                            if let Some(target_binding) = target_bindings[ns].get() {
                                if target.name == kw::Underscore
                                    && initial_binding.is_extern_crate()
                                    && !initial_binding.is_import()
                                {
                                    this.record_use(
                                        ident,
                                        target_binding,
                                        import.module_path.is_empty(),
                                    );
                                }
                            }
                            initial_binding.res()
                        });
                        let res = binding.res();
                        if let Ok(initial_res) = initial_res {
                            if res != initial_res && this.ambiguity_errors.is_empty() {
                                span_bug!(import.span, "inconsistent resolution for an import");
                            }
                        } else if res != Res::Err
                            && this.ambiguity_errors.is_empty()
                            && this.privacy_errors.is_empty()
                        {
                            let msg = "cannot determine resolution for the import";
                            let msg_note =
                                "import resolution is stuck, try simplifying other imports";
                            this.session.struct_span_err(import.span, msg).note(msg_note).emit();
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
            self.r.per_ns(|this, ns| {
                if !type_ns_only || ns == TypeNS {
                    let binding = this.resolve_ident_in_module(
                        module,
                        ident,
                        ns,
                        &import.parent_scope,
                        Some(finalize),
                        None,
                    );
                    if binding.is_ok() {
                        all_ns_failed = false;
                    }
                }
            });

            return if all_ns_failed {
                let resolutions = match module {
                    ModuleOrUniformRoot::Module(module) => {
                        Some(self.r.resolutions(module).borrow())
                    }
                    _ => None,
                };
                let resolutions = resolutions.as_ref().into_iter().flat_map(|r| r.iter());
                let names = resolutions
                    .filter_map(|(BindingKey { ident: i, .. }, resolution)| {
                        if *i == ident {
                            return None;
                        } // Never suggest the same name
                        match *resolution.borrow() {
                            NameResolution { binding: Some(name_binding), .. } => {
                                match name_binding.kind {
                                    NameBindingKind::Import { binding, .. } => {
                                        match binding.kind {
                                            // Never suggest the name that has binding error
                                            // i.e., the name that cannot be previously resolved
                                            NameBindingKind::Res(Res::Err, _) => None,
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
                            format!("no `{}` in `{}`", ident, module_str)
                        } else {
                            format!("no `{}` in the root", ident)
                        }
                    }
                    _ => {
                        if !ident.is_path_segment_keyword() {
                            format!("no external crate `{}`", ident)
                        } else {
                            // HACK(eddyb) this shows up for `self` & `super`, which
                            // should work instead - for now keep the same error message.
                            format!("no `{}` in the root", ident)
                        }
                    }
                };

                Some(UnresolvedImportError {
                    span: import.span,
                    label: Some(label),
                    note,
                    suggestion,
                })
            } else {
                // `resolve_ident_in_module` reported a privacy error.
                None
            };
        }

        let mut reexport_error = None;
        let mut any_successful_reexport = false;
        let mut crate_private_reexport = false;
        self.r.per_ns(|this, ns| {
            if let Ok(binding) = source_bindings[ns].get() {
                if !binding.vis.is_at_least(import.expect_vis(), &*this) {
                    reexport_error = Some((ns, binding));
                    if let ty::Visibility::Restricted(binding_def_id) = binding.vis {
                        if binding_def_id.is_top_level_module() {
                            crate_private_reexport = true;
                        }
                    }
                } else {
                    any_successful_reexport = true;
                }
            }
        });

        // All namespaces must be re-exported with extra visibility for an error to occur.
        if !any_successful_reexport {
            let (ns, binding) = reexport_error.unwrap();
            if pub_use_of_private_extern_crate_hack(import, binding) {
                let msg = format!(
                    "extern crate `{}` is private, and cannot be \
                                   re-exported (error E0365), consider declaring with \
                                   `pub`",
                    ident
                );
                self.r.lint_buffer.buffer_lint(
                    PUB_USE_OF_PRIVATE_EXTERN_CRATE,
                    import.id,
                    import.span,
                    &msg,
                );
            } else {
                let error_msg = if crate_private_reexport {
                    format!(
                        "`{}` is only public within the crate, and cannot be re-exported outside",
                        ident
                    )
                } else {
                    format!("`{}` is private, and cannot be re-exported", ident)
                };

                if ns == TypeNS {
                    let label_msg = if crate_private_reexport {
                        format!("re-export of crate public `{}`", ident)
                    } else {
                        format!("re-export of private `{}`", ident)
                    };

                    struct_span_err!(self.r.session, import.span, E0365, "{}", error_msg)
                        .span_label(import.span, label_msg)
                        .note(&format!("consider declaring type or module `{}` with `pub`", ident))
                        .emit();
                } else {
                    let mut err =
                        struct_span_err!(self.r.session, import.span, E0364, "{error_msg}");
                    match binding.kind {
                        NameBindingKind::Res(Res::Def(DefKind::Macro(_), def_id), _)
                            // exclude decl_macro
                            if self.r.get_macro_by_def_id(def_id).macro_rules =>
                        {
                            err.span_help(
                                binding.span,
                                "consider adding a `#[macro_export]` to the macro in the imported module",
                            );
                        }
                        _ => {
                            err.span_note(
                                import.span,
                                &format!(
                                    "consider marking `{ident}` as `pub` in the imported module"
                                ),
                            );
                        }
                    }
                    err.emit();
                }
            }
        }

        if import.module_path.len() <= 1 {
            // HACK(eddyb) `lint_if_path_starts_with_module` needs at least
            // 2 segments, so the `resolve_path` above won't trigger it.
            let mut full_path = import.module_path.clone();
            full_path.push(Segment::from_ident(ident));
            self.r.per_ns(|this, ns| {
                if let Ok(binding) = source_bindings[ns].get() {
                    this.lint_if_path_starts_with_module(Some(finalize), &full_path, Some(binding));
                }
            });
        }

        // Record what this import resolves to for later uses in documentation,
        // this may resolve to either a value or a type, but for documentation
        // purposes it's good enough to just favor one over the other.
        self.r.per_ns(|this, ns| {
            if let Ok(binding) = source_bindings[ns].get() {
                this.import_res_map.entry(import.id).or_default()[ns] = Some(binding.res());
            }
        });

        self.check_for_redundant_imports(ident, import, source_bindings, target_bindings, target);

        debug!("(resolving single import) successfully resolved import");
        None
    }

    fn check_for_redundant_imports(
        &mut self,
        ident: Ident,
        import: &'b Import<'b>,
        source_bindings: &PerNS<Cell<Result<&'b NameBinding<'b>, Determinacy>>>,
        target_bindings: &PerNS<Cell<Option<&'b NameBinding<'b>>>>,
        target: Ident,
    ) {
        // Skip if the import was produced by a macro.
        if import.parent_scope.expansion != LocalExpnId::ROOT {
            return;
        }

        // Skip if we are inside a named module (in contrast to an anonymous
        // module defined by a block).
        if let ModuleKind::Def(..) = import.parent_scope.module.kind {
            return;
        }

        let mut is_redundant = PerNS { value_ns: None, type_ns: None, macro_ns: None };

        let mut redundant_span = PerNS { value_ns: None, type_ns: None, macro_ns: None };

        self.r.per_ns(|this, ns| {
            if let Ok(binding) = source_bindings[ns].get() {
                if binding.res() == Res::Err {
                    return;
                }

                match this.early_resolve_ident_in_lexical_scope(
                    target,
                    ScopeSet::All(ns, false),
                    &import.parent_scope,
                    None,
                    false,
                    target_bindings[ns].get(),
                ) {
                    Ok(other_binding) => {
                        is_redundant[ns] = Some(
                            binding.res() == other_binding.res() && !other_binding.is_ambiguity(),
                        );
                        redundant_span[ns] = Some((other_binding.span, other_binding.is_import()));
                    }
                    Err(_) => is_redundant[ns] = Some(false),
                }
            }
        });

        if !is_redundant.is_empty() && is_redundant.present_items().all(|is_redundant| is_redundant)
        {
            let mut redundant_spans: Vec<_> = redundant_span.present_items().collect();
            redundant_spans.sort();
            redundant_spans.dedup();
            self.r.lint_buffer.buffer_lint_with_diagnostic(
                UNUSED_IMPORTS,
                import.id,
                import.span,
                &format!("the item `{}` is imported redundantly", ident),
                BuiltinLintDiagnostics::RedundantImport(redundant_spans, ident),
            );
        }
    }

    fn resolve_glob_import(&mut self, import: &'b Import<'b>) {
        let ModuleOrUniformRoot::Module(module) = import.imported_module.get().unwrap() else {
            self.r.session.span_err(import.span, "cannot glob-import all possible crates");
            return;
        };

        if module.is_trait() {
            self.r.session.span_err(import.span, "items in traits are not importable");
            return;
        } else if ptr::eq(module, import.parent_scope.module) {
            return;
        } else if let ImportKind::Glob { is_prelude: true, .. } = import.kind {
            self.r.prelude = Some(module);
            return;
        }

        // Add to module's glob_importers
        module.glob_importers.borrow_mut().push(import);

        // Ensure that `resolutions` isn't borrowed during `try_define`,
        // since it might get updated via a glob cycle.
        let bindings = self
            .r
            .resolutions(module)
            .borrow()
            .iter()
            .filter_map(|(key, resolution)| {
                resolution.borrow().binding().map(|binding| (*key, binding))
            })
            .collect::<Vec<_>>();
        for (mut key, binding) in bindings {
            let scope = match key.ident.span.reverse_glob_adjust(module.expansion, import.span) {
                Some(Some(def)) => self.r.expn_def_scope(def),
                Some(None) => import.parent_scope.module,
                None => continue,
            };
            if self.r.is_accessible_from(binding.vis, scope) {
                let imported_binding = self.r.import(binding, import);
                let _ = self.r.try_define(import.parent_scope.module, key, imported_binding);
            }
        }

        // Record the destination of this import
        self.r.record_partial_res(import.id, PartialRes::new(module.res().unwrap()));
    }

    // Miscellaneous post-processing, including recording re-exports,
    // reporting conflicts, and reporting unresolved imports.
    fn finalize_resolutions_in(&mut self, module: Module<'b>) {
        // Since import resolution is finished, globs will not define any more names.
        *module.globs.borrow_mut() = Vec::new();

        if let Some(def_id) = module.opt_def_id() {
            let mut reexports = Vec::new();

            module.for_each_child(self.r, |_, ident, _, binding| {
                // FIXME: Consider changing the binding inserted by `#[macro_export] macro_rules`
                // into the crate root to actual `NameBindingKind::Import`.
                if binding.is_import()
                    || matches!(binding.kind, NameBindingKind::Res(_, _is_macro_export @ true))
                {
                    let res = binding.res().expect_non_local();
                    // Ambiguous imports are treated as errors at this point and are
                    // not exposed to other crates (see #36837 for more details).
                    if res != def::Res::Err && !binding.is_ambiguity() {
                        reexports.push(ModChild {
                            ident,
                            res,
                            vis: binding.vis,
                            span: binding.span,
                            macro_rules: false,
                        });
                    }
                }
            });

            if !reexports.is_empty() {
                // Call to `expect_local` should be fine because current
                // code is only called for local modules.
                self.r.reexport_map.insert(def_id.expect_local(), reexports);
            }
        }
    }
}

fn import_path_to_string(names: &[Ident], import_kind: &ImportKind<'_>, span: Span) -> String {
    let pos = names.iter().position(|p| span == p.span && p.name != kw::PathRoot);
    let global = !names.is_empty() && names[0].name == kw::PathRoot;
    if let Some(pos) = pos {
        let names = if global { &names[1..pos + 1] } else { &names[..pos + 1] };
        names_to_string(&names.iter().map(|ident| ident.name).collect::<Vec<_>>())
    } else {
        let names = if global { &names[1..] } else { names };
        if names.is_empty() {
            import_kind_to_string(import_kind)
        } else {
            format!(
                "{}::{}",
                names_to_string(&names.iter().map(|ident| ident.name).collect::<Vec<_>>()),
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
        ImportKind::MacroUse => "#[macro_use]".to_string(),
    }
}
