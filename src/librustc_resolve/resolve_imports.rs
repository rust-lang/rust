use ImportDirectiveSubclass::*;

use crate::{AmbiguityError, AmbiguityKind, AmbiguityErrorMisc};
use crate::{CrateLint, Module, ModuleOrUniformRoot, PerNS, ScopeSet, Weak};
use crate::Namespace::{self, TypeNS, MacroNS};
use crate::{NameBinding, NameBindingKind, ToNameBinding, PathResult, PrivacyError};
use crate::{Resolver, Segment};
use crate::{names_to_string, module_to_string};
use crate::{resolve_error, ResolutionError, Suggestion};
use crate::ModuleKind;
use crate::macros::ParentScope;

use errors::Applicability;

use rustc_data_structures::ptr_key::PtrKey;
use rustc::ty;
use rustc::lint::builtin::BuiltinLintDiagnostics;
use rustc::lint::builtin::{
    DUPLICATE_MACRO_EXPORTS,
    PUB_USE_OF_PRIVATE_EXTERN_CRATE,
    UNUSED_IMPORTS,
};
use rustc::hir::def_id::{CrateNum, DefId};
use rustc::hir::def::{self, DefKind, PartialRes, Export};
use rustc::session::DiagnosticMessageId;
use rustc::util::nodemap::FxHashSet;
use rustc::{bug, span_bug};

use syntax::ast::{self, Ident, Name, NodeId, CRATE_NODE_ID};
use syntax::ext::base::Determinacy::{self, Determined, Undetermined};
use syntax::ext::hygiene::Mark;
use syntax::symbol::{kw, sym};
use syntax::util::lev_distance::find_best_match_for_name;
use syntax::{struct_span_err, unwrap_or};
use syntax_pos::{MultiSpan, Span};

use log::*;

use std::cell::{Cell, RefCell};
use std::{mem, ptr};

type Res = def::Res<NodeId>;

/// Contains data for specific types of import directives.
#[derive(Clone, Debug)]
pub enum ImportDirectiveSubclass<'a> {
    SingleImport {
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
    },
    GlobImport {
        is_prelude: bool,
        max_vis: Cell<ty::Visibility>, // The visibility of the greatest re-export.
        // n.b. `max_vis` is only used in `finalize_import` to check for re-export errors.
    },
    ExternCrate {
        source: Option<Name>,
        target: Ident,
    },
    MacroUse,
}

/// One import directive.
#[derive(Debug,Clone)]
crate struct ImportDirective<'a> {
    /// The ID of the `extern crate`, `UseTree` etc that imported this `ImportDirective`.
    ///
    /// In the case where the `ImportDirective` was expanded from a "nested" use tree,
    /// this id is the ID of the leaf tree. For example:
    ///
    /// ```ignore (pacify the mercilous tidy)
    /// use foo::bar::{a, b}
    /// ```
    ///
    /// If this is the import directive for `foo::bar::a`, we would have the ID of the `UseTree`
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
    pub subclass: ImportDirectiveSubclass<'a>,
    pub vis: Cell<ty::Visibility>,
    pub used: Cell<bool>,
}

impl<'a> ImportDirective<'a> {
    pub fn is_glob(&self) -> bool {
        match self.subclass { ImportDirectiveSubclass::GlobImport { .. } => true, _ => false }
    }

    pub fn is_nested(&self) -> bool {
        match self.subclass {
            ImportDirectiveSubclass::SingleImport { nested, .. } => nested,
            _ => false
        }
    }

    crate fn crate_lint(&self) -> CrateLint {
        CrateLint::UsePath { root_id: self.root_id, root_span: self.root_span }
    }
}

#[derive(Clone, Default, Debug)]
/// Records information about the resolution of a name in a namespace of a module.
pub struct NameResolution<'a> {
    /// Single imports that may define the name in the namespace.
    /// Import directives are arena-allocated, so it's ok to use pointers as keys.
    single_imports: FxHashSet<PtrKey<'a, ImportDirective<'a>>>,
    /// The least shadowable known binding for this name, or None if there are no known bindings.
    pub binding: Option<&'a NameBinding<'a>>,
    shadowed_glob: Option<&'a NameBinding<'a>>,
}

impl<'a> NameResolution<'a> {
    // Returns the binding for the name if it is known or None if it not known.
    pub(crate) fn binding(&self) -> Option<&'a NameBinding<'a>> {
        self.binding.and_then(|binding| {
            if !binding.is_glob_import() ||
               self.single_imports.is_empty() { Some(binding) } else { None }
        })
    }
}

impl<'a> Resolver<'a> {
    fn resolution(&self, module: Module<'a>, ident: Ident, ns: Namespace)
                  -> &'a RefCell<NameResolution<'a>> {
        *module.resolutions.borrow_mut().entry((ident.modern(), ns))
               .or_insert_with(|| self.arenas.alloc_name_resolution())
    }

    crate fn resolve_ident_in_module_unadjusted(
        &mut self,
        module: ModuleOrUniformRoot<'a>,
        ident: Ident,
        ns: Namespace,
        record_used: bool,
        path_span: Span,
    ) -> Result<&'a NameBinding<'a>, Determinacy> {
        self.resolve_ident_in_module_unadjusted_ext(
            module, ident, ns, None, false, record_used, path_span
        ).map_err(|(determinacy, _)| determinacy)
    }

    /// Attempts to resolve `ident` in namespaces `ns` of `module`.
    /// Invariant: if `record_used` is `Some`, expansion and import resolution must be complete.
    crate fn resolve_ident_in_module_unadjusted_ext(
        &mut self,
        module: ModuleOrUniformRoot<'a>,
        ident: Ident,
        ns: Namespace,
        parent_scope: Option<&ParentScope<'a>>,
        restricted_shadowing: bool,
        record_used: bool,
        path_span: Span,
    ) -> Result<&'a NameBinding<'a>, (Determinacy, Weak)> {
        let module = match module {
            ModuleOrUniformRoot::Module(module) => module,
            ModuleOrUniformRoot::CrateRootAndExternPrelude => {
                assert!(!restricted_shadowing);
                let parent_scope = self.dummy_parent_scope();
                let binding = self.early_resolve_ident_in_lexical_scope(
                    ident, ScopeSet::AbsolutePath(ns), &parent_scope,
                    record_used, record_used, path_span,
                );
                return binding.map_err(|determinacy| (determinacy, Weak::No));
            }
            ModuleOrUniformRoot::ExternPrelude => {
                assert!(!restricted_shadowing);
                return if ns != TypeNS {
                    Err((Determined, Weak::No))
                } else if let Some(binding) = self.extern_prelude_get(ident, !record_used) {
                    Ok(binding)
                } else if !self.graph_root.unresolved_invocations.borrow().is_empty() {
                    // Macro-expanded `extern crate` items can add names to extern prelude.
                    Err((Undetermined, Weak::No))
                } else {
                    Err((Determined, Weak::No))
                }
            }
            ModuleOrUniformRoot::CurrentScope => {
                assert!(!restricted_shadowing);
                let parent_scope =
                    parent_scope.expect("no parent scope for a single-segment import");

                if ns == TypeNS {
                    if ident.name == kw::Crate ||
                        ident.name == kw::DollarCrate {
                        let module = self.resolve_crate_root(ident);
                        let binding = (module, ty::Visibility::Public,
                                        module.span, Mark::root())
                                        .to_name_binding(self.arenas);
                        return Ok(binding);
                    } else if ident.name == kw::Super ||
                                ident.name == kw::SelfLower {
                        // FIXME: Implement these with renaming requirements so that e.g.
                        // `use super;` doesn't work, but `use super as name;` does.
                        // Fall through here to get an error from `early_resolve_...`.
                    }
                }

                let binding = self.early_resolve_ident_in_lexical_scope(
                    ident, ScopeSet::Import(ns), parent_scope, record_used, record_used, path_span
                );
                return binding.map_err(|determinacy| (determinacy, Weak::No));
            }
        };

        self.populate_module_if_necessary(module);

        let resolution = self.resolution(module, ident, ns)
            .try_borrow_mut()
            .map_err(|_| (Determined, Weak::No))?; // This happens when there is a cycle of imports.

        if let Some(binding) = resolution.binding {
            if !restricted_shadowing && binding.expansion != Mark::root() {
                if let NameBindingKind::Res(_, true) = binding.kind {
                    self.macro_expanded_macro_export_errors.insert((path_span, binding.span));
                }
            }
        }

        let check_usable = |this: &mut Self, binding: &'a NameBinding<'a>| {
            if let Some(blacklisted_binding) = this.blacklisted_binding {
                if ptr::eq(binding, blacklisted_binding) {
                    return Err((Determined, Weak::No));
                }
            }
            // `extern crate` are always usable for backwards compatibility, see issue #37020,
            // remove this together with `PUB_USE_OF_PRIVATE_EXTERN_CRATE`.
            let usable = this.is_accessible(binding.vis) || binding.is_extern_crate();
            if usable { Ok(binding) } else { Err((Determined, Weak::No)) }
        };

        if record_used {
            return resolution.binding.and_then(|binding| {
                // If the primary binding is blacklisted, search further and return the shadowed
                // glob binding if it exists. What we really want here is having two separate
                // scopes in a module - one for non-globs and one for globs, but until that's done
                // use this hack to avoid inconsistent resolution ICEs during import validation.
                if let Some(blacklisted_binding) = self.blacklisted_binding {
                    if ptr::eq(binding, blacklisted_binding) {
                        return resolution.shadowed_glob;
                    }
                }
                Some(binding)
            }).ok_or((Determined, Weak::No)).and_then(|binding| {
                if self.last_import_segment && check_usable(self, binding).is_err() {
                    Err((Determined, Weak::No))
                } else {
                    self.record_use(ident, ns, binding, restricted_shadowing);

                    if let Some(shadowed_glob) = resolution.shadowed_glob {
                        // Forbid expanded shadowing to avoid time travel.
                        if restricted_shadowing &&
                        binding.expansion != Mark::root() &&
                        binding.res() != shadowed_glob.res() {
                            self.ambiguity_errors.push(AmbiguityError {
                                kind: AmbiguityKind::GlobVsExpanded,
                                ident,
                                b1: binding,
                                b2: shadowed_glob,
                                misc1: AmbiguityErrorMisc::None,
                                misc2: AmbiguityErrorMisc::None,
                            });
                        }
                    }

                    if !self.is_accessible(binding.vis) &&
                       // Remove this together with `PUB_USE_OF_PRIVATE_EXTERN_CRATE`
                       !(self.last_import_segment && binding.is_extern_crate()) {
                        self.privacy_errors.push(PrivacyError(path_span, ident, binding));
                    }

                    Ok(binding)
                }
            })
        }

        // Items and single imports are not shadowable, if we have one, then it's determined.
        if let Some(binding) = resolution.binding {
            if !binding.is_glob_import() {
                return check_usable(self, binding);
            }
        }

        // --- From now on we either have a glob resolution or no resolution. ---

        // Check if one of single imports can still define the name,
        // if it can then our result is not determined and can be invalidated.
        for single_import in &resolution.single_imports {
            if !self.is_accessible(single_import.vis.get()) {
                continue;
            }
            let module = unwrap_or!(single_import.imported_module.get(),
                                    return Err((Undetermined, Weak::No)));
            let ident = match single_import.subclass {
                SingleImport { source, .. } => source,
                _ => unreachable!(),
            };
            match self.resolve_ident_in_module(module, ident, ns, Some(&single_import.parent_scope),
                                               false, path_span) {
                Err(Determined) => continue,
                Ok(binding) if !self.is_accessible_from(
                    binding.vis, single_import.parent_scope.module
                ) => continue,
                Ok(_) | Err(Undetermined) => return Err((Undetermined, Weak::No)),
            }
        }

        // So we have a resolution that's from a glob import. This resolution is determined
        // if it cannot be shadowed by some new item/import expanded from a macro.
        // This happens either if there are no unexpanded macros, or expanded names cannot
        // shadow globs (that happens in macro namespace or with restricted shadowing).
        //
        // Additionally, any macro in any module can plant names in the root module if it creates
        // `macro_export` macros, so the root module effectively has unresolved invocations if any
        // module has unresolved invocations.
        // However, it causes resolution/expansion to stuck too often (#53144), so, to make
        // progress, we have to ignore those potential unresolved invocations from other modules
        // and prohibit access to macro-expanded `macro_export` macros instead (unless restricted
        // shadowing is enabled, see `macro_expanded_macro_export_errors`).
        let unexpanded_macros = !module.unresolved_invocations.borrow().is_empty();
        if let Some(binding) = resolution.binding {
            if !unexpanded_macros || ns == MacroNS || restricted_shadowing {
                return check_usable(self, binding);
            } else {
                return Err((Undetermined, Weak::No));
            }
        }

        // --- From now on we have no resolution. ---

        // Now we are in situation when new item/import can appear only from a glob or a macro
        // expansion. With restricted shadowing names from globs and macro expansions cannot
        // shadow names from outer scopes, so we can freely fallback from module search to search
        // in outer scopes. For `early_resolve_ident_in_lexical_scope` to continue search in outer
        // scopes we return `Undetermined` with `Weak::Yes`.

        // Check if one of unexpanded macros can still define the name,
        // if it can then our "no resolution" result is not determined and can be invalidated.
        if unexpanded_macros {
            return Err((Undetermined, Weak::Yes));
        }

        // Check if one of glob imports can still define the name,
        // if it can then our "no resolution" result is not determined and can be invalidated.
        for glob_import in module.globs.borrow().iter() {
            if !self.is_accessible(glob_import.vis.get()) {
                continue
            }
            let module = match glob_import.imported_module.get() {
                Some(ModuleOrUniformRoot::Module(module)) => module,
                Some(_) => continue,
                None => return Err((Undetermined, Weak::Yes)),
            };
            let (orig_current_module, mut ident) = (self.current_module, ident.modern());
            match ident.span.glob_adjust(module.expansion, glob_import.span) {
                Some(Some(def)) => self.current_module = self.macro_def_scope(def),
                Some(None) => {}
                None => continue,
            };
            let result = self.resolve_ident_in_module_unadjusted(
                ModuleOrUniformRoot::Module(module),
                ident,
                ns,
                false,
                path_span,
            );
            self.current_module = orig_current_module;

            match result {
                Err(Determined) => continue,
                Ok(binding) if !self.is_accessible_from(
                    binding.vis, glob_import.parent_scope.module
                ) => continue,
                Ok(_) | Err(Undetermined) => return Err((Undetermined, Weak::Yes)),
            }
        }

        // No resolution and no one else can define the name - determinate error.
        Err((Determined, Weak::No))
    }

    // Add an import directive to the current module.
    pub fn add_import_directive(&mut self,
                                module_path: Vec<Segment>,
                                subclass: ImportDirectiveSubclass<'a>,
                                span: Span,
                                id: NodeId,
                                item: &ast::Item,
                                root_span: Span,
                                root_id: NodeId,
                                vis: ty::Visibility,
                                parent_scope: ParentScope<'a>) {
        let current_module = parent_scope.module;
        let directive = self.arenas.alloc_import_directive(ImportDirective {
            parent_scope,
            module_path,
            imported_module: Cell::new(None),
            subclass,
            span,
            id,
            use_span: item.span,
            use_span_with_attributes: item.span_with_attributes(),
            has_attributes: !item.attrs.is_empty(),
            root_span,
            root_id,
            vis: Cell::new(vis),
            used: Cell::new(false),
        });

        debug!("add_import_directive({:?})", directive);

        self.indeterminate_imports.push(directive);
        match directive.subclass {
            SingleImport { target, type_ns_only, .. } => {
                self.per_ns(|this, ns| if !type_ns_only || ns == TypeNS {
                    let mut resolution = this.resolution(current_module, target, ns).borrow_mut();
                    resolution.single_imports.insert(PtrKey(directive));
                });
            }
            // We don't add prelude imports to the globs since they only affect lexical scopes,
            // which are not relevant to import resolution.
            GlobImport { is_prelude: true, .. } => {}
            GlobImport { .. } => current_module.globs.borrow_mut().push(directive),
            _ => unreachable!(),
        }
    }

    // Given a binding and an import directive that resolves to it,
    // return the corresponding binding defined by the import directive.
    crate fn import(&self, binding: &'a NameBinding<'a>, directive: &'a ImportDirective<'a>)
                    -> &'a NameBinding<'a> {
        let vis = if binding.pseudo_vis().is_at_least(directive.vis.get(), self) ||
                     // cf. `PUB_USE_OF_PRIVATE_EXTERN_CRATE`
                     !directive.is_glob() && binding.is_extern_crate() {
            directive.vis.get()
        } else {
            binding.pseudo_vis()
        };

        if let GlobImport { ref max_vis, .. } = directive.subclass {
            if vis == directive.vis.get() || vis.is_at_least(max_vis.get(), self) {
                max_vis.set(vis)
            }
        }

        self.arenas.alloc_name_binding(NameBinding {
            kind: NameBindingKind::Import {
                binding,
                directive,
                used: Cell::new(false),
            },
            ambiguity: None,
            span: directive.span,
            vis,
            expansion: directive.parent_scope.expansion,
        })
    }

    crate fn check_reserved_macro_name(&self, ident: Ident, ns: Namespace) {
        // Reserve some names that are not quite covered by the general check
        // performed on `Resolver::builtin_attrs`.
        if ns == MacroNS &&
           (ident.name == sym::cfg || ident.name == sym::cfg_attr ||
            ident.name == sym::derive) {
            self.session.span_err(ident.span,
                                  &format!("name `{}` is reserved in macro namespace", ident));
        }
    }

    // Define the name or return the existing binding if there is a collision.
    pub fn try_define(&mut self,
                      module: Module<'a>,
                      ident: Ident,
                      ns: Namespace,
                      binding: &'a NameBinding<'a>)
                      -> Result<(), &'a NameBinding<'a>> {
        self.check_reserved_macro_name(ident, ns);
        self.set_binding_parent_module(binding, module);
        self.update_resolution(module, ident, ns, |this, resolution| {
            if let Some(old_binding) = resolution.binding {
                if binding.res() == Res::Err {
                    // Do not override real bindings with `Res::Err`s from error recovery.
                    return Ok(());
                }
                match (old_binding.is_glob_import(), binding.is_glob_import()) {
                    (true, true) => {
                        if binding.res() != old_binding.res() {
                            resolution.binding = Some(this.ambiguity(AmbiguityKind::GlobVsGlob,
                                                                     old_binding, binding));
                        } else if !old_binding.vis.is_at_least(binding.vis, &*this) {
                            // We are glob-importing the same item but with greater visibility.
                            resolution.binding = Some(binding);
                        }
                    }
                    (old_glob @ true, false) | (old_glob @ false, true) => {
                        let (glob_binding, nonglob_binding) = if old_glob {
                            (old_binding, binding)
                        } else {
                            (binding, old_binding)
                        };
                        if glob_binding.res() != nonglob_binding.res() &&
                           ns == MacroNS && nonglob_binding.expansion != Mark::root() {
                            resolution.binding = Some(this.ambiguity(AmbiguityKind::GlobVsExpanded,
                                                                    nonglob_binding, glob_binding));
                        } else {
                            resolution.binding = Some(nonglob_binding);
                        }
                        resolution.shadowed_glob = Some(glob_binding);
                    }
                    (false, false) => {
                        if let (&NameBindingKind::Res(_, true), &NameBindingKind::Res(_, true)) =
                               (&old_binding.kind, &binding.kind) {

                            this.session.buffer_lint_with_diagnostic(
                                DUPLICATE_MACRO_EXPORTS,
                                CRATE_NODE_ID,
                                binding.span,
                                &format!("a macro named `{}` has already been exported", ident),
                                BuiltinLintDiagnostics::DuplicatedMacroExports(
                                    ident, old_binding.span, binding.span));

                            resolution.binding = Some(binding);
                        } else {
                            return Err(old_binding);
                        }
                    }
                }
            } else {
                resolution.binding = Some(binding);
            }

            Ok(())
        })
    }

    fn ambiguity(&self, kind: AmbiguityKind,
                 primary_binding: &'a NameBinding<'a>, secondary_binding: &'a NameBinding<'a>)
                 -> &'a NameBinding<'a> {
        self.arenas.alloc_name_binding(NameBinding {
            ambiguity: Some((secondary_binding, kind)),
            ..primary_binding.clone()
        })
    }

    // Use `f` to mutate the resolution of the name in the module.
    // If the resolution becomes a success, define it in the module's glob importers.
    fn update_resolution<T, F>(&mut self, module: Module<'a>, ident: Ident, ns: Namespace, f: F)
                               -> T
        where F: FnOnce(&mut Resolver<'a>, &mut NameResolution<'a>) -> T
    {
        // Ensure that `resolution` isn't borrowed when defining in the module's glob importers,
        // during which the resolution might end up getting re-defined via a glob cycle.
        let (binding, t) = {
            let resolution = &mut *self.resolution(module, ident, ns).borrow_mut();
            let old_binding = resolution.binding();

            let t = f(self, resolution);

            match resolution.binding() {
                _ if old_binding.is_some() => return t,
                None => return t,
                Some(binding) => match old_binding {
                    Some(old_binding) if ptr::eq(old_binding, binding) => return t,
                    _ => (binding, t),
                }
            }
        };

        // Define `binding` in `module`s glob importers.
        for directive in module.glob_importers.borrow_mut().iter() {
            let mut ident = ident.modern();
            let scope = match ident.span.reverse_glob_adjust(module.expansion, directive.span) {
                Some(Some(def)) => self.macro_def_scope(def),
                Some(None) => directive.parent_scope.module,
                None => continue,
            };
            if self.is_accessible_from(binding.vis, scope) {
                let imported_binding = self.import(binding, directive);
                let _ = self.try_define(directive.parent_scope.module, ident, ns, imported_binding);
            }
        }

        t
    }

    // Define a "dummy" resolution containing a Res::Err as a placeholder for a
    // failed resolution
    fn import_dummy_binding(&mut self, directive: &'a ImportDirective<'a>) {
        if let SingleImport { target, .. } = directive.subclass {
            let dummy_binding = self.dummy_binding;
            let dummy_binding = self.import(dummy_binding, directive);
            self.per_ns(|this, ns| {
                let _ = this.try_define(directive.parent_scope.module, target, ns, dummy_binding);
                // Consider erroneous imports used to avoid duplicate diagnostics.
                this.record_use(target, ns, dummy_binding, false);
            });
        }
    }
}

/// An error that may be transformed into a diagnostic later. Used to combine multiple unresolved
/// import errors within the same use tree into a single diagnostic.
#[derive(Debug, Clone)]
struct UnresolvedImportError {
    span: Span,
    label: Option<String>,
    note: Vec<String>,
    suggestion: Option<Suggestion>,
}

pub struct ImportResolver<'a, 'b> {
    pub resolver: &'a mut Resolver<'b>,
}

impl<'a, 'b> std::ops::Deref for ImportResolver<'a, 'b> {
    type Target = Resolver<'b>;
    fn deref(&self) -> &Resolver<'b> {
        self.resolver
    }
}

impl<'a, 'b> std::ops::DerefMut for ImportResolver<'a, 'b> {
    fn deref_mut(&mut self) -> &mut Resolver<'b> {
        self.resolver
    }
}

impl<'a, 'b> ty::DefIdTree for &'a ImportResolver<'a, 'b> {
    fn parent(self, id: DefId) -> Option<DefId> {
        self.resolver.parent(id)
    }
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
        let mut prev_num_indeterminates = self.indeterminate_imports.len() + 1;
        while self.indeterminate_imports.len() < prev_num_indeterminates {
            prev_num_indeterminates = self.indeterminate_imports.len();
            for import in mem::take(&mut self.indeterminate_imports) {
                match self.resolve_import(&import) {
                    true => self.determined_imports.push(import),
                    false => self.indeterminate_imports.push(import),
                }
            }
        }
    }

    pub fn finalize_imports(&mut self) {
        for module in self.arenas.local_modules().iter() {
            self.finalize_resolutions_in(module);
        }

        let mut has_errors = false;
        let mut seen_spans = FxHashSet::default();
        let mut errors = vec![];
        let mut prev_root_id: NodeId = NodeId::from_u32(0);
        for i in 0 .. self.determined_imports.len() {
            let import = self.determined_imports[i];
            if let Some(err) = self.finalize_import(import) {
                has_errors = true;

                if let SingleImport { source, ref source_bindings, .. } = import.subclass {
                    if source.name == kw::SelfLower {
                        // Silence `unresolved import` error if E0429 is already emitted
                        if let Err(Determined) = source_bindings.value_ns.get() {
                            continue;
                        }
                    }
                }

                // If the error is a single failed import then create a "fake" import
                // resolution for it so that later resolve stages won't complain.
                self.import_dummy_binding(import);
                if prev_root_id.as_u32() != 0
                        && prev_root_id.as_u32() != import.root_id.as_u32()
                        && !errors.is_empty() {
                    // In the case of a new import line, throw a diagnostic message
                    // for the previous line.
                    self.throw_unresolved_import_error(errors, None);
                    errors = vec![];
                }
                if !seen_spans.contains(&err.span) {
                    let path = import_path_to_string(
                        &import.module_path.iter().map(|seg| seg.ident).collect::<Vec<_>>(),
                        &import.subclass,
                        err.span,
                    );
                    seen_spans.insert(err.span);
                    errors.push((path, err));
                    prev_root_id = import.root_id;
                }
            }
        }

        if !errors.is_empty() {
            self.throw_unresolved_import_error(errors.clone(), None);
        }

        // Report unresolved imports only if no hard error was already reported
        // to avoid generating multiple errors on the same import.
        if !has_errors {
            for import in &self.indeterminate_imports {
                self.throw_unresolved_import_error(errors, Some(MultiSpan::from(import.span)));
                break;
            }
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
            let span = MultiSpan::from_spans(
                errors
                    .iter()
                    .map(|(_, err)| err.span)
                    .collect(),
            );

            let paths = errors
                .iter()
                .map(|(path, _)| format!("`{}`", path))
                .collect::<Vec<_>>();

            let msg = format!(
                "unresolved import{} {}",
                if paths.len() > 1 { "s" } else { "" },
                paths.join(", "),
            );

            (span, msg)
        };

        let mut diag = struct_span_err!(self.resolver.session, span, E0432, "{}", &msg);

        if let Some((_, UnresolvedImportError { note, .. })) = errors.iter().last() {
            for message in note {
                diag.note(&message);
            }
        }

        for (_, err) in errors.into_iter().take(MAX_LABEL_COUNT) {
            if let Some(label) = err.label {
                diag.span_label(err.span, label);
            }

            if let Some((suggestions, msg, applicability)) = err.suggestion {
                diag.multipart_suggestion(&msg, suggestions, applicability);
            }
        }

        diag.emit();
    }

    /// Attempts to resolve the given import, returning true if its resolution is determined.
    /// If successful, the resolved bindings are written into the module.
    fn resolve_import(&mut self, directive: &'b ImportDirective<'b>) -> bool {
        debug!("(resolving import for module) resolving import `{}::...` in `{}`",
               Segment::names_to_string(&directive.module_path),
               module_to_string(self.current_module).unwrap_or_else(|| "???".to_string()));

        self.current_module = directive.parent_scope.module;

        let module = if let Some(module) = directive.imported_module.get() {
            module
        } else {
            // For better failure detection, pretend that the import will
            // not define any names while resolving its module path.
            let orig_vis = directive.vis.replace(ty::Visibility::Invisible);
            let path_res = self.resolve_path(
                &directive.module_path,
                None,
                &directive.parent_scope,
                false,
                directive.span,
                directive.crate_lint(),
            );
            directive.vis.set(orig_vis);

            match path_res {
                PathResult::Module(module) => module,
                PathResult::Indeterminate => return false,
                PathResult::NonModule(..) | PathResult::Failed { .. } => return true,
            }
        };

        directive.imported_module.set(Some(module));
        let (source, target, source_bindings, target_bindings, type_ns_only) =
                match directive.subclass {
            SingleImport { source, target, ref source_bindings,
                           ref target_bindings, type_ns_only, .. } =>
                (source, target, source_bindings, target_bindings, type_ns_only),
            GlobImport { .. } => {
                self.resolve_glob_import(directive);
                return true;
            }
            _ => unreachable!(),
        };

        let mut indeterminate = false;
        self.per_ns(|this, ns| if !type_ns_only || ns == TypeNS {
            if let Err(Undetermined) = source_bindings[ns].get() {
                // For better failure detection, pretend that the import will
                // not define any names while resolving its module path.
                let orig_vis = directive.vis.replace(ty::Visibility::Invisible);
                let binding = this.resolve_ident_in_module(
                    module, source, ns, Some(&directive.parent_scope), false, directive.span
                );
                directive.vis.set(orig_vis);

                source_bindings[ns].set(binding);
            } else {
                return
            };

            let parent = directive.parent_scope.module;
            match source_bindings[ns].get() {
                Err(Undetermined) => indeterminate = true,
                Err(Determined) => {
                    this.update_resolution(parent, target, ns, |_, resolution| {
                        resolution.single_imports.remove(&PtrKey(directive));
                    });
                }
                Ok(binding) if !binding.is_importable() => {
                    let msg = format!("`{}` is not directly importable", target);
                    struct_span_err!(this.session, directive.span, E0253, "{}", &msg)
                        .span_label(directive.span, "cannot be imported directly")
                        .emit();
                    // Do not import this illegal binding. Import a dummy binding and pretend
                    // everything is fine
                    this.import_dummy_binding(directive);
                }
                Ok(binding) => {
                    let imported_binding = this.import(binding, directive);
                    target_bindings[ns].set(Some(imported_binding));
                    this.define(parent, target, ns, imported_binding);
                }
            }
        });

        !indeterminate
    }

    /// Performs final import resolution, consistency checks and error reporting.
    ///
    /// Optionally returns an unresolved import error. This error is buffered and used to
    /// consolidate multiple unresolved import errors into a single diagnostic.
    fn finalize_import(
        &mut self,
        directive: &'b ImportDirective<'b>
    ) -> Option<UnresolvedImportError> {
        self.current_module = directive.parent_scope.module;

        let orig_vis = directive.vis.replace(ty::Visibility::Invisible);
        let prev_ambiguity_errors_len = self.ambiguity_errors.len();
        let path_res = self.resolve_path(&directive.module_path, None, &directive.parent_scope,
                                         true, directive.span, directive.crate_lint());
        let no_ambiguity = self.ambiguity_errors.len() == prev_ambiguity_errors_len;
        directive.vis.set(orig_vis);
        let module = match path_res {
            PathResult::Module(module) => {
                // Consistency checks, analogous to `finalize_current_module_macro_resolutions`.
                if let Some(initial_module) = directive.imported_module.get() {
                    if !ModuleOrUniformRoot::same_def(module, initial_module) && no_ambiguity {
                        span_bug!(directive.span, "inconsistent resolution for an import");
                    }
                } else {
                    if self.privacy_errors.is_empty() {
                        let msg = "cannot determine resolution for the import";
                        let msg_note = "import resolution is stuck, try simplifying other imports";
                        self.session.struct_span_err(directive.span, msg).note(msg_note).emit();
                    }
                }

                module
            }
            PathResult::Failed { is_error_from_last_segment: false, span, label, suggestion } => {
                if no_ambiguity {
                    assert!(directive.imported_module.get().is_none());
                    resolve_error(self, span, ResolutionError::FailedToResolve {
                        label,
                        suggestion,
                    });
                }
                return None;
            }
            PathResult::Failed { is_error_from_last_segment: true, span, label, suggestion } => {
                if no_ambiguity {
                    assert!(directive.imported_module.get().is_none());
                    let err = match self.make_path_suggestion(
                        span,
                        directive.module_path.clone(),
                        &directive.parent_scope,
                    ) {
                        Some((suggestion, note)) => {
                            UnresolvedImportError {
                                span,
                                label: None,
                                note,
                                suggestion: Some((
                                    vec![(span, Segment::names_to_string(&suggestion))],
                                    String::from("a similar path exists"),
                                    Applicability::MaybeIncorrect,
                                )),
                            }
                        }
                        None => {
                            UnresolvedImportError {
                                span,
                                label: Some(label),
                                note: Vec::new(),
                                suggestion,
                            }
                        }
                    };

                    return Some(err);
                }
                return None;
            }
            PathResult::NonModule(path_res) if path_res.base_res() == Res::Err => {
                if no_ambiguity {
                    assert!(directive.imported_module.get().is_none());
                }
                // The error was already reported earlier.
                return None;
            }
            PathResult::Indeterminate | PathResult::NonModule(..) => unreachable!(),
        };

        let (ident, target, source_bindings, target_bindings, type_ns_only) =
                match directive.subclass {
            SingleImport { source, target, ref source_bindings,
                           ref target_bindings, type_ns_only, .. } =>
                (source, target, source_bindings, target_bindings, type_ns_only),
            GlobImport { is_prelude, ref max_vis } => {
                if directive.module_path.len() <= 1 {
                    // HACK(eddyb) `lint_if_path_starts_with_module` needs at least
                    // 2 segments, so the `resolve_path` above won't trigger it.
                    let mut full_path = directive.module_path.clone();
                    full_path.push(Segment::from_ident(Ident::invalid()));
                    self.lint_if_path_starts_with_module(
                        directive.crate_lint(),
                        &full_path,
                        directive.span,
                        None,
                    );
                }

                if let ModuleOrUniformRoot::Module(module) = module {
                    if module.def_id() == directive.parent_scope.module.def_id() {
                        // Importing a module into itself is not allowed.
                        return Some(UnresolvedImportError {
                            span: directive.span,
                            label: Some(String::from("cannot glob-import a module into itself")),
                            note: Vec::new(),
                            suggestion: None,
                        });
                    }
                }
                if !is_prelude &&
                   max_vis.get() != ty::Visibility::Invisible && // Allow empty globs.
                   !max_vis.get().is_at_least(directive.vis.get(), &*self) {
                    let msg = "A non-empty glob must import something with the glob's visibility";
                    self.session.span_err(directive.span, msg);
                }
                return None;
            }
            _ => unreachable!(),
        };

        let mut all_ns_err = true;
        self.per_ns(|this, ns| if !type_ns_only || ns == TypeNS {
            let orig_vis = directive.vis.replace(ty::Visibility::Invisible);
            let orig_blacklisted_binding =
                mem::replace(&mut this.blacklisted_binding, target_bindings[ns].get());
            let orig_last_import_segment = mem::replace(&mut this.last_import_segment, true);
            let binding = this.resolve_ident_in_module(
                module, ident, ns, Some(&directive.parent_scope), true, directive.span
            );
            this.last_import_segment = orig_last_import_segment;
            this.blacklisted_binding = orig_blacklisted_binding;
            directive.vis.set(orig_vis);

            match binding {
                Ok(binding) => {
                    // Consistency checks, analogous to `finalize_current_module_macro_resolutions`.
                    let initial_res = source_bindings[ns].get().map(|initial_binding| {
                        all_ns_err = false;
                        if let Some(target_binding) = target_bindings[ns].get() {
                            // Note that as_str() de-gensyms the Symbol
                            if target.name.as_str() == "_" &&
                               initial_binding.is_extern_crate() && !initial_binding.is_import() {
                                this.record_use(ident, ns, target_binding,
                                                directive.module_path.is_empty());
                            }
                        }
                        initial_binding.res()
                    });
                    let res = binding.res();
                    if let Ok(initial_res) = initial_res {
                        if res != initial_res && this.ambiguity_errors.is_empty() {
                            span_bug!(directive.span, "inconsistent resolution for an import");
                        }
                    } else {
                        if res != Res::Err &&
                           this.ambiguity_errors.is_empty() && this.privacy_errors.is_empty() {
                            let msg = "cannot determine resolution for the import";
                            let msg_note =
                                "import resolution is stuck, try simplifying other imports";
                            this.session.struct_span_err(directive.span, msg).note(msg_note).emit();
                        }
                    }
                }
                Err(..) => {
                    // FIXME: This assert may fire if public glob is later shadowed by a private
                    // single import (see test `issue-55884-2.rs`). In theory single imports should
                    // always block globs, even if they are not yet resolved, so that this kind of
                    // self-inconsistent resolution never happens.
                    // Reenable the assert when the issue is fixed.
                    // assert!(result[ns].get().is_err());
                }
            }
        });

        if all_ns_err {
            let mut all_ns_failed = true;
            self.per_ns(|this, ns| if !type_ns_only || ns == TypeNS {
                let binding = this.resolve_ident_in_module(
                    module, ident, ns, Some(&directive.parent_scope), true, directive.span
                );
                if binding.is_ok() {
                    all_ns_failed = false;
                }
            });

            return if all_ns_failed {
                let resolutions = match module {
                    ModuleOrUniformRoot::Module(module) => Some(module.resolutions.borrow()),
                    _ => None,
                };
                let resolutions = resolutions.as_ref().into_iter().flat_map(|r| r.iter());
                let names = resolutions.filter_map(|(&(ref i, _), resolution)| {
                    if *i == ident { return None; } // Never suggest the same name
                    match *resolution.borrow() {
                        NameResolution { binding: Some(name_binding), .. } => {
                            match name_binding.kind {
                                NameBindingKind::Import { binding, .. } => {
                                    match binding.kind {
                                        // Never suggest the name that has binding error
                                        // i.e., the name that cannot be previously resolved
                                        NameBindingKind::Res(Res::Err, _) => return None,
                                        _ => Some(&i.name),
                                    }
                                },
                                _ => Some(&i.name),
                            }
                        },
                        NameResolution { ref single_imports, .. }
                            if single_imports.is_empty() => None,
                        _ => Some(&i.name),
                    }
                });

                let lev_suggestion = find_best_match_for_name(names, &ident.as_str(), None)
                   .map(|suggestion|
                        (vec![(ident.span, suggestion.to_string())],
                         String::from("a similar name exists in the module"),
                         Applicability::MaybeIncorrect)
                    );

                let (suggestion, note) = match self.check_for_module_export_macro(
                    directive, module, ident,
                ) {
                    Some((suggestion, note)) => (suggestion.or(lev_suggestion), note),
                    _ => (lev_suggestion, Vec::new()),
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
                            format!("no `{}` external crate", ident)
                        } else {
                            // HACK(eddyb) this shows up for `self` & `super`, which
                            // should work instead - for now keep the same error message.
                            format!("no `{}` in the root", ident)
                        }
                    }
                };

                Some(UnresolvedImportError {
                    span: directive.span,
                    label: Some(label),
                    note,
                    suggestion,
                })
            } else {
                // `resolve_ident_in_module` reported a privacy error.
                self.import_dummy_binding(directive);
                None
            }
        }

        let mut reexport_error = None;
        let mut any_successful_reexport = false;
        self.per_ns(|this, ns| {
            if let Ok(binding) = source_bindings[ns].get() {
                let vis = directive.vis.get();
                if !binding.pseudo_vis().is_at_least(vis, &*this) {
                    reexport_error = Some((ns, binding));
                } else {
                    any_successful_reexport = true;
                }
            }
        });

        // All namespaces must be re-exported with extra visibility for an error to occur.
        if !any_successful_reexport {
            let (ns, binding) = reexport_error.unwrap();
            if ns == TypeNS && binding.is_extern_crate() {
                let msg = format!("extern crate `{}` is private, and cannot be \
                                   re-exported (error E0365), consider declaring with \
                                   `pub`",
                                   ident);
                self.session.buffer_lint(PUB_USE_OF_PRIVATE_EXTERN_CRATE,
                                         directive.id,
                                         directive.span,
                                         &msg);
            } else if ns == TypeNS {
                struct_span_err!(self.session, directive.span, E0365,
                                 "`{}` is private, and cannot be re-exported", ident)
                    .span_label(directive.span, format!("re-export of private `{}`", ident))
                    .note(&format!("consider declaring type or module `{}` with `pub`", ident))
                    .emit();
            } else {
                let msg = format!("`{}` is private, and cannot be re-exported", ident);
                let note_msg =
                    format!("consider marking `{}` as `pub` in the imported module", ident);
                struct_span_err!(self.session, directive.span, E0364, "{}", &msg)
                    .span_note(directive.span, &note_msg)
                    .emit();
            }
        }

        if directive.module_path.len() <= 1 {
            // HACK(eddyb) `lint_if_path_starts_with_module` needs at least
            // 2 segments, so the `resolve_path` above won't trigger it.
            let mut full_path = directive.module_path.clone();
            full_path.push(Segment::from_ident(ident));
            self.per_ns(|this, ns| {
                if let Ok(binding) = source_bindings[ns].get() {
                    this.lint_if_path_starts_with_module(
                        directive.crate_lint(),
                        &full_path,
                        directive.span,
                        Some(binding),
                    );
                }
            });
        }

        // Record what this import resolves to for later uses in documentation,
        // this may resolve to either a value or a type, but for documentation
        // purposes it's good enough to just favor one over the other.
        self.per_ns(|this, ns| if let Some(binding) = source_bindings[ns].get().ok() {
            let mut res = binding.res();
            if let Res::Def(DefKind::Macro(_), def_id) = res {
                // `DefId`s from the "built-in macro crate" should not leak from resolve because
                // later stages are not ready to deal with them and produce lots of ICEs. Replace
                // them with `Res::Err` until some saner scheme is implemented for built-in macros.
                if def_id.krate == CrateNum::BuiltinMacros {
                    this.session.span_err(directive.span, "cannot import a built-in macro");
                    res = Res::Err;
                }
            }
            this.import_res_map.entry(directive.id).or_default()[ns] = Some(res);
        });

        self.check_for_redundant_imports(
            ident,
            directive,
            source_bindings,
            target_bindings,
            target,
        );

        debug!("(resolving single import) successfully resolved import");
        None
    }

    fn check_for_redundant_imports(
        &mut self,
        ident: Ident,
        directive: &'b ImportDirective<'b>,
        source_bindings: &PerNS<Cell<Result<&'b NameBinding<'b>, Determinacy>>>,
        target_bindings: &PerNS<Cell<Option<&'b NameBinding<'b>>>>,
        target: Ident,
    ) {
        // Skip if the import was produced by a macro.
        if directive.parent_scope.expansion != Mark::root() {
            return;
        }

        // Skip if we are inside a named module (in contrast to an anonymous
        // module defined by a block).
        if let ModuleKind::Def(..) = directive.parent_scope.module.kind {
            return;
        }

        let mut is_redundant = PerNS {
            value_ns: None,
            type_ns: None,
            macro_ns: None,
        };

        let mut redundant_span = PerNS {
            value_ns: None,
            type_ns: None,
            macro_ns: None,
        };

        self.per_ns(|this, ns| if let Some(binding) = source_bindings[ns].get().ok() {
            if binding.res() == Res::Err {
                return;
            }

            let orig_blacklisted_binding = mem::replace(
                &mut this.blacklisted_binding,
                target_bindings[ns].get()
            );

            match this.early_resolve_ident_in_lexical_scope(
                target,
                ScopeSet::Import(ns),
                &directive.parent_scope,
                false,
                false,
                directive.span,
            ) {
                Ok(other_binding) => {
                    is_redundant[ns] = Some(
                        binding.res() == other_binding.res()
                        && !other_binding.is_ambiguity()
                    );
                    redundant_span[ns] =
                        Some((other_binding.span, other_binding.is_import()));
                }
                Err(_) => is_redundant[ns] = Some(false)
            }

            this.blacklisted_binding = orig_blacklisted_binding;
        });

        if !is_redundant.is_empty() &&
            is_redundant.present_items().all(|is_redundant| is_redundant)
        {
            let mut redundant_spans: Vec<_> = redundant_span.present_items().collect();
            redundant_spans.sort();
            redundant_spans.dedup();
            self.session.buffer_lint_with_diagnostic(
                UNUSED_IMPORTS,
                directive.id,
                directive.span,
                &format!("the item `{}` is imported redundantly", ident),
                BuiltinLintDiagnostics::RedundantImport(redundant_spans, ident),
            );
        }
    }

    fn resolve_glob_import(&mut self, directive: &'b ImportDirective<'b>) {
        let module = match directive.imported_module.get().unwrap() {
            ModuleOrUniformRoot::Module(module) => module,
            _ => {
                self.session.span_err(directive.span, "cannot glob-import all possible crates");
                return;
            }
        };

        self.populate_module_if_necessary(module);

        if module.is_trait() {
            self.session.span_err(directive.span, "items in traits are not importable.");
            return;
        } else if module.def_id() == directive.parent_scope.module.def_id()  {
            return;
        } else if let GlobImport { is_prelude: true, .. } = directive.subclass {
            self.prelude = Some(module);
            return;
        }

        // Add to module's glob_importers
        module.glob_importers.borrow_mut().push(directive);

        // Ensure that `resolutions` isn't borrowed during `try_define`,
        // since it might get updated via a glob cycle.
        let bindings = module.resolutions.borrow().iter().filter_map(|(&ident, resolution)| {
            resolution.borrow().binding().map(|binding| (ident, binding))
        }).collect::<Vec<_>>();
        for ((mut ident, ns), binding) in bindings {
            let scope = match ident.span.reverse_glob_adjust(module.expansion, directive.span) {
                Some(Some(def)) => self.macro_def_scope(def),
                Some(None) => self.current_module,
                None => continue,
            };
            if self.is_accessible_from(binding.pseudo_vis(), scope) {
                let imported_binding = self.import(binding, directive);
                let _ = self.try_define(directive.parent_scope.module, ident, ns, imported_binding);
            }
        }

        // Record the destination of this import
        self.record_partial_res(directive.id, PartialRes::new(module.res().unwrap()));
    }

    // Miscellaneous post-processing, including recording re-exports,
    // reporting conflicts, and reporting unresolved imports.
    fn finalize_resolutions_in(&mut self, module: Module<'b>) {
        // Since import resolution is finished, globs will not define any more names.
        *module.globs.borrow_mut() = Vec::new();

        let mut reexports = Vec::new();

        for (&(ident, ns), resolution) in module.resolutions.borrow().iter() {
            let resolution = &mut *resolution.borrow_mut();
            let binding = match resolution.binding {
                Some(binding) => binding,
                None => continue,
            };

            // Filter away ambiguous and gensymed imports. Gensymed imports
            // (e.g. implicitly injected `std`) cannot be properly encoded in metadata,
            // so they can cause name conflict errors downstream.
            let is_good_import = binding.is_import() && !binding.is_ambiguity() &&
                                 // Note that as_str() de-gensyms the Symbol
                                 !(ident.is_gensymed() && ident.name.as_str() != "_");
            if is_good_import || binding.is_macro_def() {
                let res = binding.res();
                if res != Res::Err {
                    if let Some(def_id) = res.opt_def_id() {
                        if !def_id.is_local() && def_id.krate != CrateNum::BuiltinMacros {
                            self.cstore.export_macros_untracked(def_id.krate);
                        }
                    }
                    reexports.push(Export {
                        ident: ident.modern(),
                        res: res,
                        span: binding.span,
                        vis: binding.vis,
                    });
                }
            }

            if let NameBindingKind::Import { binding: orig_binding, directive, .. } = binding.kind {
                if ns == TypeNS && orig_binding.is_variant() &&
                    !orig_binding.vis.is_at_least(binding.vis, &*self) {
                        let msg = match directive.subclass {
                            ImportDirectiveSubclass::SingleImport { .. } => {
                                format!("variant `{}` is private and cannot be re-exported",
                                        ident)
                            },
                            ImportDirectiveSubclass::GlobImport { .. } => {
                                let msg = "enum is private and its variants \
                                           cannot be re-exported".to_owned();
                                let error_id = (DiagnosticMessageId::ErrorId(0), // no code?!
                                                Some(binding.span),
                                                msg.clone());
                                let fresh = self.session.one_time_diagnostics
                                    .borrow_mut().insert(error_id);
                                if !fresh {
                                    continue;
                                }
                                msg
                            },
                            ref s @ _ => bug!("unexpected import subclass {:?}", s)
                        };
                        let mut err = self.session.struct_span_err(binding.span, &msg);

                        let imported_module = match directive.imported_module.get() {
                            Some(ModuleOrUniformRoot::Module(module)) => module,
                            _ => bug!("module should exist"),
                        };
                        let resolutions = imported_module.parent.expect("parent should exist")
                            .resolutions.borrow();
                        let enum_path_segment_index = directive.module_path.len() - 1;
                        let enum_ident = directive.module_path[enum_path_segment_index].ident;

                        let enum_resolution = resolutions.get(&(enum_ident, TypeNS))
                            .expect("resolution should exist");
                        let enum_span = enum_resolution.borrow()
                            .binding.expect("binding should exist")
                            .span;
                        let enum_def_span = self.session.source_map().def_span(enum_span);
                        let enum_def_snippet = self.session.source_map()
                            .span_to_snippet(enum_def_span).expect("snippet should exist");
                        // potentially need to strip extant `crate`/`pub(path)` for suggestion
                        let after_vis_index = enum_def_snippet.find("enum")
                            .expect("`enum` keyword should exist in snippet");
                        let suggestion = format!("pub {}",
                                                 &enum_def_snippet[after_vis_index..]);

                        self.session
                            .diag_span_suggestion_once(&mut err,
                                                       DiagnosticMessageId::ErrorId(0),
                                                       enum_def_span,
                                                       "consider making the enum public",
                                                       suggestion);
                        err.emit();
                }
            }
        }

        if reexports.len() > 0 {
            if let Some(def_id) = module.def_id() {
                self.export_map.insert(def_id, reexports);
            }
        }
    }
}

fn import_path_to_string(names: &[Ident],
                         subclass: &ImportDirectiveSubclass<'_>,
                         span: Span) -> String {
    let pos = names.iter()
        .position(|p| span == p.span && p.name != kw::PathRoot);
    let global = !names.is_empty() && names[0].name == kw::PathRoot;
    if let Some(pos) = pos {
        let names = if global { &names[1..pos + 1] } else { &names[..pos + 1] };
        names_to_string(names)
    } else {
        let names = if global { &names[1..] } else { names };
        if names.is_empty() {
            import_directive_subclass_to_string(subclass)
        } else {
            format!("{}::{}",
                    names_to_string(names),
                    import_directive_subclass_to_string(subclass))
        }
    }
}

fn import_directive_subclass_to_string(subclass: &ImportDirectiveSubclass<'_>) -> String {
    match *subclass {
        SingleImport { source, .. } => source.to_string(),
        GlobImport { .. } => "*".to_string(),
        ExternCrate { .. } => "<extern crate>".to_string(),
        MacroUse => "#[macro_use]".to_string(),
    }
}
