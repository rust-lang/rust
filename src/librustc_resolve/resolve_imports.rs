// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::ImportDirectiveSubclass::*;

use Module;
use Namespace::{self, TypeNS, ValueNS};
use {NameBinding, NameBindingKind, PrivacyError, ToNameBinding};
use ResolveResult;
use ResolveResult::*;
use Resolver;
use UseLexicalScopeFlag::DontUseLexicalScope;
use {names_to_string, module_to_string};
use {resolve_error, ResolutionError};

use rustc::ty;
use rustc::lint::builtin::PRIVATE_IN_PUBLIC;
use rustc::hir::def::*;

use syntax::ast::{NodeId, Name};
use syntax::util::lev_distance::find_best_match_for_name;
use syntax_pos::{Span, DUMMY_SP};

use std::cell::{Cell, RefCell};

impl<'a> Resolver<'a> {
    pub fn resolve_imports(&mut self) {
        ImportResolver { resolver: self }.resolve_imports();
    }
}

/// Contains data for specific types of import directives.
#[derive(Clone, Debug)]
pub enum ImportDirectiveSubclass {
    SingleImport {
        target: Name,
        source: Name,
        type_determined: Cell<bool>,
        value_determined: Cell<bool>,
    },
    GlobImport { is_prelude: bool },
}

impl ImportDirectiveSubclass {
    pub fn single(target: Name, source: Name) -> Self {
        SingleImport {
            target: target,
            source: source,
            type_determined: Cell::new(false),
            value_determined: Cell::new(false),
        }
    }
}

/// One import directive.
#[derive(Debug,Clone)]
pub struct ImportDirective<'a> {
    pub id: NodeId,
    module_path: Vec<Name>,
    target_module: Cell<Option<Module<'a>>>, // the resolution of `module_path`
    subclass: ImportDirectiveSubclass,
    span: Span,
    vis: ty::Visibility, // see note in ImportResolutionPerNamespace about how to use this
}

impl<'a> ImportDirective<'a> {
    pub fn is_glob(&self) -> bool {
        match self.subclass { ImportDirectiveSubclass::GlobImport { .. } => true, _ => false }
    }
}

#[derive(Clone, Default)]
/// Records information about the resolution of a name in a namespace of a module.
pub struct NameResolution<'a> {
    /// The single imports that define the name in the namespace.
    single_imports: SingleImports<'a>,
    /// The least shadowable known binding for this name, or None if there are no known bindings.
    pub binding: Option<&'a NameBinding<'a>>,
    duplicate_globs: Vec<&'a NameBinding<'a>>,
}

#[derive(Clone, Debug)]
enum SingleImports<'a> {
    /// No single imports can define the name in the namespace.
    None,
    /// Only the given single import can define the name in the namespace.
    MaybeOne(&'a ImportDirective<'a>),
    /// At least one single import will define the name in the namespace.
    AtLeastOne,
}

impl<'a> Default for SingleImports<'a> {
    fn default() -> Self {
        SingleImports::None
    }
}

impl<'a> SingleImports<'a> {
    fn add_directive(&mut self, directive: &'a ImportDirective<'a>) {
        match *self {
            SingleImports::None => *self = SingleImports::MaybeOne(directive),
            // If two single imports can define the name in the namespace, we can assume that at
            // least one of them will define it since otherwise both would have to define only one
            // namespace, leading to a duplicate error.
            SingleImports::MaybeOne(_) => *self = SingleImports::AtLeastOne,
            SingleImports::AtLeastOne => {}
        };
    }

    fn directive_failed(&mut self) {
        match *self {
            SingleImports::None => unreachable!(),
            SingleImports::MaybeOne(_) => *self = SingleImports::None,
            SingleImports::AtLeastOne => {}
        }
    }
}

impl<'a> NameResolution<'a> {
    // Returns the binding for the name if it is known or None if it not known.
    fn binding(&self) -> Option<&'a NameBinding<'a>> {
        self.binding.and_then(|binding| match self.single_imports {
            SingleImports::None => Some(binding),
            _ if !binding.is_glob_import() => Some(binding),
            _ => None, // The binding could be shadowed by a single import, so it is not known.
        })
    }

    // Returns Some(the resolution of the name), or None if the resolution depends
    // on whether more globs can define the name.
    fn try_result(&self, ns: Namespace, allow_private_imports: bool)
                  -> Option<ResolveResult<&'a NameBinding<'a>>> {
        match self.binding {
            Some(binding) if !binding.is_glob_import() =>
                return Some(Success(binding)),
            _ => {} // Items and single imports are not shadowable
        };

        // Check if a single import can still define the name.
        match self.single_imports {
            SingleImports::None => {},
            SingleImports::AtLeastOne => return Some(Indeterminate),
            SingleImports::MaybeOne(directive) => {
                // If (1) we don't allow private imports, (2) no public single import can define
                // the name, and (3) no public glob has defined the name, the resolution depends
                // on whether more globs can define the name.
                if !allow_private_imports && directive.vis != ty::Visibility::Public &&
                   !self.binding.map(NameBinding::is_pseudo_public).unwrap_or(false) {
                    return None;
                }

                let target_module = match directive.target_module.get() {
                    Some(target_module) => target_module,
                    None => return Some(Indeterminate),
                };
                let name = match directive.subclass {
                    SingleImport { source, .. } => source,
                    GlobImport { .. } => unreachable!(),
                };
                match target_module.resolve_name(name, ns, false) {
                    Failed(_) => {}
                    _ => return Some(Indeterminate),
                }
            }
        }

        self.binding.map(Success)
    }
}

impl<'a> ::ModuleS<'a> {
    fn resolution(&self, name: Name, ns: Namespace) -> &'a RefCell<NameResolution<'a>> {
        *self.resolutions.borrow_mut().entry((name, ns))
             .or_insert_with(|| self.arenas.alloc_name_resolution())
    }

    pub fn resolve_name(&self, name: Name, ns: Namespace, allow_private_imports: bool)
                        -> ResolveResult<&'a NameBinding<'a>> {
        let resolution = self.resolution(name, ns);
        let resolution = match resolution.borrow_state() {
            ::std::cell::BorrowState::Unused => resolution.borrow_mut(),
            _ => return Failed(None), // This happens when there is a cycle of imports
        };

        if let Some(result) = resolution.try_result(ns, allow_private_imports) {
            // If the resolution doesn't depend on glob definability, check privacy and return.
            return result.and_then(|binding| {
                let allowed = allow_private_imports || !binding.is_import() ||
                                                       binding.is_pseudo_public();
                if allowed { Success(binding) } else { Failed(None) }
            });
        }

        // Check if the globs are determined
        for directive in self.globs.borrow().iter() {
            if !allow_private_imports && directive.vis != ty::Visibility::Public { continue }
            match directive.target_module.get() {
                None => return Indeterminate,
                Some(target_module) => match target_module.resolve_name(name, ns, false) {
                    Indeterminate => return Indeterminate,
                    _ => {}
                }
            }
        }

        Failed(None)
    }

    pub fn add_import_directive(&self,
                                module_path: Vec<Name>,
                                subclass: ImportDirectiveSubclass,
                                span: Span,
                                id: NodeId,
                                vis: ty::Visibility) {
        let directive = self.arenas.alloc_import_directive(ImportDirective {
            module_path: module_path,
            target_module: Cell::new(None),
            subclass: subclass,
            span: span,
            id: id,
            vis: vis,
        });

        self.unresolved_imports.borrow_mut().push(directive);
        match directive.subclass {
            SingleImport { target, .. } => {
                for &ns in &[ValueNS, TypeNS] {
                    self.resolution(target, ns).borrow_mut().single_imports
                                                            .add_directive(directive);
                }
            }
            // We don't add prelude imports to the globs since they only affect lexical scopes,
            // which are not relevant to import resolution.
            GlobImport { is_prelude: true } => {}
            GlobImport { .. } => self.globs.borrow_mut().push(directive),
        }
    }
}

impl<'a> Resolver<'a> {
    // Given a binding and an import directive that resolves to it,
    // return the corresponding binding defined by the import directive.
    fn import(&mut self, binding: &'a NameBinding<'a>, directive: &'a ImportDirective<'a>)
              -> NameBinding<'a> {
        NameBinding {
            kind: NameBindingKind::Import {
                binding: binding,
                directive: directive,
            },
            span: directive.span,
            vis: directive.vis,
        }
    }

    // Define the name or return the existing binding if there is a collision.
    pub fn try_define<T>(&mut self, module: Module<'a>, name: Name, ns: Namespace, binding: T)
                         -> Result<(), &'a NameBinding<'a>>
        where T: ToNameBinding<'a>
    {
        let binding = self.arenas.alloc_name_binding(binding.to_name_binding());
        self.update_resolution(module, name, ns, |_, resolution| {
            if let Some(old_binding) = resolution.binding {
                if binding.is_glob_import() {
                    resolution.duplicate_globs.push(binding);
                } else if old_binding.is_glob_import() {
                    resolution.duplicate_globs.push(old_binding);
                    resolution.binding = Some(binding);
                } else {
                    return Err(old_binding);
                }
            } else {
                resolution.binding = Some(binding);
            }

            Ok(())
        })
    }

    // Use `f` to mutate the resolution of the name in the module.
    // If the resolution becomes a success, define it in the module's glob importers.
    fn update_resolution<T, F>(&mut self, module: Module<'a>, name: Name, ns: Namespace, f: F) -> T
        where F: FnOnce(&mut Resolver<'a>, &mut NameResolution<'a>) -> T
    {
        // Ensure that `resolution` isn't borrowed when defining in the module's glob importers,
        // during which the resolution might end up getting re-defined via a glob cycle.
        let (new_binding, t) = {
            let mut resolution = &mut *module.resolution(name, ns).borrow_mut();
            let was_known = resolution.binding().is_some();

            let t = f(self, resolution);

            if was_known { return t; }
            match resolution.binding() {
                Some(binding) => (binding, t),
                None => return t,
            }
        };

        // Define `new_binding` in `module`s glob importers.
        if new_binding.is_importable() && new_binding.is_pseudo_public() {
            for &(importer, directive) in module.glob_importers.borrow_mut().iter() {
                let imported_binding = self.import(new_binding, directive);
                let _ = self.try_define(importer, name, ns, imported_binding);
            }
        }

        t
    }
}

struct ImportResolvingError<'a> {
    /// Module where the error happened
    source_module: Module<'a>,
    import_directive: &'a ImportDirective<'a>,
    span: Span,
    help: String,
}

struct ImportResolver<'a, 'b: 'a> {
    resolver: &'a mut Resolver<'b>,
}

impl<'a, 'b: 'a> ::std::ops::Deref for ImportResolver<'a, 'b> {
    type Target = Resolver<'b>;
    fn deref(&self) -> &Resolver<'b> {
        self.resolver
    }
}

impl<'a, 'b: 'a> ::std::ops::DerefMut for ImportResolver<'a, 'b> {
    fn deref_mut(&mut self) -> &mut Resolver<'b> {
        self.resolver
    }
}

impl<'a, 'b: 'a> ty::NodeIdTree for ImportResolver<'a, 'b> {
    fn is_descendant_of(&self, node: NodeId, ancestor: NodeId) -> bool {
        self.resolver.is_descendant_of(node, ancestor)
    }
}

impl<'a, 'b:'a> ImportResolver<'a, 'b> {
    // Import resolution
    //
    // This is a fixed-point algorithm. We resolve imports until our efforts
    // are stymied by an unresolved import; then we bail out of the current
    // module and continue. We terminate successfully once no more imports
    // remain or unsuccessfully when no forward progress in resolving imports
    // is made.

    /// Resolves all imports for the crate. This method performs the fixed-
    /// point iteration.
    fn resolve_imports(&mut self) {
        let mut i = 0;
        let mut prev_unresolved_imports = 0;
        let mut errors = Vec::new();

        loop {
            debug!("(resolving imports) iteration {}, {} imports left", i, self.unresolved_imports);

            // Attempt to resolve imports in all local modules.
            for module in self.arenas.local_modules().iter() {
                self.current_module = module;
                self.resolve_imports_in_current_module(&mut errors);
            }

            if self.unresolved_imports == 0 {
                debug!("(resolving imports) success");
                for module in self.arenas.local_modules().iter() {
                    self.finalize_resolutions_in(module, false);
                }
                break;
            }

            if self.unresolved_imports == prev_unresolved_imports {
                // resolving failed
                // Report unresolved imports only if no hard error was already reported
                // to avoid generating multiple errors on the same import.
                // Imports that are still indeterminate at this point are actually blocked
                // by errored imports, so there is no point reporting them.
                for module in self.arenas.local_modules().iter() {
                    self.finalize_resolutions_in(module, errors.len() == 0);
                }
                for e in errors {
                    self.import_resolving_error(e)
                }
                break;
            }

            i += 1;
            prev_unresolved_imports = self.unresolved_imports;
        }
    }

    // Define a "dummy" resolution containing a Def::Err as a placeholder for a
    // failed resolution
    fn import_dummy_binding(&mut self,
                            source_module: Module<'b>,
                            directive: &'b ImportDirective<'b>) {
        if let SingleImport { target, .. } = directive.subclass {
            let dummy_binding = self.arenas.alloc_name_binding(NameBinding {
                kind: NameBindingKind::Def(Def::Err),
                span: DUMMY_SP,
                vis: ty::Visibility::Public,
            });
            let dummy_binding = self.import(dummy_binding, directive);

            let _ = self.try_define(source_module, target, ValueNS, dummy_binding.clone());
            let _ = self.try_define(source_module, target, TypeNS, dummy_binding);
        }
    }

    /// Resolves an `ImportResolvingError` into the correct enum discriminant
    /// and passes that on to `resolve_error`.
    fn import_resolving_error(&mut self, e: ImportResolvingError<'b>) {
        // If the error is a single failed import then create a "fake" import
        // resolution for it so that later resolve stages won't complain.
        self.import_dummy_binding(e.source_module, e.import_directive);
        let path = import_path_to_string(&e.import_directive.module_path,
                                         &e.import_directive.subclass);
        resolve_error(self.resolver,
                      e.span,
                      ResolutionError::UnresolvedImport(Some((&path, &e.help))));
    }

    /// Attempts to resolve imports for the given module only.
    fn resolve_imports_in_current_module(&mut self, errors: &mut Vec<ImportResolvingError<'b>>) {
        let mut imports = Vec::new();
        let mut unresolved_imports = self.current_module.unresolved_imports.borrow_mut();
        ::std::mem::swap(&mut imports, &mut unresolved_imports);

        for import_directive in imports {
            match self.resolve_import(&import_directive) {
                Failed(err) => {
                    let (span, help) = match err {
                        Some((span, msg)) => (span, format!(". {}", msg)),
                        None => (import_directive.span, String::new()),
                    };
                    errors.push(ImportResolvingError {
                        source_module: self.current_module,
                        import_directive: import_directive,
                        span: span,
                        help: help,
                    });
                }
                Indeterminate => unresolved_imports.push(import_directive),
                Success(()) => {
                    // Decrement the count of unresolved imports.
                    assert!(self.unresolved_imports >= 1);
                    self.unresolved_imports -= 1;
                }
            }
        }
    }

    /// Attempts to resolve the given import. The return value indicates
    /// failure if we're certain the name does not exist, indeterminate if we
    /// don't know whether the name exists at the moment due to other
    /// currently-unresolved imports, or success if we know the name exists.
    /// If successful, the resolved bindings are written into the module.
    fn resolve_import(&mut self, directive: &'b ImportDirective<'b>) -> ResolveResult<()> {
        debug!("(resolving import for module) resolving import `{}::...` in `{}`",
               names_to_string(&directive.module_path),
               module_to_string(self.current_module));

        let target_module = match directive.target_module.get() {
            Some(module) => module,
            _ => match self.resolve_module_path(&directive.module_path,
                                                DontUseLexicalScope,
                                                directive.span) {
                Success(module) => module,
                Indeterminate => return Indeterminate,
                Failed(err) => return Failed(err),
            },
        };

        directive.target_module.set(Some(target_module));
        let (source, target, value_determined, type_determined) = match directive.subclass {
            SingleImport { source, target, ref value_determined, ref type_determined } =>
                (source, target, value_determined, type_determined),
            GlobImport { .. } => return self.resolve_glob_import(target_module, directive),
        };

        // We need to resolve both namespaces for this to succeed.
        let value_result = self.resolve_name_in_module(target_module, source, ValueNS, false, true);
        let type_result = self.resolve_name_in_module(target_module, source, TypeNS, false, true);

        let module = self.current_module;
        let mut privacy_error = true;
        for &(ns, result, determined) in &[(ValueNS, &value_result, value_determined),
                                           (TypeNS, &type_result, type_determined)] {
            match *result {
                Failed(..) if !determined.get() => {
                    determined.set(true);
                    self.update_resolution(module, target, ns, |_, resolution| {
                        resolution.single_imports.directive_failed()
                    });
                }
                Success(binding) if !binding.is_importable() => {
                    let msg = format!("`{}` is not directly importable", target);
                    struct_span_err!(self.session, directive.span, E0253, "{}", &msg)
                        .span_label(directive.span, &format!("cannot be imported directly"))
                        .emit();
                    // Do not import this illegal binding. Import a dummy binding and pretend
                    // everything is fine
                    self.import_dummy_binding(module, directive);
                    return Success(());
                }
                Success(binding) if !self.is_accessible(binding.vis) => {}
                Success(binding) if !determined.get() => {
                    determined.set(true);
                    let imported_binding = self.import(binding, directive);
                    let conflict = self.try_define(module, target, ns, imported_binding);
                    if let Err(old_binding) = conflict {
                        let binding = &self.import(binding, directive);
                        self.report_conflict(module, target, ns, binding, old_binding);
                    }
                    privacy_error = false;
                }
                Success(_) => privacy_error = false,
                _ => {}
            }
        }

        match (&value_result, &type_result) {
            (&Indeterminate, _) | (_, &Indeterminate) => return Indeterminate,
            (&Failed(_), &Failed(_)) => {
                let resolutions = target_module.resolutions.borrow();
                let names = resolutions.iter().filter_map(|(&(ref name, _), resolution)| {
                    if *name == source { return None; } // Never suggest the same name
                    match *resolution.borrow() {
                        NameResolution { binding: Some(_), .. } => Some(name),
                        NameResolution { single_imports: SingleImports::None, .. } => None,
                        _ => Some(name),
                    }
                });
                let lev_suggestion = match find_best_match_for_name(names, &source.as_str(), None) {
                    Some(name) => format!(". Did you mean to use `{}`?", name),
                    None => "".to_owned(),
                };
                let module_str = module_to_string(target_module);
                let msg = if &module_str == "???" {
                    format!("There is no `{}` in the crate root{}", source, lev_suggestion)
                } else {
                    format!("There is no `{}` in `{}`{}", source, module_str, lev_suggestion)
                };
                return Failed(Some((directive.span, msg)));
            }
            _ => (),
        }

        if privacy_error {
            for &(ns, result) in &[(ValueNS, &value_result), (TypeNS, &type_result)] {
                let binding = match *result { Success(binding) => binding, _ => continue };
                self.privacy_errors.push(PrivacyError(directive.span, source, binding));
                let imported_binding = self.import(binding, directive);
                let _ = self.try_define(module, target, ns, imported_binding);
            }
        }

        match (&value_result, &type_result) {
            (&Success(binding), _) if !binding.pseudo_vis().is_at_least(directive.vis, self) &&
                                      self.is_accessible(binding.vis) => {
                let msg = format!("`{}` is private, and cannot be reexported", source);
                let note_msg = format!("consider marking `{}` as `pub` in the imported module",
                                        source);
                struct_span_err!(self.session, directive.span, E0364, "{}", &msg)
                    .span_note(directive.span, &note_msg)
                    .emit();
            }

            (_, &Success(binding)) if !binding.pseudo_vis().is_at_least(directive.vis, self) &&
                                      self.is_accessible(binding.vis) => {
                if binding.is_extern_crate() {
                    let msg = format!("extern crate `{}` is private, and cannot be reexported \
                                       (error E0364), consider declaring with `pub`",
                                       source);
                    self.session.add_lint(PRIVATE_IN_PUBLIC, directive.id, directive.span, msg);
                } else {
                    let mut err = struct_span_err!(self.session, directive.span, E0365,
                                                     "`{}` is private, and cannot be reexported",
                                                     source);
                    err.span_label(directive.span, &format!("reexport of private `{}`", source));
                    err.note(&format!("consider declaring type or module `{}` with `pub`", source));
                    err.emit();
                }
            }

            _ => {}
        }

        // Record what this import resolves to for later uses in documentation,
        // this may resolve to either a value or a type, but for documentation
        // purposes it's good enough to just favor one over the other.
        let def = match type_result.success().and_then(NameBinding::def) {
            Some(def) => def,
            None => value_result.success().and_then(NameBinding::def).unwrap(),
        };
        let path_resolution = PathResolution::new(def);
        self.def_map.insert(directive.id, path_resolution);

        debug!("(resolving single import) successfully resolved import");
        return Success(());
    }

    // Resolves a glob import. Note that this function cannot fail; it either
    // succeeds or bails out (as importing * from an empty module or a module
    // that exports nothing is valid). target_module is the module we are
    // actually importing, i.e., `foo` in `use foo::*`.
    fn resolve_glob_import(&mut self, target_module: Module<'b>, directive: &'b ImportDirective<'b>)
                           -> ResolveResult<()> {
        if let Some(Def::Trait(_)) = target_module.def {
            self.session.span_err(directive.span, "items in traits are not importable.");
        }

        let module = self.current_module;
        if module.def_id() == target_module.def_id() {
            // This means we are trying to glob import a module into itself, and it is a no-go
            let msg = "Cannot glob-import a module into itself.".into();
            return Failed(Some((directive.span, msg)));
        }
        self.populate_module_if_necessary(target_module);

        if let GlobImport { is_prelude: true } = directive.subclass {
            self.prelude = Some(target_module);
            return Success(());
        }

        // Add to target_module's glob_importers
        target_module.glob_importers.borrow_mut().push((module, directive));

        // Ensure that `resolutions` isn't borrowed during `try_define`,
        // since it might get updated via a glob cycle.
        let bindings = target_module.resolutions.borrow().iter().filter_map(|(name, resolution)| {
            resolution.borrow().binding().map(|binding| (*name, binding))
        }).collect::<Vec<_>>();
        for ((name, ns), binding) in bindings {
            if binding.is_importable() && binding.is_pseudo_public() {
                let imported_binding = self.import(binding, directive);
                let _ = self.try_define(module, name, ns, imported_binding);
            }
        }

        // Record the destination of this import
        if let Some(did) = target_module.def_id() {
            let resolution = PathResolution::new(Def::Mod(did));
            self.def_map.insert(directive.id, resolution);
        }

        debug!("(resolving glob import) successfully resolved import");
        return Success(());
    }

    // Miscellaneous post-processing, including recording reexports, reporting conflicts,
    // reporting the PRIVATE_IN_PUBLIC lint, and reporting unresolved imports.
    fn finalize_resolutions_in(&mut self, module: Module<'b>, report_unresolved_imports: bool) {
        // Since import resolution is finished, globs will not define any more names.
        *module.globs.borrow_mut() = Vec::new();

        let mut reexports = Vec::new();
        for (&(name, ns), resolution) in module.resolutions.borrow().iter() {
            let resolution = resolution.borrow();
            let binding = match resolution.binding {
                Some(binding) => binding,
                None => continue,
            };

            // Report conflicts
            for duplicate_glob in resolution.duplicate_globs.iter() {
                // FIXME #31337: We currently allow items to shadow glob-imported re-exports.
                if !binding.is_import() {
                    if let NameBindingKind::Import { binding, .. } = duplicate_glob.kind {
                        if binding.is_import() { continue }
                    }
                }

                self.report_conflict(module, name, ns, duplicate_glob, binding);
            }

            if binding.vis == ty::Visibility::Public &&
               (binding.is_import() || binding.is_extern_crate()) {
                if let Some(def) = binding.def() {
                    reexports.push(Export { name: name, def_id: def.def_id() });
                }
            }

            if let NameBindingKind::Import { binding: orig_binding, directive, .. } = binding.kind {
                if ns == TypeNS && orig_binding.is_variant() &&
                   !orig_binding.vis.is_at_least(binding.vis, self) {
                    let msg = format!("variant `{}` is private, and cannot be reexported \
                                       (error E0364), consider declaring its enum as `pub`",
                                      name);
                    self.session.add_lint(PRIVATE_IN_PUBLIC, directive.id, binding.span, msg);
                }
            }
        }

        if reexports.len() > 0 {
            if let Some(def_id) = module.def_id() {
                let node_id = self.definitions.as_local_node_id(def_id).unwrap();
                self.export_map.insert(node_id, reexports);
            }
        }

        if report_unresolved_imports {
            for import in module.unresolved_imports.borrow().iter() {
                resolve_error(self.resolver, import.span, ResolutionError::UnresolvedImport(None));
                break;
            }
        }
    }
}

fn import_path_to_string(names: &[Name], subclass: &ImportDirectiveSubclass) -> String {
    if names.is_empty() {
        import_directive_subclass_to_string(subclass)
    } else {
        (format!("{}::{}",
                 names_to_string(names),
                 import_directive_subclass_to_string(subclass)))
            .to_string()
    }
}

fn import_directive_subclass_to_string(subclass: &ImportDirectiveSubclass) -> String {
    match *subclass {
        SingleImport { source, .. } => source.to_string(),
        GlobImport { .. } => "*".to_string(),
    }
}
