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

use DefModifiers;
use Module;
use Namespace::{self, TypeNS, ValueNS};
use NameBinding;
use ResolveResult;
use ResolveResult::*;
use Resolver;
use UseLexicalScopeFlag;
use {names_to_string, module_to_string};
use {resolve_error, ResolutionError};

use build_reduced_graph;

use rustc::lint;
use rustc::middle::def::*;
use rustc::middle::def_id::DefId;
use rustc::middle::privacy::*;

use syntax::ast::{NodeId, Name};
use syntax::attr::AttrMetaMethods;
use syntax::codemap::Span;
use syntax::util::lev_distance::find_best_match_for_name;

use std::mem::replace;

/// Contains data for specific types of import directives.
#[derive(Copy, Clone,Debug)]
pub enum ImportDirectiveSubclass {
    SingleImport(Name /* target */, Name /* source */),
    GlobImport,
}

/// Whether an import can be shadowed by another import.
#[derive(Debug,PartialEq,Clone,Copy)]
pub enum Shadowable {
    Always,
    Never,
}

/// One import directive.
#[derive(Debug)]
pub struct ImportDirective {
    pub module_path: Vec<Name>,
    pub subclass: ImportDirectiveSubclass,
    pub span: Span,
    pub id: NodeId,
    pub is_public: bool, // see note in ImportResolutionPerNamespace about how to use this
    pub shadowable: Shadowable,
}

impl ImportDirective {
    pub fn new(module_path: Vec<Name>,
               subclass: ImportDirectiveSubclass,
               span: Span,
               id: NodeId,
               is_public: bool,
               shadowable: Shadowable)
               -> ImportDirective {
        ImportDirective {
            module_path: module_path,
            subclass: subclass,
            span: span,
            id: id,
            is_public: is_public,
            shadowable: shadowable,
        }
    }
}

/// The item that an import resolves to.
#[derive(Clone,Debug)]
pub struct Target<'a> {
    pub target_module: Module<'a>,
    pub binding: NameBinding<'a>,
    pub shadowable: Shadowable,
}

impl<'a> Target<'a> {
    pub fn new(target_module: Module<'a>, binding: NameBinding<'a>, shadowable: Shadowable)
               -> Self {
        Target {
            target_module: target_module,
            binding: binding,
            shadowable: shadowable,
        }
    }
}

#[derive(Debug)]
/// An ImportResolution records what we know about an imported name in a given namespace.
/// More specifically, it records the number of unresolved `use` directives that import the name,
/// the `use` directive importing the name in the namespace, and the `NameBinding` to which the
/// name in the namespace resolves (if applicable).
/// Different `use` directives may import the same name in different namespaces.
pub struct ImportResolution<'a> {
    // When outstanding_references reaches zero, outside modules can count on the targets being
    // correct. Before then, all bets are off; future `use` directives could override the name.
    // Since shadowing is forbidden, the only way outstanding_references > 1 in a legal program
    // is if the name is imported by exactly two `use` directives, one of which resolves to a
    // value and the other of which resolves to a type.
    pub outstanding_references: usize,

    /// Whether this resolution came from a `use` or a `pub use`.
    pub is_public: bool,

    /// Resolution of the name in the namespace
    pub target: Option<Target<'a>>,

    /// The source node of the `use` directive
    pub id: NodeId,
}

impl<'a> ImportResolution<'a> {
    pub fn new(id: NodeId, is_public: bool) -> Self {
        ImportResolution {
            outstanding_references: 0,
            id: id,
            target: None,
            is_public: is_public,
        }
    }

    pub fn shadowable(&self) -> Shadowable {
        match self.target {
            Some(ref target) => target.shadowable,
            None => Shadowable::Always,
        }
    }
}

struct ImportResolvingError {
    span: Span,
    path: String,
    help: String,
}

struct ImportResolver<'a, 'b: 'a, 'tcx: 'b> {
    resolver: &'a mut Resolver<'b, 'tcx>,
}

impl<'a, 'b:'a, 'tcx:'b> ImportResolver<'a, 'b, 'tcx> {
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
        loop {
            debug!("(resolving imports) iteration {}, {} imports left",
                   i,
                   self.resolver.unresolved_imports);

            let module_root = self.resolver.graph_root;
            let errors = self.resolve_imports_for_module_subtree(module_root);

            if self.resolver.unresolved_imports == 0 {
                debug!("(resolving imports) success");
                break;
            }

            if self.resolver.unresolved_imports == prev_unresolved_imports {
                // resolving failed
                if errors.len() > 0 {
                    for e in errors {
                        resolve_error(self.resolver,
                                      e.span,
                                      ResolutionError::UnresolvedImport(Some((&e.path, &e.help))));
                    }
                } else {
                    // Report unresolved imports only if no hard error was already reported
                    // to avoid generating multiple errors on the same import.
                    // Imports that are still indeterminate at this point are actually blocked
                    // by errored imports, so there is no point reporting them.
                    self.resolver.report_unresolved_imports(module_root);
                }
                break;
            }

            i += 1;
            prev_unresolved_imports = self.resolver.unresolved_imports;
        }
    }

    /// Attempts to resolve imports for the given module and all of its
    /// submodules.
    fn resolve_imports_for_module_subtree(&mut self,
                                          module_: Module<'b>)
                                          -> Vec<ImportResolvingError> {
        let mut errors = Vec::new();
        debug!("(resolving imports for module subtree) resolving {}",
               module_to_string(&*module_));
        let orig_module = replace(&mut self.resolver.current_module, module_);
        errors.extend(self.resolve_imports_for_module(module_));
        self.resolver.current_module = orig_module;

        build_reduced_graph::populate_module_if_necessary(self.resolver, &module_);
        for (_, child_node) in module_.children.borrow().iter() {
            match child_node.module() {
                None => {
                    // Nothing to do.
                }
                Some(child_module) => {
                    errors.extend(self.resolve_imports_for_module_subtree(child_module));
                }
            }
        }

        for (_, child_module) in module_.anonymous_children.borrow().iter() {
            errors.extend(self.resolve_imports_for_module_subtree(child_module));
        }

        errors
    }

    /// Attempts to resolve imports for the given module only.
    fn resolve_imports_for_module(&mut self, module: Module<'b>) -> Vec<ImportResolvingError> {
        let mut errors = Vec::new();

        if module.all_imports_resolved() {
            debug!("(resolving imports for module) all imports resolved for {}",
                   module_to_string(&*module));
            return errors;
        }

        let mut imports = module.imports.borrow_mut();
        let import_count = imports.len();
        let mut indeterminate_imports = Vec::new();
        while module.resolved_import_count.get() + indeterminate_imports.len() < import_count {
            let import_index = module.resolved_import_count.get();
            match self.resolve_import_for_module(module, &imports[import_index]) {
                ResolveResult::Failed(err) => {
                    let import_directive = &imports[import_index];
                    let (span, help) = match err {
                        Some((span, msg)) => (span, format!(". {}", msg)),
                        None => (import_directive.span, String::new()),
                    };
                    errors.push(ImportResolvingError {
                        span: span,
                        path: import_path_to_string(&import_directive.module_path,
                                                    import_directive.subclass),
                        help: help,
                    });
                }
                ResolveResult::Indeterminate => {}
                ResolveResult::Success(()) => {
                    // count success
                    module.resolved_import_count
                          .set(module.resolved_import_count.get() + 1);
                    continue;
                }
            }
            // This resolution was not successful, keep it for later
            indeterminate_imports.push(imports.swap_remove(import_index));

        }

        imports.extend(indeterminate_imports);

        errors
    }

    /// Attempts to resolve the given import. The return value indicates
    /// failure if we're certain the name does not exist, indeterminate if we
    /// don't know whether the name exists at the moment due to other
    /// currently-unresolved imports, or success if we know the name exists.
    /// If successful, the resolved bindings are written into the module.
    fn resolve_import_for_module(&mut self,
                                 module_: Module<'b>,
                                 import_directive: &ImportDirective)
                                 -> ResolveResult<()> {
        let mut resolution_result = ResolveResult::Failed(None);
        let module_path = &import_directive.module_path;

        debug!("(resolving import for module) resolving import `{}::...` in `{}`",
               names_to_string(&module_path[..]),
               module_to_string(&*module_));

        // First, resolve the module path for the directive, if necessary.
        let container = if module_path.is_empty() {
            // Use the crate root.
            Some((self.resolver.graph_root, LastMod(AllPublic)))
        } else {
            match self.resolver.resolve_module_path(module_,
                                                    &module_path[..],
                                                    UseLexicalScopeFlag::DontUseLexicalScope,
                                                    import_directive.span) {
                ResolveResult::Failed(err) => {
                    resolution_result = ResolveResult::Failed(err);
                    None
                }
                ResolveResult::Indeterminate => {
                    resolution_result = ResolveResult::Indeterminate;
                    None
                }
                ResolveResult::Success(container) => Some(container),
            }
        };

        match container {
            None => {}
            Some((containing_module, lp)) => {
                // We found the module that the target is contained
                // within. Attempt to resolve the import within it.

                match import_directive.subclass {
                    SingleImport(target, source) => {
                        resolution_result = self.resolve_single_import(&module_,
                                                                       containing_module,
                                                                       target,
                                                                       source,
                                                                       import_directive,
                                                                       lp);
                    }
                    GlobImport => {
                        resolution_result = self.resolve_glob_import(&module_,
                                                                     containing_module,
                                                                     import_directive,
                                                                     lp);
                    }
                }
            }
        }

        // Decrement the count of unresolved imports.
        match resolution_result {
            ResolveResult::Success(()) => {
                assert!(self.resolver.unresolved_imports >= 1);
                self.resolver.unresolved_imports -= 1;
            }
            _ => {
                // Nothing to do here; just return the error.
            }
        }

        // Decrement the count of unresolved globs if necessary. But only if
        // the resolution result is a success -- other cases will
        // be handled by the main loop.

        if resolution_result.success() {
            match import_directive.subclass {
                GlobImport => {
                    module_.dec_glob_count();
                    if import_directive.is_public {
                        module_.dec_pub_glob_count();
                    }
                }
                SingleImport(..) => {
                    // Ignore.
                }
            }
            if import_directive.is_public {
                module_.dec_pub_count();
            }
        }

        return resolution_result;
    }

    /// Resolves the name in the namespace of the module because it is being imported by
    /// importing_module. Returns the module in which the name was defined (as opposed to imported),
    /// the name bindings defining the name, and whether or not the name was imported into `module`.
    fn resolve_name_in_module(&mut self,
                              module: Module<'b>, // Module containing the name
                              name: Name,
                              ns: Namespace,
                              importing_module: Module<'b>) // Module importing the name
                              -> (ResolveResult<(Module<'b>, NameBinding<'b>)>, bool) {
        build_reduced_graph::populate_module_if_necessary(self.resolver, module);
        if let Some(name_binding) = module.get_child(name, ns) {
            return (Success((module, name_binding)), false);
        }

        if ns == TypeNS {
            if let Some(extern_crate) = module.external_module_children.borrow().get(&name) {
                // track the extern crate as used.
                if let Some(DefId{ krate: kid, .. }) = extern_crate.def_id() {
                    self.resolver.used_crates.insert(kid);
                }
                let name_binding = NameBinding::create_from_module(extern_crate, None);
                return (Success((module, name_binding)), false);
            }
        }

        // If there is an unresolved glob at this point in the containing module, bail out.
        // We don't know enough to be able to resolve the name.
        if module.pub_glob_count.get() > 0 {
            return (Indeterminate, false);
        }

        match module.import_resolutions.borrow().get(&(name, ns)) {
            // The containing module definitely doesn't have an exported import with the
            // name in question. We can therefore accurately report that names are unbound.
            None => (Failed(None), false),

            // The name is an import which has been fully resolved, so we just follow it.
            Some(resolution) if resolution.outstanding_references == 0 => {
                // Import resolutions must be declared with "pub" in order to be exported.
                if !resolution.is_public {
                    return (Failed(None), false);
                }

                let target = resolution.target.clone();
                if let Some(Target { target_module, binding, shadowable: _ }) = target {
                    // track used imports and extern crates as well
                    self.resolver.used_imports.insert((resolution.id, ns));
                    self.resolver.record_import_use(resolution.id, name);
                    if let Some(DefId { krate, .. }) = target_module.def_id() {
                        self.resolver.used_crates.insert(krate);
                    }
                    (Success((target_module, binding)), true)
                } else {
                    (Failed(None), false)
                }
            }

            // If module is the same module whose import we are resolving and
            // it has an unresolved import with the same name as `name`, then the user
            // is actually trying to import an item that is declared in the same scope
            //
            // e.g
            // use self::submodule;
            // pub mod submodule;
            //
            // In this case we continue as if we resolved the import and let
            // check_for_conflicts_between_imports_and_items handle the conflict
            Some(_) => match (importing_module.def_id(), module.def_id()) {
                (Some(id1), Some(id2)) if id1 == id2 => (Failed(None), false),
                _ => (Indeterminate, false)
            },
        }
    }

    fn resolve_single_import(&mut self,
                             module_: Module<'b>,
                             target_module: Module<'b>,
                             target: Name,
                             source: Name,
                             directive: &ImportDirective,
                             lp: LastPrivate)
                             -> ResolveResult<()> {
        debug!("(resolving single import) resolving `{}` = `{}::{}` from `{}` id {}, last \
                private {:?}",
               target,
               module_to_string(&*target_module),
               source,
               module_to_string(module_),
               directive.id,
               lp);

        let lp = match lp {
            LastMod(lp) => lp,
            LastImport {..} => {
                self.resolver
                    .session
                    .span_bug(directive.span, "not expecting Import here, must be LastMod")
            }
        };

        // We need to resolve both namespaces for this to succeed.
        let (value_result, value_used_reexport) =
            self.resolve_name_in_module(&target_module, source, ValueNS, module_);
        let (type_result, type_used_reexport) =
            self.resolve_name_in_module(&target_module, source, TypeNS, module_);

        match (&value_result, &type_result) {
            (&Success((_, ref name_binding)), _) if !value_used_reexport &&
                                                    directive.is_public &&
                                                    !name_binding.is_public() => {
                let msg = format!("`{}` is private, and cannot be reexported", source);
                let note_msg = format!("Consider marking `{}` as `pub` in the imported module",
                                        source);
                struct_span_err!(self.resolver.session, directive.span, E0364, "{}", &msg)
                    .span_note(directive.span, &note_msg)
                    .emit();
            }

            (_, &Success((_, ref name_binding))) if !type_used_reexport &&
                                                    directive.is_public => {
                if !name_binding.is_public() {
                    let msg = format!("`{}` is private, and cannot be reexported", source);
                    let note_msg =
                        format!("Consider declaring type or module `{}` with `pub`", source);
                    struct_span_err!(self.resolver.session, directive.span, E0365, "{}", &msg)
                        .span_note(directive.span, &note_msg)
                        .emit();
                } else if name_binding.defined_with(DefModifiers::PRIVATE_VARIANT) {
                    let msg = format!("variant `{}` is private, and cannot be reexported \
                                       (error E0364), consider declaring its enum as `pub`",
                                       source);
                    self.resolver.session.add_lint(lint::builtin::PRIVATE_IN_PUBLIC,
                                                   directive.id,
                                                   directive.span,
                                                   msg);
                }
            }

            _ => {}
        }

        let mut lev_suggestion = "".to_owned();
        match (&value_result, &type_result) {
            (&Indeterminate, _) | (_, &Indeterminate) => return Indeterminate,
            (&Failed(_), &Failed(_)) => {
                let children = target_module.children.borrow();
                let names = children.keys().map(|&(ref name, _)| name);
                if let Some(name) = find_best_match_for_name(names, &source.as_str(), None) {
                    lev_suggestion = format!(". Did you mean to use `{}`?", name);
                } else {
                    let resolutions = target_module.import_resolutions.borrow();
                    let names = resolutions.keys().map(|&(ref name, _)| name);
                    if let Some(name) = find_best_match_for_name(names,
                                                                 &source.as_str(),
                                                                 None) {
                        lev_suggestion =
                            format!(". Did you mean to use the re-exported import `{}`?", name);
                    }
                }
            }
            _ => (),
        }

        let mut value_used_public = false;
        let mut type_used_public = false;

        // We've successfully resolved the import. Write the results in.
        let mut import_resolutions = module_.import_resolutions.borrow_mut();

        {
            let mut check_and_write_import = |namespace, result, used_public: &mut bool| {
                let result: &ResolveResult<(Module<'b>, NameBinding)> = result;

                let import_resolution = import_resolutions.get_mut(&(target, namespace)).unwrap();
                let namespace_name = match namespace {
                    TypeNS => "type",
                    ValueNS => "value",
                };

                match *result {
                    Success((ref target_module, ref name_binding)) => {
                        debug!("(resolving single import) found {:?} target: {:?}",
                               namespace_name,
                               name_binding.def());
                        self.check_for_conflicting_import(&import_resolution,
                                                          directive.span,
                                                          target,
                                                          namespace);

                        self.check_that_import_is_importable(&name_binding,
                                                             directive.span,
                                                             target);

                        import_resolution.target = Some(Target::new(target_module,
                                                                    name_binding.clone(),
                                                                    directive.shadowable));
                        import_resolution.id = directive.id;
                        import_resolution.is_public = directive.is_public;

                        self.add_export(module_, target, &import_resolution);
                        *used_public = name_binding.is_public();
                    }
                    Failed(_) => {
                        // Continue.
                    }
                    Indeterminate => {
                        panic!("{:?} result should be known at this point", namespace_name);
                    }
                }

                self.check_for_conflicts_between_imports_and_items(module_,
                                                                   import_resolution,
                                                                   directive.span,
                                                                   (target, namespace));
            };
            check_and_write_import(ValueNS, &value_result, &mut value_used_public);
            check_and_write_import(TypeNS, &type_result, &mut type_used_public);
        }

        if let (&Failed(_), &Failed(_)) = (&value_result, &type_result) {
            let msg = format!("There is no `{}` in `{}`{}",
                              source,
                              module_to_string(&target_module), lev_suggestion);
            return Failed(Some((directive.span, msg)));
        }

        let value_used_public = value_used_reexport || value_used_public;
        let type_used_public = type_used_reexport || type_used_public;

        let value_def_and_priv = {
            let import_resolution_value = import_resolutions.get_mut(&(target, ValueNS)).unwrap();
            assert!(import_resolution_value.outstanding_references >= 1);
            import_resolution_value.outstanding_references -= 1;

            // Record what this import resolves to for later uses in documentation,
            // this may resolve to either a value or a type, but for documentation
            // purposes it's good enough to just favor one over the other.
            import_resolution_value.target.as_ref().map(|target| {
                let def = target.binding.def().unwrap();
                let last_private = if value_used_public { lp } else { DependsOn(def.def_id()) };
                (def, last_private)
            })
        };

        let type_def_and_priv = {
            let import_resolution_type = import_resolutions.get_mut(&(target, TypeNS)).unwrap();
            assert!(import_resolution_type.outstanding_references >= 1);
            import_resolution_type.outstanding_references -= 1;

            import_resolution_type.target.as_ref().map(|target| {
                let def = target.binding.def().unwrap();
                let last_private = if type_used_public { lp } else { DependsOn(def.def_id()) };
                (def, last_private)
            })
        };

        let import_lp = LastImport {
            value_priv: value_def_and_priv.map(|(_, p)| p),
            value_used: Used,
            type_priv: type_def_and_priv.map(|(_, p)| p),
            type_used: Used,
        };

        if let Some((def, _)) = value_def_and_priv {
            self.resolver.def_map.borrow_mut().insert(directive.id,
                                                      PathResolution {
                                                          base_def: def,
                                                          last_private: import_lp,
                                                          depth: 0,
                                                      });
        }
        if let Some((def, _)) = type_def_and_priv {
            self.resolver.def_map.borrow_mut().insert(directive.id,
                                                      PathResolution {
                                                          base_def: def,
                                                          last_private: import_lp,
                                                          depth: 0,
                                                      });
        }

        debug!("(resolving single import) successfully resolved import");
        return Success(());
    }

    // Resolves a glob import. Note that this function cannot fail; it either
    // succeeds or bails out (as importing * from an empty module or a module
    // that exports nothing is valid). target_module is the module we are
    // actually importing, i.e., `foo` in `use foo::*`.
    fn resolve_glob_import(&mut self,
                           module_: Module<'b>,
                           target_module: Module<'b>,
                           import_directive: &ImportDirective,
                           lp: LastPrivate)
                           -> ResolveResult<()> {
        let id = import_directive.id;
        let is_public = import_directive.is_public;

        // This function works in a highly imperative manner; it eagerly adds
        // everything it can to the list of import resolutions of the module
        // node.
        debug!("(resolving glob import) resolving glob import {}", id);

        // We must bail out if the node has unresolved imports of any kind
        // (including globs).
        if (*target_module).pub_count.get() > 0 {
            debug!("(resolving glob import) target module has unresolved pub imports; bailing out");
            return ResolveResult::Indeterminate;
        }

        // Add all resolved imports from the containing module.
        let import_resolutions = target_module.import_resolutions.borrow();

        if module_.import_resolutions.borrow_state() != ::std::cell::BorrowState::Unused {
            // In this case, target_module == module_
            // This means we are trying to glob import a module into itself,
            // and it is a no-go
            debug!("(resolving glob imports) target module is current module; giving up");
            return ResolveResult::Failed(Some((import_directive.span,
                                               "Cannot glob-import a module into itself.".into())));
        }

        for (&(name, ns), target_import_resolution) in import_resolutions.iter() {
            debug!("(resolving glob import) writing module resolution {} into `{}`",
                   name,
                   module_to_string(module_));

            // Here we merge two import resolutions.
            let mut import_resolutions = module_.import_resolutions.borrow_mut();
            let mut dest_import_resolution =
                import_resolutions.entry((name, ns))
                                  .or_insert_with(|| ImportResolution::new(id, is_public));

            match target_import_resolution.target {
                Some(ref target) if target_import_resolution.is_public => {
                    self.check_for_conflicting_import(&dest_import_resolution,
                                                      import_directive.span,
                                                      name,
                                                      ns);
                    dest_import_resolution.id = id;
                    dest_import_resolution.is_public = is_public;
                    dest_import_resolution.target = Some(target.clone());
                    self.add_export(module_, name, &dest_import_resolution);
                }
                _ => {}
            }
        }

        // Add all children from the containing module.
        build_reduced_graph::populate_module_if_necessary(self.resolver, &target_module);

        for (&name, name_binding) in target_module.children.borrow().iter() {
            self.merge_import_resolution(module_,
                                         target_module,
                                         import_directive,
                                         name,
                                         name_binding.clone());
        }

        // Record the destination of this import
        if let Some(did) = target_module.def_id() {
            self.resolver.def_map.borrow_mut().insert(id,
                                                      PathResolution {
                                                          base_def: Def::Mod(did),
                                                          last_private: lp,
                                                          depth: 0,
                                                      });
        }

        debug!("(resolving glob import) successfully resolved import");
        return ResolveResult::Success(());
    }

    fn merge_import_resolution(&mut self,
                               module_: Module<'b>,
                               containing_module: Module<'b>,
                               import_directive: &ImportDirective,
                               (name, ns): (Name, Namespace),
                               name_binding: NameBinding<'b>) {
        let id = import_directive.id;
        let is_public = import_directive.is_public;

        let mut import_resolutions = module_.import_resolutions.borrow_mut();
        let dest_import_resolution = import_resolutions.entry((name, ns)).or_insert_with(|| {
            ImportResolution::new(id, is_public)
        });

        debug!("(resolving glob import) writing resolution `{}` in `{}` to `{}`",
               name,
               module_to_string(&*containing_module),
               module_to_string(module_));

        // Merge the child item into the import resolution.
        let modifier = DefModifiers::IMPORTABLE | DefModifiers::PUBLIC;

        if ns == TypeNS && is_public && name_binding.defined_with(DefModifiers::PRIVATE_VARIANT) {
            let msg = format!("variant `{}` is private, and cannot be reexported (error \
                               E0364), consider declaring its enum as `pub`", name);
            self.resolver.session.add_lint(lint::builtin::PRIVATE_IN_PUBLIC,
                                           import_directive.id,
                                           import_directive.span,
                                           msg);
        }

        if name_binding.defined_with(modifier) {
            let namespace_name = match ns {
                TypeNS => "type",
                ValueNS => "value",
            };
            debug!("(resolving glob import) ... for {} target", namespace_name);
            if dest_import_resolution.shadowable() == Shadowable::Never {
                let msg = format!("a {} named `{}` has already been imported in this module",
                                 namespace_name,
                                 name);
                span_err!(self.resolver.session, import_directive.span, E0251, "{}", msg);
           } else {
                let target = Target::new(containing_module,
                                         name_binding.clone(),
                                         import_directive.shadowable);
                dest_import_resolution.target = Some(target);
                dest_import_resolution.id = id;
                dest_import_resolution.is_public = is_public;
                self.add_export(module_, name, &dest_import_resolution);
            }
        } else {
            // FIXME #30159: This is required for backwards compatability.
            dest_import_resolution.is_public |= is_public;
        }

        self.check_for_conflicts_between_imports_and_items(module_,
                                                           dest_import_resolution,
                                                           import_directive.span,
                                                           (name, ns));
    }

    fn add_export(&mut self, module: Module<'b>, name: Name, resolution: &ImportResolution<'b>) {
        if !resolution.is_public { return }
        let node_id = match module.def_id() {
            Some(def_id) => self.resolver.ast_map.as_local_node_id(def_id).unwrap(),
            None => return,
        };
        let export = match resolution.target.as_ref().unwrap().binding.def() {
            Some(def) => Export { name: name, def_id: def.def_id() },
            None => return,
        };
        self.resolver.export_map.entry(node_id).or_insert(Vec::new()).push(export);
    }

    /// Checks that imported names and items don't have the same name.
    fn check_for_conflicting_import(&mut self,
                                    import_resolution: &ImportResolution,
                                    import_span: Span,
                                    name: Name,
                                    namespace: Namespace) {
        let target = &import_resolution.target;
        debug!("check_for_conflicting_import: {}; target exists: {}",
               name,
               target.is_some());

        match *target {
            Some(ref target) if target.shadowable != Shadowable::Always => {
                let ns_word = match namespace {
                    TypeNS => {
                        match target.binding.module() {
                            Some(ref module) if module.is_normal() => "module",
                            Some(ref module) if module.is_trait() => "trait",
                            _ => "type",
                        }
                    }
                    ValueNS => "value",
                };
                let use_id = import_resolution.id;
                let item = self.resolver.ast_map.expect_item(use_id);
                let mut err = struct_span_err!(self.resolver.session,
                                               import_span,
                                               E0252,
                                               "a {} named `{}` has already been imported \
                                                in this module",
                                               ns_word,
                                               name);
                span_note!(&mut err,
                           item.span,
                           "previous import of `{}` here",
                           name);
                err.emit();
            }
            Some(_) | None => {}
        }
    }

    /// Checks that an import is actually importable
    fn check_that_import_is_importable(&mut self,
                                       name_binding: &NameBinding,
                                       import_span: Span,
                                       name: Name) {
        if !name_binding.defined_with(DefModifiers::IMPORTABLE) {
            let msg = format!("`{}` is not directly importable", name);
            span_err!(self.resolver.session, import_span, E0253, "{}", &msg[..]);
        }
    }

    /// Checks that imported names and items don't have the same name.
    fn check_for_conflicts_between_imports_and_items(&mut self,
                                                     module: Module<'b>,
                                                     import: &ImportResolution<'b>,
                                                     import_span: Span,
                                                     (name, ns): (Name, Namespace)) {
        // First, check for conflicts between imports and `extern crate`s.
        if ns == TypeNS {
            if module.external_module_children.borrow().contains_key(&name) {
                match import.target {
                    Some(ref target) if target.shadowable != Shadowable::Always => {
                        let msg = format!("import `{0}` conflicts with imported crate \
                                           in this module (maybe you meant `use {0}::*`?)",
                                          name);
                        span_err!(self.resolver.session, import_span, E0254, "{}", &msg[..]);
                    }
                    Some(_) | None => {}
                }
            }
        }

        // Check for item conflicts.
        let name_binding = match module.get_child(name, ns) {
            None => {
                // There can't be any conflicts.
                return;
            }
            Some(name_binding) => name_binding,
        };

        if ns == ValueNS {
            match import.target {
                Some(ref target) if target.shadowable != Shadowable::Always => {
                    let mut err = struct_span_err!(self.resolver.session,
                                                   import_span,
                                                   E0255,
                                                   "import `{}` conflicts with \
                                                    value in this module",
                                                   name);
                    if let Some(span) = name_binding.span {
                        err.span_note(span, "conflicting value here");
                    }
                    err.emit();
                }
                Some(_) | None => {}
            }
        } else {
            match import.target {
                Some(ref target) if target.shadowable != Shadowable::Always => {
                    let (what, note) = match name_binding.module() {
                        Some(ref module) if module.is_normal() =>
                            ("existing submodule", "note conflicting module here"),
                        Some(ref module) if module.is_trait() =>
                            ("trait in this module", "note conflicting trait here"),
                        _ => ("type in this module", "note conflicting type here"),
                    };
                    let mut err = struct_span_err!(self.resolver.session,
                                                   import_span,
                                                   E0256,
                                                   "import `{}` conflicts with {}",
                                                   name,
                                                   what);
                    if let Some(span) = name_binding.span {
                        err.span_note(span, note);
                    }
                    err.emit();
                }
                Some(_) | None => {}
            }
        }
    }
}

fn import_path_to_string(names: &[Name], subclass: ImportDirectiveSubclass) -> String {
    if names.is_empty() {
        import_directive_subclass_to_string(subclass)
    } else {
        (format!("{}::{}",
                 names_to_string(names),
                 import_directive_subclass_to_string(subclass)))
            .to_string()
    }
}

fn import_directive_subclass_to_string(subclass: ImportDirectiveSubclass) -> String {
    match subclass {
        SingleImport(_, source) => source.to_string(),
        GlobImport => "*".to_string(),
    }
}

pub fn resolve_imports(resolver: &mut Resolver) {
    let mut import_resolver = ImportResolver { resolver: resolver };
    import_resolver.resolve_imports();
}
