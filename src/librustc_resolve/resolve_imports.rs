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
use {NameBinding, NameBindingKind};
use ResolveResult;
use ResolveResult::*;
use Resolver;
use UseLexicalScopeFlag;
use {names_to_string, module_to_string};
use {resolve_error, ResolutionError};

use build_reduced_graph;

use rustc::lint;
use rustc::middle::def::*;

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
#[derive(Debug,Clone)]
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

    // Given the binding to which this directive resolves in a particular namespace,
    // this returns the binding for the name this directive defines in that namespace.
    fn import<'a>(&self, binding: &'a NameBinding<'a>) -> NameBinding<'a> {
        let mut modifiers = match self.is_public {
            true => DefModifiers::PUBLIC | DefModifiers::IMPORTABLE,
            false => DefModifiers::empty(),
        };
        if let GlobImport = self.subclass {
            modifiers = modifiers | DefModifiers::GLOB_IMPORTED;
        }
        if self.shadowable == Shadowable::Always {
            modifiers = modifiers | DefModifiers::PRELUDE;
        }

        NameBinding {
            kind: NameBindingKind::Import { binding: binding, id: self.id },
            span: Some(self.span),
            modifiers: modifiers,
        }
    }
}

#[derive(Clone, Default)]
/// Records information about the resolution of a name in a module.
pub struct NameResolution<'a> {
    /// The number of unresolved single imports that could define the name.
    pub outstanding_references: usize,
    /// The least shadowable known binding for this name, or None if there are no known bindings.
    pub binding: Option<&'a NameBinding<'a>>,
}

impl<'a> NameResolution<'a> {
    pub fn result(&self, outstanding_globs: usize) -> ResolveResult<&'a NameBinding<'a>> {
        // If no unresolved imports (single or glob) can define the name, self.binding is final.
        if self.outstanding_references == 0 && outstanding_globs == 0 {
            return self.binding.map(Success).unwrap_or(Failed(None));
        }

        if let Some(binding) = self.binding {
            // Single imports will never be shadowable by other single or glob imports.
            if !binding.defined_with(DefModifiers::GLOB_IMPORTED) { return Success(binding); }
            // Non-PRELUDE glob imports will never be shadowable by other glob imports.
            if self.outstanding_references == 0 && !binding.defined_with(DefModifiers::PRELUDE) {
                return Success(binding);
            }
        }

        Indeterminate
    }

    // Define the name or return the existing binding if there is a collision.
    pub fn try_define(&mut self, binding: &'a NameBinding<'a>) -> Result<(), &'a NameBinding<'a>> {
        let is_prelude = |binding: &NameBinding| binding.defined_with(DefModifiers::PRELUDE);
        let old_binding = match self.binding {
            Some(_) if is_prelude(binding) => return Ok(()),
            Some(old_binding) if !is_prelude(old_binding) => old_binding,
            _ => { self.binding = Some(binding); return Ok(()); }
        };

        // FIXME #31337: We currently allow items to shadow glob-imported re-exports.
        if !old_binding.is_import() && binding.defined_with(DefModifiers::GLOB_IMPORTED) {
            if let NameBindingKind::Import { binding, .. } = binding.kind {
                if binding.is_import() { return Ok(()); }
            }
        }

        Err(old_binding)
    }
}

struct ImportResolvingError<'a> {
    /// Module where the error happened
    source_module: Module<'a>,
    import_directive: ImportDirective,
    span: Span,
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
        let mut errors = Vec::new();

        loop {
            debug!("(resolving imports) iteration {}, {} imports left",
                   i,
                   self.resolver.unresolved_imports);

            self.resolve_imports_for_module_subtree(self.resolver.graph_root, &mut errors);

            if self.resolver.unresolved_imports == 0 {
                debug!("(resolving imports) success");
                break;
            }

            if self.resolver.unresolved_imports == prev_unresolved_imports {
                // resolving failed
                if errors.len() > 0 {
                    for e in errors {
                        self.import_resolving_error(e)
                    }
                } else {
                    // Report unresolved imports only if no hard error was already reported
                    // to avoid generating multiple errors on the same import.
                    // Imports that are still indeterminate at this point are actually blocked
                    // by errored imports, so there is no point reporting them.
                    self.resolver.report_unresolved_imports(self.resolver.graph_root);
                }
                break;
            }

            i += 1;
            prev_unresolved_imports = self.resolver.unresolved_imports;
        }
    }

    /// Resolves an `ImportResolvingError` into the correct enum discriminant
    /// and passes that on to `resolve_error`.
    fn import_resolving_error(&self, e: ImportResolvingError<'b>) {
        // If it's a single failed import then create a "fake" import
        // resolution for it so that later resolve stages won't complain.
        if let SingleImport(target, _) = e.import_directive.subclass {
            let dummy_binding = self.resolver.new_name_binding(NameBinding {
                modifiers: DefModifiers::PRELUDE,
                kind: NameBindingKind::Def(Def::Err),
                span: None,
            });
            let dummy_binding =
                self.resolver.new_name_binding(e.import_directive.import(dummy_binding));

            let _ = e.source_module.try_define_child(target, ValueNS, dummy_binding);
            let _ = e.source_module.try_define_child(target, TypeNS, dummy_binding);
        }

        let path = import_path_to_string(&e.import_directive.module_path,
                                         e.import_directive.subclass);

        resolve_error(self.resolver,
                      e.span,
                      ResolutionError::UnresolvedImport(Some((&path, &e.help))));
    }

    /// Attempts to resolve imports for the given module and all of its
    /// submodules.
    fn resolve_imports_for_module_subtree(&mut self,
                                          module_: Module<'b>,
                                          errors: &mut Vec<ImportResolvingError<'b>>) {
        debug!("(resolving imports for module subtree) resolving {}",
               module_to_string(&module_));
        let orig_module = replace(&mut self.resolver.current_module, module_);
        self.resolve_imports_for_module(module_, errors);
        self.resolver.current_module = orig_module;

        for (_, child_module) in module_.module_children.borrow().iter() {
            self.resolve_imports_for_module_subtree(child_module, errors);
        }
    }

    /// Attempts to resolve imports for the given module only.
    fn resolve_imports_for_module(&mut self,
                                  module: Module<'b>,
                                  errors: &mut Vec<ImportResolvingError<'b>>) {
        let mut imports = Vec::new();
        let mut unresolved_imports = module.unresolved_imports.borrow_mut();
        ::std::mem::swap(&mut imports, &mut unresolved_imports);

        for import_directive in imports {
            match self.resolve_import_for_module(module, &import_directive) {
                Failed(err) => {
                    let (span, help) = match err {
                        Some((span, msg)) => (span, format!(". {}", msg)),
                        None => (import_directive.span, String::new()),
                    };
                    errors.push(ImportResolvingError {
                        source_module: module,
                        import_directive: import_directive,
                        span: span,
                        help: help,
                    });
                }
                Indeterminate => unresolved_imports.push(import_directive),
                Success(()) => {}
            }
        }
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
        debug!("(resolving import for module) resolving import `{}::...` in `{}`",
               names_to_string(&import_directive.module_path),
               module_to_string(&module_));

        self.resolver
            .resolve_module_path(module_,
                                 &import_directive.module_path,
                                 UseLexicalScopeFlag::DontUseLexicalScope,
                                 import_directive.span)
            .and_then(|containing_module| {
                // We found the module that the target is contained
                // within. Attempt to resolve the import within it.
                if let SingleImport(target, source) = import_directive.subclass {
                    self.resolve_single_import(module_,
                                               containing_module,
                                               target,
                                               source,
                                               import_directive)
                } else {
                    self.resolve_glob_import(module_, containing_module, import_directive)
                }
            })
            .and_then(|()| {
                // Decrement the count of unresolved imports.
                assert!(self.resolver.unresolved_imports >= 1);
                self.resolver.unresolved_imports -= 1;

                if let GlobImport = import_directive.subclass {
                    module_.dec_glob_count();
                    if import_directive.is_public {
                        module_.dec_pub_glob_count();
                    }
                }
                if import_directive.is_public {
                    module_.dec_pub_count();
                }
                Success(())
            })
    }

    fn resolve_single_import(&mut self,
                             module_: Module<'b>,
                             target_module: Module<'b>,
                             target: Name,
                             source: Name,
                             directive: &ImportDirective)
                             -> ResolveResult<()> {
        debug!("(resolving single import) resolving `{}` = `{}::{}` from `{}` id {}",
               target,
               module_to_string(&target_module),
               source,
               module_to_string(module_),
               directive.id);

        // If this is a circular import, we temporarily count it as determined so that
        // it fails (as opposed to being indeterminate) when nothing else can define it.
        if target_module.def_id() == module_.def_id() && source == target {
            module_.decrement_outstanding_references_for(target, ValueNS);
            module_.decrement_outstanding_references_for(target, TypeNS);
        }

        // We need to resolve both namespaces for this to succeed.
        let value_result =
            self.resolver.resolve_name_in_module(target_module, source, ValueNS, false, true);
        let type_result =
            self.resolver.resolve_name_in_module(target_module, source, TypeNS, false, true);

        if target_module.def_id() == module_.def_id() && source == target {
            module_.increment_outstanding_references_for(target, ValueNS);
            module_.increment_outstanding_references_for(target, TypeNS);
        }

        match (&value_result, &type_result) {
            (&Indeterminate, _) | (_, &Indeterminate) => return Indeterminate,
            (&Failed(_), &Failed(_)) => {
                let children = target_module.resolutions.borrow();
                let names = children.keys().map(|&(ref name, _)| name);
                let lev_suggestion = match find_best_match_for_name(names, &source.as_str(), None) {
                    Some(name) => format!(". Did you mean to use `{}`?", name),
                    None => "".to_owned(),
                };
                let msg = format!("There is no `{}` in `{}`{}",
                                  source,
                                  module_to_string(target_module), lev_suggestion);
                return Failed(Some((directive.span, msg)));
            }
            _ => (),
        }

        match (&value_result, &type_result) {
            (&Success(name_binding), _) if !name_binding.is_import() &&
                                           directive.is_public &&
                                           !name_binding.is_public() => {
                let msg = format!("`{}` is private, and cannot be reexported", source);
                let note_msg = format!("consider marking `{}` as `pub` in the imported module",
                                        source);
                struct_span_err!(self.resolver.session, directive.span, E0364, "{}", &msg)
                    .span_note(directive.span, &note_msg)
                    .emit();
            }

            (_, &Success(name_binding)) if !name_binding.is_import() && directive.is_public => {
                if !name_binding.is_public() {
                    if name_binding.is_extern_crate() {
                        let msg = format!("extern crate `{}` is private, and cannot be reexported \
                                           (error E0364), consider declaring with `pub`",
                                           source);
                        self.resolver.session.add_lint(lint::builtin::PRIVATE_IN_PUBLIC,
                                                       directive.id,
                                                       directive.span,
                                                       msg);
                    } else {
                        let msg = format!("`{}` is private, and cannot be reexported", source);
                        let note_msg =
                            format!("consider declaring type or module `{}` with `pub`", source);
                        struct_span_err!(self.resolver.session, directive.span, E0365, "{}", &msg)
                            .span_note(directive.span, &note_msg)
                            .emit();
                    }
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

        for &(ns, result) in &[(ValueNS, &value_result), (TypeNS, &type_result)] {
            if let Success(binding) = *result {
                if !binding.defined_with(DefModifiers::IMPORTABLE) {
                    let msg = format!("`{}` is not directly importable", target);
                    span_err!(self.resolver.session, directive.span, E0253, "{}", &msg);
                }

                self.define(module_, target, ns, directive.import(binding));
            }
        }

        // Record what this import resolves to for later uses in documentation,
        // this may resolve to either a value or a type, but for documentation
        // purposes it's good enough to just favor one over the other.
        module_.decrement_outstanding_references_for(target, ValueNS);
        module_.decrement_outstanding_references_for(target, TypeNS);

        let def = match type_result.success().and_then(NameBinding::def) {
            Some(def) => def,
            None => value_result.success().and_then(NameBinding::def).unwrap(),
        };
        let path_resolution = PathResolution { base_def: def, depth: 0 };
        self.resolver.def_map.borrow_mut().insert(directive.id, path_resolution);

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
                           directive: &ImportDirective)
                           -> ResolveResult<()> {
        // We must bail out if the node has unresolved imports of any kind (including globs).
        if target_module.pub_count.get() > 0 {
            debug!("(resolving glob import) target module has unresolved pub imports; bailing out");
            return Indeterminate;
        }

        if module_.def_id() == target_module.def_id() {
            // This means we are trying to glob import a module into itself, and it is a no-go
            let msg = "Cannot glob-import a module into itself.".into();
            return Failed(Some((directive.span, msg)));
        }

        // Add all children from the containing module.
        build_reduced_graph::populate_module_if_necessary(self.resolver, target_module);
        target_module.for_each_child(|name, ns, binding| {
            if !binding.defined_with(DefModifiers::IMPORTABLE | DefModifiers::PUBLIC) { return }
            self.define(module_, name, ns, directive.import(binding));

            if ns == TypeNS && directive.is_public &&
               binding.defined_with(DefModifiers::PRIVATE_VARIANT) {
                let msg = format!("variant `{}` is private, and cannot be reexported (error \
                                   E0364), consider declaring its enum as `pub`", name);
                self.resolver.session.add_lint(lint::builtin::PRIVATE_IN_PUBLIC,
                                               directive.id,
                                               directive.span,
                                               msg);
            }
        });

        // Record the destination of this import
        if let Some(did) = target_module.def_id() {
            self.resolver.def_map.borrow_mut().insert(directive.id,
                                                      PathResolution {
                                                          base_def: Def::Mod(did),
                                                          depth: 0,
                                                      });
        }

        debug!("(resolving glob import) successfully resolved import");
        return Success(());
    }

    fn define(&mut self,
              parent: Module<'b>,
              name: Name,
              ns: Namespace,
              binding: NameBinding<'b>) {
        let binding = self.resolver.new_name_binding(binding);
        if let Err(old_binding) = parent.try_define_child(name, ns, binding) {
            self.report_conflict(name, ns, binding, old_binding);
        } else if binding.is_public() { // Add to the export map
            if let (Some(parent_def_id), Some(def)) = (parent.def_id(), binding.def()) {
                let parent_node_id = self.resolver.ast_map.as_local_node_id(parent_def_id).unwrap();
                let export = Export { name: name, def_id: def.def_id() };
                self.resolver.export_map.entry(parent_node_id).or_insert(Vec::new()).push(export);
            }
        }
    }

    fn report_conflict(&mut self,
                       name: Name,
                       ns: Namespace,
                       binding: &'b NameBinding<'b>,
                       old_binding: &'b NameBinding<'b>) {
        // Error on the second of two conflicting imports
        if old_binding.is_import() && binding.is_import() &&
           old_binding.span.unwrap().lo > binding.span.unwrap().lo {
            self.report_conflict(name, ns, old_binding, binding);
            return;
        }

        if old_binding.is_extern_crate() {
            let msg = format!("import `{0}` conflicts with imported crate \
                               in this module (maybe you meant `use {0}::*`?)",
                              name);
            span_err!(self.resolver.session, binding.span.unwrap(), E0254, "{}", &msg);
        } else if old_binding.is_import() {
            let ns_word = match (ns, old_binding.module()) {
                (ValueNS, _) => "value",
                (TypeNS, Some(module)) if module.is_normal() => "module",
                (TypeNS, Some(module)) if module.is_trait() => "trait",
                (TypeNS, _) => "type",
            };
            let mut err = struct_span_err!(self.resolver.session,
                                           binding.span.unwrap(),
                                           E0252,
                                           "a {} named `{}` has already been imported \
                                            in this module",
                                           ns_word,
                                           name);
            err.span_note(old_binding.span.unwrap(),
                          &format!("previous import of `{}` here", name));
            err.emit();
        } else if ns == ValueNS { // Check for item conflicts in the value namespace
            let mut err = struct_span_err!(self.resolver.session,
                                           binding.span.unwrap(),
                                           E0255,
                                           "import `{}` conflicts with value in this module",
                                           name);
            err.span_note(old_binding.span.unwrap(), "conflicting value here");
            err.emit();
        } else { // Check for item conflicts in the type namespace
            let (what, note) = match old_binding.module() {
                Some(ref module) if module.is_normal() =>
                    ("existing submodule", "note conflicting module here"),
                Some(ref module) if module.is_trait() =>
                    ("trait in this module", "note conflicting trait here"),
                _ => ("type in this module", "note conflicting type here"),
            };
            let mut err = struct_span_err!(self.resolver.session,
                                           binding.span.unwrap(),
                                           E0256,
                                           "import `{}` conflicts with {}",
                                           name,
                                           what);
            err.span_note(old_binding.span.unwrap(), note);
            err.emit();
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
