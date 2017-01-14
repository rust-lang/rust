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

use {AmbiguityError, Module, PerNS};
use Namespace::{self, TypeNS, MacroNS};
use {NameBinding, NameBindingKind, PathResult, PrivacyError};
use Resolver;
use {names_to_string, module_to_string};
use {resolve_error, ResolutionError};

use rustc::ty;
use rustc::lint::builtin::PRIVATE_IN_PUBLIC;
use rustc::hir::def_id::DefId;
use rustc::hir::def::*;
use rustc::util::nodemap::FxHashSet;

use syntax::ast::{Ident, NodeId};
use syntax::ext::base::Determinacy::{self, Determined, Undetermined};
use syntax::ext::hygiene::Mark;
use syntax::parse::token;
use syntax::symbol::keywords;
use syntax::util::lev_distance::find_best_match_for_name;
use syntax_pos::Span;

use std::cell::{Cell, RefCell};
use std::mem;

/// Contains data for specific types of import directives.
#[derive(Clone, Debug)]
pub enum ImportDirectiveSubclass<'a> {
    SingleImport {
        target: Ident,
        source: Ident,
        result: PerNS<Cell<Result<&'a NameBinding<'a>, Determinacy>>>,
        type_ns_only: bool,
    },
    GlobImport {
        is_prelude: bool,
        max_vis: Cell<ty::Visibility>, // The visibility of the greatest reexport.
        // n.b. `max_vis` is only used in `finalize_import` to check for reexport errors.
    },
    ExternCrate,
    MacroUse,
}

/// One import directive.
#[derive(Debug,Clone)]
pub struct ImportDirective<'a> {
    pub id: NodeId,
    pub parent: Module<'a>,
    pub module_path: Vec<Ident>,
    pub imported_module: Cell<Option<Module<'a>>>, // the resolution of `module_path`
    pub subclass: ImportDirectiveSubclass<'a>,
    pub span: Span,
    pub vis: Cell<ty::Visibility>,
    pub expansion: Mark,
    pub used: Cell<bool>,
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
    shadows_glob: Option<&'a NameBinding<'a>>,
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
    /// Creates a `SingleImports<'a>` of None type.
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
}

impl<'a> Resolver<'a> {
    fn resolution(&self, module: Module<'a>, ident: Ident, ns: Namespace)
                  -> &'a RefCell<NameResolution<'a>> {
        let ident = ident.unhygienize();
        *module.resolutions.borrow_mut().entry((ident, ns))
               .or_insert_with(|| self.arenas.alloc_name_resolution())
    }

    /// Attempts to resolve `ident` in namespaces `ns` of `module`.
    /// Invariant: if `record_used` is `Some`, import resolution must be complete.
    pub fn resolve_ident_in_module(&mut self,
                                   module: Module<'a>,
                                   ident: Ident,
                                   ns: Namespace,
                                   ignore_unresolved_invocations: bool,
                                   record_used: Option<Span>)
                                   -> Result<&'a NameBinding<'a>, Determinacy> {
        self.populate_module_if_necessary(module);

        let resolution = self.resolution(module, ident, ns)
            .try_borrow_mut()
            .map_err(|_| Determined)?; // This happens when there is a cycle of imports

        if let Some(span) = record_used {
            if let Some(binding) = resolution.binding {
                if let Some(shadowed_glob) = resolution.shadows_glob {
                    let name = ident.name;
                    // If we ignore unresolved invocations, we must forbid
                    // expanded shadowing to avoid time travel.
                    if ignore_unresolved_invocations &&
                       binding.expansion != Mark::root() &&
                       ns != MacroNS && // In MacroNS, `try_define` always forbids this shadowing
                       binding.def() != shadowed_glob.def() {
                        self.ambiguity_errors.push(AmbiguityError {
                            span: span, name: name, lexical: false, b1: binding, b2: shadowed_glob,
                            legacy: false,
                        });
                    }
                }
                if self.record_use(ident, ns, binding, span) {
                    return Ok(self.dummy_binding);
                }
                if !self.is_accessible(binding.vis) {
                    self.privacy_errors.push(PrivacyError(span, ident.name, binding));
                }
            }

            return resolution.binding.ok_or(Determined);
        }

        let check_usable = |this: &mut Self, binding: &'a NameBinding<'a>| {
            // `extern crate` are always usable for backwards compatability, see issue #37020.
            let usable = this.is_accessible(binding.vis) || binding.is_extern_crate();
            if usable { Ok(binding) } else { Err(Determined) }
        };

        // Items and single imports are not shadowable.
        if let Some(binding) = resolution.binding {
            if !binding.is_glob_import() {
                return check_usable(self, binding);
            }
        }

        // Check if a single import can still define the name.
        match resolution.single_imports {
            SingleImports::AtLeastOne => return Err(Undetermined),
            SingleImports::MaybeOne(directive) if self.is_accessible(directive.vis.get()) => {
                let module = match directive.imported_module.get() {
                    Some(module) => module,
                    None => return Err(Undetermined),
                };
                let ident = match directive.subclass {
                    SingleImport { source, .. } => source,
                    _ => unreachable!(),
                };
                match self.resolve_ident_in_module(module, ident, ns, false, None) {
                    Err(Determined) => {}
                    _ => return Err(Undetermined),
                }
            }
            SingleImports::MaybeOne(_) | SingleImports::None => {},
        }

        let no_unresolved_invocations =
            ignore_unresolved_invocations || module.unresolved_invocations.borrow().is_empty();
        match resolution.binding {
            // In `MacroNS`, expanded bindings do not shadow (enforced in `try_define`).
            Some(binding) if no_unresolved_invocations || ns == MacroNS =>
                return check_usable(self, binding),
            None if no_unresolved_invocations => {}
            _ => return Err(Undetermined),
        }

        // Check if the globs are determined
        for directive in module.globs.borrow().iter() {
            if self.is_accessible(directive.vis.get()) {
                if let Some(module) = directive.imported_module.get() {
                    let result = self.resolve_ident_in_module(module, ident, ns, false, None);
                    if let Err(Undetermined) = result {
                        return Err(Undetermined);
                    }
                } else {
                    return Err(Undetermined);
                }
            }
        }

        Err(Determined)
    }

    // Add an import directive to the current module.
    pub fn add_import_directive(&mut self,
                                module_path: Vec<Ident>,
                                subclass: ImportDirectiveSubclass<'a>,
                                span: Span,
                                id: NodeId,
                                vis: ty::Visibility,
                                expansion: Mark) {
        let current_module = self.current_module;
        let directive = self.arenas.alloc_import_directive(ImportDirective {
            parent: current_module,
            module_path: module_path,
            imported_module: Cell::new(None),
            subclass: subclass,
            span: span,
            id: id,
            vis: Cell::new(vis),
            expansion: expansion,
            used: Cell::new(false),
        });

        self.indeterminate_imports.push(directive);
        match directive.subclass {
            SingleImport { target, .. } => {
                self.per_ns(|this, ns| {
                    let mut resolution = this.resolution(current_module, target, ns).borrow_mut();
                    resolution.single_imports.add_directive(directive);
                });
            }
            // We don't add prelude imports to the globs since they only affect lexical scopes,
            // which are not relevant to import resolution.
            GlobImport { is_prelude: true, .. } => {}
            GlobImport { .. } => self.current_module.globs.borrow_mut().push(directive),
            _ => unreachable!(),
        }
    }

    // Given a binding and an import directive that resolves to it,
    // return the corresponding binding defined by the import directive.
    pub fn import(&self, binding: &'a NameBinding<'a>, directive: &'a ImportDirective<'a>)
                  -> &'a NameBinding<'a> {
        let vis = if binding.pseudo_vis().is_at_least(directive.vis.get(), self) ||
                     !directive.is_glob() && binding.is_extern_crate() { // c.f. `PRIVATE_IN_PUBLIC`
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
                binding: binding,
                directive: directive,
                used: Cell::new(false),
                legacy_self_import: false,
            },
            span: directive.span,
            vis: vis,
            expansion: directive.expansion,
        })
    }

    // Define the name or return the existing binding if there is a collision.
    pub fn try_define(&mut self,
                      module: Module<'a>,
                      ident: Ident,
                      ns: Namespace,
                      binding: &'a NameBinding<'a>)
                      -> Result<(), &'a NameBinding<'a>> {
        self.update_resolution(module, ident, ns, |this, resolution| {
            if let Some(old_binding) = resolution.binding {
                if binding.is_glob_import() {
                    if !old_binding.is_glob_import() &&
                       !(ns == MacroNS && old_binding.expansion != Mark::root()) {
                        resolution.shadows_glob = Some(binding);
                    } else if binding.def() != old_binding.def() {
                        resolution.binding = Some(this.ambiguity(old_binding, binding));
                    } else if !old_binding.vis.is_at_least(binding.vis, &*this) {
                        // We are glob-importing the same item but with greater visibility.
                        resolution.binding = Some(binding);
                    }
                } else if old_binding.is_glob_import() {
                    if ns == MacroNS && binding.expansion != Mark::root() &&
                       binding.def() != old_binding.def() {
                        resolution.binding = Some(this.ambiguity(binding, old_binding));
                    } else {
                        resolution.binding = Some(binding);
                        resolution.shadows_glob = Some(old_binding);
                    }
                } else {
                    return Err(old_binding);
                }
            } else {
                resolution.binding = Some(binding);
            }

            Ok(())
        })
    }

    pub fn ambiguity(&self, b1: &'a NameBinding<'a>, b2: &'a NameBinding<'a>)
                     -> &'a NameBinding<'a> {
        self.arenas.alloc_name_binding(NameBinding {
            kind: NameBindingKind::Ambiguity { b1: b1, b2: b2, legacy: false },
            vis: if b1.vis.is_at_least(b2.vis, self) { b1.vis } else { b2.vis },
            span: b1.span,
            expansion: Mark::root(),
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
            let mut resolution = &mut *self.resolution(module, ident, ns).borrow_mut();
            let old_binding = resolution.binding();

            let t = f(self, resolution);

            match resolution.binding() {
                _ if old_binding.is_some() => return t,
                None => return t,
                Some(binding) => match old_binding {
                    Some(old_binding) if old_binding as *const _ == binding as *const _ => return t,
                    _ => (binding, t),
                }
            }
        };

        // Define `binding` in `module`s glob importers.
        for directive in module.glob_importers.borrow_mut().iter() {
            if self.is_accessible_from(binding.vis, directive.parent) {
                let imported_binding = self.import(binding, directive);
                let _ = self.try_define(directive.parent, ident, ns, imported_binding);
            }
        }

        t
    }

    // Define a "dummy" resolution containing a Def::Err as a placeholder for a
    // failed resolution
    fn import_dummy_binding(&mut self, directive: &'a ImportDirective<'a>) {
        if let SingleImport { target, .. } = directive.subclass {
            let dummy_binding = self.dummy_binding;
            let dummy_binding = self.import(dummy_binding, directive);
            self.per_ns(|this, ns| {
                let _ = this.try_define(directive.parent, target, ns, dummy_binding);
            });
        }
    }
}

pub struct ImportResolver<'a, 'b: 'a> {
    pub resolver: &'a mut Resolver<'b>,
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

impl<'a, 'b: 'a> ty::DefIdTree for &'a ImportResolver<'a, 'b> {
    fn parent(self, id: DefId) -> Option<DefId> {
        self.resolver.parent(id)
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
    pub fn resolve_imports(&mut self) {
        let mut prev_num_indeterminates = self.indeterminate_imports.len() + 1;
        while self.indeterminate_imports.len() < prev_num_indeterminates {
            prev_num_indeterminates = self.indeterminate_imports.len();
            for import in mem::replace(&mut self.indeterminate_imports, Vec::new()) {
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

        let mut errors = false;
        for i in 0 .. self.determined_imports.len() {
            let import = self.determined_imports[i];
            if let Some(err) = self.finalize_import(import) {
                errors = true;

                // If the error is a single failed import then create a "fake" import
                // resolution for it so that later resolve stages won't complain.
                self.import_dummy_binding(import);
                let path = import_path_to_string(&import.module_path, &import.subclass);
                let error = ResolutionError::UnresolvedImport(Some((&path, &err)));
                resolve_error(self.resolver, import.span, error);
            }
        }

        // Report unresolved imports only if no hard error was already reported
        // to avoid generating multiple errors on the same import.
        if !errors {
            if let Some(import) = self.indeterminate_imports.iter().next() {
                let error = ResolutionError::UnresolvedImport(None);
                resolve_error(self.resolver, import.span, error);
            }
        }
    }

    /// Attempts to resolve the given import, returning true if its resolution is determined.
    /// If successful, the resolved bindings are written into the module.
    fn resolve_import(&mut self, directive: &'b ImportDirective<'b>) -> bool {
        debug!("(resolving import for module) resolving import `{}::...` in `{}`",
               names_to_string(&directive.module_path),
               module_to_string(self.current_module));

        self.current_module = directive.parent;

        let module = if let Some(module) = directive.imported_module.get() {
            module
        } else {
            let vis = directive.vis.get();
            // For better failure detection, pretend that the import will not define any names
            // while resolving its module path.
            directive.vis.set(ty::Visibility::Invisible);
            let result = self.resolve_path(&directive.module_path, None, None);
            directive.vis.set(vis);

            match result {
                PathResult::Module(module) => module,
                PathResult::Indeterminate => return false,
                _ => return true,
            }
        };

        directive.imported_module.set(Some(module));
        let (source, target, result, type_ns_only) = match directive.subclass {
            SingleImport { source, target, ref result, type_ns_only } =>
                (source, target, result, type_ns_only),
            GlobImport { .. } => {
                self.resolve_glob_import(directive);
                return true;
            }
            _ => unreachable!(),
        };

        let mut indeterminate = false;
        self.per_ns(|this, ns| if !type_ns_only || ns == TypeNS {
            if let Err(Undetermined) = result[ns].get() {
                result[ns].set(this.resolve_ident_in_module(module, source, ns, false, None));
            } else {
                return
            };

            let parent = directive.parent;
            match result[ns].get() {
                Err(Undetermined) => indeterminate = true,
                Err(Determined) => {
                    this.update_resolution(parent, target, ns, |_, resolution| {
                        resolution.single_imports.directive_failed()
                    });
                }
                Ok(binding) if !binding.is_importable() => {
                    let msg = format!("`{}` is not directly importable", target);
                    struct_span_err!(this.session, directive.span, E0253, "{}", &msg)
                        .span_label(directive.span, &format!("cannot be imported directly"))
                        .emit();
                    // Do not import this illegal binding. Import a dummy binding and pretend
                    // everything is fine
                    this.import_dummy_binding(directive);
                }
                Ok(binding) => {
                    let imported_binding = this.import(binding, directive);
                    let conflict = this.try_define(parent, target, ns, imported_binding);
                    if let Err(old_binding) = conflict {
                        this.report_conflict(parent, target, ns, imported_binding, old_binding);
                    }
                }
            }
        });

        !indeterminate
    }

    // If appropriate, returns an error to report.
    fn finalize_import(&mut self, directive: &'b ImportDirective<'b>) -> Option<String> {
        self.current_module = directive.parent;

        let ImportDirective { ref module_path, span, .. } = *directive;
        let module_result = self.resolve_path(&module_path, None, Some(span));
        let module = match module_result {
            PathResult::Module(module) => module,
            PathResult::Failed(msg, _) => {
                let (mut self_path, mut self_result) = (module_path.clone(), None);
                if !self_path.is_empty() && !token::Ident(self_path[0]).is_path_segment_keyword() {
                    self_path[0].name = keywords::SelfValue.name();
                    self_result = Some(self.resolve_path(&self_path, None, None));
                }
                return if let Some(PathResult::Module(..)) = self_result {
                    Some(format!("Did you mean `{}`?", names_to_string(&self_path)))
                } else {
                    Some(msg)
                };
            },
            _ => return None,
        };

        let (ident, result, type_ns_only) = match directive.subclass {
            SingleImport { source, ref result, type_ns_only, .. } => (source, result, type_ns_only),
            GlobImport { .. } if module.def_id() == directive.parent.def_id() => {
                // Importing a module into itself is not allowed.
                return Some("Cannot glob-import a module into itself.".to_string());
            }
            GlobImport { is_prelude, ref max_vis } => {
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
        let mut legacy_self_import = None;
        self.per_ns(|this, ns| if !type_ns_only || ns == TypeNS {
            if let Ok(binding) = result[ns].get() {
                all_ns_err = false;
                if this.record_use(ident, ns, binding, directive.span) {
                    this.resolution(module, ident, ns).borrow_mut().binding =
                        Some(this.dummy_binding);
                }
            }
        } else if let Ok(binding) = this.resolve_ident_in_module(module, ident, ns, false, None) {
            legacy_self_import = Some(directive);
            let binding = this.arenas.alloc_name_binding(NameBinding {
                kind: NameBindingKind::Import {
                    binding: binding,
                    directive: directive,
                    used: Cell::new(false),
                    legacy_self_import: true,
                },
                ..*binding
            });
            let _ = this.try_define(directive.parent, ident, ns, binding);
        });

        if all_ns_err {
            if let Some(directive) = legacy_self_import {
                self.warn_legacy_self_import(directive);
                return None;
            }
            let mut all_ns_failed = true;
            self.per_ns(|this, ns| if !type_ns_only || ns == TypeNS {
                match this.resolve_ident_in_module(module, ident, ns, false, Some(span)) {
                    Ok(_) => all_ns_failed = false,
                    _ => {}
                }
            });

            return if all_ns_failed {
                let resolutions = module.resolutions.borrow();
                let names = resolutions.iter().filter_map(|(&(ref i, _), resolution)| {
                    if *i == ident { return None; } // Never suggest the same name
                    match *resolution.borrow() {
                        NameResolution { binding: Some(_), .. } => Some(&i.name),
                        NameResolution { single_imports: SingleImports::None, .. } => None,
                        _ => Some(&i.name),
                    }
                });
                let lev_suggestion =
                    match find_best_match_for_name(names, &ident.name.as_str(), None) {
                        Some(name) => format!(". Did you mean to use `{}`?", name),
                        None => "".to_owned(),
                    };
                let module_str = module_to_string(module);
                let msg = if &module_str == "???" {
                    format!("no `{}` in the root{}", ident, lev_suggestion)
                } else {
                    format!("no `{}` in `{}`{}", ident, module_str, lev_suggestion)
                };
                Some(msg)
            } else {
                // `resolve_ident_in_module` reported a privacy error.
                self.import_dummy_binding(directive);
                None
            }
        }

        let mut reexport_error = None;
        let mut any_successful_reexport = false;
        self.per_ns(|this, ns| {
            if let Ok(binding) = result[ns].get() {
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
                let msg = format!("extern crate `{}` is private, and cannot be reexported \
                                   (error E0364), consider declaring with `pub`",
                                   ident);
                self.session.add_lint(PRIVATE_IN_PUBLIC, directive.id, directive.span, msg);
            } else if ns == TypeNS {
                struct_span_err!(self.session, directive.span, E0365,
                                 "`{}` is private, and cannot be reexported", ident)
                    .span_label(directive.span, &format!("reexport of private `{}`", ident))
                    .note(&format!("consider declaring type or module `{}` with `pub`", ident))
                    .emit();
            } else {
                let msg = format!("`{}` is private, and cannot be reexported", ident);
                let note_msg =
                    format!("consider marking `{}` as `pub` in the imported module", ident);
                struct_span_err!(self.session, directive.span, E0364, "{}", &msg)
                    .span_note(directive.span, &note_msg)
                    .emit();
            }
        }

        // Record what this import resolves to for later uses in documentation,
        // this may resolve to either a value or a type, but for documentation
        // purposes it's good enough to just favor one over the other.
        self.per_ns(|this, ns| if let Some(binding) = result[ns].get().ok() {
            this.def_map.entry(directive.id).or_insert(PathResolution::new(binding.def()));
        });

        debug!("(resolving single import) successfully resolved import");
        None
    }

    fn resolve_glob_import(&mut self, directive: &'b ImportDirective<'b>) {
        let module = directive.imported_module.get().unwrap();
        self.populate_module_if_necessary(module);

        if let Some(Def::Trait(_)) = module.def() {
            self.session.span_err(directive.span, "items in traits are not importable.");
            return;
        } else if module.def_id() == directive.parent.def_id()  {
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
        for ((ident, ns), binding) in bindings {
            if binding.pseudo_vis() == ty::Visibility::Public || self.is_accessible(binding.vis) {
                let imported_binding = self.import(binding, directive);
                let _ = self.try_define(directive.parent, ident, ns, imported_binding);
            }
        }

        // Record the destination of this import
        self.record_def(directive.id, PathResolution::new(module.def().unwrap()));
    }

    // Miscellaneous post-processing, including recording reexports, reporting conflicts,
    // reporting the PRIVATE_IN_PUBLIC lint, and reporting unresolved imports.
    fn finalize_resolutions_in(&mut self, module: Module<'b>) {
        // Since import resolution is finished, globs will not define any more names.
        *module.globs.borrow_mut() = Vec::new();

        let mut reexports = Vec::new();
        if module as *const _ == self.graph_root as *const _ {
            let mut exported_macro_names = FxHashSet();
            for export in mem::replace(&mut self.macro_exports, Vec::new()).into_iter().rev() {
                if exported_macro_names.insert(export.name) {
                    reexports.push(export);
                }
            }
        }

        for (&(ident, ns), resolution) in module.resolutions.borrow().iter() {
            let resolution = &mut *resolution.borrow_mut();
            let binding = match resolution.binding {
                Some(binding) => binding,
                None => continue,
            };

            if binding.vis == ty::Visibility::Public &&
               (binding.is_import() || binding.is_extern_crate()) {
                let def = binding.def();
                if def != Def::Err {
                    if !def.def_id().is_local() {
                        self.session.cstore.export_macros(def.def_id().krate);
                    }
                    reexports.push(Export { name: ident.name, def: def });
                }
            }

            match binding.kind {
                NameBindingKind::Import { binding: orig_binding, directive, .. } => {
                    if ns == TypeNS && orig_binding.is_variant() &&
                       !orig_binding.vis.is_at_least(binding.vis, &*self) {
                        let msg = format!("variant `{}` is private, and cannot be reexported \
                                           (error E0364), consider declaring its enum as `pub`",
                                          ident);
                        self.session.add_lint(PRIVATE_IN_PUBLIC, directive.id, binding.span, msg);
                    }
                }
                NameBindingKind::Ambiguity { b1, b2, .. }
                        if b1.is_glob_import() && b2.is_glob_import() => {
                    let (orig_b1, orig_b2) = match (&b1.kind, &b2.kind) {
                        (&NameBindingKind::Import { binding: b1, .. },
                         &NameBindingKind::Import { binding: b2, .. }) => (b1, b2),
                        _ => continue,
                    };
                    let (b1, b2) = match (orig_b1.vis, orig_b2.vis) {
                        (ty::Visibility::Public, ty::Visibility::Public) => continue,
                        (ty::Visibility::Public, _) => (b1, b2),
                        (_, ty::Visibility::Public) => (b2, b1),
                        _ => continue,
                    };
                    resolution.binding = Some(self.arenas.alloc_name_binding(NameBinding {
                        kind: NameBindingKind::Ambiguity { b1: b1, b2: b2, legacy: true }, ..*b1
                    }));
                }
                _ => {}
            }
        }

        if reexports.len() > 0 {
            if let Some(def_id) = module.def_id() {
                let node_id = self.definitions.as_local_node_id(def_id).unwrap();
                self.export_map.insert(node_id, reexports);
            }
        }
    }
}

fn import_path_to_string(names: &[Ident], subclass: &ImportDirectiveSubclass) -> String {
    let global = !names.is_empty() && names[0].name == keywords::CrateRoot.name();
    let names = if global { &names[1..] } else { names };
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
        ExternCrate => "<extern crate>".to_string(),
        MacroUse => "#[macro_use]".to_string(),
    }
}
