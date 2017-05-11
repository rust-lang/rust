// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use {AmbiguityError, Resolver, ResolutionError, resolve_error};
use {Module, ModuleKind, NameBinding, NameBindingKind, PathResult};
use Namespace::{self, MacroNS};
use build_reduced_graph::BuildReducedGraphVisitor;
use resolve_imports::ImportResolver;
use rustc::hir::def_id::{DefId, BUILTIN_MACROS_CRATE, CRATE_DEF_INDEX, DefIndex};
use rustc::hir::def::{Def, Export};
use rustc::hir::map::{self, DefCollector};
use rustc::{ty, lint};
use syntax::ast::{self, Name, Ident};
use syntax::attr::{self, HasAttrs};
use syntax::errors::DiagnosticBuilder;
use syntax::ext::base::{self, Annotatable, Determinacy, MultiModifier, MultiDecorator};
use syntax::ext::base::{MacroKind, SyntaxExtension, Resolver as SyntaxResolver};
use syntax::ext::expand::{Expansion, ExpansionKind, Invocation, InvocationKind, find_attr_invoc};
use syntax::ext::hygiene::Mark;
use syntax::ext::placeholders::placeholder;
use syntax::ext::tt::macro_rules;
use syntax::feature_gate::{self, emit_feature_err, GateIssue};
use syntax::fold::{self, Folder};
use syntax::parse::parser::PathStyle;
use syntax::parse::token::{self, Token};
use syntax::ptr::P;
use syntax::symbol::{Symbol, keywords};
use syntax::tokenstream::{TokenStream, TokenTree, Delimited};
use syntax::util::lev_distance::find_best_match_for_name;
use syntax_pos::{Span, DUMMY_SP};

use std::cell::Cell;
use std::mem;
use std::rc::Rc;

#[derive(Clone)]
pub struct InvocationData<'a> {
    pub module: Cell<Module<'a>>,
    pub def_index: DefIndex,
    // True if this expansion is in a `const_expr` position, for example `[u32; m!()]`.
    // c.f. `DefCollector::visit_const_expr`.
    pub const_expr: bool,
    // The scope in which the invocation path is resolved.
    pub legacy_scope: Cell<LegacyScope<'a>>,
    // The smallest scope that includes this invocation's expansion,
    // or `Empty` if this invocation has not been expanded yet.
    pub expansion: Cell<LegacyScope<'a>>,
}

impl<'a> InvocationData<'a> {
    pub fn root(graph_root: Module<'a>) -> Self {
        InvocationData {
            module: Cell::new(graph_root),
            def_index: CRATE_DEF_INDEX,
            const_expr: false,
            legacy_scope: Cell::new(LegacyScope::Empty),
            expansion: Cell::new(LegacyScope::Empty),
        }
    }
}

#[derive(Copy, Clone)]
pub enum LegacyScope<'a> {
    Empty,
    Invocation(&'a InvocationData<'a>), // The scope of the invocation, not including its expansion
    Expansion(&'a InvocationData<'a>), // The scope of the invocation, including its expansion
    Binding(&'a LegacyBinding<'a>),
}

pub struct LegacyBinding<'a> {
    pub parent: Cell<LegacyScope<'a>>,
    pub name: ast::Name,
    def_id: DefId,
    pub span: Span,
}

#[derive(Copy, Clone)]
pub enum MacroBinding<'a> {
    Legacy(&'a LegacyBinding<'a>),
    Global(&'a NameBinding<'a>),
    Modern(&'a NameBinding<'a>),
}

impl<'a> MacroBinding<'a> {
    pub fn span(self) -> Span {
        match self {
            MacroBinding::Legacy(binding) => binding.span,
            MacroBinding::Global(binding) | MacroBinding::Modern(binding) => binding.span,
        }
    }

    pub fn binding(self) -> &'a NameBinding<'a> {
        match self {
            MacroBinding::Global(binding) | MacroBinding::Modern(binding) => binding,
            MacroBinding::Legacy(_) => panic!("unexpected MacroBinding::Legacy"),
        }
    }
}

impl<'a> base::Resolver for Resolver<'a> {
    fn next_node_id(&mut self) -> ast::NodeId {
        self.session.next_node_id()
    }

    fn get_module_scope(&mut self, id: ast::NodeId) -> Mark {
        let mark = Mark::fresh();
        let module = self.module_map[&self.definitions.local_def_id(id)];
        self.invocations.insert(mark, self.arenas.alloc_invocation_data(InvocationData {
            module: Cell::new(module),
            def_index: module.def_id().unwrap().index,
            const_expr: false,
            legacy_scope: Cell::new(LegacyScope::Empty),
            expansion: Cell::new(LegacyScope::Empty),
        }));
        mark
    }

    fn eliminate_crate_var(&mut self, item: P<ast::Item>) -> P<ast::Item> {
        struct EliminateCrateVar<'b, 'a: 'b>(&'b mut Resolver<'a>, Span);

        impl<'a, 'b> Folder for EliminateCrateVar<'a, 'b> {
            fn fold_path(&mut self, mut path: ast::Path) -> ast::Path {
                let ident = path.segments[0].identifier;
                if ident.name == "$crate" {
                    path.segments[0].identifier.name = keywords::CrateRoot.name();
                    let module = self.0.resolve_crate_var(ident.ctxt, self.1);
                    if !module.is_local() {
                        let span = path.segments[0].span;
                        path.segments.insert(1, match module.kind {
                            ModuleKind::Def(_, name) => ast::PathSegment::from_ident(
                                ast::Ident::with_empty_ctxt(name), span
                            ),
                            _ => unreachable!(),
                        })
                    }
                }
                path
            }

            fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
                fold::noop_fold_mac(mac, self)
            }
        }

        EliminateCrateVar(self, item.span).fold_item(item).expect_one("")
    }

    fn is_whitelisted_legacy_custom_derive(&self, name: Name) -> bool {
        self.whitelisted_legacy_custom_derives.contains(&name)
    }

    fn visit_expansion(&mut self, mark: Mark, expansion: &Expansion, derives: &[Mark]) {
        let invocation = self.invocations[&mark];
        self.collect_def_ids(invocation, expansion);

        self.current_module = invocation.module.get();
        self.current_module.unresolved_invocations.borrow_mut().remove(&mark);
        self.current_module.unresolved_invocations.borrow_mut().extend(derives);
        for &derive in derives {
            self.invocations.insert(derive, invocation);
        }
        let mut visitor = BuildReducedGraphVisitor {
            resolver: self,
            legacy_scope: LegacyScope::Invocation(invocation),
            expansion: mark,
        };
        expansion.visit_with(&mut visitor);
        invocation.expansion.set(visitor.legacy_scope);
    }

    fn add_builtin(&mut self, ident: ast::Ident, ext: Rc<SyntaxExtension>) {
        let def_id = DefId {
            krate: BUILTIN_MACROS_CRATE,
            index: DefIndex::new(self.macro_map.len()),
        };
        let kind = ext.kind();
        self.macro_map.insert(def_id, ext);
        let binding = self.arenas.alloc_name_binding(NameBinding {
            kind: NameBindingKind::Def(Def::Macro(def_id, kind)),
            span: DUMMY_SP,
            vis: ty::Visibility::Invisible,
            expansion: Mark::root(),
        });
        self.global_macros.insert(ident.name, binding);
    }

    fn resolve_imports(&mut self) {
        ImportResolver { resolver: self }.resolve_imports()
    }

    // Resolves attribute and derive legacy macros from `#![plugin(..)]`.
    fn find_legacy_attr_invoc(&mut self, attrs: &mut Vec<ast::Attribute>)
                              -> Option<ast::Attribute> {
        for i in 0..attrs.len() {
            let name = unwrap_or!(attrs[i].name(), continue);

            if self.session.plugin_attributes.borrow().iter()
                    .any(|&(ref attr_nm, _)| name == &**attr_nm) {
                attr::mark_known(&attrs[i]);
            }

            match self.global_macros.get(&name).cloned() {
                Some(binding) => match *binding.get_macro(self) {
                    MultiModifier(..) | MultiDecorator(..) | SyntaxExtension::AttrProcMacro(..) => {
                        return Some(attrs.remove(i))
                    }
                    _ => {}
                },
                None => {}
            }
        }

        // Check for legacy derives
        for i in 0..attrs.len() {
            let name = unwrap_or!(attrs[i].name(), continue);

            if name == "derive" {
                let result = attrs[i].parse_list(&self.session.parse_sess, |parser| {
                    parser.parse_path_allowing_meta(PathStyle::Mod)
                });

                let mut traits = match result {
                    Ok(traits) => traits,
                    Err(mut e) => {
                        e.cancel();
                        continue
                    }
                };

                for j in 0..traits.len() {
                    if traits[j].segments.len() > 1 {
                        continue
                    }
                    let trait_name = traits[j].segments[0].identifier.name;
                    let legacy_name = Symbol::intern(&format!("derive_{}", trait_name));
                    if !self.global_macros.contains_key(&legacy_name) {
                        continue
                    }
                    let span = traits.remove(j).span;
                    self.gate_legacy_custom_derive(legacy_name, span);
                    if traits.is_empty() {
                        attrs.remove(i);
                    } else {
                        let mut tokens = Vec::new();
                        for (j, path) in traits.iter().enumerate() {
                            if j > 0 {
                                tokens.push(TokenTree::Token(attrs[i].span, Token::Comma).into());
                            }
                            for (k, segment) in path.segments.iter().enumerate() {
                                if k > 0 {
                                    tokens.push(TokenTree::Token(path.span, Token::ModSep).into());
                                }
                                let tok = Token::Ident(segment.identifier);
                                tokens.push(TokenTree::Token(path.span, tok).into());
                            }
                        }
                        attrs[i].tokens = TokenTree::Delimited(attrs[i].span, Delimited {
                            delim: token::Paren,
                            tts: TokenStream::concat(tokens).into(),
                        }).into();
                    }
                    return Some(ast::Attribute {
                        path: ast::Path::from_ident(span, Ident::with_empty_ctxt(legacy_name)),
                        tokens: TokenStream::empty(),
                        id: attr::mk_attr_id(),
                        style: ast::AttrStyle::Outer,
                        is_sugared_doc: false,
                        span: span,
                    });
                }
            }
        }

        None
    }

    fn resolve_invoc(&mut self, invoc: &mut Invocation, scope: Mark, force: bool)
                     -> Result<Option<Rc<SyntaxExtension>>, Determinacy> {
        let def = match invoc.kind {
            InvocationKind::Attr { attr: None, .. } => return Ok(None),
            _ => match self.resolve_invoc_to_def(invoc, scope, force) {
                Ok(def) => def,
                Err(determinacy) => return Err(determinacy),
            },
        };
        self.macro_defs.insert(invoc.expansion_data.mark, def.def_id());
        self.unused_macros.get_mut(&def.def_id()).map(|m| *m = false);
        Ok(Some(self.get_macro(def)))
    }

    fn resolve_macro(&mut self, scope: Mark, path: &ast::Path, kind: MacroKind, force: bool)
                     -> Result<Rc<SyntaxExtension>, Determinacy> {
        self.resolve_macro_to_def(scope, path, kind, force).map(|def| {
            self.unused_macros.get_mut(&def.def_id()).map(|m| *m = false);
            self.get_macro(def)
        })
    }

    fn check_unused_macros(&self) {
        for (did, _) in self.unused_macros.iter().filter(|&(_, b)| *b) {
            let span = match *self.macro_map[did] {
                           SyntaxExtension::NormalTT(_, sp, _) => sp,
                           SyntaxExtension::IdentTT(_, sp, _) => sp,
                           _ => None
                       };
            if let Some(span) = span {
                let lint = lint::builtin::UNUSED_MACROS;
                let msg = "unused macro".to_string();
                // We are using CRATE_NODE_ID here even though its inaccurate, as we
                // sadly don't have the NodeId of the macro definition.
                self.session.add_lint(lint, ast::CRATE_NODE_ID, span, msg);
            } else {
                bug!("attempted to create unused macro error, but span not available");
            }
        }
    }
}

impl<'a> Resolver<'a> {
    fn resolve_invoc_to_def(&mut self, invoc: &mut Invocation, scope: Mark, force: bool)
                            -> Result<Def, Determinacy> {
        let (attr, traits, item) = match invoc.kind {
            InvocationKind::Attr { ref mut attr, ref traits, ref mut item } => (attr, traits, item),
            InvocationKind::Bang { ref mac, .. } => {
                return self.resolve_macro_to_def(scope, &mac.node.path, MacroKind::Bang, force);
            }
            InvocationKind::Derive { ref path, .. } => {
                return self.resolve_macro_to_def(scope, path, MacroKind::Derive, force);
            }
        };


        let path = attr.as_ref().unwrap().path.clone();
        let mut determinacy = Determinacy::Determined;
        match self.resolve_macro_to_def(scope, &path, MacroKind::Attr, force) {
            Ok(def) => return Ok(def),
            Err(Determinacy::Undetermined) => determinacy = Determinacy::Undetermined,
            Err(Determinacy::Determined) if force => return Err(Determinacy::Determined),
            Err(Determinacy::Determined) => {}
        }

        let attr_name = match path.segments.len() {
            1 => path.segments[0].identifier.name,
            _ => return Err(determinacy),
        };
        for path in traits {
            match self.resolve_macro(scope, path, MacroKind::Derive, force) {
                Ok(ext) => if let SyntaxExtension::ProcMacroDerive(_, ref inert_attrs) = *ext {
                    if inert_attrs.contains(&attr_name) {
                        // FIXME(jseyfried) Avoid `mem::replace` here.
                        let dummy_item = placeholder(ExpansionKind::Items, ast::DUMMY_NODE_ID)
                            .make_items().pop().unwrap();
                        let dummy_item = Annotatable::Item(dummy_item);
                        *item = mem::replace(item, dummy_item).map_attrs(|mut attrs| {
                            let inert_attr = attr.take().unwrap();
                            attr::mark_known(&inert_attr);
                            if self.proc_macro_enabled {
                                *attr = find_attr_invoc(&mut attrs);
                            }
                            attrs.push(inert_attr);
                            attrs
                        });
                    }
                    return Err(Determinacy::Undetermined);
                },
                Err(Determinacy::Undetermined) => determinacy = Determinacy::Undetermined,
                Err(Determinacy::Determined) => {}
            }
        }

        Err(determinacy)
    }

    fn resolve_macro_to_def(&mut self, scope: Mark, path: &ast::Path, kind: MacroKind, force: bool)
                            -> Result<Def, Determinacy> {
        let ast::Path { ref segments, span } = *path;
        if segments.iter().any(|segment| segment.parameters.is_some()) {
            let kind =
                if segments.last().unwrap().parameters.is_some() { "macro" } else { "module" };
            let msg = format!("type parameters are not allowed on {}s", kind);
            self.session.span_err(path.span, &msg);
            return Err(Determinacy::Determined);
        }

        let path: Vec<_> = segments.iter().map(|seg| seg.identifier).collect();
        let invocation = self.invocations[&scope];
        self.current_module = invocation.module.get();

        if path.len() > 1 {
            if !self.use_extern_macros && self.gated_errors.insert(span) {
                let msg = "non-ident macro paths are experimental";
                let feature = "use_extern_macros";
                emit_feature_err(&self.session.parse_sess, feature, span, GateIssue::Language, msg);
                self.found_unresolved_macro = true;
                return Err(Determinacy::Determined);
            }

            let def = match self.resolve_path(&path, Some(MacroNS), false, span) {
                PathResult::NonModule(path_res) => match path_res.base_def() {
                    Def::Err => Err(Determinacy::Determined),
                    def @ _ => Ok(def),
                },
                PathResult::Module(..) => unreachable!(),
                PathResult::Indeterminate if !force => return Err(Determinacy::Undetermined),
                _ => {
                    self.found_unresolved_macro = true;
                    Err(Determinacy::Determined)
                },
            };
            self.current_module.nearest_item_scope().macro_resolutions.borrow_mut()
                .push((path.into_boxed_slice(), span));
            return def;
        }

        let name = path[0].name;
        let legacy_resolution = self.resolve_legacy_scope(&invocation.legacy_scope, name, false);
        let result = if let Some(MacroBinding::Legacy(binding)) = legacy_resolution {
            Ok(Def::Macro(binding.def_id, MacroKind::Bang))
        } else {
            match self.resolve_lexical_macro_path_segment(path[0], MacroNS, false, span) {
                Ok(binding) => Ok(binding.binding().def_ignoring_ambiguity()),
                Err(Determinacy::Undetermined) if !force => return Err(Determinacy::Undetermined),
                Err(_) => {
                    self.found_unresolved_macro = true;
                    Err(Determinacy::Determined)
                }
            }
        };

        self.current_module.nearest_item_scope().legacy_macro_resolutions.borrow_mut()
            .push((scope, path[0], span, kind));

        result
    }

    // Resolve the initial segment of a non-global macro path (e.g. `foo` in `foo::bar!();`)
    pub fn resolve_lexical_macro_path_segment(&mut self,
                                              ident: Ident,
                                              ns: Namespace,
                                              record_used: bool,
                                              path_span: Span)
                                              -> Result<MacroBinding<'a>, Determinacy> {
        let mut module = Some(self.current_module);
        let mut potential_illegal_shadower = Err(Determinacy::Determined);
        let determinacy =
            if record_used { Determinacy::Determined } else { Determinacy::Undetermined };
        loop {
            let result = if let Some(module) = module {
                // Since expanded macros may not shadow the lexical scope and
                // globs may not shadow global macros (both enforced below),
                // we resolve with restricted shadowing (indicated by the penultimate argument).
                self.resolve_ident_in_module(module, ident, ns, true, record_used, path_span)
                    .map(MacroBinding::Modern)
            } else {
                self.global_macros.get(&ident.name).cloned().ok_or(determinacy)
                    .map(MacroBinding::Global)
            };

            match result.map(MacroBinding::binding) {
                Ok(binding) => {
                    if !record_used {
                        return result;
                    }
                    if let Ok(MacroBinding::Modern(shadower)) = potential_illegal_shadower {
                        if shadower.def() != binding.def() {
                            let name = ident.name;
                            self.ambiguity_errors.push(AmbiguityError {
                                span: path_span,
                                name: name,
                                b1: shadower,
                                b2: binding,
                                lexical: true,
                                legacy: false,
                            });
                            return potential_illegal_shadower;
                        }
                    }
                    if binding.expansion != Mark::root() ||
                       (binding.is_glob_import() && module.unwrap().def().is_some()) {
                        potential_illegal_shadower = result;
                    } else {
                        return result;
                    }
                },
                Err(Determinacy::Undetermined) => return Err(Determinacy::Undetermined),
                Err(Determinacy::Determined) => {}
            }

            module = match module {
                Some(module) => match module.kind {
                    ModuleKind::Block(..) => module.parent,
                    ModuleKind::Def(..) => None,
                },
                None => return potential_illegal_shadower,
            }
        }
    }

    pub fn resolve_legacy_scope(&mut self,
                                mut scope: &'a Cell<LegacyScope<'a>>,
                                name: Name,
                                record_used: bool)
                                -> Option<MacroBinding<'a>> {
        let mut possible_time_travel = None;
        let mut relative_depth: u32 = 0;
        let mut binding = None;
        loop {
            match scope.get() {
                LegacyScope::Empty => break,
                LegacyScope::Expansion(invocation) => {
                    match invocation.expansion.get() {
                        LegacyScope::Invocation(_) => scope.set(invocation.legacy_scope.get()),
                        LegacyScope::Empty => {
                            if possible_time_travel.is_none() {
                                possible_time_travel = Some(scope);
                            }
                            scope = &invocation.legacy_scope;
                        }
                        _ => {
                            relative_depth += 1;
                            scope = &invocation.expansion;
                        }
                    }
                }
                LegacyScope::Invocation(invocation) => {
                    relative_depth = relative_depth.saturating_sub(1);
                    scope = &invocation.legacy_scope;
                }
                LegacyScope::Binding(potential_binding) => {
                    if potential_binding.name == name {
                        if (!self.use_extern_macros || record_used) && relative_depth > 0 {
                            self.disallowed_shadowing.push(potential_binding);
                        }
                        binding = Some(potential_binding);
                        break
                    }
                    scope = &potential_binding.parent;
                }
            };
        }

        let binding = if let Some(binding) = binding {
            MacroBinding::Legacy(binding)
        } else if let Some(binding) = self.global_macros.get(&name).cloned() {
            if !self.use_extern_macros {
                self.record_use(Ident::with_empty_ctxt(name), MacroNS, binding, DUMMY_SP);
            }
            MacroBinding::Global(binding)
        } else {
            return None;
        };

        if !self.use_extern_macros {
            if let Some(scope) = possible_time_travel {
                // Check for disallowed shadowing later
                self.lexical_macro_resolutions.push((name, scope));
            }
        }

        Some(binding)
    }

    pub fn finalize_current_module_macro_resolutions(&mut self) {
        let module = self.current_module;
        for &(ref path, span) in module.macro_resolutions.borrow().iter() {
            match self.resolve_path(path, Some(MacroNS), true, span) {
                PathResult::NonModule(_) => {},
                PathResult::Failed(msg, _) => {
                    resolve_error(self, span, ResolutionError::FailedToResolve(&msg));
                }
                _ => unreachable!(),
            }
        }

        for &(mark, ident, span, kind) in module.legacy_macro_resolutions.borrow().iter() {
            let legacy_scope = &self.invocations[&mark].legacy_scope;
            let legacy_resolution = self.resolve_legacy_scope(legacy_scope, ident.name, true);
            let resolution = self.resolve_lexical_macro_path_segment(ident, MacroNS, true, span);
            match (legacy_resolution, resolution) {
                (Some(MacroBinding::Legacy(legacy_binding)), Ok(MacroBinding::Modern(binding))) => {
                    let msg1 = format!("`{}` could refer to the macro defined here", ident);
                    let msg2 = format!("`{}` could also refer to the macro imported here", ident);
                    self.session.struct_span_err(span, &format!("`{}` is ambiguous", ident))
                        .span_note(legacy_binding.span, &msg1)
                        .span_note(binding.span, &msg2)
                        .emit();
                },
                (Some(MacroBinding::Global(binding)), Ok(MacroBinding::Global(_))) => {
                    self.record_use(ident, MacroNS, binding, span);
                    self.err_if_macro_use_proc_macro(ident.name, span, binding);
                },
                (None, Err(_)) => {
                    let msg = match kind {
                        MacroKind::Bang =>
                            format!("cannot find macro `{}!` in this scope", ident),
                        MacroKind::Attr =>
                            format!("cannot find attribute macro `{}` in this scope", ident),
                        MacroKind::Derive =>
                            format!("cannot find derive macro `{}` in this scope", ident),
                    };
                    let mut err = self.session.struct_span_err(span, &msg);
                    self.suggest_macro_name(&ident.name.as_str(), kind, &mut err, span);
                    err.emit();
                },
                _ => {},
            };
        }
    }

    fn suggest_macro_name(&mut self, name: &str, kind: MacroKind,
                          err: &mut DiagnosticBuilder<'a>, span: Span) {
        // First check if this is a locally-defined bang macro.
        let suggestion = if let MacroKind::Bang = kind {
            find_best_match_for_name(self.macro_names.iter(), name, None)
        } else {
            None
        // Then check global macros.
        }.or_else(|| {
            // FIXME: get_macro needs an &mut Resolver, can we do it without cloning?
            let global_macros = self.global_macros.clone();
            let names = global_macros.iter().filter_map(|(name, binding)| {
                if binding.get_macro(self).kind() == kind {
                    Some(name)
                } else {
                    None
                }
            });
            find_best_match_for_name(names, name, None)
        // Then check modules.
        }).or_else(|| {
            if !self.use_extern_macros {
                return None;
            }
            let is_macro = |def| {
                if let Def::Macro(_, def_kind) = def {
                    def_kind == kind
                } else {
                    false
                }
            };
            let ident = Ident::from_str(name);
            self.lookup_typo_candidate(&vec![ident], MacroNS, is_macro, span)
        });

        if let Some(suggestion) = suggestion {
            if suggestion != name {
                if let MacroKind::Bang = kind {
                    err.help(&format!("did you mean `{}!`?", suggestion));
                } else {
                    err.help(&format!("did you mean `{}`?", suggestion));
                }
            } else {
                err.help("have you added the `#[macro_use]` on the module/import?");
            }
        }
    }

    fn collect_def_ids(&mut self, invocation: &'a InvocationData<'a>, expansion: &Expansion) {
        let Resolver { ref mut invocations, arenas, graph_root, .. } = *self;
        let InvocationData { def_index, const_expr, .. } = *invocation;

        let visit_macro_invoc = &mut |invoc: map::MacroInvocationData| {
            invocations.entry(invoc.mark).or_insert_with(|| {
                arenas.alloc_invocation_data(InvocationData {
                    def_index: invoc.def_index,
                    const_expr: invoc.const_expr,
                    module: Cell::new(graph_root),
                    expansion: Cell::new(LegacyScope::Empty),
                    legacy_scope: Cell::new(LegacyScope::Empty),
                })
            });
        };

        let mut def_collector = DefCollector::new(&mut self.definitions);
        def_collector.visit_macro_invoc = Some(visit_macro_invoc);
        def_collector.with_parent(def_index, |def_collector| {
            if const_expr {
                if let Expansion::Expr(ref expr) = *expansion {
                    def_collector.visit_const_expr(expr);
                }
            }
            expansion.visit_with(def_collector)
        });
    }

    pub fn define_macro(&mut self, item: &ast::Item, legacy_scope: &mut LegacyScope<'a>) {
        self.local_macro_def_scopes.insert(item.id, self.current_module);
        let ident = item.ident;
        if ident.name == "macro_rules" {
            self.session.span_err(item.span, "user-defined macros may not be named `macro_rules`");
        }

        let def_id = self.definitions.local_def_id(item.id);
        let ext = Rc::new(macro_rules::compile(&self.session.parse_sess,
                                               &self.session.features,
                                               item));
        self.macro_map.insert(def_id, ext);
        *legacy_scope = LegacyScope::Binding(self.arenas.alloc_legacy_binding(LegacyBinding {
            parent: Cell::new(*legacy_scope), name: ident.name, def_id: def_id, span: item.span,
        }));
        self.macro_names.insert(ident.name);

        if attr::contains_name(&item.attrs, "macro_export") {
            let def = Def::Macro(def_id, MacroKind::Bang);
            self.macro_exports.push(Export { name: ident.name, def: def, span: item.span });
        } else {
            self.unused_macros.insert(def_id, true);
        }
    }

    /// Error if `ext` is a Macros 1.1 procedural macro being imported by `#[macro_use]`
    fn err_if_macro_use_proc_macro(&mut self, name: Name, use_span: Span,
                                   binding: &NameBinding<'a>) {
        use self::SyntaxExtension::*;

        let krate = binding.def().def_id().krate;

        // Plugin-based syntax extensions are exempt from this check
        if krate == BUILTIN_MACROS_CRATE { return; }

        let ext = binding.get_macro(self);

        match *ext {
            // If `ext` is a procedural macro, check if we've already warned about it
            AttrProcMacro(_) | ProcMacro(_) => if !self.warned_proc_macros.insert(name) { return; },
            _ => return,
        }

        let warn_msg = match *ext {
            AttrProcMacro(_) => "attribute procedural macros cannot be \
                                 imported with `#[macro_use]`",
            ProcMacro(_) => "procedural macros cannot be imported with `#[macro_use]`",
            _ => return,
        };

        let crate_name = self.session.cstore.crate_name(krate);

        self.session.struct_span_err(use_span, warn_msg)
            .help(&format!("instead, import the procedural macro like any other item: \
                             `use {}::{};`", crate_name, name))
            .emit();
    }

    fn gate_legacy_custom_derive(&mut self, name: Symbol, span: Span) {
        if !self.session.features.borrow().custom_derive {
            let sess = &self.session.parse_sess;
            let explain = feature_gate::EXPLAIN_CUSTOM_DERIVE;
            emit_feature_err(sess, "custom_derive", span, GateIssue::Language, explain);
        } else if !self.is_whitelisted_legacy_custom_derive(name) {
            self.session.span_warn(span, feature_gate::EXPLAIN_DEPR_CUSTOM_DERIVE);
        }
    }
}
