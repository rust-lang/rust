// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use {Module, Resolver};
use build_reduced_graph::BuildReducedGraphVisitor;
use rustc::hir::def_id::{CRATE_DEF_INDEX, DefIndex};
use rustc::hir::map::{self, DefCollector};
use rustc::util::nodemap::FnvHashMap;
use std::cell::Cell;
use std::rc::Rc;
use syntax::ast;
use syntax::errors::DiagnosticBuilder;
use syntax::ext::base::{self, Determinacy, MultiModifier, MultiDecorator, MultiItemModifier};
use syntax::ext::base::{NormalTT, SyntaxExtension};
use syntax::ext::expand::{Expansion, Invocation, InvocationKind};
use syntax::ext::hygiene::Mark;
use syntax::ext::tt::macro_rules;
use syntax::parse::token::intern;
use syntax::util::lev_distance::find_best_match_for_name;
use syntax_pos::Span;

#[derive(Clone)]
pub struct InvocationData<'a> {
    pub module: Cell<Module<'a>>,
    def_index: DefIndex,
    // True if this expansion is in a `const_integer` position, for example `[u32; m!()]`.
    // c.f. `DefCollector::visit_ast_const_integer`.
    const_integer: bool,
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
            const_integer: false,
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

impl<'a> LegacyScope<'a> {
    fn simplify_expansion(mut invoc: &'a InvocationData<'a>) -> Self {
        while let LegacyScope::Invocation(_) = invoc.expansion.get() {
            match invoc.legacy_scope.get() {
                LegacyScope::Expansion(new_invoc) => invoc = new_invoc,
                LegacyScope::Binding(_) => break,
                scope @ _ => return scope,
            }
        }
        LegacyScope::Expansion(invoc)
    }
}

pub struct LegacyBinding<'a> {
    parent: LegacyScope<'a>,
    name: ast::Name,
    ext: Rc<SyntaxExtension>,
    span: Span,
}

pub type LegacyImports = FnvHashMap<ast::Name, (Rc<SyntaxExtension>, Span)>;

impl<'a> base::Resolver for Resolver<'a> {
    fn next_node_id(&mut self) -> ast::NodeId {
        self.session.next_node_id()
    }

    fn get_module_scope(&mut self, id: ast::NodeId) -> Mark {
        let mark = Mark::fresh();
        let module = self.module_map[&id];
        self.invocations.insert(mark, self.arenas.alloc_invocation_data(InvocationData {
            module: Cell::new(module),
            def_index: module.def_id().unwrap().index,
            const_integer: false,
            legacy_scope: Cell::new(LegacyScope::Empty),
            expansion: Cell::new(LegacyScope::Empty),
        }));
        mark
    }

    fn visit_expansion(&mut self, mark: Mark, expansion: &Expansion) {
        let invocation = self.invocations[&mark];
        self.collect_def_ids(invocation, expansion);

        self.current_module = invocation.module.get();
        let mut visitor = BuildReducedGraphVisitor {
            resolver: self,
            legacy_scope: LegacyScope::Invocation(invocation),
            expansion: mark,
        };
        expansion.visit_with(&mut visitor);
        invocation.expansion.set(visitor.legacy_scope);
    }

    fn add_macro(&mut self, scope: Mark, mut def: ast::MacroDef) {
        if &def.ident.name.as_str() == "macro_rules" {
            self.session.span_err(def.span, "user-defined macros may not be named `macro_rules`");
        }
        if def.use_locally {
            let invocation = self.invocations[&scope];
            let binding = self.arenas.alloc_legacy_binding(LegacyBinding {
                parent: invocation.legacy_scope.get(),
                name: def.ident.name,
                ext: Rc::new(macro_rules::compile(&self.session.parse_sess, &def)),
                span: def.span,
            });
            invocation.legacy_scope.set(LegacyScope::Binding(binding));
            self.macro_names.insert(def.ident.name);
        }
        if def.export {
            def.id = self.next_node_id();
            self.exported_macros.push(def);
        }
    }

    fn add_ext(&mut self, ident: ast::Ident, ext: Rc<SyntaxExtension>) {
        if let NormalTT(..) = *ext {
            self.macro_names.insert(ident.name);
        }
        self.builtin_macros.insert(ident.name, ext);
    }

    fn add_expansions_at_stmt(&mut self, id: ast::NodeId, macros: Vec<Mark>) {
        self.macros_at_scope.insert(id, macros);
    }

    fn find_attr_invoc(&mut self, attrs: &mut Vec<ast::Attribute>) -> Option<ast::Attribute> {
        for i in 0..attrs.len() {
            let name = intern(&attrs[i].name());
            match self.builtin_macros.get(&name) {
                Some(ext) => match **ext {
                    MultiModifier(..) | MultiDecorator(..) | SyntaxExtension::AttrProcMacro(..) => {
                        return Some(attrs.remove(i))
                    }
                    _ => {}
                },
                None => {}
            }
        }
        None
    }

    fn resolve_invoc(&mut self, scope: Mark, invoc: &Invocation, force: bool)
                     -> Result<Rc<SyntaxExtension>, Determinacy> {
        let (name, span) = match invoc.kind {
            InvocationKind::Bang { ref mac, .. } => {
                let path = &mac.node.path;
                if path.segments.len() > 1 || path.global ||
                   !path.segments[0].parameters.is_empty() {
                    self.session.span_err(path.span,
                                          "expected macro name without module separators");
                    return Err(Determinacy::Determined);
                }
                (path.segments[0].identifier.name, path.span)
            }
            InvocationKind::Attr { ref attr, .. } => (intern(&*attr.name()), attr.span),
        };

        let invocation = self.invocations[&scope];
        if let LegacyScope::Expansion(parent) = invocation.legacy_scope.get() {
            invocation.legacy_scope.set(LegacyScope::simplify_expansion(parent));
        }
        self.resolve_macro_name(invocation.legacy_scope.get(), name, true).ok_or_else(|| {
            if force {
                let mut err =
                    self.session.struct_span_err(span, &format!("macro undefined: '{}!'", name));
                self.suggest_macro_name(&name.as_str(), &mut err);
                err.emit();
                Determinacy::Determined
            } else {
                Determinacy::Undetermined
            }
        })
    }

    fn resolve_derive_mode(&mut self, ident: ast::Ident) -> Option<Rc<MultiItemModifier>> {
        self.derive_modes.get(&ident.name).cloned()
    }
}

impl<'a> Resolver<'a> {
    pub fn resolve_macro_name(&mut self,
                              mut scope: LegacyScope<'a>,
                              name: ast::Name,
                              record_used: bool)
                              -> Option<Rc<SyntaxExtension>> {
        let mut relative_depth: u32 = 0;
        loop {
            scope = match scope {
                LegacyScope::Empty => break,
                LegacyScope::Expansion(invocation) => {
                    if let LegacyScope::Empty = invocation.expansion.get() {
                        invocation.legacy_scope.get()
                    } else {
                        relative_depth += 1;
                        invocation.expansion.get()
                    }
                }
                LegacyScope::Invocation(invocation) => {
                    relative_depth = relative_depth.saturating_sub(1);
                    invocation.legacy_scope.get()
                }
                LegacyScope::Binding(binding) => {
                    if binding.name == name {
                        if record_used && relative_depth > 0 {
                            self.disallowed_shadowing.push((name, binding.span, binding.parent));
                        }
                        return Some(binding.ext.clone());
                    }
                    binding.parent
                }
            };
        }

        self.builtin_macros.get(&name).cloned()
    }

    fn suggest_macro_name(&mut self, name: &str, err: &mut DiagnosticBuilder<'a>) {
        if let Some(suggestion) = find_best_match_for_name(self.macro_names.iter(), name, None) {
            if suggestion != name {
                err.help(&format!("did you mean `{}!`?", suggestion));
            } else {
                err.help(&format!("have you added the `#[macro_use]` on the module/import?"));
            }
        }
    }

    fn collect_def_ids(&mut self, invocation: &'a InvocationData<'a>, expansion: &Expansion) {
        let Resolver { ref mut invocations, arenas, graph_root, .. } = *self;
        let InvocationData { def_index, const_integer, .. } = *invocation;

        let visit_macro_invoc = &mut |invoc: map::MacroInvocationData| {
            invocations.entry(invoc.mark).or_insert_with(|| {
                arenas.alloc_invocation_data(InvocationData {
                    def_index: invoc.def_index,
                    const_integer: invoc.const_integer,
                    module: Cell::new(graph_root),
                    expansion: Cell::new(LegacyScope::Empty),
                    legacy_scope: Cell::new(LegacyScope::Empty),
                })
            });
        };

        let mut def_collector = DefCollector::new(&mut self.definitions);
        def_collector.visit_macro_invoc = Some(visit_macro_invoc);
        def_collector.with_parent(def_index, |def_collector| {
            if const_integer {
                if let Expansion::Expr(ref expr) = *expansion {
                    def_collector.visit_ast_const_integer(expr);
                }
            }
            expansion.visit_with(def_collector)
        });
    }
}
