// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use {Module, ModuleKind, Resolver};
use build_reduced_graph::BuildReducedGraphVisitor;
use rustc::hir::def_id::{DefId, BUILTIN_MACROS_CRATE, CRATE_DEF_INDEX, DefIndex};
use rustc::hir::def::{Def, Export};
use rustc::hir::map::{self, DefCollector};
use std::cell::Cell;
use std::rc::Rc;
use syntax::ast;
use syntax::errors::DiagnosticBuilder;
use syntax::ext::base::{self, Determinacy, MultiModifier, MultiDecorator};
use syntax::ext::base::{NormalTT, SyntaxExtension};
use syntax::ext::expand::Expansion;
use syntax::ext::hygiene::Mark;
use syntax::ext::tt::macro_rules;
use syntax::fold::Folder;
use syntax::parse::token::intern;
use syntax::ptr::P;
use syntax::util::lev_distance::find_best_match_for_name;
use syntax::visit::Visitor;
use syntax_pos::Span;

#[derive(Clone)]
pub struct InvocationData<'a> {
    pub module: Cell<Module<'a>>,
    pub def_index: DefIndex,
    // True if this expansion is in a `const_integer` position, for example `[u32; m!()]`.
    // c.f. `DefCollector::visit_ast_const_integer`.
    pub const_integer: bool,
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
    pub parent: LegacyScope<'a>,
    pub name: ast::Name,
    ext: Rc<SyntaxExtension>,
    pub span: Span,
}

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

    fn eliminate_crate_var(&mut self, item: P<ast::Item>) -> P<ast::Item> {
        struct EliminateCrateVar<'b, 'a: 'b>(&'b mut Resolver<'a>);

        impl<'a, 'b> Folder for EliminateCrateVar<'a, 'b> {
            fn fold_path(&mut self, mut path: ast::Path) -> ast::Path {
                let ident = path.segments[0].identifier;
                if &ident.name.as_str() == "$crate" {
                    path.global = true;
                    let module = self.0.resolve_crate_var(ident.ctxt);
                    if module.is_local() {
                        path.segments.remove(0);
                    } else {
                        path.segments[0].identifier = match module.kind {
                            ModuleKind::Def(_, name) => ast::Ident::with_empty_ctxt(name),
                            _ => unreachable!(),
                        };
                    }
                }
                path
            }
        }

        EliminateCrateVar(self).fold_item(item).expect_one("")
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

    fn add_macro(&mut self, scope: Mark, mut def: ast::MacroDef, export: bool) {
        if &def.ident.name.as_str() == "macro_rules" {
            self.session.span_err(def.span, "user-defined macros may not be named `macro_rules`");
        }

        let invocation = self.invocations[&scope];
        let binding = self.arenas.alloc_legacy_binding(LegacyBinding {
            parent: invocation.legacy_scope.get(),
            name: def.ident.name,
            ext: Rc::new(macro_rules::compile(&self.session.parse_sess, &def)),
            span: def.span,
        });
        invocation.legacy_scope.set(LegacyScope::Binding(binding));
        self.macro_names.insert(def.ident.name);

        if export {
            def.id = self.next_node_id();
            DefCollector::new(&mut self.definitions).with_parent(CRATE_DEF_INDEX, |collector| {
                collector.visit_macro_def(&def)
            });
            self.macro_exports.push(Export {
                name: def.ident.name,
                def: Def::Macro(self.definitions.local_def_id(def.id)),
            });
            self.exported_macros.push(def);
        }
    }

    fn add_ext(&mut self, ident: ast::Ident, ext: Rc<SyntaxExtension>) {
        if let NormalTT(..) = *ext {
            self.macro_names.insert(ident.name);
        }
        let def_id = DefId {
            krate: BUILTIN_MACROS_CRATE,
            index: DefIndex::new(self.macro_map.len()),
        };
        self.macro_map.insert(def_id, ext);
        self.builtin_macros.insert(ident.name, def_id);
    }

    fn add_expansions_at_stmt(&mut self, id: ast::NodeId, macros: Vec<Mark>) {
        self.macros_at_scope.insert(id, macros);
    }

    fn find_attr_invoc(&mut self, attrs: &mut Vec<ast::Attribute>) -> Option<ast::Attribute> {
        for i in 0..attrs.len() {
            let name = intern(&attrs[i].name());
            match self.builtin_macros.get(&name) {
                Some(&def_id) => match *self.get_macro(Def::Macro(def_id)) {
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

    fn resolve_macro(&mut self, scope: Mark, path: &ast::Path, force: bool)
                     -> Result<Rc<SyntaxExtension>, Determinacy> {
        if path.segments.len() > 1 || path.global || !path.segments[0].parameters.is_empty() {
            self.session.span_err(path.span, "expected macro name without module separators");
            return Err(Determinacy::Determined);
        }
        let name = path.segments[0].identifier.name;

        let invocation = self.invocations[&scope];
        if let LegacyScope::Expansion(parent) = invocation.legacy_scope.get() {
            invocation.legacy_scope.set(LegacyScope::simplify_expansion(parent));
        }
        self.resolve_macro_name(invocation.legacy_scope.get(), name).ok_or_else(|| {
            if force {
                let msg = format!("macro undefined: '{}!'", name);
                let mut err = self.session.struct_span_err(path.span, &msg);
                self.suggest_macro_name(&name.as_str(), &mut err);
                err.emit();
                Determinacy::Determined
            } else {
                Determinacy::Undetermined
            }
        })
    }
}

impl<'a> Resolver<'a> {
    pub fn resolve_macro_name(&mut self, mut scope: LegacyScope<'a>, name: ast::Name)
                              -> Option<Rc<SyntaxExtension>> {
        let mut possible_time_travel = None;
        let mut relative_depth: u32 = 0;
        loop {
            scope = match scope {
                LegacyScope::Empty => break,
                LegacyScope::Expansion(invocation) => {
                    if let LegacyScope::Empty = invocation.expansion.get() {
                        if possible_time_travel.is_none() {
                            possible_time_travel = Some(scope);
                        }
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
                        if let Some(scope) = possible_time_travel {
                            // Check for disallowed shadowing later
                            self.lexical_macro_resolutions.push((name, scope));
                        } else if relative_depth > 0 {
                            self.disallowed_shadowing.push(binding);
                        }
                        return Some(binding.ext.clone());
                    }
                    binding.parent
                }
            };
        }

        if let Some(scope) = possible_time_travel {
            self.lexical_macro_resolutions.push((name, scope));
        }
        self.builtin_macros.get(&name).cloned().map(|def_id| self.get_macro(Def::Macro(def_id)))
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
