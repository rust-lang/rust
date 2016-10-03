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
use std::rc::Rc;
use syntax::ast;
use syntax::errors::DiagnosticBuilder;
use syntax::ext::base::{self, MultiModifier, MultiDecorator, MultiItemModifier};
use syntax::ext::base::{NormalTT, SyntaxExtension};
use syntax::ext::expand::{Expansion, Invocation, InvocationKind};
use syntax::ext::hygiene::{Mark, SyntaxContext};
use syntax::ext::tt::macro_rules;
use syntax::parse::token::intern;
use syntax::util::lev_distance::find_best_match_for_name;
use syntax_pos::{Span, DUMMY_SP};

// FIXME(jseyfried) Merge with `::NameBinding`.
pub struct NameBinding {
    pub ext: Rc<SyntaxExtension>,
    pub expansion: Mark,
    pub shadowing: bool,
    pub span: Span,
}

#[derive(Clone)]
pub struct ExpansionData<'a> {
    backtrace: SyntaxContext,
    pub module: Module<'a>,
    def_index: DefIndex,
    // True if this expansion is in a `const_integer` position, for example `[u32; m!()]`.
    // c.f. `DefCollector::visit_ast_const_integer`.
    const_integer: bool,
}

impl<'a> ExpansionData<'a> {
    pub fn root(graph_root: Module<'a>) -> Self {
        ExpansionData {
            backtrace: SyntaxContext::empty(),
            module: graph_root,
            def_index: CRATE_DEF_INDEX,
            const_integer: false,
        }
    }
}

impl<'a> base::Resolver for Resolver<'a> {
    fn next_node_id(&mut self) -> ast::NodeId {
        self.session.next_node_id()
    }

    fn get_module_scope(&mut self, id: ast::NodeId) -> Mark {
        let mark = Mark::fresh();
        let module = self.module_map[&id];
        self.expansion_data.insert(mark, ExpansionData {
            backtrace: SyntaxContext::empty(),
            module: module,
            def_index: module.def_id().unwrap().index,
            const_integer: false,
        });
        mark
    }

    fn visit_expansion(&mut self, mark: Mark, expansion: &Expansion) {
        self.collect_def_ids(mark, expansion);
        self.current_module = self.expansion_data[&mark].module;
        expansion.visit_with(&mut BuildReducedGraphVisitor { resolver: self, expansion: mark });
    }

    fn add_macro(&mut self, scope: Mark, mut def: ast::MacroDef) {
        if &def.ident.name.as_str() == "macro_rules" {
            self.session.span_err(def.span, "user-defined macros may not be named `macro_rules`");
        }
        if def.use_locally {
            let ExpansionData { mut module, backtrace, .. } = self.expansion_data[&scope];
            while module.macros_escape {
                module = module.parent.unwrap();
            }
            let binding = NameBinding {
                ext: Rc::new(macro_rules::compile(&self.session.parse_sess, &def)),
                expansion: backtrace.data().prev_ctxt.data().outer_mark,
                shadowing: self.resolve_macro_name(scope, def.ident.name, false).is_some(),
                span: def.span,
            };
            module.macros.borrow_mut().insert(def.ident.name, binding);
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
        self.graph_root.macros.borrow_mut().insert(ident.name, NameBinding {
            ext: ext,
            expansion: Mark::root(),
            shadowing: false,
            span: DUMMY_SP,
        });
    }

    fn add_expansions_at_stmt(&mut self, id: ast::NodeId, macros: Vec<Mark>) {
        self.macros_at_scope.insert(id, macros);
    }

    fn find_attr_invoc(&mut self, attrs: &mut Vec<ast::Attribute>) -> Option<ast::Attribute> {
        for i in 0..attrs.len() {
            let name = intern(&attrs[i].name());
            match self.expansion_data[&Mark::root()].module.macros.borrow().get(&name) {
                Some(binding) => match *binding.ext {
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

    fn resolve_invoc(&mut self, scope: Mark, invoc: &Invocation) -> Option<Rc<SyntaxExtension>> {
        let (name, span) = match invoc.kind {
            InvocationKind::Bang { ref mac, .. } => {
                let path = &mac.node.path;
                if path.segments.len() > 1 || path.global ||
                   !path.segments[0].parameters.is_empty() {
                    self.session.span_err(path.span,
                                          "expected macro name without module separators");
                    return None;
                }
                (path.segments[0].identifier.name, path.span)
            }
            InvocationKind::Attr { ref attr, .. } => (intern(&*attr.name()), attr.span),
        };

        self.resolve_macro_name(scope, name, true).or_else(|| {
            let mut err =
                self.session.struct_span_err(span, &format!("macro undefined: '{}!'", name));
            self.suggest_macro_name(&name.as_str(), &mut err);
            err.emit();
            None
        })
    }

    fn resolve_derive_mode(&mut self, ident: ast::Ident) -> Option<Rc<MultiItemModifier>> {
        self.derive_modes.get(&ident.name).cloned()
    }
}

impl<'a> Resolver<'a> {
    pub fn resolve_macro_name(&mut self, scope: Mark, name: ast::Name, record_used: bool)
                              -> Option<Rc<SyntaxExtension>> {
        let ExpansionData { mut module, backtrace, .. } = self.expansion_data[&scope];
        loop {
            if let Some(binding) = module.macros.borrow().get(&name) {
                let mut backtrace = backtrace.data();
                while binding.expansion != backtrace.outer_mark {
                    if backtrace.outer_mark != Mark::root() {
                        backtrace = backtrace.prev_ctxt.data();
                        continue
                    }

                    if record_used && binding.shadowing &&
                       self.macro_shadowing_errors.insert(binding.span) {
                        let msg = format!("`{}` is already in scope", name);
                        self.session.struct_span_err(binding.span, &msg)
                            .note("macro-expanded `macro_rules!`s and `#[macro_use]`s \
                                   may not shadow existing macros (see RFC 1560)")
                            .emit();
                    }
                    break
                }
                return Some(binding.ext.clone());
            }
            match module.parent {
                Some(parent) => module = parent,
                None => break,
            }
        }
        None
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

    fn collect_def_ids(&mut self, mark: Mark, expansion: &Expansion) {
        let expansion_data = &mut self.expansion_data;
        let ExpansionData { backtrace, def_index, const_integer, module } = expansion_data[&mark];
        let visit_macro_invoc = &mut |invoc: map::MacroInvocationData| {
            expansion_data.entry(invoc.mark).or_insert(ExpansionData {
                backtrace: backtrace.apply_mark(invoc.mark),
                def_index: invoc.def_index,
                const_integer: invoc.const_integer,
                module: module,
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
