// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use Resolver;
use rustc::middle::cstore::LoadedMacro;
use rustc::util::nodemap::FnvHashMap;
use std::cell::RefCell;
use std::mem;
use std::rc::Rc;
use syntax::ast::{self, Name};
use syntax::errors::DiagnosticBuilder;
use syntax::ext::base::{self, MultiModifier, MultiDecorator, MultiItemModifier};
use syntax::ext::base::{NormalTT, Resolver as SyntaxResolver, SyntaxExtension};
use syntax::ext::expand::{Expansion, Invocation, InvocationKind};
use syntax::ext::hygiene::Mark;
use syntax::ext::tt::macro_rules;
use syntax::feature_gate::{self, emit_feature_err};
use syntax::parse::token::{self, intern};
use syntax::util::lev_distance::find_best_match_for_name;
use syntax::visit::{self, Visitor};
use syntax_pos::Span;

#[derive(Clone, Default)]
pub struct ExpansionData {
    module: Rc<ModuleData>,
}

// FIXME(jseyfried): merge with `::ModuleS`.
#[derive(Default)]
struct ModuleData {
    parent: Option<Rc<ModuleData>>,
    macros: RefCell<FnvHashMap<Name, Rc<SyntaxExtension>>>,
    macros_escape: bool,
}

impl<'a> base::Resolver for Resolver<'a> {
    fn next_node_id(&mut self) -> ast::NodeId {
        self.session.next_node_id()
    }

    fn visit_expansion(&mut self, mark: Mark, expansion: &Expansion) {
        expansion.visit_with(&mut ExpansionVisitor {
            current_module: self.expansion_data[&mark.as_u32()].module.clone(),
            resolver: self,
        });
    }

    fn add_macro(&mut self, scope: Mark, mut def: ast::MacroDef) {
        if &def.ident.name.as_str() == "macro_rules" {
            self.session.span_err(def.span, "user-defined macros may not be named `macro_rules`");
        }
        if def.use_locally {
            let ext = macro_rules::compile(&self.session.parse_sess, &def);
            self.add_ext(scope, def.ident, Rc::new(ext));
        }
        if def.export {
            def.id = self.next_node_id();
            self.exported_macros.push(def);
        }
    }

    fn add_ext(&mut self, scope: Mark, ident: ast::Ident, ext: Rc<SyntaxExtension>) {
        if let NormalTT(..) = *ext {
            self.macro_names.insert(ident.name);
        }

        let mut module = self.expansion_data[&scope.as_u32()].module.clone();
        while module.macros_escape {
            module = module.parent.clone().unwrap();
        }
        module.macros.borrow_mut().insert(ident.name, ext);
    }

    fn add_expansions_at_stmt(&mut self, id: ast::NodeId, macros: Vec<Mark>) {
        self.macros_at_scope.insert(id, macros);
    }

    fn find_attr_invoc(&mut self, attrs: &mut Vec<ast::Attribute>) -> Option<ast::Attribute> {
        for i in 0..attrs.len() {
            let name = intern(&attrs[i].name());
            match self.expansion_data[&0].module.macros.borrow().get(&name) {
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

        let mut module = self.expansion_data[&scope.as_u32()].module.clone();
        loop {
            if let Some(ext) = module.macros.borrow().get(&name) {
                return Some(ext.clone());
            }
            match module.parent.clone() {
                Some(parent) => module = parent,
                None => break,
            }
        }

        let mut err =
            self.session.struct_span_err(span, &format!("macro undefined: '{}!'", name));
        self.suggest_macro_name(&name.as_str(), &mut err);
        err.emit();
        None
    }

    fn resolve_derive_mode(&mut self, ident: ast::Ident) -> Option<Rc<MultiItemModifier>> {
        self.derive_modes.get(&ident.name).cloned()
    }
}

impl<'a> Resolver<'a> {
    fn suggest_macro_name(&mut self, name: &str, err: &mut DiagnosticBuilder<'a>) {
        if let Some(suggestion) = find_best_match_for_name(self.macro_names.iter(), name, None) {
            if suggestion != name {
                err.help(&format!("did you mean `{}!`?", suggestion));
            } else {
                err.help(&format!("have you added the `#[macro_use]` on the module/import?"));
            }
        }
    }

    fn insert_custom_derive(&mut self, name: &str, ext: Rc<MultiItemModifier>, sp: Span) {
        if !self.session.features.borrow().rustc_macro {
            let diagnostic = &self.session.parse_sess.span_diagnostic;
            let msg = "loading custom derive macro crates is experimentally supported";
            emit_feature_err(diagnostic, "rustc_macro", sp, feature_gate::GateIssue::Language, msg);
        }
        if self.derive_modes.insert(token::intern(name), ext).is_some() {
            self.session.span_err(sp, &format!("cannot shadow existing derive mode `{}`", name));
        }
    }
}

struct ExpansionVisitor<'b, 'a: 'b> {
    resolver: &'b mut Resolver<'a>,
    current_module: Rc<ModuleData>,
}

impl<'a, 'b> ExpansionVisitor<'a, 'b> {
    fn visit_invoc(&mut self, id: ast::NodeId) {
        self.resolver.expansion_data.insert(id.as_u32(), ExpansionData {
            module: self.current_module.clone(),
        });
    }

    // does this attribute list contain "macro_use"?
    fn contains_macro_use(&mut self, attrs: &[ast::Attribute]) -> bool {
        for attr in attrs {
            if attr.check_name("macro_escape") {
                let msg = "macro_escape is a deprecated synonym for macro_use";
                let mut err = self.resolver.session.struct_span_warn(attr.span, msg);
                if let ast::AttrStyle::Inner = attr.node.style {
                    err.help("consider an outer attribute, #[macro_use] mod ...").emit();
                } else {
                    err.emit();
                }
            } else if !attr.check_name("macro_use") {
                continue;
            }

            if !attr.is_word() {
                self.resolver.session.span_err(attr.span,
                                               "arguments to macro_use are not allowed here");
            }
            return true;
        }

        false
    }
}

macro_rules! method {
    ($visit:ident: $ty:ty, $invoc:path, $walk:ident) => {
        fn $visit(&mut self, node: &$ty) {
            match node.node {
                $invoc(..) => self.visit_invoc(node.id),
                _ => visit::$walk(self, node),
            }
        }
    }
}

impl<'a, 'b> Visitor for ExpansionVisitor<'a, 'b>  {
    method!(visit_trait_item: ast::TraitItem, ast::TraitItemKind::Macro, walk_trait_item);
    method!(visit_impl_item:  ast::ImplItem,  ast::ImplItemKind::Macro,  walk_impl_item);
    method!(visit_stmt:       ast::Stmt,      ast::StmtKind::Mac,        walk_stmt);
    method!(visit_expr:       ast::Expr,      ast::ExprKind::Mac,        walk_expr);
    method!(visit_pat:        ast::Pat,       ast::PatKind::Mac,         walk_pat);
    method!(visit_ty:         ast::Ty,        ast::TyKind::Mac,          walk_ty);

    fn visit_item(&mut self, item: &ast::Item) {
        match item.node {
            ast::ItemKind::Mac(..) if item.id == ast::DUMMY_NODE_ID => {} // Scope placeholder
            ast::ItemKind::Mac(..) => self.visit_invoc(item.id),
            ast::ItemKind::Mod(..) => {
                let module_data = ModuleData {
                    parent: Some(self.current_module.clone()),
                    macros: RefCell::new(FnvHashMap()),
                    macros_escape: self.contains_macro_use(&item.attrs),
                };
                let orig_module = mem::replace(&mut self.current_module, Rc::new(module_data));
                visit::walk_item(self, item);
                self.current_module = orig_module;
            }
            ast::ItemKind::ExternCrate(..) => {
                // We need to error on `#[macro_use] extern crate` when it isn't at the
                // crate root, because `$crate` won't work properly.
                // FIXME(jseyfried): This will be nicer once `ModuleData` is merged with `ModuleS`.
                let is_crate_root = self.current_module.parent.as_ref().unwrap().parent.is_none();
                for def in self.resolver.crate_loader.load_macros(item, is_crate_root) {
                    match def {
                        LoadedMacro::Def(def) => self.resolver.add_macro(Mark::root(), def),
                        LoadedMacro::CustomDerive(name, ext) => {
                            self.resolver.insert_custom_derive(&name, ext, item.span);
                        }
                    }
                }
                visit::walk_item(self, item);
            }
            _ => visit::walk_item(self, item),
        }
    }

    fn visit_block(&mut self, block: &ast::Block) {
        let module_data = ModuleData {
            parent: Some(self.current_module.clone()),
            macros: RefCell::new(FnvHashMap()),
            macros_escape: false,
        };
        let orig_module = mem::replace(&mut self.current_module, Rc::new(module_data));
        visit::walk_block(self, block);
        self.current_module = orig_module;
    }
}
