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
use rustc::util::nodemap::FnvHashMap;
use std::cell::RefCell;
use std::mem;
use std::rc::Rc;
use syntax::ast::{self, Name};
use syntax::errors::DiagnosticBuilder;
use syntax::ext::base::{self, LoadedMacro, MultiModifier, MultiDecorator};
use syntax::ext::base::{NormalTT, SyntaxExtension};
use syntax::ext::expand::{Expansion, Invocation, InvocationKind};
use syntax::ext::hygiene::Mark;
use syntax::parse::token::intern;
use syntax::util::lev_distance::find_best_match_for_name;
use syntax::visit::{self, Visitor};

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
    fn load_crate(&mut self, extern_crate: &ast::Item, allows_macros: bool) -> Vec<LoadedMacro> {
        self.macro_loader.load_crate(extern_crate, allows_macros)
    }

    fn next_node_id(&mut self) -> ast::NodeId {
        self.session.next_node_id()
    }

    fn visit_expansion(&mut self, mark: Mark, expansion: &Expansion) {
        expansion.visit_with(&mut ExpansionVisitor {
            current_module: self.expansion_data[mark.as_u32() as usize].module.clone(),
            resolver: self,
        });
    }

    fn add_macro(&mut self, scope: Mark, ident: ast::Ident, ext: Rc<SyntaxExtension>) {
        if let NormalTT(..) = *ext {
            self.macro_names.insert(ident.name);
        }

        let mut module = self.expansion_data[scope.as_u32() as usize].module.clone();
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
            match self.expansion_data[0].module.macros.borrow().get(&name) {
                Some(ext) => match **ext {
                    MultiModifier(..) | MultiDecorator(..) => return Some(attrs.remove(i)),
                    _ => {}
                },
                None => {}
            }
        }
        None
    }

    fn resolve_invoc(&mut self, invoc: &Invocation) -> Option<Rc<SyntaxExtension>> {
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

        let mut module = self.expansion_data[invoc.mark().as_u32() as usize].module.clone();
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
}

struct ExpansionVisitor<'b, 'a: 'b> {
    resolver: &'b mut Resolver<'a>,
    current_module: Rc<ModuleData>,
}

impl<'a, 'b> ExpansionVisitor<'a, 'b> {
    fn visit_invoc(&mut self, id: ast::NodeId) {
        assert_eq!(id, self.resolver.expansion_data.len() as u32);
        self.resolver.expansion_data.push(ExpansionData {
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
