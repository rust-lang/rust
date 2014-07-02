// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This implements the dead-code warning pass. It follows middle::reachable
// closely. The idea is that all reachable symbols are live, codes called
// from live codes are live, and everything else is dead.

use middle::def;
use lint;
use middle::privacy;
use middle::ty;
use middle::typeck;
use util::nodemap::NodeSet;

use std::collections::HashSet;
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::{local_def, is_local};
use syntax::attr::AttrMetaMethods;
use syntax::attr;
use syntax::codemap;
use syntax::parse::token;
use syntax::visit::Visitor;
use syntax::visit;

// Any local node that may call something in its body block should be
// explored. For example, if it's a live NodeItem that is a
// function, then we should explore its block to check for codes that
// may need to be marked as live.
fn should_explore(tcx: &ty::ctxt, def_id: ast::DefId) -> bool {
    if !is_local(def_id) {
        return false;
    }

    match tcx.map.find(def_id.node) {
        Some(ast_map::NodeItem(..))
        | Some(ast_map::NodeMethod(..))
        | Some(ast_map::NodeForeignItem(..))
        | Some(ast_map::NodeTraitMethod(..)) => true,
        _ => false
    }
}

struct MarkSymbolVisitor<'a> {
    worklist: Vec<ast::NodeId>,
    tcx: &'a ty::ctxt,
    live_symbols: Box<HashSet<ast::NodeId>>,
}

#[deriving(Clone)]
struct MarkSymbolVisitorContext {
    struct_has_extern_repr: bool
}

impl<'a> MarkSymbolVisitor<'a> {
    fn new(tcx: &'a ty::ctxt,
           worklist: Vec<ast::NodeId>) -> MarkSymbolVisitor<'a> {
        MarkSymbolVisitor {
            worklist: worklist,
            tcx: tcx,
            live_symbols: box HashSet::new(),
        }
    }

    fn check_def_id(&mut self, def_id: ast::DefId) {
        if should_explore(self.tcx, def_id) {
            self.worklist.push(def_id.node);
        }
        self.live_symbols.insert(def_id.node);
    }

    fn lookup_and_handle_definition(&mut self, id: &ast::NodeId) {
        let def = match self.tcx.def_map.borrow().find(id) {
            Some(&def) => def,
            None => return
        };
        let def_id = match def {
            def::DefVariant(enum_id, _, _) => Some(enum_id),
            def::DefPrimTy(_) => None,
            _ => Some(def.def_id())
        };
        match def_id {
            Some(def_id) => self.check_def_id(def_id),
            None => (),
        }
    }

    fn lookup_and_handle_method(&mut self, id: ast::NodeId,
                                span: codemap::Span) {
        let method_call = typeck::MethodCall::expr(id);
        match self.tcx.method_map.borrow().find(&method_call) {
            Some(method) => {
                match method.origin {
                    typeck::MethodStatic(def_id) => {
                        match ty::provided_source(self.tcx, def_id) {
                            Some(p_did) => self.check_def_id(p_did),
                            None => self.check_def_id(def_id)
                        }
                    }
                    typeck::MethodParam(typeck::MethodParam {
                        trait_id: trait_id,
                        method_num: index,
                        ..
                    })
                    | typeck::MethodObject(typeck::MethodObject {
                        trait_id: trait_id,
                        method_num: index,
                        ..
                    }) => {
                        let def_id = ty::trait_method(self.tcx,
                                                      trait_id, index).def_id;
                        self.check_def_id(def_id);
                    }
                }
            }
            None => {
                self.tcx.sess.span_bug(span,
                                       "method call expression not \
                                        in method map?!")
            }
        }
    }

    fn handle_field_access(&mut self, lhs: &ast::Expr, name: &ast::Ident) {
        match ty::get(ty::expr_ty_adjusted(self.tcx, lhs)).sty {
            ty::ty_struct(id, _) => {
                let fields = ty::lookup_struct_fields(self.tcx, id);
                let field_id = fields.iter()
                    .find(|field| field.name == name.name).unwrap().id;
                self.live_symbols.insert(field_id.node);
            },
            _ => ()
        }
    }

    fn handle_field_pattern_match(&mut self, lhs: &ast::Pat, pats: &[ast::FieldPat]) {
        match self.tcx.def_map.borrow().get(&lhs.id) {
            &def::DefStruct(id) | &def::DefVariant(_, id, _) => {
                let fields = ty::lookup_struct_fields(self.tcx, id);
                for pat in pats.iter() {
                    let field_id = fields.iter()
                        .find(|field| field.name == pat.ident.name).unwrap().id;
                    self.live_symbols.insert(field_id.node);
                }
            }
            _ => ()
        }
    }

    fn mark_live_symbols(&mut self) {
        let mut scanned = HashSet::new();
        while self.worklist.len() > 0 {
            let id = self.worklist.pop().unwrap();
            if scanned.contains(&id) {
                continue
            }
            scanned.insert(id);

            match self.tcx.map.find(id) {
                Some(ref node) => {
                    self.live_symbols.insert(id);
                    self.visit_node(node);
                }
                None => (),
            }
        }
    }

    fn visit_node(&mut self, node: &ast_map::Node) {
        let ctxt = MarkSymbolVisitorContext {
            struct_has_extern_repr: false
        };
        match *node {
            ast_map::NodeItem(item) => {
                match item.node {
                    ast::ItemStruct(..) => {
                        let has_extern_repr = item.attrs.iter().fold(attr::ReprAny, |acc, attr| {
                            attr::find_repr_attr(self.tcx.sess.diagnostic(), attr, acc)
                        }) == attr::ReprExtern;

                        visit::walk_item(self, &*item, MarkSymbolVisitorContext {
                            struct_has_extern_repr: has_extern_repr,
                            ..(ctxt)
                        });
                    }
                    ast::ItemFn(..)
                    | ast::ItemEnum(..)
                    | ast::ItemTy(..)
                    | ast::ItemStatic(..) => {
                        visit::walk_item(self, &*item, ctxt);
                    }
                    _ => ()
                }
            }
            ast_map::NodeTraitMethod(trait_method) => {
                visit::walk_trait_method(self, &*trait_method, ctxt);
            }
            ast_map::NodeMethod(method) => {
                visit::walk_block(self, &*method.body, ctxt);
            }
            ast_map::NodeForeignItem(foreign_item) => {
                visit::walk_foreign_item(self, &*foreign_item, ctxt);
            }
            _ => ()
        }
    }
}

impl<'a> Visitor<MarkSymbolVisitorContext> for MarkSymbolVisitor<'a> {

    fn visit_struct_def(&mut self, def: &ast::StructDef, _: ast::Ident, _: &ast::Generics,
                        _: ast::NodeId, ctxt: MarkSymbolVisitorContext) {
        let live_fields = def.fields.iter().filter(|f| {
            ctxt.struct_has_extern_repr || match f.node.kind {
                ast::NamedField(_, ast::Public) => true,
                _ => false
            }
        });
        self.live_symbols.extend(live_fields.map(|f| f.node.id));

        visit::walk_struct_def(self, def, ctxt);
    }

    fn visit_expr(&mut self, expr: &ast::Expr, ctxt: MarkSymbolVisitorContext) {
        match expr.node {
            ast::ExprMethodCall(..) => {
                self.lookup_and_handle_method(expr.id, expr.span);
            }
            ast::ExprField(ref lhs, ref ident, _) => {
                self.handle_field_access(&**lhs, &ident.node);
            }
            _ => ()
        }

        visit::walk_expr(self, expr, ctxt);
    }

    fn visit_pat(&mut self, pat: &ast::Pat, ctxt: MarkSymbolVisitorContext) {
        match pat.node {
            ast::PatStruct(_, ref fields, _) => {
                self.handle_field_pattern_match(pat, fields.as_slice());
            }
            _ => ()
        }

        visit::walk_pat(self, pat, ctxt);
    }

    fn visit_path(&mut self, path: &ast::Path, id: ast::NodeId, ctxt: MarkSymbolVisitorContext) {
        self.lookup_and_handle_definition(&id);
        visit::walk_path(self, path, ctxt);
    }

    fn visit_item(&mut self, _: &ast::Item, _: MarkSymbolVisitorContext) {
        // Do not recurse into items. These items will be added to the
        // worklist and recursed into manually if necessary.
    }
}

fn has_allow_dead_code_or_lang_attr(attrs: &[ast::Attribute]) -> bool {
    if attr::contains_name(attrs.as_slice(), "lang") {
        return true;
    }

    let dead_code = lint::builtin::DEAD_CODE.name_lower();
    for attr in lint::gather_attrs(attrs).move_iter() {
        match attr {
            Ok((ref name, lint::Allow, _))
                if name.get() == dead_code.as_slice() => return true,
            _ => (),
        }
    }
    false
}

// This visitor seeds items that
//   1) We want to explicitly consider as live:
//     * Item annotated with #[allow(dead_code)]
//         - This is done so that if we want to suppress warnings for a
//           group of dead functions, we only have to annotate the "root".
//           For example, if both `f` and `g` are dead and `f` calls `g`,
//           then annotating `f` with `#[allow(dead_code)]` will suppress
//           warning for both `f` and `g`.
//     * Item annotated with #[lang=".."]
//         - This is because lang items are always callable from elsewhere.
//   or
//   2) We are not sure to be live or not
//     * Implementation of a trait method
struct LifeSeeder {
    worklist: Vec<ast::NodeId> ,
}

impl Visitor<()> for LifeSeeder {
    fn visit_item(&mut self, item: &ast::Item, _: ()) {
        if has_allow_dead_code_or_lang_attr(item.attrs.as_slice()) {
            self.worklist.push(item.id);
        }
        match item.node {
            ast::ItemImpl(_, Some(ref _trait_ref), _, ref methods) => {
                for method in methods.iter() {
                    self.worklist.push(method.id);
                }
            }
            _ => ()
        }
        visit::walk_item(self, item, ());
    }

    fn visit_fn(&mut self, fk: &visit::FnKind,
                _: &ast::FnDecl, block: &ast::Block,
                _: codemap::Span, id: ast::NodeId, _: ()) {
        // Check for method here because methods are not ast::Item
        match *fk {
            visit::FkMethod(_, _, method) => {
                if has_allow_dead_code_or_lang_attr(method.attrs.as_slice()) {
                    self.worklist.push(id);
                }
            }
            _ => ()
        }
        visit::walk_block(self, block, ());
    }
}

fn create_and_seed_worklist(tcx: &ty::ctxt,
                            exported_items: &privacy::ExportedItems,
                            reachable_symbols: &NodeSet,
                            krate: &ast::Crate) -> Vec<ast::NodeId> {
    let mut worklist = Vec::new();

    // Preferably, we would only need to seed the worklist with reachable
    // symbols. However, since the set of reachable symbols differs
    // depending on whether a crate is built as bin or lib, and we want
    // the warning to be consistent, we also seed the worklist with
    // exported symbols.
    for &id in exported_items.iter() {
        worklist.push(id);
    }
    for &id in reachable_symbols.iter() {
        worklist.push(id);
    }

    // Seed entry point
    match *tcx.sess.entry_fn.borrow() {
        Some((id, _)) => worklist.push(id),
        None => ()
    }

    // Seed implemented trait methods
    let mut life_seeder = LifeSeeder {
        worklist: worklist
    };
    visit::walk_crate(&mut life_seeder, krate, ());

    return life_seeder.worklist;
}

fn find_live(tcx: &ty::ctxt,
             exported_items: &privacy::ExportedItems,
             reachable_symbols: &NodeSet,
             krate: &ast::Crate)
             -> Box<HashSet<ast::NodeId>> {
    let worklist = create_and_seed_worklist(tcx, exported_items,
                                            reachable_symbols, krate);
    let mut symbol_visitor = MarkSymbolVisitor::new(tcx, worklist);
    symbol_visitor.mark_live_symbols();
    symbol_visitor.live_symbols
}

fn should_warn(item: &ast::Item) -> bool {
    match item.node {
        ast::ItemStatic(..)
        | ast::ItemFn(..)
        | ast::ItemEnum(..)
        | ast::ItemStruct(..) => true,
        _ => false
    }
}

fn get_struct_ctor_id(item: &ast::Item) -> Option<ast::NodeId> {
    match item.node {
        ast::ItemStruct(struct_def, _) => struct_def.ctor_id,
        _ => None
    }
}

struct DeadVisitor<'a> {
    tcx: &'a ty::ctxt,
    live_symbols: Box<HashSet<ast::NodeId>>,
}

impl<'a> DeadVisitor<'a> {
    fn should_warn_about_field(&mut self, node: &ast::StructField_) -> bool {
        let (is_named, has_leading_underscore) = match node.ident() {
            Some(ref ident) => (true, token::get_ident(*ident).get().as_bytes()[0] == ('_' as u8)),
            _ => (false, false)
        };
        let field_type = ty::node_id_to_type(self.tcx, node.id);
        let is_marker_field = match ty::ty_to_def_id(field_type) {
            Some(def_id) => self.tcx.lang_items.items().any(|(_, item)| *item == Some(def_id)),
            _ => false
        };
        is_named
            && !self.symbol_is_live(node.id, None)
            && !has_leading_underscore
            && !is_marker_field
            && !has_allow_dead_code_or_lang_attr(node.attrs.as_slice())
    }

    // id := node id of an item's definition.
    // ctor_id := `Some` if the item is a struct_ctor (tuple struct),
    //            `None` otherwise.
    // If the item is a struct_ctor, then either its `id` or
    // `ctor_id` (unwrapped) is in the live_symbols set. More specifically,
    // DefMap maps the ExprPath of a struct_ctor to the node referred by
    // `ctor_id`. On the other hand, in a statement like
    // `type <ident> <generics> = <ty>;` where <ty> refers to a struct_ctor,
    // DefMap maps <ty> to `id` instead.
    fn symbol_is_live(&mut self, id: ast::NodeId,
                      ctor_id: Option<ast::NodeId>) -> bool {
        if self.live_symbols.contains(&id)
           || ctor_id.map_or(false,
                             |ctor| self.live_symbols.contains(&ctor)) {
            return true;
        }
        // If it's a type whose methods are live, then it's live, too.
        // This is done to handle the case where, for example, the static
        // method of a private type is used, but the type itself is never
        // called directly.
        let impl_methods = self.tcx.impl_methods.borrow();
        match self.tcx.inherent_impls.borrow().find(&local_def(id)) {
            None => (),
            Some(impl_list) => {
                for impl_did in impl_list.borrow().iter() {
                    for method_did in impl_methods.get(impl_did).iter() {
                        if self.live_symbols.contains(&method_did.node) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn warn_dead_code(&mut self,
                      id: ast::NodeId,
                      span: codemap::Span,
                      ident: ast::Ident) {
        self.tcx
            .sess
            .add_lint(lint::builtin::DEAD_CODE,
                      id,
                      span,
                      format!("code is never used: `{}`",
                              token::get_ident(ident)));
    }
}

impl<'a> Visitor<()> for DeadVisitor<'a> {
    fn visit_item(&mut self, item: &ast::Item, _: ()) {
        let ctor_id = get_struct_ctor_id(item);
        if !self.symbol_is_live(item.id, ctor_id) && should_warn(item) {
            self.warn_dead_code(item.id, item.span, item.ident);
        }
        visit::walk_item(self, item, ());
    }

    fn visit_foreign_item(&mut self, fi: &ast::ForeignItem, _: ()) {
        if !self.symbol_is_live(fi.id, None) {
            self.warn_dead_code(fi.id, fi.span, fi.ident);
        }
        visit::walk_foreign_item(self, fi, ());
    }

    fn visit_fn(&mut self, fk: &visit::FnKind,
                _: &ast::FnDecl, block: &ast::Block,
                span: codemap::Span, id: ast::NodeId, _: ()) {
        // Have to warn method here because methods are not ast::Item
        match *fk {
            visit::FkMethod(..) => {
                let ident = visit::name_of_fn(fk);
                if !self.symbol_is_live(id, None) {
                    self.warn_dead_code(id, span, ident);
                }
            }
            _ => ()
        }
        visit::walk_block(self, block, ());
    }

    fn visit_struct_field(&mut self, field: &ast::StructField, _: ()) {
        if self.should_warn_about_field(&field.node) {
            self.warn_dead_code(field.node.id, field.span, field.node.ident().unwrap());
        }

        visit::walk_struct_field(self, field, ());
    }

    // Overwrite so that we don't warn the trait method itself.
    fn visit_trait_method(&mut self, trait_method: &ast::TraitMethod, _: ()) {
        match *trait_method {
            ast::Provided(ref method) => visit::walk_block(self, &*method.body, ()),
            ast::Required(_) => ()
        }
    }
}

pub fn check_crate(tcx: &ty::ctxt,
                   exported_items: &privacy::ExportedItems,
                   reachable_symbols: &NodeSet,
                   krate: &ast::Crate) {
    let live_symbols = find_live(tcx, exported_items,
                                 reachable_symbols, krate);
    let mut visitor = DeadVisitor { tcx: tcx, live_symbols: live_symbols };
    visit::walk_crate(&mut visitor, krate, ());
}
