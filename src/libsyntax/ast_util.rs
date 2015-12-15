// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::*;
use ast;
use codemap;
use codemap::Span;
use owned_slice::OwnedSlice;
use parse::token;
use print::pprust;
use ptr::P;
use visit::{FnKind, Visitor};
use visit;

use std::cmp;
use std::u32;

pub fn path_name_i(idents: &[Ident]) -> String {
    // FIXME: Bad copies (#2543 -- same for everything else that says "bad")
    idents.iter().map(|i| i.to_string()).collect::<Vec<String>>().join("::")
}

pub fn is_path(e: P<Expr>) -> bool {
    match e.node { ExprPath(..) => true, _ => false }
}


// convert a span and an identifier to the corresponding
// 1-segment path
pub fn ident_to_path(s: Span, identifier: Ident) -> Path {
    ast::Path {
        span: s,
        global: false,
        segments: vec!(
            ast::PathSegment {
                identifier: identifier,
                parameters: ast::AngleBracketedParameters(ast::AngleBracketedParameterData {
                    lifetimes: Vec::new(),
                    types: OwnedSlice::empty(),
                    bindings: OwnedSlice::empty(),
                })
            }
        ),
    }
}

// If path is a single segment ident path, return that ident. Otherwise, return
// None.
pub fn path_to_ident(path: &Path) -> Option<Ident> {
    if path.segments.len() != 1 {
        return None;
    }

    let segment = &path.segments[0];
    if !segment.parameters.is_empty() {
        return None;
    }

    Some(segment.identifier)
}

pub fn ident_to_pat(id: NodeId, s: Span, i: Ident) -> P<Pat> {
    P(Pat {
        id: id,
        node: PatIdent(BindByValue(MutImmutable), codemap::Spanned{span:s, node:i}, None),
        span: s
    })
}

/// Generate a "pretty" name for an `impl` from its type and trait.
/// This is designed so that symbols of `impl`'d methods give some
/// hint of where they came from, (previously they would all just be
/// listed as `__extensions__::method_name::hash`, with no indication
/// of the type).
pub fn impl_pretty_name(trait_ref: &Option<TraitRef>, ty: Option<&Ty>) -> Ident {
    let mut pretty = match ty {
        Some(t) => pprust::ty_to_string(t),
        None => String::from("..")
    };

    match *trait_ref {
        Some(ref trait_ref) => {
            pretty.push('.');
            pretty.push_str(&pprust::path_to_string(&trait_ref.path));
        }
        None => {}
    }
    token::gensym_ident(&pretty[..])
}

pub fn struct_field_visibility(field: ast::StructField) -> Visibility {
    match field.node.kind {
        ast::NamedField(_, v) | ast::UnnamedField(v) => v
    }
}

// ______________________________________________________________________
// Enumerating the IDs which appear in an AST

#[derive(Copy, Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct IdRange {
    pub min: NodeId,
    pub max: NodeId,
}

impl IdRange {
    pub fn max() -> IdRange {
        IdRange {
            min: u32::MAX,
            max: u32::MIN,
        }
    }

    pub fn empty(&self) -> bool {
        self.min >= self.max
    }

    pub fn add(&mut self, id: NodeId) {
        self.min = cmp::min(self.min, id);
        self.max = cmp::max(self.max, id + 1);
    }
}

pub trait IdVisitingOperation {
    fn visit_id(&mut self, node_id: NodeId);
}

/// A visitor that applies its operation to all of the node IDs
/// in a visitable thing.

pub struct IdVisitor<'a, O:'a> {
    pub operation: &'a mut O,
    pub visited_outermost: bool,
}

impl<'a, O: IdVisitingOperation> IdVisitor<'a, O> {
    fn visit_generics_helper(&mut self, generics: &Generics) {
        for type_parameter in generics.ty_params.iter() {
            self.operation.visit_id(type_parameter.id)
        }
        for lifetime in &generics.lifetimes {
            self.operation.visit_id(lifetime.lifetime.id)
        }
    }
}

impl<'a, 'v, O: IdVisitingOperation> Visitor<'v> for IdVisitor<'a, O> {
    fn visit_mod(&mut self,
                 module: &Mod,
                 _: Span,
                 node_id: NodeId) {
        self.operation.visit_id(node_id);
        visit::walk_mod(self, module)
    }

    fn visit_foreign_item(&mut self, foreign_item: &ForeignItem) {
        self.operation.visit_id(foreign_item.id);
        visit::walk_foreign_item(self, foreign_item)
    }

    fn visit_item(&mut self, item: &Item) {
        if self.visited_outermost {
            return
        } else {
            self.visited_outermost = true
        }

        self.operation.visit_id(item.id);
        match item.node {
            ItemUse(ref view_path) => {
                match view_path.node {
                    ViewPathSimple(_, _) |
                    ViewPathGlob(_) => {}
                    ViewPathList(_, ref paths) => {
                        for path in paths {
                            self.operation.visit_id(path.node.id())
                        }
                    }
                }
            }
            _ => {}
        }

        visit::walk_item(self, item);

        self.visited_outermost = false
    }

    fn visit_local(&mut self, local: &Local) {
        self.operation.visit_id(local.id);
        visit::walk_local(self, local)
    }

    fn visit_block(&mut self, block: &Block) {
        self.operation.visit_id(block.id);
        visit::walk_block(self, block)
    }

    fn visit_stmt(&mut self, statement: &Stmt) {
        self.operation
            .visit_id(statement.node.id().expect("attempted to visit unexpanded stmt"));
        visit::walk_stmt(self, statement)
    }

    fn visit_pat(&mut self, pattern: &Pat) {
        self.operation.visit_id(pattern.id);
        visit::walk_pat(self, pattern)
    }

    fn visit_expr(&mut self, expression: &Expr) {
        self.operation.visit_id(expression.id);
        visit::walk_expr(self, expression)
    }

    fn visit_ty(&mut self, typ: &Ty) {
        self.operation.visit_id(typ.id);
        visit::walk_ty(self, typ)
    }

    fn visit_generics(&mut self, generics: &Generics) {
        self.visit_generics_helper(generics);
        visit::walk_generics(self, generics)
    }

    fn visit_fn(&mut self,
                function_kind: visit::FnKind<'v>,
                function_declaration: &'v FnDecl,
                block: &'v Block,
                span: Span,
                node_id: NodeId) {
        match function_kind {
            FnKind::Method(..) if self.visited_outermost => return,
            FnKind::Method(..) => self.visited_outermost = true,
            _ => {}
        }

        self.operation.visit_id(node_id);

        match function_kind {
            FnKind::ItemFn(_, generics, _, _, _, _) => {
                self.visit_generics_helper(generics)
            }
            FnKind::Method(_, sig, _) => {
                self.visit_generics_helper(&sig.generics)
            }
            FnKind::Closure => {}
        }

        for argument in &function_declaration.inputs {
            self.operation.visit_id(argument.id)
        }

        visit::walk_fn(self,
                       function_kind,
                       function_declaration,
                       block,
                       span);

        if let FnKind::Method(..) = function_kind {
            self.visited_outermost = false;
        }
    }

    fn visit_struct_field(&mut self, struct_field: &StructField) {
        self.operation.visit_id(struct_field.node.id);
        visit::walk_struct_field(self, struct_field)
    }

    fn visit_variant_data(&mut self,
                        struct_def: &VariantData,
                        _: ast::Ident,
                        _: &ast::Generics,
                        _: NodeId,
                        _: Span) {
        self.operation.visit_id(struct_def.id());
        visit::walk_struct_def(self, struct_def);
    }

    fn visit_trait_item(&mut self, ti: &ast::TraitItem) {
        self.operation.visit_id(ti.id);
        visit::walk_trait_item(self, ti);
    }

    fn visit_impl_item(&mut self, ii: &ast::ImplItem) {
        self.operation.visit_id(ii.id);
        visit::walk_impl_item(self, ii);
    }

    fn visit_lifetime(&mut self, lifetime: &Lifetime) {
        self.operation.visit_id(lifetime.id);
    }

    fn visit_lifetime_def(&mut self, def: &LifetimeDef) {
        self.visit_lifetime(&def.lifetime);
    }

    fn visit_trait_ref(&mut self, trait_ref: &TraitRef) {
        self.operation.visit_id(trait_ref.ref_id);
        visit::walk_trait_ref(self, trait_ref);
    }
}

pub struct IdRangeComputingVisitor {
    pub result: IdRange,
}

impl IdRangeComputingVisitor {
    pub fn new() -> IdRangeComputingVisitor {
        IdRangeComputingVisitor { result: IdRange::max() }
    }

    pub fn result(&self) -> IdRange {
        self.result
    }
}

impl IdVisitingOperation for IdRangeComputingVisitor {
    fn visit_id(&mut self, id: NodeId) {
        self.result.add(id);
    }
}

/// Computes the id range for a single fn body, ignoring nested items.
pub fn compute_id_range_for_fn_body(fk: FnKind,
                                    decl: &FnDecl,
                                    body: &Block,
                                    sp: Span,
                                    id: NodeId)
                                    -> IdRange
{
    let mut visitor = IdRangeComputingVisitor::new();
    let mut id_visitor = IdVisitor {
        operation: &mut visitor,
        visited_outermost: false,
    };
    id_visitor.visit_fn(fk, decl, body, sp, id);
    id_visitor.operation.result
}

/// Returns true if the given pattern consists solely of an identifier
/// and false otherwise.
pub fn pat_is_ident(pat: P<ast::Pat>) -> bool {
    match pat.node {
        ast::PatIdent(..) => true,
        _ => false,
    }
}

// are two paths equal when compared unhygienically?
// since I'm using this to replace ==, it seems appropriate
// to compare the span, global, etc. fields as well.
pub fn path_name_eq(a : &ast::Path, b : &ast::Path) -> bool {
    (a.span == b.span)
    && (a.global == b.global)
    && (segments_name_eq(&a.segments[..], &b.segments[..]))
}

// are two arrays of segments equal when compared unhygienically?
pub fn segments_name_eq(a : &[ast::PathSegment], b : &[ast::PathSegment]) -> bool {
    a.len() == b.len() &&
    a.iter().zip(b).all(|(s, t)| {
        s.identifier.name == t.identifier.name &&
        // FIXME #7743: ident -> name problems in lifetime comparison?
        // can types contain idents?
        s.parameters == t.parameters
    })
}

#[cfg(test)]
mod tests {
    use ast::*;
    use super::*;

    fn ident_to_segment(id: Ident) -> PathSegment {
        PathSegment {identifier: id,
                     parameters: PathParameters::none()}
    }

    #[test] fn idents_name_eq_test() {
        assert!(segments_name_eq(
            &[Ident::new(Name(3),SyntaxContext(4)), Ident::new(Name(78),SyntaxContext(82))]
                .iter().cloned().map(ident_to_segment).collect::<Vec<PathSegment>>(),
            &[Ident::new(Name(3),SyntaxContext(104)), Ident::new(Name(78),SyntaxContext(182))]
                .iter().cloned().map(ident_to_segment).collect::<Vec<PathSegment>>()));
        assert!(!segments_name_eq(
            &[Ident::new(Name(3),SyntaxContext(4)), Ident::new(Name(78),SyntaxContext(82))]
                .iter().cloned().map(ident_to_segment).collect::<Vec<PathSegment>>(),
            &[Ident::new(Name(3),SyntaxContext(104)), Ident::new(Name(77),SyntaxContext(182))]
                .iter().cloned().map(ident_to_segment).collect::<Vec<PathSegment>>()));
    }
}
