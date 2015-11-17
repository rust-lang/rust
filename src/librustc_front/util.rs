// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir;
use hir::*;
use intravisit::{self, Visitor, FnKind};
use syntax::ast_util;
use syntax::ast::{Ident, Name, NodeId, DUMMY_NODE_ID};
use syntax::codemap::Span;
use syntax::ptr::P;
use syntax::owned_slice::OwnedSlice;

pub fn walk_pat<F>(pat: &Pat, mut it: F) -> bool
    where F: FnMut(&Pat) -> bool
{
    // FIXME(#19596) this is a workaround, but there should be a better way
    fn walk_pat_<G>(pat: &Pat, it: &mut G) -> bool
        where G: FnMut(&Pat) -> bool
    {
        if !(*it)(pat) {
            return false;
        }

        match pat.node {
            PatIdent(_, _, Some(ref p)) => walk_pat_(&**p, it),
            PatStruct(_, ref fields, _) => {
                fields.iter().all(|field| walk_pat_(&*field.node.pat, it))
            }
            PatEnum(_, Some(ref s)) | PatTup(ref s) => {
                s.iter().all(|p| walk_pat_(&**p, it))
            }
            PatBox(ref s) | PatRegion(ref s, _) => {
                walk_pat_(&**s, it)
            }
            PatVec(ref before, ref slice, ref after) => {
                before.iter().all(|p| walk_pat_(&**p, it)) &&
                slice.iter().all(|p| walk_pat_(&**p, it)) &&
                after.iter().all(|p| walk_pat_(&**p, it))
            }
            PatWild |
            PatLit(_) |
            PatRange(_, _) |
            PatIdent(_, _, _) |
            PatEnum(_, _) |
            PatQPath(_, _) => {
                true
            }
        }
    }

    walk_pat_(pat, &mut it)
}

pub fn binop_to_string(op: BinOp_) -> &'static str {
    match op {
        BiAdd => "+",
        BiSub => "-",
        BiMul => "*",
        BiDiv => "/",
        BiRem => "%",
        BiAnd => "&&",
        BiOr => "||",
        BiBitXor => "^",
        BiBitAnd => "&",
        BiBitOr => "|",
        BiShl => "<<",
        BiShr => ">>",
        BiEq => "==",
        BiLt => "<",
        BiLe => "<=",
        BiNe => "!=",
        BiGe => ">=",
        BiGt => ">",
    }
}

pub fn stmt_id(s: &Stmt) -> NodeId {
    match s.node {
        StmtDecl(_, id) => id,
        StmtExpr(_, id) => id,
        StmtSemi(_, id) => id,
    }
}

pub fn lazy_binop(b: BinOp_) -> bool {
    match b {
        BiAnd => true,
        BiOr => true,
        _ => false,
    }
}

pub fn is_shift_binop(b: BinOp_) -> bool {
    match b {
        BiShl => true,
        BiShr => true,
        _ => false,
    }
}

pub fn is_comparison_binop(b: BinOp_) -> bool {
    match b {
        BiEq | BiLt | BiLe | BiNe | BiGt | BiGe => true,
        BiAnd |
        BiOr |
        BiAdd |
        BiSub |
        BiMul |
        BiDiv |
        BiRem |
        BiBitXor |
        BiBitAnd |
        BiBitOr |
        BiShl |
        BiShr => false,
    }
}

/// Returns `true` if the binary operator takes its arguments by value
pub fn is_by_value_binop(b: BinOp_) -> bool {
    !is_comparison_binop(b)
}

/// Returns `true` if the unary operator takes its argument by value
pub fn is_by_value_unop(u: UnOp) -> bool {
    match u {
        UnNeg | UnNot => true,
        _ => false,
    }
}

pub fn unop_to_string(op: UnOp) -> &'static str {
    match op {
        UnDeref => "*",
        UnNot => "!",
        UnNeg => "-",
    }
}

pub struct IdVisitor<'a, O: 'a> {
    operation: &'a mut O,

    // In general, the id visitor visits the contents of an item, but
    // not including nested trait/impl items, nor other nested items.
    // The base visitor itself always skips nested items, but not
    // trait/impl items. This means in particular that if you start by
    // visiting a trait or an impl, you should not visit the
    // trait/impl items respectively.  This is handled by setting
    // `skip_members` to true when `visit_item` is on the stack. This
    // way, if the user begins by calling `visit_trait_item`, we will
    // visit the trait item, but if they begin with `visit_item`, we
    // won't visit the (nested) trait items.
    skip_members: bool,
}

impl<'a, O: ast_util::IdVisitingOperation> IdVisitor<'a, O> {
    pub fn new(operation: &'a mut O) -> IdVisitor<'a, O> {
        IdVisitor { operation: operation, skip_members: false }
    }

    fn visit_generics_helper(&mut self, generics: &Generics) {
        for type_parameter in generics.ty_params.iter() {
            self.operation.visit_id(type_parameter.id)
        }
        for lifetime in &generics.lifetimes {
            self.operation.visit_id(lifetime.lifetime.id)
        }
    }
}

impl<'a, 'v, O: ast_util::IdVisitingOperation> Visitor<'v> for IdVisitor<'a, O> {
    fn visit_mod(&mut self, module: &Mod, _: Span, node_id: NodeId) {
        self.operation.visit_id(node_id);
        intravisit::walk_mod(self, module)
    }

    fn visit_foreign_item(&mut self, foreign_item: &ForeignItem) {
        self.operation.visit_id(foreign_item.id);
        intravisit::walk_foreign_item(self, foreign_item)
    }

    fn visit_item(&mut self, item: &Item) {
        assert!(!self.skip_members);
        self.skip_members = true;

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
        intravisit::walk_item(self, item);

        self.skip_members = false;
    }

    fn visit_local(&mut self, local: &Local) {
        self.operation.visit_id(local.id);
        intravisit::walk_local(self, local)
    }

    fn visit_block(&mut self, block: &Block) {
        self.operation.visit_id(block.id);
        intravisit::walk_block(self, block)
    }

    fn visit_stmt(&mut self, statement: &Stmt) {
        self.operation.visit_id(stmt_id(statement));
        intravisit::walk_stmt(self, statement)
    }

    fn visit_pat(&mut self, pattern: &Pat) {
        self.operation.visit_id(pattern.id);
        intravisit::walk_pat(self, pattern)
    }

    fn visit_expr(&mut self, expression: &Expr) {
        self.operation.visit_id(expression.id);
        intravisit::walk_expr(self, expression)
    }

    fn visit_ty(&mut self, typ: &Ty) {
        self.operation.visit_id(typ.id);
        intravisit::walk_ty(self, typ)
    }

    fn visit_generics(&mut self, generics: &Generics) {
        self.visit_generics_helper(generics);
        intravisit::walk_generics(self, generics)
    }

    fn visit_fn(&mut self,
                function_kind: FnKind<'v>,
                function_declaration: &'v FnDecl,
                block: &'v Block,
                span: Span,
                node_id: NodeId) {
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

        intravisit::walk_fn(self, function_kind, function_declaration, block, span);
    }

    fn visit_struct_field(&mut self, struct_field: &StructField) {
        self.operation.visit_id(struct_field.node.id);
        intravisit::walk_struct_field(self, struct_field)
    }

    fn visit_variant_data(&mut self,
                          struct_def: &VariantData,
                          _: Name,
                          _: &hir::Generics,
                          _: NodeId,
                          _: Span) {
        self.operation.visit_id(struct_def.id());
        intravisit::walk_struct_def(self, struct_def);
    }

    fn visit_trait_item(&mut self, ti: &hir::TraitItem) {
        if !self.skip_members {
            self.operation.visit_id(ti.id);
            intravisit::walk_trait_item(self, ti);
        }
    }

    fn visit_impl_item(&mut self, ii: &hir::ImplItem) {
        if !self.skip_members {
            self.operation.visit_id(ii.id);
            intravisit::walk_impl_item(self, ii);
        }
    }

    fn visit_lifetime(&mut self, lifetime: &Lifetime) {
        self.operation.visit_id(lifetime.id);
    }

    fn visit_lifetime_def(&mut self, def: &LifetimeDef) {
        self.visit_lifetime(&def.lifetime);
    }

    fn visit_trait_ref(&mut self, trait_ref: &TraitRef) {
        self.operation.visit_id(trait_ref.ref_id);
        intravisit::walk_trait_ref(self, trait_ref);
    }
}

/// Computes the id range for a single fn body, ignoring nested items.
pub fn compute_id_range_for_fn_body(fk: FnKind,
                                    decl: &FnDecl,
                                    body: &Block,
                                    sp: Span,
                                    id: NodeId)
                                    -> ast_util::IdRange {
    let mut visitor = ast_util::IdRangeComputingVisitor { result: ast_util::IdRange::max() };
    let mut id_visitor = IdVisitor::new(&mut visitor);
    id_visitor.visit_fn(fk, decl, body, sp, id);
    id_visitor.operation.result
}

pub fn is_path(e: P<Expr>) -> bool {
    match e.node {
        ExprPath(..) => true,
        _ => false,
    }
}

pub fn empty_generics() -> Generics {
    Generics {
        lifetimes: Vec::new(),
        ty_params: OwnedSlice::empty(),
        where_clause: WhereClause {
            id: DUMMY_NODE_ID,
            predicates: Vec::new(),
        },
    }
}

// convert a span and an identifier to the corresponding
// 1-segment path
pub fn ident_to_path(s: Span, ident: Ident) -> Path {
    hir::Path {
        span: s,
        global: false,
        segments: vec!(hir::PathSegment {
            identifier: ident,
            parameters: hir::AngleBracketedParameters(hir::AngleBracketedParameterData {
                lifetimes: Vec::new(),
                types: OwnedSlice::empty(),
                bindings: OwnedSlice::empty(),
            }),
        }),
    }
}
