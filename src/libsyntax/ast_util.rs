// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::Abi;
use ast::*;
use ast;
use ast_util;
use codemap;
use codemap::Span;
use owned_slice::OwnedSlice;
use parse::token;
use print::pprust;
use ptr::P;
use visit::Visitor;
use visit;

use std::cmp;
use std::u32;

pub fn path_name_i(idents: &[Ident]) -> String {
    // FIXME: Bad copies (#2543 -- same for everything else that says "bad")
    idents.iter().map(|i| {
        token::get_ident(*i).to_string()
    }).collect::<Vec<String>>().connect("::")
}

pub fn local_def(id: NodeId) -> DefId {
    ast::DefId { krate: LOCAL_CRATE, node: id }
}

pub fn is_local(did: ast::DefId) -> bool { did.krate == LOCAL_CRATE }

pub fn stmt_id(s: &Stmt) -> NodeId {
    match s.node {
      StmtDecl(_, id) => id,
      StmtExpr(_, id) => id,
      StmtSemi(_, id) => id,
      StmtMac(..) => panic!("attempted to analyze unexpanded stmt")
    }
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
        BiGt => ">"
    }
}

pub fn lazy_binop(b: BinOp_) -> bool {
    match b {
      BiAnd => true,
      BiOr => true,
      _ => false
    }
}

pub fn is_shift_binop(b: BinOp_) -> bool {
    match b {
      BiShl => true,
      BiShr => true,
      _ => false
    }
}

pub fn is_comparison_binop(b: BinOp_) -> bool {
    match b {
        BiEq | BiLt | BiLe | BiNe | BiGt | BiGe => true,
        _ => false
    }
}

/// Returns `true` if the binary operator takes its arguments by value
pub fn is_by_value_binop(b: BinOp_) -> bool {
    match b {
        BiAdd | BiSub | BiMul | BiDiv | BiRem | BiBitXor | BiBitAnd | BiBitOr | BiShl | BiShr => {
            true
        }
        _ => false
    }
}

/// Returns `true` if the binary operator is symmetric in the sense that LHS
/// and RHS must have the same type. So the type of LHS can serve as an hint
/// for the type of RHS and vice versa.
pub fn is_symmetric_binop(b: BinOp_) -> bool {
    match b {
        BiAdd | BiSub | BiMul | BiDiv | BiRem |
        BiBitXor | BiBitAnd | BiBitOr |
        BiEq | BiLt | BiLe | BiNe | BiGt | BiGe => {
            true
        }
        _ => false
    }
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
      UnUniq => "box() ",
      UnDeref => "*",
      UnNot => "!",
      UnNeg => "-",
    }
}

pub fn is_path(e: P<Expr>) -> bool {
    return match e.node { ExprPath(_) => true, _ => false };
}

/// Get a string representation of a signed int type, with its value.
/// We want to avoid "45int" and "-3int" in favor of "45" and "-3"
pub fn int_ty_to_string(t: IntTy, val: Option<i64>) -> String {
    let s = match t {
        TyIs(_) => "isize",
        TyI8 => "i8",
        TyI16 => "i16",
        TyI32 => "i32",
        TyI64 => "i64"
    };

    match val {
        // cast to a u64 so we can correctly print INT64_MIN. All integral types
        // are parsed as u64, so we wouldn't want to print an extra negative
        // sign.
        Some(n) => format!("{}{}", n as u64, s),
        None => s.to_string()
    }
}

pub fn int_ty_max(t: IntTy) -> u64 {
    match t {
        TyI8 => 0x80u64,
        TyI16 => 0x8000u64,
        TyIs(_) | TyI32 => 0x80000000u64, // actually ni about TyIs
        TyI64 => 0x8000000000000000u64
    }
}

/// Get a string representation of an unsigned int type, with its value.
/// We want to avoid "42u" in favor of "42us". "42uint" is right out.
pub fn uint_ty_to_string(t: UintTy, val: Option<u64>) -> String {
    let s = match t {
        TyUs(_) => "usize",
        TyU8 => "u8",
        TyU16 => "u16",
        TyU32 => "u32",
        TyU64 => "u64"
    };

    match val {
        Some(n) => format!("{}{}", n, s),
        None => s.to_string()
    }
}

pub fn uint_ty_max(t: UintTy) -> u64 {
    match t {
        TyU8 => 0xffu64,
        TyU16 => 0xffffu64,
        TyUs(_) | TyU32 => 0xffffffffu64, // actually ni about TyUs
        TyU64 => 0xffffffffffffffffu64
    }
}

pub fn float_ty_to_string(t: FloatTy) -> String {
    match t {
        TyF32 => "f32".to_string(),
        TyF64 => "f64".to_string(),
    }
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

pub fn name_to_dummy_lifetime(name: Name) -> Lifetime {
    Lifetime { id: DUMMY_NODE_ID,
               span: codemap::DUMMY_SP,
               name: name }
}

/// Generate a "pretty" name for an `impl` from its type and trait.
/// This is designed so that symbols of `impl`'d methods give some
/// hint of where they came from, (previously they would all just be
/// listed as `__extensions__::method_name::hash`, with no indication
/// of the type).
pub fn impl_pretty_name(trait_ref: &Option<TraitRef>, ty: &Ty) -> Ident {
    let mut pretty = pprust::ty_to_string(ty);
    match *trait_ref {
        Some(ref trait_ref) => {
            pretty.push('.');
            pretty.push_str(&pprust::path_to_string(&trait_ref.path));
        }
        None => {}
    }
    token::gensym_ident(&pretty[..])
}

pub fn trait_method_to_ty_method(method: &Method) -> TypeMethod {
    match method.node {
        MethDecl(ident,
                 ref generics,
                 abi,
                 ref explicit_self,
                 unsafety,
                 ref decl,
                 _,
                 vis) => {
            TypeMethod {
                ident: ident,
                attrs: method.attrs.clone(),
                unsafety: unsafety,
                decl: (*decl).clone(),
                generics: generics.clone(),
                explicit_self: (*explicit_self).clone(),
                id: method.id,
                span: method.span,
                vis: vis,
                abi: abi,
            }
        },
        MethMac(_) => panic!("expected non-macro method declaration")
    }
}

/// extract a TypeMethod from a TraitItem. if the TraitItem is
/// a default, pull out the useful fields to make a TypeMethod
//
// NB: to be used only after expansion is complete, and macros are gone.
pub fn trait_item_to_ty_method(method: &TraitItem) -> TypeMethod {
    match *method {
        RequiredMethod(ref m) => (*m).clone(),
        ProvidedMethod(ref m) => trait_method_to_ty_method(&**m),
        TypeTraitItem(_) => {
            panic!("trait_method_to_ty_method(): expected method but found \
                   typedef")
        }
    }
}

pub fn split_trait_methods(trait_methods: &[TraitItem])
                           -> (Vec<TypeMethod>, Vec<P<Method>> ) {
    let mut reqd = Vec::new();
    let mut provd = Vec::new();
    for trt_method in trait_methods {
        match *trt_method {
            RequiredMethod(ref tm) => reqd.push((*tm).clone()),
            ProvidedMethod(ref m) => provd.push((*m).clone()),
            TypeTraitItem(_) => {}
        }
    };
    (reqd, provd)
}

pub fn struct_field_visibility(field: ast::StructField) -> Visibility {
    match field.node.kind {
        ast::NamedField(_, v) | ast::UnnamedField(v) => v
    }
}

/// Maps a binary operator to its precedence
pub fn operator_prec(op: ast::BinOp_) -> usize {
  match op {
      // 'as' sits here with 12
      BiMul | BiDiv | BiRem     => 11,
      BiAdd | BiSub             => 10,
      BiShl | BiShr             =>  9,
      BiBitAnd                  =>  8,
      BiBitXor                  =>  7,
      BiBitOr                   =>  6,
      BiLt | BiLe | BiGe | BiGt | BiEq | BiNe => 3,
      BiAnd                     =>  2,
      BiOr                      =>  1
  }
}

/// Precedence of the `as` operator, which is a binary operator
/// not appearing in the prior table.
pub const AS_PREC: usize = 12;

pub fn empty_generics() -> Generics {
    Generics {
        lifetimes: Vec::new(),
        ty_params: OwnedSlice::empty(),
        where_clause: WhereClause {
            id: DUMMY_NODE_ID,
            predicates: Vec::new(),
        }
    }
}

// ______________________________________________________________________
// Enumerating the IDs which appear in an AST

#[derive(RustcEncodable, RustcDecodable, Debug, Copy)]
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
    pub pass_through_items: bool,
    pub visited_outermost: bool,
}

impl<'a, O: IdVisitingOperation> IdVisitor<'a, O> {
    fn visit_generics_helper(&mut self, generics: &Generics) {
        for type_parameter in &*generics.ty_params {
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
        if !self.pass_through_items {
            if self.visited_outermost {
                return
            } else {
                self.visited_outermost = true
            }
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
            ItemEnum(ref enum_definition, _) => {
                for variant in &enum_definition.variants {
                    self.operation.visit_id(variant.node.id)
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
        self.operation.visit_id(ast_util::stmt_id(statement));
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
        if let TyPath(_, id) = typ.node {
            self.operation.visit_id(id);
        }
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
        if !self.pass_through_items {
            match function_kind {
                visit::FkMethod(..) if self.visited_outermost => return,
                visit::FkMethod(..) => self.visited_outermost = true,
                _ => {}
            }
        }

        self.operation.visit_id(node_id);

        match function_kind {
            visit::FkItemFn(_, generics, _, _) |
            visit::FkMethod(_, generics, _) => {
                self.visit_generics_helper(generics)
            }
            visit::FkFnBlock => {}
        }

        for argument in &function_declaration.inputs {
            self.operation.visit_id(argument.id)
        }

        visit::walk_fn(self,
                       function_kind,
                       function_declaration,
                       block,
                       span);

        if !self.pass_through_items {
            if let visit::FkMethod(..) = function_kind {
                self.visited_outermost = false;
            }
        }
    }

    fn visit_struct_field(&mut self, struct_field: &StructField) {
        self.operation.visit_id(struct_field.node.id);
        visit::walk_struct_field(self, struct_field)
    }

    fn visit_struct_def(&mut self,
                        struct_def: &StructDef,
                        _: ast::Ident,
                        _: &ast::Generics,
                        id: NodeId) {
        self.operation.visit_id(id);
        struct_def.ctor_id.map(|ctor_id| self.operation.visit_id(ctor_id));
        visit::walk_struct_def(self, struct_def);
    }

    fn visit_trait_item(&mut self, tm: &ast::TraitItem) {
        match *tm {
            ast::RequiredMethod(ref m) => self.operation.visit_id(m.id),
            ast::ProvidedMethod(ref m) => self.operation.visit_id(m.id),
            ast::TypeTraitItem(ref typ) => self.operation.visit_id(typ.ty_param.id),
        }
        visit::walk_trait_item(self, tm);
    }

    fn visit_lifetime_ref(&mut self, lifetime: &'v Lifetime) {
        self.operation.visit_id(lifetime.id);
    }

    fn visit_lifetime_def(&mut self, def: &'v LifetimeDef) {
        self.visit_lifetime_ref(&def.lifetime);
    }
}

pub fn visit_ids_for_inlined_item<O: IdVisitingOperation>(item: &InlinedItem,
                                                          operation: &mut O) {
    let mut id_visitor = IdVisitor {
        operation: operation,
        pass_through_items: true,
        visited_outermost: false,
    };

    visit::walk_inlined_item(&mut id_visitor, item);
}

struct IdRangeComputingVisitor {
    result: IdRange,
}

impl IdVisitingOperation for IdRangeComputingVisitor {
    fn visit_id(&mut self, id: NodeId) {
        self.result.add(id);
    }
}

pub fn compute_id_range_for_inlined_item(item: &InlinedItem) -> IdRange {
    let mut visitor = IdRangeComputingVisitor {
        result: IdRange::max()
    };
    visit_ids_for_inlined_item(item, &mut visitor);
    visitor.result
}

/// Computes the id range for a single fn body, ignoring nested items.
pub fn compute_id_range_for_fn_body(fk: visit::FnKind,
                                    decl: &FnDecl,
                                    body: &Block,
                                    sp: Span,
                                    id: NodeId)
                                    -> IdRange
{
    let mut visitor = IdRangeComputingVisitor {
        result: IdRange::max()
    };
    let mut id_visitor = IdVisitor {
        operation: &mut visitor,
        pass_through_items: false,
        visited_outermost: false,
    };
    id_visitor.visit_fn(fk, decl, body, sp, id);
    id_visitor.operation.result
}

pub fn walk_pat<F>(pat: &Pat, mut it: F) -> bool where F: FnMut(&Pat) -> bool {
    // FIXME(#19596) this is a workaround, but there should be a better way
    fn walk_pat_<G>(pat: &Pat, it: &mut G) -> bool where G: FnMut(&Pat) -> bool {
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
            PatMac(_) => panic!("attempted to analyze unexpanded pattern"),
            PatWild(_) | PatLit(_) | PatRange(_, _) | PatIdent(_, _, _) |
            PatEnum(_, _) => {
                true
            }
        }
    }

    walk_pat_(pat, &mut it)
}

/// Returns true if the given struct def is tuple-like; i.e. that its fields
/// are unnamed.
pub fn struct_def_is_tuple_like(struct_def: &ast::StructDef) -> bool {
    struct_def.ctor_id.is_some()
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
    a.iter().zip(b.iter()).all(|(s, t)| {
        s.identifier.name == t.identifier.name &&
        // FIXME #7743: ident -> name problems in lifetime comparison?
        // can types contain idents?
        s.parameters == t.parameters
    })
}

/// Returns true if this literal is a string and false otherwise.
pub fn lit_is_str(lit: &Lit) -> bool {
    match lit.node {
        LitStr(..) => true,
        _ => false,
    }
}

/// Macro invocations are guaranteed not to occur after expansion is complete.
/// Extracting fields of a method requires a dynamic check to make sure that it's
/// not a macro invocation. This check is guaranteed to succeed, assuming
/// that the invocations are indeed gone.
pub trait PostExpansionMethod {
    fn pe_ident(&self) -> ast::Ident;
    fn pe_generics<'a>(&'a self) -> &'a ast::Generics;
    fn pe_abi(&self) -> Abi;
    fn pe_explicit_self<'a>(&'a self) -> &'a ast::ExplicitSelf;
    fn pe_unsafety(&self) -> ast::Unsafety;
    fn pe_fn_decl<'a>(&'a self) -> &'a ast::FnDecl;
    fn pe_body<'a>(&'a self) -> &'a ast::Block;
    fn pe_vis(&self) -> ast::Visibility;
}

macro_rules! mf_method{
    ($meth_name:ident, $field_ty:ty, $field_pat:pat, $result:expr) => {
        fn $meth_name<'a>(&'a self) -> $field_ty {
            match self.node {
                $field_pat => $result,
                MethMac(_) => {
                    panic!("expected an AST without macro invocations");
                }
            }
        }
    }
}


impl PostExpansionMethod for Method {
    mf_method! { pe_ident,ast::Ident,MethDecl(ident,_,_,_,_,_,_,_),ident }
    mf_method! {
        pe_generics,&'a ast::Generics,
        MethDecl(_,ref generics,_,_,_,_,_,_),generics
    }
    mf_method! { pe_abi,Abi,MethDecl(_,_,abi,_,_,_,_,_),abi }
    mf_method! {
        pe_explicit_self,&'a ast::ExplicitSelf,
        MethDecl(_,_,_,ref explicit_self,_,_,_,_),explicit_self
    }
    mf_method! { pe_unsafety,ast::Unsafety,MethDecl(_,_,_,_,unsafety,_,_,_),unsafety }
    mf_method! { pe_fn_decl,&'a ast::FnDecl,MethDecl(_,_,_,_,_,ref decl,_,_),&**decl }
    mf_method! { pe_body,&'a ast::Block,MethDecl(_,_,_,_,_,_,ref body,_),&**body }
    mf_method! { pe_vis,ast::Visibility,MethDecl(_,_,_,_,_,_,_,vis),vis }
}

#[cfg(test)]
mod test {
    use ast::*;
    use super::*;

    fn ident_to_segment(id : &Ident) -> PathSegment {
        PathSegment {identifier: id.clone(),
                     parameters: PathParameters::none()}
    }

    #[test] fn idents_name_eq_test() {
        assert!(segments_name_eq(
            &[Ident{name:Name(3),ctxt:4}, Ident{name:Name(78),ctxt:82}]
                .iter().map(ident_to_segment).collect::<Vec<PathSegment>>()[],
            &[Ident{name:Name(3),ctxt:104}, Ident{name:Name(78),ctxt:182}]
                .iter().map(ident_to_segment).collect::<Vec<PathSegment>>()[]));
        assert!(!segments_name_eq(
            &[Ident{name:Name(3),ctxt:4}, Ident{name:Name(78),ctxt:82}]
                .iter().map(ident_to_segment).collect::<Vec<PathSegment>>()[],
            &[Ident{name:Name(3),ctxt:104}, Ident{name:Name(77),ctxt:182}]
                .iter().map(ident_to_segment).collect::<Vec<PathSegment>>()[]));
    }
}
