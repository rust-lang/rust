// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use ast_util;
use codemap;
use codemap::Span;
use owned_slice::OwnedSlice;
use parse::token;
use print::pprust;
use visit::Visitor;
use visit;

use std::cell::Cell;
use std::cmp;
use std::u32;

pub fn path_name_i(idents: &[ast::Ident]) -> String {
    // FIXME: Bad copies (#2543 -- same for everything else that says "bad")
    idents.iter().map(|i| {
        token::get_ident(*i).get().to_string()
    }).collect::<Vec<String>>().connect("::").to_string()
}

// totally scary function: ignores all but the last element, should have
// a different name
pub fn path_to_ident(path: &ast::Path) -> ast::Ident {
    path.segments.last().unwrap().identifier
}

pub fn local_def(id: ast::NodeId) -> ast::DefId {
    ast::DefId { krate: ast::LOCAL_CRATE, node: id }
}

pub fn is_local(did: ast::DefId) -> bool { did.krate == ast::LOCAL_CRATE }

pub fn stmt_id(s: &ast::Stmt) -> ast::NodeId {
    match s.node {
      ast::StmtDecl(_, id) => id,
      ast::StmtExpr(_, id) => id,
      ast::StmtSemi(_, id) => id,
      ast::StmtMac(..) => fail!("attempted to analyze unexpanded stmt")
    }
}

pub fn variant_def_ids(d: ast::Def) -> Option<(ast::DefId, ast::DefId)> {
    match d {
      ast::DefVariant(enum_id, var_id, _) => {
          Some((enum_id, var_id))
      }
      _ => None
    }
}

pub fn def_id_of_def(d: ast::Def) -> ast::DefId {
    match d {
        ast::DefFn(id, _) | ast::DefStaticMethod(id, _, _) | ast::DefMod(id) |
        ast::DefForeignMod(id) | ast::DefStatic(id, _) |
        ast::DefVariant(_, id, _) | ast::DefTy(id) | ast::DefTyParam(id, _) |
        ast::DefUse(id) | ast::DefStruct(id) | ast::DefTrait(id) | ast::DefMethod(id, _) => {
            id
        }
        ast::DefArg(id, _) | ast::DefLocal(id, _) | ast::DefSelfTy(id) |
        ast::DefUpvar(id, _, _, _) | ast::DefBinding(id, _) | ast::DefRegion(id) |
        ast::DefTyParamBinder(id) | ast::DefLabel(id) => {
            local_def(id)
        }

        ast::DefPrimTy(_) => fail!()
    }
}

pub fn binop_to_str(op: ast::BinOp) -> &'static str {
    match op {
        ast::BiAdd => "+",
        ast::BiSub => "-",
        ast::BiMul => "*",
        ast::BiDiv => "/",
        ast::BiRem => "%",
        ast::BiAnd => "&&",
        ast::BiOr => "||",
        ast::BiBitXor => "^",
        ast::BiBitAnd => "&",
        ast::BiBitOr => "|",
        ast::BiShl => "<<",
        ast::BiShr => ">>",
        ast::BiEq => "==",
        ast::BiLt => "<",
        ast::BiLe => "<=",
        ast::BiNe => "!=",
        ast::BiGe => ">=",
        ast::BiGt => ">"
    }
}

pub fn lazy_binop(b: ast::BinOp) -> bool {
    match b {
      ast::BiAnd => true,
      ast::BiOr => true,
      _ => false
    }
}

pub fn is_shift_binop(b: ast::BinOp) -> bool {
    match b {
      ast::BiShl => true,
      ast::BiShr => true,
      _ => false
    }
}

pub fn unop_to_str(op: ast::UnOp) -> &'static str {
    match op {
      ast::UnBox => "@",
      ast::UnUniq => "box() ",
      ast::UnDeref => "*",
      ast::UnNot => "!",
      ast::UnNeg => "-",
    }
}

pub fn is_path(e: @ast::Expr) -> bool {
    return match e.node { ast::ExprPath(_) => true, _ => false };
}

pub enum SuffixMode {
    ForceSuffix,
    AutoSuffix,
}

// Get a string representation of a signed int type, with its value.
// We want to avoid "45int" and "-3int" in favor of "45" and "-3"
pub fn int_ty_to_str(t: ast::IntTy, val: Option<i64>, mode: SuffixMode) -> String {
    let s = match t {
        ast::TyI if val.is_some() => match mode {
            AutoSuffix => "",
            ForceSuffix => "i",
        },
        ast::TyI => "int",
        ast::TyI8 => "i8",
        ast::TyI16 => "i16",
        ast::TyI32 => "i32",
        ast::TyI64 => "i64"
    };

    match val {
        // cast to a u64 so we can correctly print INT64_MIN. All integral types
        // are parsed as u64, so we wouldn't want to print an extra negative
        // sign.
        Some(n) => format!("{}{}", n as u64, s).to_string(),
        None => s.to_string()
    }
}

pub fn int_ty_max(t: ast::IntTy) -> u64 {
    match t {
        ast::TyI8 => 0x80u64,
        ast::TyI16 => 0x8000u64,
        ast::TyI | ast::TyI32 => 0x80000000u64, // actually ni about TyI
        ast::TyI64 => 0x8000000000000000u64
    }
}

// Get a string representation of an unsigned int type, with its value.
// We want to avoid "42uint" in favor of "42u"
pub fn uint_ty_to_str(t: ast::UintTy, val: Option<u64>, mode: SuffixMode) -> String {
    let s = match t {
        ast::TyU if val.is_some() => match mode {
            AutoSuffix => "",
            ForceSuffix => "u",
        },
        ast::TyU => "uint",
        ast::TyU8 => "u8",
        ast::TyU16 => "u16",
        ast::TyU32 => "u32",
        ast::TyU64 => "u64"
    };

    match val {
        Some(n) => format!("{}{}", n, s).to_string(),
        None => s.to_string()
    }
}

pub fn uint_ty_max(t: ast::UintTy) -> u64 {
    match t {
        ast::TyU8 => 0xffu64,
        ast::TyU16 => 0xffffu64,
        ast::TyU | ast::TyU32 => 0xffffffffu64, // actually ni about TyU
        ast::TyU64 => 0xffffffffffffffffu64
    }
}

pub fn float_ty_to_str(t: ast::FloatTy) -> String {
    match t {
        ast::TyF32 => "f32".to_string(),
        ast::TyF64 => "f64".to_string(),
        ast::TyF128 => "f128".to_string(),
    }
}

pub fn is_call_expr(e: @ast::Expr) -> bool {
    match e.node { ast::ExprCall(..) => true, _ => false }
}

pub fn block_from_expr(e: @ast::Expr) -> ast::P<ast::Block> {
    ast::P(ast::Block {
        view_items: Vec::new(),
        stmts: Vec::new(),
        expr: Some(e),
        id: e.id,
        rules: ast::DefaultBlock,
        span: e.span
    })
}

pub fn ident_to_path(s: Span, identifier: ast::Ident) -> ast::Path {
    ast::Path {
        span: s,
        global: false,
        segments: vec!(
            ast::PathSegment {
                identifier: identifier,
                lifetimes: Vec::new(),
                types: OwnedSlice::empty(),
            }
        ),
    }
}

pub fn ident_to_pat(id: ast::NodeId, s: Span, i: ast::Ident) -> @ast::Pat {
    @ast::Pat { id: id,
                node: ast::PatIdent(ast::BindByValue(ast::MutImmutable), ident_to_path(s, i), None),
                span: s }
}

pub fn name_to_dummy_lifetime(name: ast::Name) -> ast::Lifetime {
    ast::Lifetime { id: ast::DUMMY_NODE_ID,
                    span: codemap::DUMMY_SP,
                    name: name }
}

pub fn is_unguarded(a: &ast::Arm) -> bool {
    match a.guard {
      None => true,
      _    => false
    }
}

pub fn unguarded_pat(a: &ast::Arm) -> Option<Vec<@ast::Pat> > {
    if is_unguarded(a) {
        Some(/* FIXME (#2543) */ a.pats.clone())
    } else {
        None
    }
}

/// Generate a "pretty" name for an `impl` from its type and trait.
/// This is designed so that symbols of `impl`'d methods give some
/// hint of where they came from, (previously they would all just be
/// listed as `__extensions__::method_name::hash`, with no indication
/// of the type).
pub fn impl_pretty_name(trait_ref: &Option<ast::TraitRef>, ty: &ast::Ty) -> ast::Ident {
    let mut pretty = pprust::ty_to_str(ty);
    match *trait_ref {
        Some(ref trait_ref) => {
            pretty.push_char('.');
            pretty.push_str(pprust::path_to_str(&trait_ref.path).as_slice());
        }
        None => {}
    }
    token::gensym_ident(pretty.as_slice())
}

pub fn public_methods(ms: Vec<@ast::Method> ) -> Vec<@ast::Method> {
    ms.move_iter().filter(|m| {
        match m.vis {
            ast::Public => true,
            _   => false
        }
    }).collect()
}

// extract a TypeMethod from a TraitMethod. if the TraitMethod is
// a default, pull out the useful fields to make a TypeMethod
pub fn trait_method_to_ty_method(method: &ast::TraitMethod) -> ast::TypeMethod {
    match *method {
        ast::Required(ref m) => (*m).clone(),
        ast::Provided(ref m) => {
            ast::TypeMethod {
                ident: m.ident,
                attrs: m.attrs.clone(),
                fn_style: m.fn_style,
                decl: m.decl,
                generics: m.generics.clone(),
                explicit_self: m.explicit_self,
                id: m.id,
                span: m.span,
                vis: m.vis,
            }
        }
    }
}

pub fn split_trait_methods(trait_methods: &[ast::TraitMethod])
    -> (Vec<ast::TypeMethod> , Vec<@ast::Method> ) {
    let mut reqd = Vec::new();
    let mut provd = Vec::new();
    for trt_method in trait_methods.iter() {
        match *trt_method {
            ast::Required(ref tm) => reqd.push((*tm).clone()),
            ast::Provided(m) => provd.push(m)
        }
    };
    (reqd, provd)
}

pub fn struct_field_visibility(field: ast::StructField) -> ast::Visibility {
    match field.node.kind {
        ast::NamedField(_, v) | ast::UnnamedField(v) => v
    }
}

/// Maps a binary operator to its precedence
pub fn operator_prec(op: ast::BinOp) -> uint {
  match op {
      // 'as' sits here with 12
      ast::BiMul | ast::BiDiv | ast::BiRem          => 11u,
      ast::BiAdd | ast::BiSub                       => 10u,
      ast::BiShl | ast::BiShr                       =>  9u,
      ast::BiBitAnd                                 =>  8u,
      ast::BiBitXor                                 =>  7u,
      ast::BiBitOr                                  =>  6u,
      ast::BiLt | ast::BiLe | ast::BiGe | ast::BiGt =>  4u,
      ast::BiEq | ast::BiNe                         =>  3u,
      ast::BiAnd                                    =>  2u,
      ast::BiOr                                     =>  1u
  }
}

/// Precedence of the `as` operator, which is a binary operator
/// not appearing in the prior table.
pub static as_prec: uint = 12u;

pub fn empty_generics() -> ast::Generics {
    ast::Generics {lifetimes: Vec::new(),
              ty_params: OwnedSlice::empty()}
}

// ______________________________________________________________________
// Enumerating the IDs which appear in an AST

#[deriving(Encodable, Decodable)]
pub struct IdRange {
    pub min: ast::NodeId,
    pub max: ast::NodeId,
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

    pub fn add(&mut self, id: ast::NodeId) {
        self.min = cmp::min(self.min, id);
        self.max = cmp::max(self.max, id + 1);
    }
}

pub trait IdVisitingOperation {
    fn visit_id(&self, node_id: ast::NodeId);
}

pub struct IdVisitor<'a, O> {
    pub operation: &'a O,
    pub pass_through_items: bool,
    pub visited_outermost: bool,
}

impl<'a, O: IdVisitingOperation> IdVisitor<'a, O> {
    fn visit_generics_helper(&self, generics: &ast::Generics) {
        for type_parameter in generics.ty_params.iter() {
            self.operation.visit_id(type_parameter.id)
        }
        for lifetime in generics.lifetimes.iter() {
            self.operation.visit_id(lifetime.id)
        }
    }
}

impl<'a, O: IdVisitingOperation> Visitor<()> for IdVisitor<'a, O> {
    fn visit_mod(&mut self,
                 module: &ast::Mod,
                 _: Span,
                 node_id: ast::NodeId,
                 env: ()) {
        self.operation.visit_id(node_id);
        visit::walk_mod(self, module, env)
    }

    fn visit_view_item(&mut self, view_item: &ast::ViewItem, env: ()) {
        if !self.pass_through_items {
            if self.visited_outermost {
                return;
            } else {
                self.visited_outermost = true;
            }
        }
        match view_item.node {
            ast::ViewItemExternCrate(_, _, node_id) => {
                self.operation.visit_id(node_id)
            }
            ast::ViewItemUse(ref view_path) => {
                match view_path.node {
                    ast::ViewPathSimple(_, _, node_id) |
                    ast::ViewPathGlob(_, node_id) => {
                        self.operation.visit_id(node_id)
                    }
                    ast::ViewPathList(_, ref paths, node_id) => {
                        self.operation.visit_id(node_id);
                        for path in paths.iter() {
                            self.operation.visit_id(path.node.id)
                        }
                    }
                }
            }
        }
        visit::walk_view_item(self, view_item, env);
        self.visited_outermost = false;
    }

    fn visit_foreign_item(&mut self, foreign_item: &ast::ForeignItem, env: ()) {
        self.operation.visit_id(foreign_item.id);
        visit::walk_foreign_item(self, foreign_item, env)
    }

    fn visit_item(&mut self, item: &ast::Item, env: ()) {
        if !self.pass_through_items {
            if self.visited_outermost {
                return
            } else {
                self.visited_outermost = true
            }
        }

        self.operation.visit_id(item.id);
        match item.node {
            ast::ItemEnum(ref enum_definition, _) => {
                for variant in enum_definition.variants.iter() {
                    self.operation.visit_id(variant.node.id)
                }
            }
            _ => {}
        }

        visit::walk_item(self, item, env);

        self.visited_outermost = false
    }

    fn visit_local(&mut self, local: &ast::Local, env: ()) {
        self.operation.visit_id(local.id);
        visit::walk_local(self, local, env)
    }

    fn visit_block(&mut self, block: &ast::Block, env: ()) {
        self.operation.visit_id(block.id);
        visit::walk_block(self, block, env)
    }

    fn visit_stmt(&mut self, statement: &ast::Stmt, env: ()) {
        self.operation.visit_id(ast_util::stmt_id(statement));
        visit::walk_stmt(self, statement, env)
    }

    fn visit_pat(&mut self, pattern: &ast::Pat, env: ()) {
        self.operation.visit_id(pattern.id);
        visit::walk_pat(self, pattern, env)
    }

    fn visit_expr(&mut self, expression: &ast::Expr, env: ()) {
        self.operation.visit_id(expression.id);
        visit::walk_expr(self, expression, env)
    }

    fn visit_ty(&mut self, typ: &ast::Ty, env: ()) {
        self.operation.visit_id(typ.id);
        match typ.node {
            ast::TyPath(_, _, id) => self.operation.visit_id(id),
            _ => {}
        }
        visit::walk_ty(self, typ, env)
    }

    fn visit_generics(&mut self, generics: &ast::Generics, env: ()) {
        self.visit_generics_helper(generics);
        visit::walk_generics(self, generics, env)
    }

    fn visit_fn(&mut self,
                function_kind: &visit::FnKind,
                function_declaration: &ast::FnDecl,
                block: &ast::Block,
                span: Span,
                node_id: ast::NodeId,
                env: ()) {
        if !self.pass_through_items {
            match *function_kind {
                visit::FkMethod(..) if self.visited_outermost => return,
                visit::FkMethod(..) => self.visited_outermost = true,
                _ => {}
            }
        }

        self.operation.visit_id(node_id);

        match *function_kind {
            visit::FkItemFn(_, generics, _, _) |
            visit::FkMethod(_, generics, _) => {
                self.visit_generics_helper(generics)
            }
            visit::FkFnBlock => {}
        }

        for argument in function_declaration.inputs.iter() {
            self.operation.visit_id(argument.id)
        }

        visit::walk_fn(self,
                        function_kind,
                        function_declaration,
                        block,
                        span,
                        env);

        if !self.pass_through_items {
            match *function_kind {
                visit::FkMethod(..) => self.visited_outermost = false,
                _ => {}
            }
        }
    }

    fn visit_struct_field(&mut self, struct_field: &ast::StructField, env: ()) {
        self.operation.visit_id(struct_field.node.id);
        visit::walk_struct_field(self, struct_field, env)
    }

    fn visit_struct_def(&mut self,
                        struct_def: &ast::StructDef,
                        _: ast::Ident,
                        _: &ast::Generics,
                        id: ast::NodeId,
                        _: ()) {
        self.operation.visit_id(id);
        struct_def.ctor_id.map(|ctor_id| self.operation.visit_id(ctor_id));
        visit::walk_struct_def(self, struct_def, ());
    }

    fn visit_trait_method(&mut self, tm: &ast::TraitMethod, _: ()) {
        match *tm {
            ast::Required(ref m) => self.operation.visit_id(m.id),
            ast::Provided(ref m) => self.operation.visit_id(m.id),
        }
        visit::walk_trait_method(self, tm, ());
    }
}

pub fn visit_ids_for_inlined_item<O: IdVisitingOperation>(item: &ast::InlinedItem,
                                                          operation: &O) {
    let mut id_visitor = IdVisitor {
        operation: operation,
        pass_through_items: true,
        visited_outermost: false,
    };

    visit::walk_inlined_item(&mut id_visitor, item, ());
}

struct IdRangeComputingVisitor {
    result: Cell<IdRange>,
}

impl IdVisitingOperation for IdRangeComputingVisitor {
    fn visit_id(&self, id: ast::NodeId) {
        let mut id_range = self.result.get();
        id_range.add(id);
        self.result.set(id_range)
    }
}

pub fn compute_id_range_for_inlined_item(item: &ast::InlinedItem) -> IdRange {
    let visitor = IdRangeComputingVisitor {
        result: Cell::new(IdRange::max())
    };
    visit_ids_for_inlined_item(item, &visitor);
    visitor.result.get()
}

pub fn compute_id_range_for_fn_body(fk: &visit::FnKind,
                                    decl: &ast::FnDecl,
                                    body: &ast::Block,
                                    sp: Span,
                                    id: ast::NodeId)
                                    -> IdRange
{
    /*!
     * Computes the id range for a single fn body,
     * ignoring nested items.
     */

    let visitor = IdRangeComputingVisitor {
        result: Cell::new(IdRange::max())
    };
    let mut id_visitor = IdVisitor {
        operation: &visitor,
        pass_through_items: false,
        visited_outermost: false,
    };
    id_visitor.visit_fn(fk, decl, body, sp, id, ());
    visitor.result.get()
}

pub fn is_item_impl(item: @ast::Item) -> bool {
    match item.node {
        ast::ItemImpl(..) => true,
        _                 => false
    }
}

pub fn walk_pat(pat: &ast::Pat, it: |&ast::Pat| -> bool) -> bool {
    if !it(pat) {
        return false;
    }

    match pat.node {
        ast::PatIdent(_, _, Some(p)) => walk_pat(p, it),
        ast::PatStruct(_, ref fields, _) => {
            fields.iter().advance(|f| walk_pat(f.pat, |p| it(p)))
        }
        ast::PatEnum(_, Some(ref s)) | ast::PatTup(ref s) => {
            s.iter().advance(|&p| walk_pat(p, |p| it(p)))
        }
        ast::PatBox(s) | ast::PatRegion(s) => {
            walk_pat(s, it)
        }
        ast::PatVec(ref before, ref slice, ref after) => {
            before.iter().advance(|&p| walk_pat(p, |p| it(p))) &&
                slice.iter().advance(|&p| walk_pat(p, |p| it(p))) &&
                after.iter().advance(|&p| walk_pat(p, |p| it(p)))
        }
        ast::PatMac(_) => fail!("attempted to analyze unexpanded pattern"),
        ast::PatWild | ast::PatWildMulti | ast::PatLit(_) | ast::PatRange(_, _) |
        ast::PatIdent(_, _, _) | ast::PatEnum(_, _) => {
            true
        }
    }
}

pub trait EachViewItem {
    fn each_view_item(&self, f: |&ast::ViewItem| -> bool) -> bool;
}

struct EachViewItemData<'a> {
    callback: |&ast::ViewItem|: 'a -> bool,
}

impl<'a> Visitor<()> for EachViewItemData<'a> {
    fn visit_view_item(&mut self, view_item: &ast::ViewItem, _: ()) {
        let _ = (self.callback)(view_item);
    }
}

impl EachViewItem for ast::Crate {
    fn each_view_item(&self, f: |&ast::ViewItem| -> bool) -> bool {
        let mut visit = EachViewItemData {
            callback: f,
        };
        visit::walk_crate(&mut visit, self, ());
        true
    }
}

pub fn view_path_id(p: &ast::ViewPath) -> ast::NodeId {
    match p.node {
        ast::ViewPathSimple(_, _, id) | ast::ViewPathGlob(_, id) |
        ast::ViewPathList(_, _, id) => id
    }
}

/// Returns true if the given struct def is tuple-like; i.e. that its fields
/// are unnamed.
pub fn struct_def_is_tuple_like(struct_def: &ast::StructDef) -> bool {
    struct_def.ctor_id.is_some()
}

/// Returns true if the given pattern consists solely of an identifier
/// and false otherwise.
pub fn pat_is_ident(pat: @ast::Pat) -> bool {
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
    && (segments_name_eq(a.segments.as_slice(), b.segments.as_slice()))
}

// are two arrays of segments equal when compared unhygienically?
pub fn segments_name_eq(a : &[ast::PathSegment], b : &[ast::PathSegment]) -> bool {
    if a.len() != b.len() {
        false
    } else {
        for (idx,seg) in a.iter().enumerate() {
            if (seg.identifier.name != b[idx].identifier.name)
                // FIXME #7743: ident -> name problems in lifetime comparison?
                || (seg.lifetimes != b[idx].lifetimes)
                // can types contain idents?
                || (seg.types != b[idx].types) {
                return false;
            }
        }
        true
    }
}

// Returns true if this literal is a string and false otherwise.
pub fn lit_is_str(lit: @ast::Lit) -> bool {
    match lit.node {
        ast::LitStr(..) => true,
        _ => false,
    }
}

pub fn get_inner_tys(ty: ast::P<ast::Ty>) -> Vec<ast::P<ast::Ty>> {
    match ty.node {
        ast::TyRptr(_, mut_ty) | ast::TyPtr(mut_ty) => {
            vec!(mut_ty.ty)
        }
        ast::TyBox(ty)
        | ast::TyVec(ty)
        | ast::TyUniq(ty)
        | ast::TyFixedLengthVec(ty, _) => vec!(ty),
        ast::TyTup(ref tys) => tys.clone(),
        _ => Vec::new()
    }
}


#[cfg(test)]
mod test {
    use owned_slice::OwnedSlice;

    fn ident_to_segment(id : &ast::Ident) -> PathSegment {
        PathSegment {identifier:id.clone(),
                     lifetimes: Vec::new(),
                     types: OwnedSlice::empty()}
    }

    #[test] fn idents_name_eq_test() {
        assert!(segments_name_eq(
            [ast::Ident{name:3,ctxt:4}, ast::Ident{name:78,ctxt:82}]
                .iter().map(ident_to_segment).collect::<Vec<PathSegment>>().as_slice(),
            [ast::Ident{name:3,ctxt:104}, ast::Ident{name:78,ctxt:182}]
                .iter().map(ident_to_segment).collect::<Vec<PathSegment>>().as_slice()));
        assert!(!segments_name_eq(
            [ast::Ident{name:3,ctxt:4}, ast::Ident{name:78,ctxt:82}]
                .iter().map(ident_to_segment).collect::<Vec<PathSegment>>().as_slice(),
            [ast::Ident{name:3,ctxt:104}, ast::Ident{name:77,ctxt:182}]
                .iter().map(ident_to_segment).collect::<Vec<PathSegment>>().as_slice()));
    }
}
