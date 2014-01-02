// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
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
use ast_util;
use codemap::Span;
use opt_vec;
use parse::token;
use visit::Visitor;
use visit;

use std::hashmap::HashMap;
use std::u32;
use std::local_data;
use std::num;

pub fn path_name_i(idents: &[Ident]) -> ~str {
    // FIXME: Bad copies (#2543 -- same for everything else that says "bad")
    idents.map(|i| token::interner_get(i.name)).connect("::")
}

// totally scary function: ignores all but the last element, should have
// a different name
pub fn path_to_ident(path: &Path) -> Ident {
    path.segments.last().identifier
}

pub fn local_def(id: NodeId) -> DefId {
    ast::DefId { crate: LOCAL_CRATE, node: id }
}

pub fn is_local(did: ast::DefId) -> bool { did.crate == LOCAL_CRATE }

pub fn stmt_id(s: &Stmt) -> NodeId {
    match s.node {
      StmtDecl(_, id) => id,
      StmtExpr(_, id) => id,
      StmtSemi(_, id) => id,
      StmtMac(..) => fail!("attempted to analyze unexpanded stmt")
    }
}

pub fn variant_def_ids(d: Def) -> Option<(DefId, DefId)> {
    match d {
      DefVariant(enum_id, var_id, _) => {
          Some((enum_id, var_id))
      }
      _ => None
    }
}

pub fn def_id_of_def(d: Def) -> DefId {
    match d {
      DefFn(id, _) | DefStaticMethod(id, _, _) | DefMod(id) |
      DefForeignMod(id) | DefStatic(id, _) |
      DefVariant(_, id, _) | DefTy(id) | DefTyParam(id, _) |
      DefUse(id) | DefStruct(id) | DefTrait(id) | DefMethod(id, _) => {
        id
      }
      DefArg(id, _) | DefLocal(id, _) | DefSelf(id, _) | DefSelfTy(id)
      | DefUpvar(id, _, _, _) | DefBinding(id, _) | DefRegion(id)
      | DefTyParamBinder(id) | DefLabel(id) => {
        local_def(id)
      }

      DefPrimTy(_) => fail!()
    }
}

pub fn binop_to_str(op: BinOp) -> ~str {
    match op {
      BiAdd => return ~"+",
      BiSub => return ~"-",
      BiMul => return ~"*",
      BiDiv => return ~"/",
      BiRem => return ~"%",
      BiAnd => return ~"&&",
      BiOr => return ~"||",
      BiBitXor => return ~"^",
      BiBitAnd => return ~"&",
      BiBitOr => return ~"|",
      BiShl => return ~"<<",
      BiShr => return ~">>",
      BiEq => return ~"==",
      BiLt => return ~"<",
      BiLe => return ~"<=",
      BiNe => return ~"!=",
      BiGe => return ~">=",
      BiGt => return ~">"
    }
}

pub fn binop_to_method_name(op: BinOp) -> Option<~str> {
    match op {
      BiAdd => return Some(~"add"),
      BiSub => return Some(~"sub"),
      BiMul => return Some(~"mul"),
      BiDiv => return Some(~"div"),
      BiRem => return Some(~"rem"),
      BiBitXor => return Some(~"bitxor"),
      BiBitAnd => return Some(~"bitand"),
      BiBitOr => return Some(~"bitor"),
      BiShl => return Some(~"shl"),
      BiShr => return Some(~"shr"),
      BiLt => return Some(~"lt"),
      BiLe => return Some(~"le"),
      BiGe => return Some(~"ge"),
      BiGt => return Some(~"gt"),
      BiEq => return Some(~"eq"),
      BiNe => return Some(~"ne"),
      BiAnd | BiOr => return None
    }
}

pub fn lazy_binop(b: BinOp) -> bool {
    match b {
      BiAnd => true,
      BiOr => true,
      _ => false
    }
}

pub fn is_shift_binop(b: BinOp) -> bool {
    match b {
      BiShl => true,
      BiShr => true,
      _ => false
    }
}

pub fn unop_to_str(op: UnOp) -> ~str {
    match op {
      UnBox(mt) => if mt == MutMutable { ~"@mut " } else { ~"@" },
      UnUniq => ~"~",
      UnDeref => ~"*",
      UnNot => ~"!",
      UnNeg => ~"-"
    }
}

pub fn is_path(e: @Expr) -> bool {
    return match e.node { ExprPath(_) => true, _ => false };
}

pub fn int_ty_to_str(t: int_ty) -> ~str {
    match t {
      ty_i => ~"",
      ty_i8 => ~"i8",
      ty_i16 => ~"i16",
      ty_i32 => ~"i32",
      ty_i64 => ~"i64"
    }
}

pub fn int_ty_max(t: int_ty) -> u64 {
    match t {
      ty_i8 => 0x80u64,
      ty_i16 => 0x8000u64,
      ty_i | ty_i32 => 0x80000000u64, // actually ni about ty_i
      ty_i64 => 0x8000000000000000u64
    }
}

pub fn uint_ty_to_str(t: uint_ty) -> ~str {
    match t {
      ty_u => ~"u",
      ty_u8 => ~"u8",
      ty_u16 => ~"u16",
      ty_u32 => ~"u32",
      ty_u64 => ~"u64"
    }
}

pub fn uint_ty_max(t: uint_ty) -> u64 {
    match t {
      ty_u8 => 0xffu64,
      ty_u16 => 0xffffu64,
      ty_u | ty_u32 => 0xffffffffu64, // actually ni about ty_u
      ty_u64 => 0xffffffffffffffffu64
    }
}

pub fn float_ty_to_str(t: float_ty) -> ~str {
    match t { ty_f32 => ~"f32", ty_f64 => ~"f64" }
}

pub fn is_call_expr(e: @Expr) -> bool {
    match e.node { ExprCall(..) => true, _ => false }
}

pub fn block_from_expr(e: @Expr) -> P<Block> {
    P(Block {
        view_items: ~[],
        stmts: ~[],
        expr: Some(e),
        id: e.id,
        rules: DefaultBlock,
        span: e.span
    })
}

pub fn ident_to_path(s: Span, identifier: Ident) -> Path {
    ast::Path {
        span: s,
        global: false,
        segments: ~[
            ast::PathSegment {
                identifier: identifier,
                lifetimes: opt_vec::Empty,
                types: opt_vec::Empty,
            }
        ],
    }
}

pub fn ident_to_pat(id: NodeId, s: Span, i: Ident) -> @Pat {
    @ast::Pat { id: id,
                node: PatIdent(BindByValue(MutImmutable), ident_to_path(s, i), None),
                span: s }
}

pub fn is_unguarded(a: &Arm) -> bool {
    match a.guard {
      None => true,
      _    => false
    }
}

pub fn unguarded_pat(a: &Arm) -> Option<~[@Pat]> {
    if is_unguarded(a) {
        Some(/* FIXME (#2543) */ a.pats.clone())
    } else {
        None
    }
}

pub fn public_methods(ms: ~[@method]) -> ~[@method] {
    ms.move_iter().filter(|m| {
        match m.vis {
            public => true,
            _   => false
        }
    }).collect()
}

// extract a TypeMethod from a trait_method. if the trait_method is
// a default, pull out the useful fields to make a TypeMethod
pub fn trait_method_to_ty_method(method: &trait_method) -> TypeMethod {
    match *method {
        required(ref m) => (*m).clone(),
        provided(ref m) => {
            TypeMethod {
                ident: m.ident,
                attrs: m.attrs.clone(),
                purity: m.purity,
                decl: m.decl,
                generics: m.generics.clone(),
                explicit_self: m.explicit_self,
                id: m.id,
                span: m.span,
            }
        }
    }
}

pub fn split_trait_methods(trait_methods: &[trait_method])
    -> (~[TypeMethod], ~[@method]) {
    let mut reqd = ~[];
    let mut provd = ~[];
    for trt_method in trait_methods.iter() {
        match *trt_method {
          required(ref tm) => reqd.push((*tm).clone()),
          provided(m) => provd.push(m)
        }
    };
    (reqd, provd)
}

pub fn struct_field_visibility(field: ast::struct_field) -> visibility {
    match field.node.kind {
        ast::named_field(_, visibility) => visibility,
        ast::unnamed_field => ast::public
    }
}

pub trait inlined_item_utils {
    fn ident(&self) -> Ident;
    fn id(&self) -> ast::NodeId;
    fn accept<E: Clone, V:Visitor<E>>(&self, e: E, v: &mut V);
}

impl inlined_item_utils for inlined_item {
    fn ident(&self) -> Ident {
        match *self {
            ii_item(i) => i.ident,
            ii_foreign(i) => i.ident,
            ii_method(_, _, m) => m.ident,
        }
    }

    fn id(&self) -> ast::NodeId {
        match *self {
            ii_item(i) => i.id,
            ii_foreign(i) => i.id,
            ii_method(_, _, m) => m.id,
        }
    }

    fn accept<E: Clone, V:Visitor<E>>(&self, e: E, v: &mut V) {
        match *self {
            ii_item(i) => v.visit_item(i, e),
            ii_foreign(i) => v.visit_foreign_item(i, e),
            ii_method(_, _, m) => visit::walk_method_helper(v, m, e),
        }
    }
}

/* True if d is either a def_self, or a chain of def_upvars
 referring to a def_self */
pub fn is_self(d: ast::Def) -> bool {
  match d {
    DefSelf(..)           => true,
    DefUpvar(_, d, _, _) => is_self(*d),
    _                     => false
  }
}

/// Maps a binary operator to its precedence
pub fn operator_prec(op: ast::BinOp) -> uint {
  match op {
      // 'as' sits here with 12
      BiMul | BiDiv | BiRem     => 11u,
      BiAdd | BiSub             => 10u,
      BiShl | BiShr             =>  9u,
      BiBitAnd                  =>  8u,
      BiBitXor                  =>  7u,
      BiBitOr                   =>  6u,
      BiLt | BiLe | BiGe | BiGt =>  4u,
      BiEq | BiNe               =>  3u,
      BiAnd                     =>  2u,
      BiOr                      =>  1u
  }
}

/// Precedence of the `as` operator, which is a binary operator
/// not appearing in the prior table.
pub static as_prec: uint = 12u;

pub fn empty_generics() -> Generics {
    Generics {lifetimes: opt_vec::Empty,
              ty_params: opt_vec::Empty}
}

// ______________________________________________________________________
// Enumerating the IDs which appear in an AST

#[deriving(Encodable, Decodable)]
pub struct id_range {
    min: NodeId,
    max: NodeId,
}

impl id_range {
    pub fn max() -> id_range {
        id_range {
            min: u32::max_value,
            max: u32::min_value,
        }
    }

    pub fn empty(&self) -> bool {
        self.min >= self.max
    }

    pub fn add(&mut self, id: NodeId) {
        self.min = num::min(self.min, id);
        self.max = num::max(self.max, id + 1);
    }
}

pub trait IdVisitingOperation {
    fn visit_id(&self, node_id: NodeId);
}

pub struct IdVisitor<'a, O> {
    operation: &'a O,
    pass_through_items: bool,
    visited_outermost: bool,
}

impl<'a, O: IdVisitingOperation> IdVisitor<'a, O> {
    fn visit_generics_helper(&self, generics: &Generics) {
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
                 module: &_mod,
                 _: Span,
                 node_id: NodeId,
                 env: ()) {
        self.operation.visit_id(node_id);
        visit::walk_mod(self, module, env)
    }

    fn visit_view_item(&mut self, view_item: &view_item, env: ()) {
        match view_item.node {
            view_item_extern_mod(_, _, node_id) => {
                self.operation.visit_id(node_id)
            }
            view_item_use(ref view_paths) => {
                for view_path in view_paths.iter() {
                    match view_path.node {
                        view_path_simple(_, _, node_id) |
                        view_path_glob(_, node_id) => {
                            self.operation.visit_id(node_id)
                        }
                        view_path_list(_, ref paths, node_id) => {
                            self.operation.visit_id(node_id);
                            for path in paths.iter() {
                                self.operation.visit_id(path.node.id)
                            }
                        }
                    }
                }
            }
        }
        visit::walk_view_item(self, view_item, env)
    }

    fn visit_foreign_item(&mut self, foreign_item: @foreign_item, env: ()) {
        self.operation.visit_id(foreign_item.id);
        visit::walk_foreign_item(self, foreign_item, env)
    }

    fn visit_item(&mut self, item: @item, env: ()) {
        if !self.pass_through_items {
            if self.visited_outermost {
                return
            } else {
                self.visited_outermost = true
            }
        }

        self.operation.visit_id(item.id);
        match item.node {
            item_enum(ref enum_definition, _) => {
                for variant in enum_definition.variants.iter() {
                    self.operation.visit_id(variant.node.id)
                }
            }
            _ => {}
        }

        visit::walk_item(self, item, env);

        self.visited_outermost = false
    }

    fn visit_local(&mut self, local: @Local, env: ()) {
        self.operation.visit_id(local.id);
        visit::walk_local(self, local, env)
    }

    fn visit_block(&mut self, block: P<Block>, env: ()) {
        self.operation.visit_id(block.id);
        visit::walk_block(self, block, env)
    }

    fn visit_stmt(&mut self, statement: @Stmt, env: ()) {
        self.operation.visit_id(ast_util::stmt_id(statement));
        visit::walk_stmt(self, statement, env)
    }

    fn visit_pat(&mut self, pattern: &Pat, env: ()) {
        self.operation.visit_id(pattern.id);
        visit::walk_pat(self, pattern, env)
    }


    fn visit_expr(&mut self, expression: @Expr, env: ()) {
        {
            let optional_callee_id = expression.get_callee_id();
            for callee_id in optional_callee_id.iter() {
                self.operation.visit_id(*callee_id)
            }
        }
        self.operation.visit_id(expression.id);
        visit::walk_expr(self, expression, env)
    }

    fn visit_ty(&mut self, typ: &Ty, env: ()) {
        self.operation.visit_id(typ.id);
        match typ.node {
            ty_path(_, _, id) => self.operation.visit_id(id),
            _ => {}
        }
        visit::walk_ty(self, typ, env)
    }

    fn visit_generics(&mut self, generics: &Generics, env: ()) {
        self.visit_generics_helper(generics);
        visit::walk_generics(self, generics, env)
    }

    fn visit_fn(&mut self,
                function_kind: &visit::fn_kind,
                function_declaration: &fn_decl,
                block: P<Block>,
                span: Span,
                node_id: NodeId,
                env: ()) {
        if !self.pass_through_items {
            match *function_kind {
                visit::fk_method(..) if self.visited_outermost => return,
                visit::fk_method(..) => self.visited_outermost = true,
                _ => {}
            }
        }

        self.operation.visit_id(node_id);

        match *function_kind {
            visit::fk_item_fn(_, generics, _, _) => {
                self.visit_generics_helper(generics)
            }
            visit::fk_method(_, generics, method) => {
                self.operation.visit_id(method.self_id);
                self.visit_generics_helper(generics)
            }
            visit::fk_fn_block => {}
        }

        for argument in function_declaration.inputs.iter() {
            self.operation.visit_id(argument.id)
        }

        visit::walk_fn(self,
                        function_kind,
                        function_declaration,
                        block,
                        span,
                        node_id,
                        env);

        if !self.pass_through_items {
            match *function_kind {
                visit::fk_method(..) => self.visited_outermost = false,
                _ => {}
            }
        }
    }

    fn visit_struct_field(&mut self, struct_field: &struct_field, env: ()) {
        self.operation.visit_id(struct_field.node.id);
        visit::walk_struct_field(self, struct_field, env)
    }

    fn visit_struct_def(&mut self,
                        struct_def: @struct_def,
                        ident: ast::Ident,
                        generics: &ast::Generics,
                        id: NodeId,
                        _: ()) {
        self.operation.visit_id(id);
        struct_def.ctor_id.map(|ctor_id| self.operation.visit_id(ctor_id));
        visit::walk_struct_def(self, struct_def, ident, generics, id, ());
    }

    fn visit_trait_method(&mut self, tm: &ast::trait_method, _: ()) {
        match *tm {
            ast::required(ref m) => self.operation.visit_id(m.id),
            ast::provided(ref m) => self.operation.visit_id(m.id),
        }
        visit::walk_trait_method(self, tm, ());
    }
}

pub fn visit_ids_for_inlined_item<O: IdVisitingOperation>(item: &inlined_item,
                                                          operation: &O) {
    let mut id_visitor = IdVisitor {
        operation: operation,
        pass_through_items: true,
        visited_outermost: false,
    };
    item.accept((), &mut id_visitor);
}

struct IdRangeComputingVisitor {
    result: @mut id_range,
}

impl IdVisitingOperation for IdRangeComputingVisitor {
    fn visit_id(&self, id: NodeId) {
        self.result.add(id)
    }
}

pub fn compute_id_range_for_inlined_item(item: &inlined_item) -> id_range {
    let result = @mut id_range::max();
    visit_ids_for_inlined_item(item, &IdRangeComputingVisitor {
        result: result,
    });
    *result
}

pub fn is_item_impl(item: @ast::item) -> bool {
    match item.node {
       item_impl(..) => true,
       _            => false
    }
}

pub fn walk_pat(pat: &Pat, it: |&Pat| -> bool) -> bool {
    if !it(pat) {
        return false;
    }

    match pat.node {
        PatIdent(_, _, Some(p)) => walk_pat(p, it),
        PatStruct(_, ref fields, _) => {
            fields.iter().advance(|f| walk_pat(f.pat, |p| it(p)))
        }
        PatEnum(_, Some(ref s)) | PatTup(ref s) => {
            s.iter().advance(|&p| walk_pat(p, |p| it(p)))
        }
        PatBox(s) | PatUniq(s) | PatRegion(s) => {
            walk_pat(s, it)
        }
        PatVec(ref before, ref slice, ref after) => {
            before.iter().advance(|&p| walk_pat(p, |p| it(p))) &&
                slice.iter().advance(|&p| walk_pat(p, |p| it(p))) &&
                after.iter().advance(|&p| walk_pat(p, |p| it(p)))
        }
        PatWild | PatWildMulti | PatLit(_) | PatRange(_, _) | PatIdent(_, _, _) |
        PatEnum(_, _) => {
            true
        }
    }
}

pub trait EachViewItem {
    fn each_view_item(&self, f: |&ast::view_item| -> bool) -> bool;
}

struct EachViewItemData<'a> {
    callback: 'a |&ast::view_item| -> bool,
}

impl<'a> Visitor<()> for EachViewItemData<'a> {
    fn visit_view_item(&mut self, view_item: &ast::view_item, _: ()) {
        let _ = (self.callback)(view_item);
    }
}

impl EachViewItem for ast::Crate {
    fn each_view_item(&self, f: |&ast::view_item| -> bool) -> bool {
        let mut visit = EachViewItemData {
            callback: f,
        };
        visit::walk_crate(&mut visit, self, ());
        true
    }
}

pub fn view_path_id(p: &view_path) -> NodeId {
    match p.node {
      view_path_simple(_, _, id) |
      view_path_glob(_, id) |
      view_path_list(_, _, id) => id
    }
}

/// Returns true if the given struct def is tuple-like; i.e. that its fields
/// are unnamed.
pub fn struct_def_is_tuple_like(struct_def: &ast::struct_def) -> bool {
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

// HYGIENE FUNCTIONS

/// Extend a syntax context with a given mark
pub fn new_mark(m:Mrk, tail:SyntaxContext) -> SyntaxContext {
    new_mark_internal(m,tail,get_sctable())
}

// Extend a syntax context with a given mark and table
// FIXME #8215 : currently pub to allow testing
pub fn new_mark_internal(m:Mrk, tail:SyntaxContext,table:&mut SCTable)
    -> SyntaxContext {
    let key = (tail,m);
    // FIXME #5074 : can't use more natural style because we're missing
    // flow-sensitivity. Results in two lookups on a hash table hit.
    // also applies to new_rename_internal, below.
    // let try_lookup = table.mark_memo.find(&key);
    match table.mark_memo.contains_key(&key) {
        false => {
            let new_idx = idx_push(&mut table.table,Mark(m,tail));
            table.mark_memo.insert(key,new_idx);
            new_idx
        }
        true => {
            match table.mark_memo.find(&key) {
                None => fail!("internal error: key disappeared 2013042901"),
                Some(idxptr) => {*idxptr}
            }
        }
    }
}

/// Extend a syntax context with a given rename
pub fn new_rename(id:Ident, to:Name, tail:SyntaxContext) -> SyntaxContext {
    new_rename_internal(id, to, tail, get_sctable())
}

// Extend a syntax context with a given rename and sctable
// FIXME #8215 : currently pub to allow testing
pub fn new_rename_internal(id:Ident, to:Name, tail:SyntaxContext, table: &mut SCTable)
    -> SyntaxContext {
    let key = (tail,id,to);
    // FIXME #5074
    //let try_lookup = table.rename_memo.find(&key);
    match table.rename_memo.contains_key(&key) {
        false => {
            let new_idx = idx_push(&mut table.table,Rename(id,to,tail));
            table.rename_memo.insert(key,new_idx);
            new_idx
        }
        true => {
            match table.rename_memo.find(&key) {
                None => fail!("internal error: key disappeared 2013042902"),
                Some(idxptr) => {*idxptr}
            }
        }
    }
}

/// Make a fresh syntax context table with EmptyCtxt in slot zero
/// and IllegalCtxt in slot one.
// FIXME #8215 : currently pub to allow testing
pub fn new_sctable_internal() -> SCTable {
    SCTable {
        table: ~[EmptyCtxt,IllegalCtxt],
        mark_memo: HashMap::new(),
        rename_memo: HashMap::new()
    }
}

// fetch the SCTable from TLS, create one if it doesn't yet exist.
pub fn get_sctable() -> @mut SCTable {
    local_data_key!(sctable_key: @@mut SCTable)
    match local_data::get(sctable_key, |k| k.map(|k| *k)) {
        None => {
            let new_table = @@mut new_sctable_internal();
            local_data::set(sctable_key,new_table);
            *new_table
        },
        Some(intr) => *intr
    }
}

/// print out an SCTable for debugging
pub fn display_sctable(table : &SCTable) {
    error!("SC table:");
    for (idx,val) in table.table.iter().enumerate() {
        error!("{:4u} : {:?}",idx,val);
    }
}


/// Add a value to the end of a vec, return its index
fn idx_push<T>(vec: &mut ~[T], val: T) -> u32 {
    vec.push(val);
    (vec.len() - 1) as u32
}

/// Resolve a syntax object to a name, per MTWT.
pub fn mtwt_resolve(id : Ident) -> Name {
    resolve_internal(id, get_sctable(), get_resolve_table())
}

// FIXME #8215: must be pub for testing
pub type ResolveTable = HashMap<(Name,SyntaxContext),Name>;

// okay, I admit, putting this in TLS is not so nice:
// fetch the SCTable from TLS, create one if it doesn't yet exist.
pub fn get_resolve_table() -> @mut ResolveTable {
    local_data_key!(resolve_table_key: @@mut ResolveTable)
    match local_data::get(resolve_table_key, |k| k.map(|k| *k)) {
        None => {
            let new_table = @@mut HashMap::new();
            local_data::set(resolve_table_key,new_table);
            *new_table
        },
        Some(intr) => *intr
    }
}

// Resolve a syntax object to a name, per MTWT.
// adding memoization to possibly resolve 500+ seconds in resolve for librustc (!)
// FIXME #8215 : currently pub to allow testing
pub fn resolve_internal(id : Ident,
                        table : &mut SCTable,
                        resolve_table : &mut ResolveTable) -> Name {
    let key = (id.name,id.ctxt);
    match resolve_table.contains_key(&key) {
        false => {
            let resolved = {
                match table.table[id.ctxt] {
                    EmptyCtxt => id.name,
                    // ignore marks here:
                    Mark(_,subctxt) =>
                        resolve_internal(Ident{name:id.name, ctxt: subctxt},table,resolve_table),
                    // do the rename if necessary:
                    Rename(Ident{name,ctxt},toname,subctxt) => {
                        let resolvedfrom =
                            resolve_internal(Ident{name:name,ctxt:ctxt},table,resolve_table);
                        let resolvedthis =
                            resolve_internal(Ident{name:id.name,ctxt:subctxt},table,resolve_table);
                        if ((resolvedthis == resolvedfrom)
                            && (marksof(ctxt,resolvedthis,table)
                                == marksof(subctxt,resolvedthis,table))) {
                            toname
                        } else {
                            resolvedthis
                        }
                    }
                    IllegalCtxt() => fail!("expected resolvable context, got IllegalCtxt")
                }
            };
            resolve_table.insert(key,resolved);
            resolved
        }
        true => {
            // it's guaranteed to be there, because we just checked that it was
            // there and we never remove anything from the table:
            *(resolve_table.find(&key).unwrap())
        }
    }
}

/// Compute the marks associated with a syntax context.
pub fn mtwt_marksof(ctxt: SyntaxContext, stopname: Name) -> ~[Mrk] {
    marksof(ctxt, stopname, get_sctable())
}

// the internal function for computing marks
// it's not clear to me whether it's better to use a [] mutable
// vector or a cons-list for this.
pub fn marksof(ctxt: SyntaxContext, stopname: Name, table: &SCTable) -> ~[Mrk] {
    let mut result = ~[];
    let mut loopvar = ctxt;
    loop {
        match table.table[loopvar] {
            EmptyCtxt => {return result;},
            Mark(mark,tl) => {
                xorPush(&mut result,mark);
                loopvar = tl;
            },
            Rename(_,name,tl) => {
                // see MTWT for details on the purpose of the stopname.
                // short version: it prevents duplication of effort.
                if (name == stopname) {
                    return result;
                } else {
                    loopvar = tl;
                }
            }
            IllegalCtxt => fail!("expected resolvable context, got IllegalCtxt")
        }
    }
}

/// Return the outer mark for a context with a mark at the outside.
/// FAILS when outside is not a mark.
pub fn mtwt_outer_mark(ctxt: SyntaxContext) -> Mrk {
    let sctable = get_sctable();
    match sctable.table[ctxt] {
        ast::Mark(mrk,_) => mrk,
        _ => fail!("can't retrieve outer mark when outside is not a mark")
    }
}

/// Push a name... unless it matches the one on top, in which
/// case pop and discard (so two of the same marks cancel)
pub fn xorPush(marks: &mut ~[Mrk], mark: Mrk) {
    if ((marks.len() > 0) && (getLast(marks) == mark)) {
        marks.pop();
    } else {
        marks.push(mark);
    }
}

// get the last element of a mutable array.
// FIXME #4903: , must be a separate procedure for now.
pub fn getLast(arr: &~[Mrk]) -> Mrk {
    *arr.last()
}

// are two paths equal when compared unhygienically?
// since I'm using this to replace ==, it seems appropriate
// to compare the span, global, etc. fields as well.
pub fn path_name_eq(a : &ast::Path, b : &ast::Path) -> bool {
    (a.span == b.span)
    && (a.global == b.global)
    && (segments_name_eq(a.segments, b.segments))
}

// are two arrays of segments equal when compared unhygienically?
pub fn segments_name_eq(a : &[ast::PathSegment], b : &[ast::PathSegment]) -> bool {
    if (a.len() != b.len()) {
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

#[cfg(test)]
mod test {
    use ast::*;
    use super::*;
    use opt_vec;
    use std::hashmap::HashMap;

    fn ident_to_segment(id : &Ident) -> PathSegment {
        PathSegment {identifier:id.clone(),
                     lifetimes: opt_vec::Empty,
                     types: opt_vec::Empty}
    }

    #[test] fn idents_name_eq_test() {
        assert!(segments_name_eq([Ident{name:3,ctxt:4},
                                   Ident{name:78,ctxt:82}].map(ident_to_segment),
                                 [Ident{name:3,ctxt:104},
                                   Ident{name:78,ctxt:182}].map(ident_to_segment)));
        assert!(!segments_name_eq([Ident{name:3,ctxt:4},
                                    Ident{name:78,ctxt:82}].map(ident_to_segment),
                                  [Ident{name:3,ctxt:104},
                                    Ident{name:77,ctxt:182}].map(ident_to_segment)));
    }

    #[test] fn xorpush_test () {
        let mut s = ~[];
        xorPush(&mut s,14);
        assert_eq!(s.clone(),~[14]);
        xorPush(&mut s,14);
        assert_eq!(s.clone(),~[]);
        xorPush(&mut s,14);
        assert_eq!(s.clone(),~[14]);
        xorPush(&mut s,15);
        assert_eq!(s.clone(),~[14,15]);
        xorPush (&mut s,16);
        assert_eq!(s.clone(),~[14,15,16]);
        xorPush (&mut s,16);
        assert_eq!(s.clone(),~[14,15]);
        xorPush (&mut s,15);
        assert_eq!(s.clone(),~[14]);
    }

    fn id(n: Name, s: SyntaxContext) -> Ident {
        Ident {name: n, ctxt: s}
    }

    // because of the SCTable, I now need a tidy way of
    // creating syntax objects. Sigh.
    #[deriving(Clone, Eq)]
    enum TestSC {
        M(Mrk),
        R(Ident,Name)
    }

    // unfold a vector of TestSC values into a SCTable,
    // returning the resulting index
    fn unfold_test_sc(tscs : ~[TestSC], tail: SyntaxContext, table : &mut SCTable)
        -> SyntaxContext {
        tscs.rev_iter().fold(tail, |tail : SyntaxContext, tsc : &TestSC|
                  {match *tsc {
                      M(mrk) => new_mark_internal(mrk,tail,table),
                      R(ident,name) => new_rename_internal(ident,name,tail,table)}})
    }

    // gather a SyntaxContext back into a vector of TestSCs
    fn refold_test_sc(mut sc: SyntaxContext, table : &SCTable) -> ~[TestSC] {
        let mut result = ~[];
        loop {
            match table.table[sc] {
                EmptyCtxt => {return result;},
                Mark(mrk,tail) => {
                    result.push(M(mrk));
                    sc = tail;
                    continue;
                },
                Rename(id,name,tail) => {
                    result.push(R(id,name));
                    sc = tail;
                    continue;
                }
                IllegalCtxt => fail!("expected resolvable context, got IllegalCtxt")
            }
        }
    }

    #[test] fn test_unfold_refold(){
        let mut t = new_sctable_internal();

        let test_sc = ~[M(3),R(id(101,0),14),M(9)];
        assert_eq!(unfold_test_sc(test_sc.clone(),EMPTY_CTXT,&mut t),4);
        assert_eq!(t.table[2],Mark(9,0));
        assert_eq!(t.table[3],Rename(id(101,0),14,2));
        assert_eq!(t.table[4],Mark(3,3));
        assert_eq!(refold_test_sc(4,&t),test_sc);
    }

    // extend a syntax context with a sequence of marks given
    // in a vector. v[0] will be the outermost mark.
    fn unfold_marks(mrks:~[Mrk],tail:SyntaxContext,table: &mut SCTable) -> SyntaxContext {
        mrks.rev_iter().fold(tail, |tail:SyntaxContext, mrk:&Mrk|
                   {new_mark_internal(*mrk,tail,table)})
    }

    #[test] fn unfold_marks_test() {
        let mut t = new_sctable_internal();

        assert_eq!(unfold_marks(~[3,7],EMPTY_CTXT,&mut t),3);
        assert_eq!(t.table[2],Mark(7,0));
        assert_eq!(t.table[3],Mark(3,2));
    }

    #[test] fn test_marksof () {
        let stopname = 242;
        let name1 = 243;
        let mut t = new_sctable_internal();
        assert_eq!(marksof (EMPTY_CTXT,stopname,&t),~[]);
        // FIXME #5074: ANF'd to dodge nested calls
        { let ans = unfold_marks(~[4,98],EMPTY_CTXT,&mut t);
         assert_eq! (marksof (ans,stopname,&t),~[4,98]);}
        // does xoring work?
        { let ans = unfold_marks(~[5,5,16],EMPTY_CTXT,&mut t);
         assert_eq! (marksof (ans,stopname,&t), ~[16]);}
        // does nested xoring work?
        { let ans = unfold_marks(~[5,10,10,5,16],EMPTY_CTXT,&mut t);
         assert_eq! (marksof (ans, stopname,&t), ~[16]);}
        // rename where stop doesn't match:
        { let chain = ~[M(9),
                        R(id(name1,
                             new_mark_internal (4, EMPTY_CTXT,&mut t)),
                          100101102),
                        M(14)];
         let ans = unfold_test_sc(chain,EMPTY_CTXT,&mut t);
         assert_eq! (marksof (ans, stopname, &t), ~[9,14]);}
        // rename where stop does match
        { let name1sc = new_mark_internal(4, EMPTY_CTXT, &mut t);
         let chain = ~[M(9),
                       R(id(name1, name1sc),
                         stopname),
                       M(14)];
         let ans = unfold_test_sc(chain,EMPTY_CTXT,&mut t);
         assert_eq! (marksof (ans, stopname, &t), ~[9]); }
    }


    #[test] fn resolve_tests () {
        let a = 40;
        let mut t = new_sctable_internal();
        let mut rt = HashMap::new();
        // - ctxt is MT
        assert_eq!(resolve_internal(id(a,EMPTY_CTXT),&mut t, &mut rt),a);
        // - simple ignored marks
        { let sc = unfold_marks(~[1,2,3],EMPTY_CTXT,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt),a);}
        // - orthogonal rename where names don't match
        { let sc = unfold_test_sc(~[R(id(50,EMPTY_CTXT),51),M(12)],EMPTY_CTXT,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt),a);}
        // - rename where names do match, but marks don't
        { let sc1 = new_mark_internal(1,EMPTY_CTXT,&mut t);
         let sc = unfold_test_sc(~[R(id(a,sc1),50),
                                   M(1),
                                   M(2)],
                                 EMPTY_CTXT,&mut t);
        assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt), a);}
        // - rename where names and marks match
        { let sc1 = unfold_test_sc(~[M(1),M(2)],EMPTY_CTXT,&mut t);
         let sc = unfold_test_sc(~[R(id(a,sc1),50),M(1),M(2)],EMPTY_CTXT,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt), 50); }
        // - rename where names and marks match by literal sharing
        { let sc1 = unfold_test_sc(~[M(1),M(2)],EMPTY_CTXT,&mut t);
         let sc = unfold_test_sc(~[R(id(a,sc1),50)],sc1,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt), 50); }
        // - two renames of the same var.. can only happen if you use
        // local-expand to prevent the inner binding from being renamed
        // during the rename-pass caused by the first:
        println("about to run bad test");
        { let sc = unfold_test_sc(~[R(id(a,EMPTY_CTXT),50),
                                    R(id(a,EMPTY_CTXT),51)],
                                  EMPTY_CTXT,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt), 51); }
        // the simplest double-rename:
        { let a_to_a50 = new_rename_internal(id(a,EMPTY_CTXT),50,EMPTY_CTXT,&mut t);
         let a50_to_a51 = new_rename_internal(id(a,a_to_a50),51,a_to_a50,&mut t);
         assert_eq!(resolve_internal(id(a,a50_to_a51),&mut t, &mut rt),51);
         // mark on the outside doesn't stop rename:
         let sc = new_mark_internal(9,a50_to_a51,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt),51);
         // but mark on the inside does:
         let a50_to_a51_b = unfold_test_sc(~[R(id(a,a_to_a50),51),
                                              M(9)],
                                           a_to_a50,
                                           &mut t);
         assert_eq!(resolve_internal(id(a,a50_to_a51_b),&mut t, &mut rt),50);}
    }

    #[test] fn mtwt_resolve_test(){
        let a = 40;
        assert_eq!(mtwt_resolve(id(a,EMPTY_CTXT)),a);
    }


    #[test] fn hashing_tests () {
        let mut t = new_sctable_internal();
        assert_eq!(new_mark_internal(12,EMPTY_CTXT,&mut t),2);
        assert_eq!(new_mark_internal(13,EMPTY_CTXT,&mut t),3);
        // using the same one again should result in the same index:
        assert_eq!(new_mark_internal(12,EMPTY_CTXT,&mut t),2);
        // I'm assuming that the rename table will behave the same....
    }

    #[test] fn resolve_table_hashing_tests() {
        let mut t = new_sctable_internal();
        let mut rt = HashMap::new();
        assert_eq!(rt.len(),0);
        resolve_internal(id(30,EMPTY_CTXT),&mut t, &mut rt);
        assert_eq!(rt.len(),1);
        resolve_internal(id(39,EMPTY_CTXT),&mut t, &mut rt);
        assert_eq!(rt.len(),2);
        resolve_internal(id(30,EMPTY_CTXT),&mut t, &mut rt);
        assert_eq!(rt.len(),2);
    }

}
