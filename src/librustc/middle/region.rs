// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This file actually contains two passes related to regions.  The first
//! pass builds up the `scope_map`, which describes the parent links in
//! the region hierarchy.  The second pass infers which types must be
//! region parameterized.
//!
//! Most of the documentation on regions can be found in
//! `middle/typeck/infer/region_inference.rs`

use session::Session;
use middle::ty::{self, Ty, FreeRegion};
use util::nodemap::{FnvHashMap, FnvHashSet, NodeMap};
use util::common::can_reach;

use std::cell::RefCell;
use syntax::codemap::{self, Span};
use syntax::{ast, visit};
use syntax::ast::{Block, Item, FnDecl, NodeId, Arm, Pat, Stmt, Expr, Local};
use syntax::ast_util::{stmt_id};
use syntax::ast_map;
use syntax::ptr::P;
use syntax::visit::{Visitor, FnKind};

/// CodeExtent represents a statically-describable extent that can be
/// used to bound the lifetime/region for values.
///
/// `Misc(node_id)`: Any AST node that has any extent at all has the
/// `Misc(node_id)` extent. Other variants represent special cases not
/// immediately derivable from the abstract syntax tree structure.
///
/// `DestructionScope(node_id)` represents the extent of destructors
/// implicitly-attached to `node_id` that run immediately after the
/// expression for `node_id` itself. Not every AST node carries a
/// `DestructionScope`, but those that are `terminating_scopes` do;
/// see discussion with `RegionMaps`.
///
/// `Remainder(BlockRemainder { block, statement_index })` represents
/// the extent of user code running immediately after the initializer
/// expression for the indexed statement, until the end of the block.
///
/// So: the following code can be broken down into the extents beneath:
/// ```
/// let a = f().g( 'b: { let x = d(); let y = d(); x.h(y)  }   ) ;
/// ```
///
///                                                              +-+ (D12.)
///                                                        +-+       (D11.)
///                                              +---------+         (R10.)
///                                              +-+                  (D9.)
///                                   +----------+                    (M8.)
///                                 +----------------------+          (R7.)
///                                 +-+                               (D6.)
///                      +----------+                                 (M5.)
///                    +-----------------------------------+          (M4.)
///         +--------------------------------------------------+      (M3.)
///         +--+                                                      (M2.)
/// +-----------------------------------------------------------+     (M1.)
///
///  (M1.): Misc extent of the whole `let a = ...;` statement.
///  (M2.): Misc extent of the `f()` expression.
///  (M3.): Misc extent of the `f().g(..)` expression.
///  (M4.): Misc extent of the block labelled `'b:`.
///  (M5.): Misc extent of the `let x = d();` statement
///  (D6.): DestructionScope for temporaries created during M5.
///  (R7.): Remainder extent for block `'b:`, stmt 0 (let x = ...).
///  (M8.): Misc Extent of the `let y = d();` statement.
///  (D9.): DestructionScope for temporaries created during M8.
/// (R10.): Remainder extent for block `'b:`, stmt 1 (let y = ...).
/// (D11.): DestructionScope for temporaries and bindings from block `'b:`.
/// (D12.): DestructionScope for temporaries created during M1 (e.g. f()).
///
/// Note that while the above picture shows the destruction scopes
/// as following their corresponding misc extents, in the internal
/// data structures of the compiler the destruction scopes are
/// represented as enclosing parents. This is sound because we use the
/// enclosing parent relationship just to ensure that referenced
/// values live long enough; phrased another way, the starting point
/// of each range is not really the important thing in the above
/// picture, but rather the ending point.
///
/// FIXME (pnkfelix): This currently derives `PartialOrd` and `Ord` to
/// placate the same deriving in `ty::FreeRegion`, but we may want to
/// actually attach a more meaningful ordering to scopes than the one
/// generated via deriving here.
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, RustcEncodable,
           RustcDecodable, Debug, Copy)]
pub enum CodeExtent {
    Misc(ast::NodeId),
    DestructionScope(ast::NodeId), // extent of destructors for temporaries of node-id
    Remainder(BlockRemainder)
}

/// extent of destructors for temporaries of node-id
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, RustcEncodable,
           RustcDecodable, Debug, Copy)]
pub struct DestructionScopeData {
    pub node_id: ast::NodeId
}

impl DestructionScopeData {
    pub fn new(node_id: ast::NodeId) -> DestructionScopeData {
        DestructionScopeData { node_id: node_id }
    }
    pub fn to_code_extent(&self) -> CodeExtent {
        CodeExtent::DestructionScope(self.node_id)
    }
}

/// Represents a subscope of `block` for a binding that is introduced
/// by `block.stmts[first_statement_index]`. Such subscopes represent
/// a suffix of the block. Note that each subscope does not include
/// the initializer expression, if any, for the statement indexed by
/// `first_statement_index`.
///
/// For example, given `{ let (a, b) = EXPR_1; let c = EXPR_2; ... }`:
///
/// * the subscope with `first_statement_index == 0` is scope of both
///   `a` and `b`; it does not include EXPR_1, but does include
///   everything after that first `let`. (If you want a scope that
///   includes EXPR_1 as well, then do not use `CodeExtent::Remainder`,
///   but instead another `CodeExtent` that encompasses the whole block,
///   e.g. `CodeExtent::Misc`.
///
/// * the subscope with `first_statement_index == 1` is scope of `c`,
///   and thus does not include EXPR_2, but covers the `...`.
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, RustcEncodable,
         RustcDecodable, Debug, Copy)]
pub struct BlockRemainder {
    pub block: ast::NodeId,
    pub first_statement_index: uint,
}

impl CodeExtent {
    /// Creates a scope that represents the dynamic extent associated
    /// with `node_id`.
    pub fn from_node_id(node_id: ast::NodeId) -> CodeExtent {
        CodeExtent::Misc(node_id)
    }

    /// Returns a node id associated with this scope.
    ///
    /// NB: likely to be replaced as API is refined; e.g. pnkfelix
    /// anticipates `fn entry_node_id` and `fn each_exit_node_id`.
    pub fn node_id(&self) -> ast::NodeId {
        match *self {
            CodeExtent::Misc(node_id) => node_id,
            CodeExtent::Remainder(br) => br.block,
            CodeExtent::DestructionScope(node_id) => node_id,
        }
    }

    /// Maps this scope to a potentially new one according to the
    /// NodeId transformer `f_id`.
    pub fn map_id<F>(&self, f_id: F) -> CodeExtent where
        F: FnOnce(ast::NodeId) -> ast::NodeId,
    {
        match *self {
            CodeExtent::Misc(node_id) => CodeExtent::Misc(f_id(node_id)),
            CodeExtent::Remainder(br) =>
                CodeExtent::Remainder(BlockRemainder {
                    block: f_id(br.block), first_statement_index: br.first_statement_index }),
            CodeExtent::DestructionScope(node_id) =>
                CodeExtent::DestructionScope(f_id(node_id)),
        }
    }

    /// Returns the span of this CodeExtent.  Note that in general the
    /// returned span may not correspond to the span of any node id in
    /// the AST.
    pub fn span(&self, ast_map: &ast_map::Map) -> Option<Span> {
        match ast_map.find(self.node_id()) {
            Some(ast_map::NodeBlock(ref blk)) => {
                match *self {
                    CodeExtent::Misc(_) |
                    CodeExtent::DestructionScope(_) => Some(blk.span),

                    CodeExtent::Remainder(r) => {
                        assert_eq!(r.block, blk.id);
                        // Want span for extent starting after the
                        // indexed statement and ending at end of
                        // `blk`; reuse span of `blk` and shift `lo`
                        // forward to end of indexed statement.
                        //
                        // (This is the special case aluded to in the
                        // doc-comment for this method)
                        let stmt_span = blk.stmts[r.first_statement_index].span;
                        Some(Span { lo: stmt_span.hi, ..blk.span })
                    }
                }
            }
            Some(ast_map::NodeExpr(ref expr)) => Some(expr.span),
            Some(ast_map::NodeStmt(ref stmt)) => Some(stmt.span),
            Some(ast_map::NodeItem(ref item)) => Some(item.span),
            Some(_) | None => None,
         }
    }
}

/// The region maps encode information about region relationships.
///
/// - `scope_map` maps from a scope id to the enclosing scope id; this is
///   usually corresponding to the lexical nesting, though in the case of
///   closures the parent scope is the innermost conditional expression or repeating
///   block. (Note that the enclosing scope id for the block
///   associated with a closure is the closure itself.)
///
/// - `var_map` maps from a variable or binding id to the block in which
///   that variable is declared.
///
/// - `free_region_map` maps from a free region `a` to a list of free
///   regions `bs` such that `a <= b for all b in bs`
///   - the free region map is populated during type check as we check
///     each function. See the function `relate_free_regions` for
///     more information.
///
/// - `rvalue_scopes` includes entries for those expressions whose cleanup
///   scope is larger than the default. The map goes from the expression
///   id to the cleanup scope id. For rvalues not present in this table,
///   the appropriate cleanup scope is the innermost enclosing statement,
///   conditional expression, or repeating block (see `terminating_scopes`).
///
/// - `terminating_scopes` is a set containing the ids of each statement,
///   or conditional/repeating expression. These scopes are calling "terminating
///   scopes" because, when attempting to find the scope of a temporary, by
///   default we search up the enclosing scopes until we encounter the
///   terminating scope. A conditional/repeating
///   expression is one which is not guaranteed to execute exactly once
///   upon entering the parent scope. This could be because the expression
///   only executes conditionally, such as the expression `b` in `a && b`,
///   or because the expression may execute many times, such as a loop
///   body. The reason that we distinguish such expressions is that, upon
///   exiting the parent scope, we cannot statically know how many times
///   the expression executed, and thus if the expression creates
///   temporaries we cannot know statically how many such temporaries we
///   would have to cleanup. Therefore we ensure that the temporaries never
///   outlast the conditional/repeating expression, preventing the need
///   for dynamic checks and/or arbitrary amounts of stack space.
pub struct RegionMaps {
    scope_map: RefCell<FnvHashMap<CodeExtent, CodeExtent>>,
    var_map: RefCell<NodeMap<CodeExtent>>,
    free_region_map: RefCell<FnvHashMap<FreeRegion, Vec<FreeRegion>>>,
    rvalue_scopes: RefCell<NodeMap<CodeExtent>>,
    terminating_scopes: RefCell<FnvHashSet<CodeExtent>>,
}

/// Carries the node id for the innermost block or match expression,
/// for building up the `var_map` which maps ids to the blocks in
/// which they were declared.
#[derive(PartialEq, Eq, Debug, Copy)]
enum InnermostDeclaringBlock {
    None,
    Block(ast::NodeId),
    Statement(DeclaringStatementContext),
    Match(ast::NodeId),
}

impl InnermostDeclaringBlock {
    fn to_code_extent(&self) -> Option<CodeExtent> {
        let extent = match *self {
            InnermostDeclaringBlock::None => {
                return Option::None;
            }
            InnermostDeclaringBlock::Block(id) |
            InnermostDeclaringBlock::Match(id) => CodeExtent::from_node_id(id),
            InnermostDeclaringBlock::Statement(s) =>  s.to_code_extent(),
        };
        Option::Some(extent)
    }
}

/// Contextual information for declarations introduced by a statement
/// (i.e. `let`). It carries node-id's for statement and enclosing
/// block both, as well as the statement's index within the block.
#[derive(PartialEq, Eq, Debug, Copy)]
struct DeclaringStatementContext {
    stmt_id: ast::NodeId,
    block_id: ast::NodeId,
    stmt_index: uint,
}

impl DeclaringStatementContext {
    fn to_code_extent(&self) -> CodeExtent {
        CodeExtent::Remainder(BlockRemainder {
            block: self.block_id,
            first_statement_index: self.stmt_index,
        })
    }
}

#[derive(PartialEq, Eq, Debug, Copy)]
enum InnermostEnclosingExpr {
    None,
    Some(ast::NodeId),
    Statement(DeclaringStatementContext),
}

impl InnermostEnclosingExpr {
    fn to_code_extent(&self) -> Option<CodeExtent> {
        let extent = match *self {
            InnermostEnclosingExpr::None => {
                return Option::None;
            }
            InnermostEnclosingExpr::Statement(s) =>
                s.to_code_extent(),
            InnermostEnclosingExpr::Some(parent_id) =>
                CodeExtent::from_node_id(parent_id),
        };
        Some(extent)
    }
}

#[derive(Debug, Copy)]
pub struct Context {
    var_parent: InnermostDeclaringBlock,

    parent: InnermostEnclosingExpr,
}

struct RegionResolutionVisitor<'a> {
    sess: &'a Session,

    // Generated maps:
    region_maps: &'a RegionMaps,

    cx: Context
}


impl RegionMaps {
    pub fn each_encl_scope<E>(&self, mut e:E) where E: FnMut(&CodeExtent, &CodeExtent) {
        for (child, parent) in self.scope_map.borrow().iter() {
            e(child, parent)
        }
    }
    pub fn each_var_scope<E>(&self, mut e:E) where E: FnMut(&ast::NodeId, &CodeExtent) {
        for (child, parent) in self.var_map.borrow().iter() {
            e(child, parent)
        }
    }
    pub fn each_encl_free_region<E>(&self, mut e:E) where E: FnMut(&FreeRegion, &FreeRegion) {
        for (child, parents) in self.free_region_map.borrow().iter() {
            for parent in parents.iter() {
                e(child, parent)
            }
        }
    }
    pub fn each_rvalue_scope<E>(&self, mut e:E) where E: FnMut(&ast::NodeId, &CodeExtent) {
        for (child, parent) in self.rvalue_scopes.borrow().iter() {
            e(child, parent)
        }
    }
    pub fn each_terminating_scope<E>(&self, mut e:E) where E: FnMut(&CodeExtent) {
        for scope in self.terminating_scopes.borrow().iter() {
            e(scope)
        }
    }

    pub fn relate_free_regions(&self, sub: FreeRegion, sup: FreeRegion) {
        match self.free_region_map.borrow_mut().get_mut(&sub) {
            Some(sups) => {
                if !sups.iter().any(|x| x == &sup) {
                    sups.push(sup);
                }
                return;
            }
            None => {}
        }

        debug!("relate_free_regions(sub={:?}, sup={:?})", sub, sup);
        self.free_region_map.borrow_mut().insert(sub, vec!(sup));
    }

    pub fn record_encl_scope(&self, sub: CodeExtent, sup: CodeExtent) {
        debug!("record_encl_scope(sub={:?}, sup={:?})", sub, sup);
        assert!(sub != sup);
        self.scope_map.borrow_mut().insert(sub, sup);
    }

    pub fn record_var_scope(&self, var: ast::NodeId, lifetime: CodeExtent) {
        debug!("record_var_scope(sub={:?}, sup={:?})", var, lifetime);
        assert!(var != lifetime.node_id());
        self.var_map.borrow_mut().insert(var, lifetime);
    }

    pub fn record_rvalue_scope(&self, var: ast::NodeId, lifetime: CodeExtent) {
        debug!("record_rvalue_scope(sub={:?}, sup={:?})", var, lifetime);
        assert!(var != lifetime.node_id());
        self.rvalue_scopes.borrow_mut().insert(var, lifetime);
    }

    /// Records that a scope is a TERMINATING SCOPE. Whenever we create automatic temporaries --
    /// e.g. by an expression like `a().f` -- they will be freed within the innermost terminating
    /// scope.
    pub fn mark_as_terminating_scope(&self, scope_id: CodeExtent) {
        debug!("record_terminating_scope(scope_id={:?})", scope_id);
        self.terminating_scopes.borrow_mut().insert(scope_id);
    }

    pub fn opt_encl_scope(&self, id: CodeExtent) -> Option<CodeExtent> {
        //! Returns the narrowest scope that encloses `id`, if any.
        self.scope_map.borrow().get(&id).cloned()
    }

    #[allow(dead_code)] // used in middle::cfg
    pub fn encl_scope(&self, id: CodeExtent) -> CodeExtent {
        //! Returns the narrowest scope that encloses `id`, if any.
        match self.scope_map.borrow().get(&id) {
            Some(&r) => r,
            None => { panic!("no enclosing scope for id {:?}", id); }
        }
    }

    /// Returns the lifetime of the local variable `var_id`
    pub fn var_scope(&self, var_id: ast::NodeId) -> CodeExtent {
        match self.var_map.borrow().get(&var_id) {
            Some(&r) => r,
            None => { panic!("no enclosing scope for id {:?}", var_id); }
        }
    }

    pub fn temporary_scope(&self, expr_id: ast::NodeId) -> Option<CodeExtent> {
        //! Returns the scope when temp created by expr_id will be cleaned up

        // check for a designated rvalue scope
        match self.rvalue_scopes.borrow().get(&expr_id) {
            Some(&s) => {
                debug!("temporary_scope({:?}) = {:?} [custom]", expr_id, s);
                return Some(s);
            }
            None => { }
        }

        // else, locate the innermost terminating scope
        // if there's one. Static items, for instance, won't
        // have an enclosing scope, hence no scope will be
        // returned.
        let mut id = match self.opt_encl_scope(CodeExtent::from_node_id(expr_id)) {
            Some(i) => i,
            None => { return None; }
        };

        while !self.terminating_scopes.borrow().contains(&id) {
            match self.opt_encl_scope(id) {
                Some(p) => {
                    id = p;
                }
                None => {
                    debug!("temporary_scope({:?}) = None", expr_id);
                    return None;
                }
            }
        }
        debug!("temporary_scope({:?}) = {:?} [enclosing]", expr_id, id);
        return Some(id);
    }

    pub fn var_region(&self, id: ast::NodeId) -> ty::Region {
        //! Returns the lifetime of the variable `id`.

        let scope = ty::ReScope(self.var_scope(id));
        debug!("var_region({:?}) = {:?}", id, scope);
        scope
    }

    pub fn scopes_intersect(&self, scope1: CodeExtent, scope2: CodeExtent)
                            -> bool {
        self.is_subscope_of(scope1, scope2) ||
        self.is_subscope_of(scope2, scope1)
    }

    /// Returns true if `subscope` is equal to or is lexically nested inside `superscope` and false
    /// otherwise.
    pub fn is_subscope_of(&self,
                          subscope: CodeExtent,
                          superscope: CodeExtent)
                          -> bool {
        let mut s = subscope;
        while superscope != s {
            match self.scope_map.borrow().get(&s) {
                None => {
                    debug!("is_subscope_of({:?}, {:?}, s={:?})=false",
                           subscope, superscope, s);

                    return false;
                }
                Some(&scope) => s = scope
            }
        }

        debug!("is_subscope_of({:?}, {:?})=true",
               subscope, superscope);

        return true;
    }

    /// Determines whether two free regions have a subregion relationship
    /// by walking the graph encoded in `free_region_map`.  Note that
    /// it is possible that `sub != sup` and `sub <= sup` and `sup <= sub`
    /// (that is, the user can give two different names to the same lifetime).
    pub fn sub_free_region(&self, sub: FreeRegion, sup: FreeRegion) -> bool {
        can_reach(&*self.free_region_map.borrow(), sub, sup)
    }

    /// Determines whether one region is a subregion of another.  This is intended to run *after
    /// inference* and sadly the logic is somewhat duplicated with the code in infer.rs.
    pub fn is_subregion_of(&self,
                           sub_region: ty::Region,
                           super_region: ty::Region)
                           -> bool {
        debug!("is_subregion_of(sub_region={:?}, super_region={:?})",
               sub_region, super_region);

        sub_region == super_region || {
            match (sub_region, super_region) {
                (ty::ReEmpty, _) |
                (_, ty::ReStatic) => {
                    true
                }

                (ty::ReScope(sub_scope), ty::ReScope(super_scope)) => {
                    self.is_subscope_of(sub_scope, super_scope)
                }

                (ty::ReScope(sub_scope), ty::ReFree(ref fr)) => {
                    self.is_subscope_of(sub_scope, fr.scope.to_code_extent())
                }

                (ty::ReFree(sub_fr), ty::ReFree(super_fr)) => {
                    self.sub_free_region(sub_fr, super_fr)
                }

                (ty::ReEarlyBound(param_id_a, param_space_a, index_a, _),
                 ty::ReEarlyBound(param_id_b, param_space_b, index_b, _)) => {
                    // This case is used only to make sure that explicitly-
                    // specified `Self` types match the real self type in
                    // implementations.
                    param_id_a == param_id_b &&
                        param_space_a == param_space_b &&
                        index_a == index_b
                }

                _ => {
                    false
                }
            }
        }
    }

    /// Finds the nearest common ancestor (if any) of two scopes.  That is, finds the smallest
    /// scope which is greater than or equal to both `scope_a` and `scope_b`.
    pub fn nearest_common_ancestor(&self,
                                   scope_a: CodeExtent,
                                   scope_b: CodeExtent)
                                   -> Option<CodeExtent> {
        if scope_a == scope_b { return Some(scope_a); }

        let a_ancestors = ancestors_of(self, scope_a);
        let b_ancestors = ancestors_of(self, scope_b);
        let mut a_index = a_ancestors.len() - 1;
        let mut b_index = b_ancestors.len() - 1;

        // Here, ~[ab]_ancestors is a vector going from narrow to broad.
        // The end of each vector will be the item where the scope is
        // defined; if there are any common ancestors, then the tails of
        // the vector will be the same.  So basically we want to walk
        // backwards from the tail of each vector and find the first point
        // where they diverge.  If one vector is a suffix of the other,
        // then the corresponding scope is a superscope of the other.

        if a_ancestors[a_index] != b_ancestors[b_index] {
            return None;
        }

        loop {
            // Loop invariant: a_ancestors[a_index] == b_ancestors[b_index]
            // for all indices between a_index and the end of the array
            if a_index == 0 { return Some(scope_a); }
            if b_index == 0 { return Some(scope_b); }
            a_index -= 1;
            b_index -= 1;
            if a_ancestors[a_index] != b_ancestors[b_index] {
                return Some(a_ancestors[a_index + 1]);
            }
        }

        fn ancestors_of(this: &RegionMaps, scope: CodeExtent)
            -> Vec<CodeExtent> {
            // debug!("ancestors_of(scope={:?})", scope);
            let mut result = vec!(scope);
            let mut scope = scope;
            loop {
                match this.scope_map.borrow().get(&scope) {
                    None => return result,
                    Some(&superscope) => {
                        result.push(superscope);
                        scope = superscope;
                    }
                }
                // debug!("ancestors_of_loop(scope={:?})", scope);
            }
        }
    }
}

/// Records the current parent (if any) as the parent of `child_scope`.
fn record_superlifetime(visitor: &mut RegionResolutionVisitor,
                        child_scope: CodeExtent,
                        _sp: Span) {
    match visitor.cx.parent.to_code_extent() {
        Some(parent_scope) =>
            visitor.region_maps.record_encl_scope(child_scope, parent_scope),
        None => {}
    }
}

/// Records the lifetime of a local variable as `cx.var_parent`
fn record_var_lifetime(visitor: &mut RegionResolutionVisitor,
                       var_id: ast::NodeId,
                       _sp: Span) {
    match visitor.cx.var_parent.to_code_extent() {
        Some(parent_scope) =>
            visitor.region_maps.record_var_scope(var_id, parent_scope),
        None => {
            // this can happen in extern fn declarations like
            //
            // extern fn isalnum(c: c_int) -> c_int
        }
    }
}

fn resolve_block(visitor: &mut RegionResolutionVisitor, blk: &ast::Block) {
    debug!("resolve_block(blk.id={:?})", blk.id);

    let prev_cx = visitor.cx;

    let blk_scope = CodeExtent::Misc(blk.id);
    // If block was previously marked as a terminating scope during
    // the recursive visit of its parent node in the AST, then we need
    // to account for the destruction scope representing the extent of
    // the destructors that run immediately after the the block itself
    // completes.
    if visitor.region_maps.terminating_scopes.borrow().contains(&blk_scope) {
        let dtor_scope = CodeExtent::DestructionScope(blk.id);
        record_superlifetime(visitor, dtor_scope, blk.span);
        visitor.region_maps.record_encl_scope(blk_scope, dtor_scope);
    } else {
        record_superlifetime(visitor, blk_scope, blk.span);
    }

    // We treat the tail expression in the block (if any) somewhat
    // differently from the statements. The issue has to do with
    // temporary lifetimes. Consider the following:
    //
    //    quux({
    //        let inner = ... (&bar()) ...;
    //
    //        (... (&foo()) ...) // (the tail expression)
    //    }, other_argument());
    //
    // Each of the statements within the block is a terminating
    // scope, and thus a temporary (e.g. the result of calling
    // `bar()` in the initalizer expression for `let inner = ...;`)
    // will be cleaned up immediately after its corresponding
    // statement (i.e. `let inner = ...;`) executes.
    //
    // On the other hand, temporaries associated with evaluating the
    // tail expression for the block are assigned lifetimes so that
    // they will be cleaned up as part of the terminating scope
    // *surrounding* the block expression. Here, the terminating
    // scope for the block expression is the `quux(..)` call; so
    // those temporaries will only be cleaned up *after* both
    // `other_argument()` has run and also the call to `quux(..)`
    // itself has returned.

    visitor.cx = Context {
        var_parent: InnermostDeclaringBlock::Block(blk.id),
        parent: InnermostEnclosingExpr::Some(blk.id),
    };

    {
        // This block should be kept approximately in sync with
        // `visit::walk_block`. (We manually walk the block, rather
        // than call `walk_block`, in order to maintain precise
        // `InnermostDeclaringBlock` information.)

        for (i, statement) in blk.stmts.iter().enumerate() {
            if let ast::StmtDecl(_, stmt_id) = statement.node {
                // Each StmtDecl introduces a subscope for bindings
                // introduced by the declaration; this subscope covers
                // a suffix of the block . Each subscope in a block
                // has the previous subscope in the block as a parent,
                // except for the first such subscope, which has the
                // block itself as a parent.
                let declaring = DeclaringStatementContext {
                    stmt_id: stmt_id,
                    block_id: blk.id,
                    stmt_index: i,
                };
                record_superlifetime(
                    visitor, declaring.to_code_extent(), statement.span);
                visitor.cx = Context {
                    var_parent: InnermostDeclaringBlock::Statement(declaring),
                    parent: InnermostEnclosingExpr::Statement(declaring),
                };
            }
            visitor.visit_stmt(&**statement)
        }
        visit::walk_expr_opt(visitor, &blk.expr)
    }

    visitor.cx = prev_cx;
}

fn resolve_arm(visitor: &mut RegionResolutionVisitor, arm: &ast::Arm) {
    let arm_body_scope = CodeExtent::from_node_id(arm.body.id);
    visitor.region_maps.mark_as_terminating_scope(arm_body_scope);

    match arm.guard {
        Some(ref expr) => {
            let guard_scope = CodeExtent::from_node_id(expr.id);
            visitor.region_maps.mark_as_terminating_scope(guard_scope);
        }
        None => { }
    }

    visit::walk_arm(visitor, arm);
}

fn resolve_pat(visitor: &mut RegionResolutionVisitor, pat: &ast::Pat) {
    record_superlifetime(visitor, CodeExtent::from_node_id(pat.id), pat.span);

    // If this is a binding (or maybe a binding, I'm too lazy to check
    // the def map) then record the lifetime of that binding.
    match pat.node {
        ast::PatIdent(..) => {
            record_var_lifetime(visitor, pat.id, pat.span);
        }
        _ => { }
    }

    visit::walk_pat(visitor, pat);
}

fn resolve_stmt(visitor: &mut RegionResolutionVisitor, stmt: &ast::Stmt) {
    let stmt_id = stmt_id(stmt);
    debug!("resolve_stmt(stmt.id={:?})", stmt_id);

    let stmt_scope = CodeExtent::from_node_id(stmt_id);

    // Every statement will clean up the temporaries created during
    // execution of that statement. Therefore each statement has an
    // associated destruction scope that represents the extent of the
    // statement plus its destructors, and thus the extent for which
    // regions referenced by the destructors need to survive.
    visitor.region_maps.mark_as_terminating_scope(stmt_scope);
    let dtor_scope = CodeExtent::DestructionScope(stmt_id);
    visitor.region_maps.record_encl_scope(stmt_scope, dtor_scope);
    record_superlifetime(visitor, dtor_scope, stmt.span);

    let prev_parent = visitor.cx.parent;
    visitor.cx.parent = InnermostEnclosingExpr::Some(stmt_id);
    visit::walk_stmt(visitor, stmt);
    visitor.cx.parent = prev_parent;
}

fn resolve_expr(visitor: &mut RegionResolutionVisitor, expr: &ast::Expr) {
    debug!("resolve_expr(expr.id={:?})", expr.id);

    let expr_scope = CodeExtent::Misc(expr.id);
    // If expr was previously marked as a terminating scope during the
    // recursive visit of its parent node in the AST, then we need to
    // account for the destruction scope representing the extent of
    // the destructors that run immediately after the the expression
    // itself completes.
    if visitor.region_maps.terminating_scopes.borrow().contains(&expr_scope) {
        let dtor_scope = CodeExtent::DestructionScope(expr.id);
        record_superlifetime(visitor, dtor_scope, expr.span);
        visitor.region_maps.record_encl_scope(expr_scope, dtor_scope);
    } else {
        record_superlifetime(visitor, expr_scope, expr.span);
    }

    let prev_cx = visitor.cx;
    visitor.cx.parent = InnermostEnclosingExpr::Some(expr.id);

    {
        let region_maps = &mut visitor.region_maps;
        let terminating = |e: &P<ast::Expr>| {
            let scope = CodeExtent::from_node_id(e.id);
            region_maps.mark_as_terminating_scope(scope)
        };
        let terminating_block = |b: &P<ast::Block>| {
            let scope = CodeExtent::from_node_id(b.id);
            region_maps.mark_as_terminating_scope(scope)
        };
        match expr.node {
            // Conditional or repeating scopes are always terminating
            // scopes, meaning that temporaries cannot outlive them.
            // This ensures fixed size stacks.

            ast::ExprBinary(codemap::Spanned { node: ast::BiAnd, .. }, _, ref r) |
            ast::ExprBinary(codemap::Spanned { node: ast::BiOr, .. }, _, ref r) => {
                // For shortcircuiting operators, mark the RHS as a terminating
                // scope since it only executes conditionally.
                terminating(r);
            }

            ast::ExprIf(_, ref then, Some(ref otherwise)) => {
                terminating_block(then);
                terminating(otherwise);
            }

            ast::ExprIf(ref expr, ref then, None) => {
                terminating(expr);
                terminating_block(then);
            }

            ast::ExprLoop(ref body, _) => {
                terminating_block(body);
            }

            ast::ExprWhile(ref expr, ref body, _) => {
                terminating(expr);
                terminating_block(body);
            }

            ast::ExprMatch(..) => {
                visitor.cx.var_parent = InnermostDeclaringBlock::Match(expr.id);
            }

            ast::ExprAssignOp(..) | ast::ExprIndex(..) |
            ast::ExprUnary(..) | ast::ExprCall(..) | ast::ExprMethodCall(..) => {
                // FIXME(#6268) Nested method calls
                //
                // The lifetimes for a call or method call look as follows:
                //
                // call.id
                // - arg0.id
                // - ...
                // - argN.id
                // - call.callee_id
                //
                // The idea is that call.callee_id represents *the time when
                // the invoked function is actually running* and call.id
                // represents *the time to prepare the arguments and make the
                // call*.  See the section "Borrows in Calls" borrowck/doc.rs
                // for an extended explanation of why this distinction is
                // important.
                //
                // record_superlifetime(new_cx, expr.callee_id);
            }

            _ => {}
        }
    }

    visit::walk_expr(visitor, expr);
    visitor.cx = prev_cx;
}

fn resolve_local(visitor: &mut RegionResolutionVisitor, local: &ast::Local) {
    debug!("resolve_local(local.id={:?},local.init={:?})",
           local.id,local.init.is_some());

    // For convenience in trans, associate with the local-id the var
    // scope that will be used for any bindings declared in this
    // pattern.
    let blk_scope = visitor.cx.var_parent.to_code_extent()
        .unwrap_or_else(|| visitor.sess.span_bug(
            local.span, "local without enclosing block"));

    visitor.region_maps.record_var_scope(local.id, blk_scope);

    // As an exception to the normal rules governing temporary
    // lifetimes, initializers in a let have a temporary lifetime
    // of the enclosing block. This means that e.g. a program
    // like the following is legal:
    //
    //     let ref x = HashMap::new();
    //
    // Because the hash map will be freed in the enclosing block.
    //
    // We express the rules more formally based on 3 grammars (defined
    // fully in the helpers below that implement them):
    //
    // 1. `E&`, which matches expressions like `&<rvalue>` that
    //    own a pointer into the stack.
    //
    // 2. `P&`, which matches patterns like `ref x` or `(ref x, ref
    //    y)` that produce ref bindings into the value they are
    //    matched against or something (at least partially) owned by
    //    the value they are matched against. (By partially owned,
    //    I mean that creating a binding into a ref-counted or managed value
    //    would still count.)
    //
    // 3. `ET`, which matches both rvalues like `foo()` as well as lvalues
    //    based on rvalues like `foo().x[2].y`.
    //
    // A subexpression `<rvalue>` that appears in a let initializer
    // `let pat [: ty] = expr` has an extended temporary lifetime if
    // any of the following conditions are met:
    //
    // A. `pat` matches `P&` and `expr` matches `ET`
    //    (covers cases where `pat` creates ref bindings into an rvalue
    //     produced by `expr`)
    // B. `ty` is a borrowed pointer and `expr` matches `ET`
    //    (covers cases where coercion creates a borrow)
    // C. `expr` matches `E&`
    //    (covers cases `expr` borrows an rvalue that is then assigned
    //     to memory (at least partially) owned by the binding)
    //
    // Here are some examples hopefully giving an intuition where each
    // rule comes into play and why:
    //
    // Rule A. `let (ref x, ref y) = (foo().x, 44)`. The rvalue `(22, 44)`
    // would have an extended lifetime, but not `foo()`.
    //
    // Rule B. `let x: &[...] = [foo().x]`. The rvalue `[foo().x]`
    // would have an extended lifetime, but not `foo()`.
    //
    // Rule C. `let x = &foo().x`. The rvalue ``foo()` would have extended
    // lifetime.
    //
    // In some cases, multiple rules may apply (though not to the same
    // rvalue). For example:
    //
    //     let ref x = [&a(), &b()];
    //
    // Here, the expression `[...]` has an extended lifetime due to rule
    // A, but the inner rvalues `a()` and `b()` have an extended lifetime
    // due to rule C.
    //
    // FIXME(#6308) -- Note that `[]` patterns work more smoothly post-DST.

    match local.init {
        Some(ref expr) => {
            record_rvalue_scope_if_borrow_expr(visitor, &**expr, blk_scope);

            let is_borrow =
                if let Some(ref ty) = local.ty { is_borrowed_ty(&**ty) } else { false };

            if is_binding_pat(&*local.pat) || is_borrow {
                record_rvalue_scope(visitor, &**expr, blk_scope);
            }
        }

        None => { }
    }

    visit::walk_local(visitor, local);

    /// True if `pat` match the `P&` nonterminal:
    ///
    ///     P& = ref X
    ///        | StructName { ..., P&, ... }
    ///        | VariantName(..., P&, ...)
    ///        | [ ..., P&, ... ]
    ///        | ( ..., P&, ... )
    ///        | box P&
    fn is_binding_pat(pat: &ast::Pat) -> bool {
        match pat.node {
            ast::PatIdent(ast::BindByRef(_), _, _) => true,

            ast::PatStruct(_, ref field_pats, _) => {
                field_pats.iter().any(|fp| is_binding_pat(&*fp.node.pat))
            }

            ast::PatVec(ref pats1, ref pats2, ref pats3) => {
                pats1.iter().any(|p| is_binding_pat(&**p)) ||
                pats2.iter().any(|p| is_binding_pat(&**p)) ||
                pats3.iter().any(|p| is_binding_pat(&**p))
            }

            ast::PatEnum(_, Some(ref subpats)) |
            ast::PatTup(ref subpats) => {
                subpats.iter().any(|p| is_binding_pat(&**p))
            }

            ast::PatBox(ref subpat) => {
                is_binding_pat(&**subpat)
            }

            _ => false,
        }
    }

    /// True if `ty` is a borrowed pointer type like `&int` or `&[...]`.
    fn is_borrowed_ty(ty: &ast::Ty) -> bool {
        match ty.node {
            ast::TyRptr(..) => true,
            _ => false
        }
    }

    /// If `expr` matches the `E&` grammar, then records an extended rvalue scope as appropriate:
    ///
    ///     E& = & ET
    ///        | StructName { ..., f: E&, ... }
    ///        | [ ..., E&, ... ]
    ///        | ( ..., E&, ... )
    ///        | {...; E&}
    ///        | box E&
    ///        | E& as ...
    ///        | ( E& )
    fn record_rvalue_scope_if_borrow_expr(visitor: &mut RegionResolutionVisitor,
                                          expr: &ast::Expr,
                                          blk_id: CodeExtent) {
        match expr.node {
            ast::ExprAddrOf(_, ref subexpr) => {
                record_rvalue_scope_if_borrow_expr(visitor, &**subexpr, blk_id);
                record_rvalue_scope(visitor, &**subexpr, blk_id);
            }
            ast::ExprStruct(_, ref fields, _) => {
                for field in fields {
                    record_rvalue_scope_if_borrow_expr(
                        visitor, &*field.expr, blk_id);
                }
            }
            ast::ExprVec(ref subexprs) |
            ast::ExprTup(ref subexprs) => {
                for subexpr in subexprs {
                    record_rvalue_scope_if_borrow_expr(
                        visitor, &**subexpr, blk_id);
                }
            }
            ast::ExprUnary(ast::UnUniq, ref subexpr) => {
                record_rvalue_scope_if_borrow_expr(visitor, &**subexpr, blk_id);
            }
            ast::ExprCast(ref subexpr, _) |
            ast::ExprParen(ref subexpr) => {
                record_rvalue_scope_if_borrow_expr(visitor, &**subexpr, blk_id)
            }
            ast::ExprBlock(ref block) => {
                match block.expr {
                    Some(ref subexpr) => {
                        record_rvalue_scope_if_borrow_expr(
                            visitor, &**subexpr, blk_id);
                    }
                    None => { }
                }
            }
            _ => {
            }
        }
    }

    /// Applied to an expression `expr` if `expr` -- or something owned or partially owned by
    /// `expr` -- is going to be indirectly referenced by a variable in a let statement. In that
    /// case, the "temporary lifetime" or `expr` is extended to be the block enclosing the `let`
    /// statement.
    ///
    /// More formally, if `expr` matches the grammar `ET`, record the rvalue scope of the matching
    /// `<rvalue>` as `blk_id`:
    ///
    ///     ET = *ET
    ///        | ET[...]
    ///        | ET.f
    ///        | (ET)
    ///        | <rvalue>
    ///
    /// Note: ET is intended to match "rvalues or lvalues based on rvalues".
    fn record_rvalue_scope<'a>(visitor: &mut RegionResolutionVisitor,
                               expr: &'a ast::Expr,
                               blk_scope: CodeExtent) {
        let mut expr = expr;
        loop {
            // Note: give all the expressions matching `ET` with the
            // extended temporary lifetime, not just the innermost rvalue,
            // because in trans if we must compile e.g. `*rvalue()`
            // into a temporary, we request the temporary scope of the
            // outer expression.
            visitor.region_maps.record_rvalue_scope(expr.id, blk_scope);

            match expr.node {
                ast::ExprAddrOf(_, ref subexpr) |
                ast::ExprUnary(ast::UnDeref, ref subexpr) |
                ast::ExprField(ref subexpr, _) |
                ast::ExprTupField(ref subexpr, _) |
                ast::ExprIndex(ref subexpr, _) |
                ast::ExprParen(ref subexpr) => {
                    expr = &**subexpr;
                }
                _ => {
                    return;
                }
            }
        }
    }
}

fn resolve_item(visitor: &mut RegionResolutionVisitor, item: &ast::Item) {
    // Items create a new outer block scope as far as we're concerned.
    let prev_cx = visitor.cx;
    visitor.cx = Context {
        var_parent: InnermostDeclaringBlock::None,
        parent: InnermostEnclosingExpr::None
    };
    visit::walk_item(visitor, item);
    visitor.cx = prev_cx;
}

fn resolve_fn(visitor: &mut RegionResolutionVisitor,
              fk: FnKind,
              decl: &ast::FnDecl,
              body: &ast::Block,
              sp: Span,
              id: ast::NodeId) {
    debug!("region::resolve_fn(id={:?}, \
                               span={:?}, \
                               body.id={:?}, \
                               cx.parent={:?})",
           id,
           visitor.sess.codemap().span_to_string(sp),
           body.id,
           visitor.cx.parent);

    let body_scope = CodeExtent::from_node_id(body.id);
    visitor.region_maps.mark_as_terminating_scope(body_scope);
    let dtor_scope = CodeExtent::DestructionScope(body.id);
    visitor.region_maps.record_encl_scope(body_scope, dtor_scope);
    record_superlifetime(visitor, dtor_scope, body.span);

    let outer_cx = visitor.cx;

    // The arguments and `self` are parented to the body of the fn.
    visitor.cx = Context {
        parent: InnermostEnclosingExpr::Some(body.id),
        var_parent: InnermostDeclaringBlock::Block(body.id)
    };
    visit::walk_fn_decl(visitor, decl);

    // The body of the fn itself is either a root scope (top-level fn)
    // or it continues with the inherited scope (closures).
    match fk {
        visit::FkItemFn(..) | visit::FkMethod(..) => {
            visitor.cx = Context {
                parent: InnermostEnclosingExpr::None,
                var_parent: InnermostDeclaringBlock::None
            };
            visitor.visit_block(body);
            visitor.cx = outer_cx;
        }
        visit::FkFnBlock(..) => {
            // FIXME(#3696) -- at present we are place the closure body
            // within the region hierarchy exactly where it appears lexically.
            // This is wrong because the closure may live longer
            // than the enclosing expression. We should probably fix this,
            // but the correct fix is a bit subtle, and I am also not sure
            // that the present approach is unsound -- it may not permit
            // any illegal programs. See issue for more details.
            visitor.cx = outer_cx;
            visitor.visit_block(body);
        }
    }
}

impl<'a, 'v> Visitor<'v> for RegionResolutionVisitor<'a> {

    fn visit_block(&mut self, b: &Block) {
        resolve_block(self, b);
    }

    fn visit_item(&mut self, i: &Item) {
        resolve_item(self, i);
    }

    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v FnDecl,
                b: &'v Block, s: Span, n: NodeId) {
        resolve_fn(self, fk, fd, b, s, n);
    }
    fn visit_arm(&mut self, a: &Arm) {
        resolve_arm(self, a);
    }
    fn visit_pat(&mut self, p: &Pat) {
        resolve_pat(self, p);
    }
    fn visit_stmt(&mut self, s: &Stmt) {
        resolve_stmt(self, s);
    }
    fn visit_expr(&mut self, ex: &Expr) {
        resolve_expr(self, ex);
    }
    fn visit_local(&mut self, l: &Local) {
        resolve_local(self, l);
    }
}

pub fn resolve_crate(sess: &Session, krate: &ast::Crate) -> RegionMaps {
    let maps = RegionMaps {
        scope_map: RefCell::new(FnvHashMap()),
        var_map: RefCell::new(NodeMap()),
        free_region_map: RefCell::new(FnvHashMap()),
        rvalue_scopes: RefCell::new(NodeMap()),
        terminating_scopes: RefCell::new(FnvHashSet()),
    };
    {
        let mut visitor = RegionResolutionVisitor {
            sess: sess,
            region_maps: &maps,
            cx: Context {
                parent: InnermostEnclosingExpr::None,
                var_parent: InnermostDeclaringBlock::None,
            }
        };
        visit::walk_crate(&mut visitor, krate);
    }
    return maps;
}

pub fn resolve_inlined_item(sess: &Session,
                            region_maps: &RegionMaps,
                            item: &ast::InlinedItem) {
    let mut visitor = RegionResolutionVisitor {
        sess: sess,
        region_maps: region_maps,
        cx: Context {
            parent: InnermostEnclosingExpr::None,
            var_parent: InnermostDeclaringBlock::None
        }
    };
    visit::walk_inlined_item(&mut visitor, item);
}
