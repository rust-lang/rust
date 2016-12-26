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
//! `middle/infer/region_inference/README.md`

use hir::map as hir_map;
use util::nodemap::{FxHashMap, NodeMap, NodeSet};
use ty;

use std::mem;
use std::rc::Rc;
use syntax::codemap;
use syntax::ast;
use syntax_pos::Span;
use ty::TyCtxt;
use ty::maps::Providers;

use hir;
use hir::def_id::DefId;
use hir::intravisit::{self, Visitor, NestedVisitorMap};
use hir::{Block, Arm, Pat, PatKind, Stmt, Expr, Local};
use hir::map::Node;
use mir::transform::MirSource;

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
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Debug, Copy, RustcEncodable, RustcDecodable)]
pub enum CodeExtent {
    Misc(ast::NodeId),

    // extent of the call-site for a function or closure (outlives
    // the parameters as well as the body).
    CallSiteScope(hir::BodyId),

    // extent of parameters passed to a function or closure (they
    // outlive its body)
    ParameterScope(hir::BodyId),

    // extent of destructors for temporaries of node-id
    DestructionScope(ast::NodeId),

    // extent of code following a `let id = expr;` binding in a block
    Remainder(BlockRemainder)
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
    pub first_statement_index: u32,
}

impl CodeExtent {
    /// Returns a node id associated with this scope.
    ///
    /// NB: likely to be replaced as API is refined; e.g. pnkfelix
    /// anticipates `fn entry_node_id` and `fn each_exit_node_id`.
    pub fn node_id(&self) -> ast::NodeId {
        match *self {
            CodeExtent::Misc(node_id) => node_id,

            // These cases all return rough approximations to the
            // precise extent denoted by `self`.
            CodeExtent::Remainder(br) => br.block,
            CodeExtent::DestructionScope(node_id) => node_id,
            CodeExtent::CallSiteScope(body_id) |
            CodeExtent::ParameterScope(body_id) => body_id.node_id,
        }
    }

    /// Returns the span of this CodeExtent.  Note that in general the
    /// returned span may not correspond to the span of any node id in
    /// the AST.
    pub fn span(&self, hir_map: &hir_map::Map) -> Option<Span> {
        match hir_map.find(self.node_id()) {
            Some(hir_map::NodeBlock(ref blk)) => {
                match *self {
                    CodeExtent::CallSiteScope(_) |
                    CodeExtent::ParameterScope(_) |
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
                        let stmt_span = blk.stmts[r.first_statement_index as usize].span;
                        Some(Span { lo: stmt_span.hi, hi: blk.span.hi, ctxt: stmt_span.ctxt })
                    }
                }
            }
            Some(hir_map::NodeExpr(ref expr)) => Some(expr.span),
            Some(hir_map::NodeStmt(ref stmt)) => Some(stmt.span),
            Some(hir_map::NodeItem(ref item)) => Some(item.span),
            Some(_) | None => None,
         }
    }
}

/// The region maps encode information about region relationships.
pub struct RegionMaps {
    /// If not empty, this body is the root of this region hierarchy.
    root_body: Option<hir::BodyId>,

    /// The parent of the root body owner, if the latter is an
    /// an associated const or method, as impls/traits can also
    /// have lifetime parameters free in this body.
    root_parent: Option<ast::NodeId>,

    /// `scope_map` maps from a scope id to the enclosing scope id;
    /// this is usually corresponding to the lexical nesting, though
    /// in the case of closures the parent scope is the innermost
    /// conditional expression or repeating block. (Note that the
    /// enclosing scope id for the block associated with a closure is
    /// the closure itself.)
    scope_map: FxHashMap<CodeExtent, CodeExtent>,

    /// `var_map` maps from a variable or binding id to the block in
    /// which that variable is declared.
    var_map: NodeMap<CodeExtent>,

    /// maps from a node-id to the associated destruction scope (if any)
    destruction_scopes: NodeMap<CodeExtent>,

    /// `rvalue_scopes` includes entries for those expressions whose cleanup scope is
    /// larger than the default. The map goes from the expression id
    /// to the cleanup scope id. For rvalues not present in this
    /// table, the appropriate cleanup scope is the innermost
    /// enclosing statement, conditional expression, or repeating
    /// block (see `terminating_scopes`).
    rvalue_scopes: NodeMap<CodeExtent>,

    /// Encodes the hierarchy of fn bodies. Every fn body (including
    /// closures) forms its own distinct region hierarchy, rooted in
    /// the block that is the fn body. This map points from the id of
    /// that root block to the id of the root block for the enclosing
    /// fn, if any. Thus the map structures the fn bodies into a
    /// hierarchy based on their lexical mapping. This is used to
    /// handle the relationships between regions in a fn and in a
    /// closure defined by that fn. See the "Modeling closures"
    /// section of the README in infer::region_inference for
    /// more details.
    fn_tree: NodeMap<ast::NodeId>,
}

#[derive(Debug, Copy, Clone)]
pub struct Context {
    /// the root of the current region tree. This is typically the id
    /// of the innermost fn body. Each fn forms its own disjoint tree
    /// in the region hierarchy. These fn bodies are themselves
    /// arranged into a tree. See the "Modeling closures" section of
    /// the README in infer::region_inference for more
    /// details.
    root_id: Option<ast::NodeId>,

    /// the scope that contains any new variables declared
    var_parent: Option<CodeExtent>,

    /// region parent of expressions etc
    parent: Option<CodeExtent>,
}

struct RegionResolutionVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,

    // Generated maps:
    region_maps: RegionMaps,

    cx: Context,

    /// `terminating_scopes` is a set containing the ids of each
    /// statement, or conditional/repeating expression. These scopes
    /// are calling "terminating scopes" because, when attempting to
    /// find the scope of a temporary, by default we search up the
    /// enclosing scopes until we encounter the terminating scope. A
    /// conditional/repeating expression is one which is not
    /// guaranteed to execute exactly once upon entering the parent
    /// scope. This could be because the expression only executes
    /// conditionally, such as the expression `b` in `a && b`, or
    /// because the expression may execute many times, such as a loop
    /// body. The reason that we distinguish such expressions is that,
    /// upon exiting the parent scope, we cannot statically know how
    /// many times the expression executed, and thus if the expression
    /// creates temporaries we cannot know statically how many such
    /// temporaries we would have to cleanup. Therefore we ensure that
    /// the temporaries never outlast the conditional/repeating
    /// expression, preventing the need for dynamic checks and/or
    /// arbitrary amounts of stack space. Terminating scopes end
    /// up being contained in a DestructionScope that contains the
    /// destructor's execution.
    terminating_scopes: NodeSet,
}


impl<'tcx> RegionMaps {
    pub fn new() -> Self {
        RegionMaps {
            root_body: None,
            root_parent: None,
            scope_map: FxHashMap(),
            destruction_scopes: FxHashMap(),
            var_map: NodeMap(),
            rvalue_scopes: NodeMap(),
            fn_tree: NodeMap(),
        }
    }

    pub fn record_code_extent(&mut self,
                              child: CodeExtent,
                              parent: Option<CodeExtent>) {
        debug!("{:?}.parent = {:?}", child, parent);

        if let Some(p) = parent {
            let prev = self.scope_map.insert(child, p);
            assert!(prev.is_none());
        }

        // record the destruction scopes for later so we can query them
        if let CodeExtent::DestructionScope(n) = child {
            self.destruction_scopes.insert(n, child);
        }
    }

    pub fn each_encl_scope<E>(&self, mut e:E) where E: FnMut(CodeExtent, CodeExtent) {
        for (&child, &parent) in &self.scope_map {
            e(child, parent)
        }
    }

    pub fn each_var_scope<E>(&self, mut e:E) where E: FnMut(&ast::NodeId, CodeExtent) {
        for (child, &parent) in self.var_map.iter() {
            e(child, parent)
        }
    }

    pub fn opt_destruction_extent(&self, n: ast::NodeId) -> Option<CodeExtent> {
        self.destruction_scopes.get(&n).cloned()
    }

    /// Records that `sub_fn` is defined within `sup_fn`. These ids
    /// should be the id of the block that is the fn body, which is
    /// also the root of the region hierarchy for that fn.
    fn record_fn_parent(&mut self, sub_fn: ast::NodeId, sup_fn: ast::NodeId) {
        debug!("record_fn_parent(sub_fn={:?}, sup_fn={:?})", sub_fn, sup_fn);
        assert!(sub_fn != sup_fn);
        let previous = self.fn_tree.insert(sub_fn, sup_fn);
        assert!(previous.is_none());
    }

    fn fn_is_enclosed_by(&self, mut sub_fn: ast::NodeId, sup_fn: ast::NodeId) -> bool {
        loop {
            if sub_fn == sup_fn { return true; }
            match self.fn_tree.get(&sub_fn) {
                Some(&s) => { sub_fn = s; }
                None => { return false; }
            }
        }
    }

    fn record_var_scope(&mut self, var: ast::NodeId, lifetime: CodeExtent) {
        debug!("record_var_scope(sub={:?}, sup={:?})", var, lifetime);
        assert!(var != lifetime.node_id());
        self.var_map.insert(var, lifetime);
    }

    fn record_rvalue_scope(&mut self, var: ast::NodeId, lifetime: CodeExtent) {
        debug!("record_rvalue_scope(sub={:?}, sup={:?})", var, lifetime);
        assert!(var != lifetime.node_id());
        self.rvalue_scopes.insert(var, lifetime);
    }

    pub fn opt_encl_scope(&self, id: CodeExtent) -> Option<CodeExtent> {
        //! Returns the narrowest scope that encloses `id`, if any.
        self.scope_map.get(&id).cloned()
    }

    #[allow(dead_code)] // used in cfg
    pub fn encl_scope(&self, id: CodeExtent) -> CodeExtent {
        //! Returns the narrowest scope that encloses `id`, if any.
        self.opt_encl_scope(id).unwrap()
    }

    /// Returns the lifetime of the local variable `var_id`
    pub fn var_scope(&self, var_id: ast::NodeId) -> CodeExtent {
        match self.var_map.get(&var_id) {
            Some(&r) => r,
            None => { bug!("no enclosing scope for id {:?}", var_id); }
        }
    }

    pub fn temporary_scope(&self, expr_id: ast::NodeId) -> Option<CodeExtent> {
        //! Returns the scope when temp created by expr_id will be cleaned up

        // check for a designated rvalue scope
        if let Some(&s) = self.rvalue_scopes.get(&expr_id) {
            debug!("temporary_scope({:?}) = {:?} [custom]", expr_id, s);
            return Some(s);
        }

        // else, locate the innermost terminating scope
        // if there's one. Static items, for instance, won't
        // have an enclosing scope, hence no scope will be
        // returned.
        let mut id = CodeExtent::Misc(expr_id);

        while let Some(&p) = self.scope_map.get(&id) {
            match p {
                CodeExtent::DestructionScope(..) => {
                    debug!("temporary_scope({:?}) = {:?} [enclosing]",
                           expr_id, id);
                    return Some(id);
                }
                _ => id = p
            }
        }

        debug!("temporary_scope({:?}) = None", expr_id);
        return None;
    }

    pub fn var_region(&self, id: ast::NodeId) -> ty::RegionKind {
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
        debug!("is_subscope_of({:?}, {:?})", subscope, superscope);
        while superscope != s {
            match self.opt_encl_scope(s) {
                None => {
                    debug!("is_subscope_of({:?}, {:?}, s={:?})=false",
                           subscope, superscope, s);
                    return false;
                }
                Some(scope) => s = scope
            }
        }

        debug!("is_subscope_of({:?}, {:?})=true",
               subscope, superscope);

        return true;
    }

    /// Finds the nearest common ancestor (if any) of two scopes.  That is, finds the smallest
    /// scope which is greater than or equal to both `scope_a` and `scope_b`.
    pub fn nearest_common_ancestor(&self,
                                   scope_a: CodeExtent,
                                   scope_b: CodeExtent)
                                   -> CodeExtent {
        if scope_a == scope_b { return scope_a; }

        /// [1] The initial values for `a_buf` and `b_buf` are not used.
        /// The `ancestors_of` function will return some prefix that
        /// is re-initialized with new values (or else fallback to a
        /// heap-allocated vector).
        let mut a_buf: [CodeExtent; 32] = [scope_a /* [1] */; 32];
        let mut a_vec: Vec<CodeExtent> = vec![];
        let mut b_buf: [CodeExtent; 32] = [scope_b /* [1] */; 32];
        let mut b_vec: Vec<CodeExtent> = vec![];
        let scope_map = &self.scope_map;
        let a_ancestors = ancestors_of(scope_map, scope_a, &mut a_buf, &mut a_vec);
        let b_ancestors = ancestors_of(scope_map, scope_b, &mut b_buf, &mut b_vec);
        let mut a_index = a_ancestors.len() - 1;
        let mut b_index = b_ancestors.len() - 1;

        // Here, [ab]_ancestors is a vector going from narrow to broad.
        // The end of each vector will be the item where the scope is
        // defined; if there are any common ancestors, then the tails of
        // the vector will be the same.  So basically we want to walk
        // backwards from the tail of each vector and find the first point
        // where they diverge.  If one vector is a suffix of the other,
        // then the corresponding scope is a superscope of the other.

        if a_ancestors[a_index] != b_ancestors[b_index] {
            // In this case, the two regions belong to completely
            // different functions.  Compare those fn for lexical
            // nesting. The reasoning behind this is subtle.  See the
            // "Modeling closures" section of the README in
            // infer::region_inference for more details.
            let a_root_scope = a_ancestors[a_index];
            let b_root_scope = a_ancestors[a_index];
            return match (a_root_scope, b_root_scope) {
                (CodeExtent::DestructionScope(a_root_id),
                 CodeExtent::DestructionScope(b_root_id)) => {
                    if self.fn_is_enclosed_by(a_root_id, b_root_id) {
                        // `a` is enclosed by `b`, hence `b` is the ancestor of everything in `a`
                        scope_b
                    } else if self.fn_is_enclosed_by(b_root_id, a_root_id) {
                        // `b` is enclosed by `a`, hence `a` is the ancestor of everything in `b`
                        scope_a
                    } else {
                        // neither fn encloses the other
                        bug!()
                    }
                }
                _ => {
                    // root ids are always Misc right now
                    bug!()
                }
            };
        }

        loop {
            // Loop invariant: a_ancestors[a_index] == b_ancestors[b_index]
            // for all indices between a_index and the end of the array
            if a_index == 0 { return scope_a; }
            if b_index == 0 { return scope_b; }
            a_index -= 1;
            b_index -= 1;
            if a_ancestors[a_index] != b_ancestors[b_index] {
                return a_ancestors[a_index + 1];
            }
        }

        fn ancestors_of<'a, 'tcx>(scope_map: &FxHashMap<CodeExtent, CodeExtent>,
                                  scope: CodeExtent,
                                  buf: &'a mut [CodeExtent; 32],
                                  vec: &'a mut Vec<CodeExtent>)
                                  -> &'a [CodeExtent] {
            // debug!("ancestors_of(scope={:?})", scope);
            let mut scope = scope;

            let mut i = 0;
            while i < 32 {
                buf[i] = scope;
                match scope_map.get(&scope) {
                    Some(&superscope) => scope = superscope,
                    _ => return &buf[..i+1]
                }
                i += 1;
            }

            *vec = Vec::with_capacity(64);
            vec.extend_from_slice(buf);
            loop {
                vec.push(scope);
                match scope_map.get(&scope) {
                    Some(&superscope) => scope = superscope,
                    _ => return &*vec
                }
            }
        }
    }

    /// Assuming that the provided region was defined within this `RegionMaps`,
    /// returns the outermost `CodeExtent` that the region outlives.
    pub fn early_free_extent<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                       br: &ty::EarlyBoundRegion)
                                       -> CodeExtent {
        let param_owner = tcx.parent_def_id(br.def_id).unwrap();

        let param_owner_id = tcx.hir.as_local_node_id(param_owner).unwrap();
        let body_id = tcx.hir.maybe_body_owned_by(param_owner_id).unwrap_or_else(|| {
            // The lifetime was defined on node that doesn't own a body,
            // which in practice can only mean a trait or an impl, that
            // is the parent of a method, and that is enforced below.
            assert_eq!(Some(param_owner_id), self.root_parent,
                       "free_extent: {:?} not recognized by the region maps for {:?}",
                       param_owner,
                       self.root_body.map(|body| tcx.hir.body_owner_def_id(body)));

            // The trait/impl lifetime is in scope for the method's body.
            self.root_body.unwrap()
        });

        CodeExtent::CallSiteScope(body_id)
    }

    /// Assuming that the provided region was defined within this `RegionMaps`,
    /// returns the outermost `CodeExtent` that the region outlives.
    pub fn free_extent<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, fr: &ty::FreeRegion)
                                 -> CodeExtent {
        let param_owner = match fr.bound_region {
            ty::BoundRegion::BrNamed(def_id, _) => {
                tcx.parent_def_id(def_id).unwrap()
            }
            _ => fr.scope
        };

        // Ensure that the named late-bound lifetimes were defined
        // on the same function that they ended up being freed in.
        assert_eq!(param_owner, fr.scope);

        let param_owner_id = tcx.hir.as_local_node_id(param_owner).unwrap();
        CodeExtent::CallSiteScope(tcx.hir.body_owned_by(param_owner_id))
    }
}

/// Records the lifetime of a local variable as `cx.var_parent`
fn record_var_lifetime(visitor: &mut RegionResolutionVisitor,
                       var_id: ast::NodeId,
                       _sp: Span) {
    match visitor.cx.var_parent {
        None => {
            // this can happen in extern fn declarations like
            //
            // extern fn isalnum(c: c_int) -> c_int
        }
        Some(parent_scope) =>
            visitor.region_maps.record_var_scope(var_id, parent_scope),
    }
}

fn resolve_block<'a, 'tcx>(visitor: &mut RegionResolutionVisitor<'a, 'tcx>, blk: &'tcx hir::Block) {
    debug!("resolve_block(blk.id={:?})", blk.id);

    let prev_cx = visitor.cx;

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

    visitor.enter_node_extent_with_dtor(blk.id);
    visitor.cx.var_parent = visitor.cx.parent;

    {
        // This block should be kept approximately in sync with
        // `intravisit::walk_block`. (We manually walk the block, rather
        // than call `walk_block`, in order to maintain precise
        // index information.)

        for (i, statement) in blk.stmts.iter().enumerate() {
            if let hir::StmtDecl(..) = statement.node {
                // Each StmtDecl introduces a subscope for bindings
                // introduced by the declaration; this subscope covers
                // a suffix of the block . Each subscope in a block
                // has the previous subscope in the block as a parent,
                // except for the first such subscope, which has the
                // block itself as a parent.
                visitor.enter_code_extent(
                    CodeExtent::Remainder(BlockRemainder {
                        block: blk.id,
                        first_statement_index: i as u32
                    })
                );
                visitor.cx.var_parent = visitor.cx.parent;
            }
            visitor.visit_stmt(statement)
        }
        walk_list!(visitor, visit_expr, &blk.expr);
    }

    visitor.cx = prev_cx;
}

fn resolve_arm<'a, 'tcx>(visitor: &mut RegionResolutionVisitor<'a, 'tcx>, arm: &'tcx hir::Arm) {
    visitor.terminating_scopes.insert(arm.body.id);

    if let Some(ref expr) = arm.guard {
        visitor.terminating_scopes.insert(expr.id);
    }

    intravisit::walk_arm(visitor, arm);
}

fn resolve_pat<'a, 'tcx>(visitor: &mut RegionResolutionVisitor<'a, 'tcx>, pat: &'tcx hir::Pat) {
    visitor.record_code_extent(CodeExtent::Misc(pat.id));

    // If this is a binding then record the lifetime of that binding.
    if let PatKind::Binding(..) = pat.node {
        record_var_lifetime(visitor, pat.id, pat.span);
    }

    intravisit::walk_pat(visitor, pat);
}

fn resolve_stmt<'a, 'tcx>(visitor: &mut RegionResolutionVisitor<'a, 'tcx>, stmt: &'tcx hir::Stmt) {
    let stmt_id = stmt.node.id();
    debug!("resolve_stmt(stmt.id={:?})", stmt_id);

    // Every statement will clean up the temporaries created during
    // execution of that statement. Therefore each statement has an
    // associated destruction scope that represents the extent of the
    // statement plus its destructors, and thus the extent for which
    // regions referenced by the destructors need to survive.
    visitor.terminating_scopes.insert(stmt_id);

    let prev_parent = visitor.cx.parent;
    visitor.enter_node_extent_with_dtor(stmt_id);

    intravisit::walk_stmt(visitor, stmt);

    visitor.cx.parent = prev_parent;
}

fn resolve_expr<'a, 'tcx>(visitor: &mut RegionResolutionVisitor<'a, 'tcx>, expr: &'tcx hir::Expr) {
    debug!("resolve_expr(expr.id={:?})", expr.id);

    let prev_cx = visitor.cx;
    visitor.enter_node_extent_with_dtor(expr.id);

    {
        let terminating_scopes = &mut visitor.terminating_scopes;
        let mut terminating = |id: ast::NodeId| {
            terminating_scopes.insert(id);
        };
        match expr.node {
            // Conditional or repeating scopes are always terminating
            // scopes, meaning that temporaries cannot outlive them.
            // This ensures fixed size stacks.

            hir::ExprBinary(codemap::Spanned { node: hir::BiAnd, .. }, _, ref r) |
            hir::ExprBinary(codemap::Spanned { node: hir::BiOr, .. }, _, ref r) => {
                // For shortcircuiting operators, mark the RHS as a terminating
                // scope since it only executes conditionally.
                terminating(r.id);
            }

            hir::ExprIf(ref expr, ref then, Some(ref otherwise)) => {
                terminating(expr.id);
                terminating(then.id);
                terminating(otherwise.id);
            }

            hir::ExprIf(ref expr, ref then, None) => {
                terminating(expr.id);
                terminating(then.id);
            }

            hir::ExprLoop(ref body, _, _) => {
                terminating(body.id);
            }

            hir::ExprWhile(ref expr, ref body, _) => {
                terminating(expr.id);
                terminating(body.id);
            }

            hir::ExprMatch(..) => {
                visitor.cx.var_parent = visitor.cx.parent;
            }

            hir::ExprAssignOp(..) | hir::ExprIndex(..) |
            hir::ExprUnary(..) | hir::ExprCall(..) | hir::ExprMethodCall(..) => {
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
                // call*.  See the section "Borrows in Calls" borrowck/README.md
                // for an extended explanation of why this distinction is
                // important.
                //
                // record_superlifetime(new_cx, expr.callee_id);
            }

            _ => {}
        }
    }

    match expr.node {
        // Manually recurse over closures, because they are the only
        // case of nested bodies that share the parent environment.
        hir::ExprClosure(.., body, _, _) => {
            let body = visitor.tcx.hir.body(body);
            visitor.visit_body(body);
        }

        _ => intravisit::walk_expr(visitor, expr)
    }

    visitor.cx = prev_cx;
}

fn resolve_local<'a, 'tcx>(visitor: &mut RegionResolutionVisitor<'a, 'tcx>,
                           local: &'tcx hir::Local) {
    debug!("resolve_local(local.id={:?},local.init={:?})",
           local.id,local.init.is_some());

    // For convenience in trans, associate with the local-id the var
    // scope that will be used for any bindings declared in this
    // pattern.
    let blk_scope = visitor.cx.var_parent;
    let blk_scope = blk_scope.expect("locals must be within a block");
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
    // Rule B. `let x = &foo().x`. The rvalue ``foo()` would have extended
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

    if let Some(ref expr) = local.init {
        record_rvalue_scope_if_borrow_expr(visitor, &expr, blk_scope);

        if is_binding_pat(&local.pat) {
            record_rvalue_scope(visitor, &expr, blk_scope);
        }
    }

    intravisit::walk_local(visitor, local);

    /// True if `pat` match the `P&` nonterminal:
    ///
    ///     P& = ref X
    ///        | StructName { ..., P&, ... }
    ///        | VariantName(..., P&, ...)
    ///        | [ ..., P&, ... ]
    ///        | ( ..., P&, ... )
    ///        | box P&
    fn is_binding_pat(pat: &hir::Pat) -> bool {
        match pat.node {
            PatKind::Binding(hir::BindByRef(_), ..) => true,

            PatKind::Struct(_, ref field_pats, _) => {
                field_pats.iter().any(|fp| is_binding_pat(&fp.node.pat))
            }

            PatKind::Slice(ref pats1, ref pats2, ref pats3) => {
                pats1.iter().any(|p| is_binding_pat(&p)) ||
                pats2.iter().any(|p| is_binding_pat(&p)) ||
                pats3.iter().any(|p| is_binding_pat(&p))
            }

            PatKind::TupleStruct(_, ref subpats, _) |
            PatKind::Tuple(ref subpats, _) => {
                subpats.iter().any(|p| is_binding_pat(&p))
            }

            PatKind::Box(ref subpat) => {
                is_binding_pat(&subpat)
            }

            _ => false,
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
    fn record_rvalue_scope_if_borrow_expr<'a, 'tcx>(
        visitor: &mut RegionResolutionVisitor<'a, 'tcx>,
        expr: &hir::Expr,
        blk_id: CodeExtent)
    {
        match expr.node {
            hir::ExprAddrOf(_, ref subexpr) => {
                record_rvalue_scope_if_borrow_expr(visitor, &subexpr, blk_id);
                record_rvalue_scope(visitor, &subexpr, blk_id);
            }
            hir::ExprStruct(_, ref fields, _) => {
                for field in fields {
                    record_rvalue_scope_if_borrow_expr(
                        visitor, &field.expr, blk_id);
                }
            }
            hir::ExprArray(ref subexprs) |
            hir::ExprTup(ref subexprs) => {
                for subexpr in subexprs {
                    record_rvalue_scope_if_borrow_expr(
                        visitor, &subexpr, blk_id);
                }
            }
            hir::ExprCast(ref subexpr, _) => {
                record_rvalue_scope_if_borrow_expr(visitor, &subexpr, blk_id)
            }
            hir::ExprBlock(ref block) => {
                if let Some(ref subexpr) = block.expr {
                    record_rvalue_scope_if_borrow_expr(
                        visitor, &subexpr, blk_id);
                }
            }
            _ => {}
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
    fn record_rvalue_scope<'a, 'tcx>(visitor: &mut RegionResolutionVisitor<'a, 'tcx>,
                                     expr: &hir::Expr,
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
                hir::ExprAddrOf(_, ref subexpr) |
                hir::ExprUnary(hir::UnDeref, ref subexpr) |
                hir::ExprField(ref subexpr, _) |
                hir::ExprTupField(ref subexpr, _) |
                hir::ExprIndex(ref subexpr, _) => {
                    expr = &subexpr;
                }
                _ => {
                    return;
                }
            }
        }
    }
}

impl<'a, 'tcx> RegionResolutionVisitor<'a, 'tcx> {
    /// Records the current parent (if any) as the parent of `child_scope`.
    fn record_code_extent(&mut self, child_scope: CodeExtent) {
        let parent = self.cx.parent;
        self.region_maps.record_code_extent(child_scope, parent);
    }

    /// Records the current parent (if any) as the parent of `child_scope`,
    /// and sets `child_scope` as the new current parent.
    fn enter_code_extent(&mut self, child_scope: CodeExtent) {
        self.record_code_extent(child_scope);
        self.cx.parent = Some(child_scope);
    }

    fn enter_node_extent_with_dtor(&mut self, id: ast::NodeId) {
        // If node was previously marked as a terminating scope during the
        // recursive visit of its parent node in the AST, then we need to
        // account for the destruction scope representing the extent of
        // the destructors that run immediately after it completes.
        if self.terminating_scopes.contains(&id) {
            self.enter_code_extent(CodeExtent::DestructionScope(id));
        }
        self.enter_code_extent(CodeExtent::Misc(id));
    }
}

impl<'a, 'tcx> Visitor<'tcx> for RegionResolutionVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_block(&mut self, b: &'tcx Block) {
        resolve_block(self, b);
    }

    fn visit_body(&mut self, body: &'tcx hir::Body) {
        let body_id = body.id();
        let owner_id = self.tcx.hir.body_owner(body_id);

        debug!("visit_body(id={:?}, span={:?}, body.id={:?}, cx.parent={:?})",
               owner_id,
               self.tcx.sess.codemap().span_to_string(body.value.span),
               body_id,
               self.cx.parent);

        let outer_cx = self.cx;
        let outer_ts = mem::replace(&mut self.terminating_scopes, NodeSet());

        // Only functions have an outer terminating (drop) scope,
        // while temporaries in constant initializers are 'static.
        if let MirSource::Fn(_) = MirSource::from_node(self.tcx, owner_id) {
            self.terminating_scopes.insert(body_id.node_id);
        }

        if let Some(root_id) = self.cx.root_id {
            self.region_maps.record_fn_parent(body_id.node_id, root_id);
        }
        self.cx.root_id = Some(body_id.node_id);

        self.enter_code_extent(CodeExtent::CallSiteScope(body_id));
        self.enter_code_extent(CodeExtent::ParameterScope(body_id));

        // The arguments and `self` are parented to the fn.
        self.cx.var_parent = self.cx.parent.take();
        for argument in &body.arguments {
            self.visit_pat(&argument.pat);
        }
        if let Some(ref impl_arg) = body.impl_arg {
            record_var_lifetime(self, impl_arg.id, impl_arg.span);
        }
        
        // The body of the every fn is a root scope.
        self.cx.parent = self.cx.var_parent;
        self.visit_expr(&body.value);

        // Restore context we had at the start.
        self.cx = outer_cx;
        self.terminating_scopes = outer_ts;
    }

    fn visit_arm(&mut self, a: &'tcx Arm) {
        resolve_arm(self, a);
    }
    fn visit_pat(&mut self, p: &'tcx Pat) {
        resolve_pat(self, p);
    }
    fn visit_stmt(&mut self, s: &'tcx Stmt) {
        resolve_stmt(self, s);
    }
    fn visit_expr(&mut self, ex: &'tcx Expr) {
        resolve_expr(self, ex);
    }
    fn visit_local(&mut self, l: &'tcx Local) {
        resolve_local(self, l);
    }
}

fn region_maps<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId)
    -> Rc<RegionMaps>
{
    let closure_base_def_id = tcx.closure_base_def_id(def_id);
    if closure_base_def_id != def_id {
        return tcx.region_maps(closure_base_def_id);
    }

    let id = tcx.hir.as_local_node_id(def_id).unwrap();
    let maps = if let Some(body) = tcx.hir.maybe_body_owned_by(id) {
        let mut visitor = RegionResolutionVisitor {
            tcx,
            region_maps: RegionMaps::new(),
            cx: Context {
                root_id: None,
                parent: None,
                var_parent: None,
            },
            terminating_scopes: NodeSet(),
        };

        visitor.region_maps.root_body = Some(body);

        // If the item is an associated const or a method,
        // record its impl/trait parent, as it can also have
        // lifetime parameters free in this body.
        match tcx.hir.get(id) {
            hir::map::NodeImplItem(_) |
            hir::map::NodeTraitItem(_) => {
                visitor.region_maps.root_parent = Some(tcx.hir.get_parent(id));
            }
            _ => {}
        }

        visitor.visit_body(tcx.hir.body(body));

        visitor.region_maps
    } else {
        RegionMaps::new()
    };

    Rc::new(maps)
}

struct YieldFinder(bool);

impl<'tcx> Visitor<'tcx> for YieldFinder {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_body(&mut self, _body: &'tcx hir::Body) {
        // Closures don't execute
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        if let hir::ExprSuspend(..) = expr.node {
            self.0 = true;
        }
        
        intravisit::walk_expr(self, expr);
    }
}

pub fn extent_has_yield<'a, 'gcx: 'a+'tcx, 'tcx: 'a>(tcx: TyCtxt<'a, 'gcx, 'tcx>, extent: CodeExtent) -> bool {
    let mut finder = YieldFinder(false);

    match extent {
        CodeExtent::DestructionScope(node_id) |
        CodeExtent::Misc(node_id) => {
            match tcx.hir.get(node_id) {
                Node::NodeItem(_) |
                Node::NodeTraitItem(_) |
                Node::NodeImplItem(_) => {
                    let body = tcx.hir.body(tcx.hir.body_owned_by(node_id));
                    intravisit::walk_body(&mut finder, body);
                }
                Node::NodeExpr(expr) => intravisit::walk_expr(&mut finder, expr),
                Node::NodeStmt(stmt) => intravisit::walk_stmt(&mut finder, stmt),
                Node::NodeBlock(block) => intravisit::walk_block(&mut finder, block), 
                _ => bug!(),
            }
        }

        CodeExtent::CallSiteScope(body_id) |
        CodeExtent::ParameterScope(body_id) => {
            intravisit::walk_body(&mut finder, tcx.hir.body(body_id))
        }

        CodeExtent::Remainder(r) => {
            if let Node::NodeBlock(block) = tcx.hir.get(r.block) {
                for stmt in &block.stmts[(r.first_statement_index as usize + 1)..] {
                    intravisit::walk_stmt(&mut finder, stmt);
                }
                block.expr.as_ref().map(|e| intravisit::walk_expr(&mut finder, e));
            } else {
                bug!()
            }
        }
    }

    finder.0
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        region_maps,
        ..*providers
    };
}
