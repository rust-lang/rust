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

use front::map as ast_map;
use session::Session;
use util::nodemap::{FnvHashMap, NodeMap, NodeSet};
use middle::cstore::InlinedItem;
use middle::ty::{self, Ty};

use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::fmt;
use std::mem;
use syntax::codemap::{self, Span};
use syntax::ast::{self, NodeId};

use rustc_front::hir;
use rustc_front::intravisit::{self, Visitor, FnKind};
use rustc_front::hir::{Block, Item, FnDecl, Arm, Pat, Stmt, Expr, Local};
use rustc_front::util::stmt_id;

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, RustcEncodable,
           RustcDecodable, Copy)]
pub struct CodeExtent(u32);

impl fmt::Debug for CodeExtent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "CodeExtent({:?}", self.0));

        try!(ty::tls::with_opt(|opt_tcx| {
            if let Some(tcx) = opt_tcx {
                let data = tcx.region_maps.code_extents.borrow()[self.0 as usize];
                try!(write!(f, "/{:?}", data));
            }
            Ok(())
        }));

        write!(f, ")")
    }
}

/// The root of everything. I should be using NonZero or profiling
/// instead of this (probably).
pub const ROOT_CODE_EXTENT : CodeExtent = CodeExtent(0);
/// A placeholder used in trans to stand for real code extents
pub const DUMMY_CODE_EXTENT : CodeExtent = CodeExtent(1);

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
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Debug, Copy)]
pub enum CodeExtentData {
    Misc(ast::NodeId),

    // extent of the call-site for a function or closure (outlives
    // the parameters as well as the body).
    CallSiteScope { fn_id: ast::NodeId, body_id: ast::NodeId },

    // extent of parameters passed to a function or closure (they
    // outlive its body)
    ParameterScope { fn_id: ast::NodeId, body_id: ast::NodeId },

    // extent of destructors for temporaries of node-id
    DestructionScope(ast::NodeId),

    // extent of code following a `let id = expr;` binding in a block
    Remainder(BlockRemainder)
}

/// extent of call-site for a function/method.
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, RustcEncodable,
           RustcDecodable, Debug, Copy)]
pub struct CallSiteScopeData {
    pub fn_id: ast::NodeId, pub body_id: ast::NodeId,
}

impl CallSiteScopeData {
    pub fn to_code_extent(&self, region_maps: &RegionMaps) -> CodeExtent {
        region_maps.lookup_code_extent(
            match *self {
                CallSiteScopeData { fn_id, body_id } =>
                    CodeExtentData::CallSiteScope { fn_id: fn_id, body_id: body_id },
            })
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
///   includes EXPR_1 as well, then do not use `CodeExtentData::Remainder`,
///   but instead another `CodeExtent` that encompasses the whole block,
///   e.g. `CodeExtentData::Misc`.
///
/// * the subscope with `first_statement_index == 1` is scope of `c`,
///   and thus does not include EXPR_2, but covers the `...`.
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, RustcEncodable,
         RustcDecodable, Debug, Copy)]
pub struct BlockRemainder {
    pub block: ast::NodeId,
    pub first_statement_index: u32,
}

impl CodeExtentData {
    /// Returns a node id associated with this scope.
    ///
    /// NB: likely to be replaced as API is refined; e.g. pnkfelix
    /// anticipates `fn entry_node_id` and `fn each_exit_node_id`.
    pub fn node_id(&self) -> ast::NodeId {
        match *self {
            CodeExtentData::Misc(node_id) => node_id,

            // These cases all return rough approximations to the
            // precise extent denoted by `self`.
            CodeExtentData::Remainder(br) => br.block,
            CodeExtentData::DestructionScope(node_id) => node_id,
            CodeExtentData::CallSiteScope { fn_id: _, body_id } |
            CodeExtentData::ParameterScope { fn_id: _, body_id } => body_id,
        }
    }
}

impl CodeExtent {
    #[inline]
    fn into_option(self) -> Option<CodeExtent> {
        if self == ROOT_CODE_EXTENT {
            None
        } else {
            Some(self)
        }
    }
    pub fn node_id(&self, region_maps: &RegionMaps) -> ast::NodeId {
        region_maps.code_extent_data(*self).node_id()
    }

    /// Returns the span of this CodeExtent.  Note that in general the
    /// returned span may not correspond to the span of any node id in
    /// the AST.
    pub fn span(&self, region_maps: &RegionMaps, ast_map: &ast_map::Map) -> Option<Span> {
        match ast_map.find(self.node_id(region_maps)) {
            Some(ast_map::NodeBlock(ref blk)) => {
                match region_maps.code_extent_data(*self) {
                    CodeExtentData::CallSiteScope { .. } |
                    CodeExtentData::ParameterScope { .. } |
                    CodeExtentData::Misc(_) |
                    CodeExtentData::DestructionScope(_) => Some(blk.span),

                    CodeExtentData::Remainder(r) => {
                        assert_eq!(r.block, blk.id);
                        // Want span for extent starting after the
                        // indexed statement and ending at end of
                        // `blk`; reuse span of `blk` and shift `lo`
                        // forward to end of indexed statement.
                        //
                        // (This is the special case aluded to in the
                        // doc-comment for this method)
                        let stmt_span = blk.stmts[r.first_statement_index as usize].span;
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
pub struct RegionMaps {
    code_extents: RefCell<Vec<CodeExtentData>>,
    code_extent_interner: RefCell<FnvHashMap<CodeExtentData, CodeExtent>>,
    /// `scope_map` maps from a scope id to the enclosing scope id;
    /// this is usually corresponding to the lexical nesting, though
    /// in the case of closures the parent scope is the innermost
    /// conditional expression or repeating block. (Note that the
    /// enclosing scope id for the block associated with a closure is
    /// the closure itself.)
    scope_map: RefCell<Vec<CodeExtent>>,

    /// `var_map` maps from a variable or binding id to the block in
    /// which that variable is declared.
    var_map: RefCell<NodeMap<CodeExtent>>,

    /// `rvalue_scopes` includes entries for those expressions whose cleanup scope is
    /// larger than the default. The map goes from the expression id
    /// to the cleanup scope id. For rvalues not present in this
    /// table, the appropriate cleanup scope is the innermost
    /// enclosing statement, conditional expression, or repeating
    /// block (see `terminating_scopes`).
    rvalue_scopes: RefCell<NodeMap<CodeExtent>>,

    /// Encodes the hierarchy of fn bodies. Every fn body (including
    /// closures) forms its own distinct region hierarchy, rooted in
    /// the block that is the fn body. This map points from the id of
    /// that root block to the id of the root block for the enclosing
    /// fn, if any. Thus the map structures the fn bodies into a
    /// hierarchy based on their lexical mapping. This is used to
    /// handle the relationships between regions in a fn and in a
    /// closure defined by that fn. See the "Modeling closures"
    /// section of the README in middle::infer::region_inference for
    /// more details.
    fn_tree: RefCell<NodeMap<ast::NodeId>>,
}

#[derive(Debug, Copy, Clone)]
pub struct Context {
    /// the root of the current region tree. This is typically the id
    /// of the innermost fn body. Each fn forms its own disjoint tree
    /// in the region hierarchy. These fn bodies are themselves
    /// arranged into a tree. See the "Modeling closures" section of
    /// the README in middle::infer::region_inference for more
    /// details.
    root_id: Option<ast::NodeId>,

    /// the scope that contains any new variables declared
    var_parent: CodeExtent,

    /// region parent of expressions etc
    parent: CodeExtent
}

struct RegionResolutionVisitor<'a> {
    sess: &'a Session,

    // Generated maps:
    region_maps: &'a RegionMaps,

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
    terminating_scopes: NodeSet
}


impl RegionMaps {
    /// create a bogus code extent for the regions in astencode types. Nobody
    /// really cares about the contents of these.
    pub fn bogus_code_extent(&self, e: CodeExtentData) -> CodeExtent {
        self.intern_code_extent(e, DUMMY_CODE_EXTENT)
    }
    pub fn lookup_code_extent(&self, e: CodeExtentData) -> CodeExtent {
        match self.code_extent_interner.borrow().get(&e) {
            Some(&d) => d,
            None => panic!("unknown code extent {:?}", e)
        }
    }
    pub fn node_extent(&self, n: ast::NodeId) -> CodeExtent {
        self.lookup_code_extent(CodeExtentData::Misc(n))
    }
    // Returns the code extent for an item - the destruction scope.
    pub fn item_extent(&self, n: ast::NodeId) -> CodeExtent {
        self.lookup_code_extent(CodeExtentData::DestructionScope(n))
    }
    pub fn call_site_extent(&self, fn_id: ast::NodeId, body_id: ast::NodeId) -> CodeExtent {
        assert!(fn_id != body_id);
        self.lookup_code_extent(CodeExtentData::CallSiteScope { fn_id: fn_id, body_id: body_id })
    }
    pub fn opt_destruction_extent(&self, n: ast::NodeId) -> Option<CodeExtent> {
        self.code_extent_interner.borrow().get(&CodeExtentData::DestructionScope(n)).cloned()
    }
    pub fn intern_code_extent(&self,
                              e: CodeExtentData,
                              parent: CodeExtent) -> CodeExtent {
        match self.code_extent_interner.borrow_mut().entry(e) {
            Entry::Occupied(o) => {
                // this can happen when the bogus code extents from tydecode
                // have (bogus) NodeId-s that overlap items created during
                // inlining.
                // We probably shouldn't be creating bogus code extents
                // though.
                let idx = *o.get();
                if parent == DUMMY_CODE_EXTENT {
                    info!("CodeExtent({}) = {:?} [parent={}] BOGUS!",
                          idx.0, e, parent.0);
                } else {
                    assert_eq!(self.scope_map.borrow()[idx.0 as usize],
                               DUMMY_CODE_EXTENT);
                    info!("CodeExtent({}) = {:?} [parent={}] RECLAIMED!",
                          idx.0, e, parent.0);
                    self.scope_map.borrow_mut()[idx.0 as usize] = parent;
                }
                idx
            }
            Entry::Vacant(v) => {
                if self.code_extents.borrow().len() > 0xffffffffusize {
                    unreachable!() // should pass a sess,
                                   // but this isn't the only place
                }
                let idx = CodeExtent(self.code_extents.borrow().len() as u32);
                info!("CodeExtent({}) = {:?} [parent={}]", idx.0, e, parent.0);
                self.code_extents.borrow_mut().push(e);
                self.scope_map.borrow_mut().push(parent);
                *v.insert(idx)
            }
        }
    }
    pub fn intern_node(&self,
                       n: ast::NodeId,
                       parent: CodeExtent) -> CodeExtent {
        self.intern_code_extent(CodeExtentData::Misc(n), parent)
    }
    pub fn code_extent_data(&self, e: CodeExtent) -> CodeExtentData {
        self.code_extents.borrow()[e.0 as usize]
    }
    pub fn each_encl_scope<E>(&self, mut e:E) where E: FnMut(&CodeExtent, &CodeExtent) {
        for child_id in 1..self.code_extents.borrow().len() {
            let child = CodeExtent(child_id as u32);
            if let Some(parent) = self.opt_encl_scope(child) {
                e(&child, &parent)
            }
        }
    }
    pub fn each_var_scope<E>(&self, mut e:E) where E: FnMut(&ast::NodeId, &CodeExtent) {
        for (child, parent) in self.var_map.borrow().iter() {
            e(child, parent)
        }
    }
    pub fn each_rvalue_scope<E>(&self, mut e:E) where E: FnMut(&ast::NodeId, &CodeExtent) {
        for (child, parent) in self.rvalue_scopes.borrow().iter() {
            e(child, parent)
        }
    }
    /// Records that `sub_fn` is defined within `sup_fn`. These ids
    /// should be the id of the block that is the fn body, which is
    /// also the root of the region hierarchy for that fn.
    fn record_fn_parent(&self, sub_fn: ast::NodeId, sup_fn: ast::NodeId) {
        debug!("record_fn_parent(sub_fn={:?}, sup_fn={:?})", sub_fn, sup_fn);
        assert!(sub_fn != sup_fn);
        let previous = self.fn_tree.borrow_mut().insert(sub_fn, sup_fn);
        assert!(previous.is_none());
    }

    fn fn_is_enclosed_by(&self, mut sub_fn: ast::NodeId, sup_fn: ast::NodeId) -> bool {
        let fn_tree = self.fn_tree.borrow();
        loop {
            if sub_fn == sup_fn { return true; }
            match fn_tree.get(&sub_fn) {
                Some(&s) => { sub_fn = s; }
                None => { return false; }
            }
        }
    }

    fn record_var_scope(&self, var: ast::NodeId, lifetime: CodeExtent) {
        debug!("record_var_scope(sub={:?}, sup={:?})", var, lifetime);
        assert!(var != lifetime.node_id(self));
        self.var_map.borrow_mut().insert(var, lifetime);
    }

    fn record_rvalue_scope(&self, var: ast::NodeId, lifetime: CodeExtent) {
        debug!("record_rvalue_scope(sub={:?}, sup={:?})", var, lifetime);
        assert!(var != lifetime.node_id(self));
        self.rvalue_scopes.borrow_mut().insert(var, lifetime);
    }

    pub fn opt_encl_scope(&self, id: CodeExtent) -> Option<CodeExtent> {
        //! Returns the narrowest scope that encloses `id`, if any.
        self.scope_map.borrow()[id.0 as usize].into_option()
    }

    #[allow(dead_code)] // used in middle::cfg
    pub fn encl_scope(&self, id: CodeExtent) -> CodeExtent {
        //! Returns the narrowest scope that encloses `id`, if any.
        self.opt_encl_scope(id).unwrap()
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

        let scope_map : &[CodeExtent] = &self.scope_map.borrow();
        let code_extents: &[CodeExtentData] = &self.code_extents.borrow();

        // else, locate the innermost terminating scope
        // if there's one. Static items, for instance, won't
        // have an enclosing scope, hence no scope will be
        // returned.
        let expr_extent = self.node_extent(expr_id);
        // For some reason, the expr's scope itself is skipped here.
        let mut id = match scope_map[expr_extent.0 as usize].into_option() {
            Some(i) => i,
            _ => return None
        };

        while let Some(p) = scope_map[id.0 as usize].into_option() {
            match code_extents[p.0 as usize] {
                CodeExtentData::DestructionScope(..) => {
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

        let mut a_buf: [CodeExtent; 32] = [ROOT_CODE_EXTENT; 32];
        let mut a_vec: Vec<CodeExtent> = vec![];
        let mut b_buf: [CodeExtent; 32] = [ROOT_CODE_EXTENT; 32];
        let mut b_vec: Vec<CodeExtent> = vec![];
        let scope_map : &[CodeExtent] = &self.scope_map.borrow();
        let a_ancestors = ancestors_of(scope_map,
                                       scope_a, &mut a_buf, &mut a_vec);
        let b_ancestors = ancestors_of(scope_map,
                                       scope_b, &mut b_buf, &mut b_vec);
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
            // middle::infer::region_inference for more details.
            let a_root_scope = self.code_extent_data(a_ancestors[a_index]);
            let b_root_scope = self.code_extent_data(a_ancestors[a_index]);
            return match (a_root_scope, b_root_scope) {
                (CodeExtentData::DestructionScope(a_root_id),
                 CodeExtentData::DestructionScope(b_root_id)) => {
                    if self.fn_is_enclosed_by(a_root_id, b_root_id) {
                        // `a` is enclosed by `b`, hence `b` is the ancestor of everything in `a`
                        scope_b
                    } else if self.fn_is_enclosed_by(b_root_id, a_root_id) {
                        // `b` is enclosed by `a`, hence `a` is the ancestor of everything in `b`
                        scope_a
                    } else {
                        // neither fn encloses the other
                        unreachable!()
                    }
                }
                _ => {
                    // root ids are always Misc right now
                    unreachable!()
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

        fn ancestors_of<'a>(scope_map: &[CodeExtent],
                            scope: CodeExtent,
                            buf: &'a mut [CodeExtent; 32],
                            vec: &'a mut Vec<CodeExtent>) -> &'a [CodeExtent] {
            // debug!("ancestors_of(scope={:?})", scope);
            let mut scope = scope;

            let mut i = 0;
            while i < 32 {
                buf[i] = scope;
                match scope_map[scope.0 as usize].into_option() {
                    Some(superscope) => scope = superscope,
                    _ => return &buf[..i+1]
                }
                i += 1;
            }

            *vec = Vec::with_capacity(64);
            vec.extend_from_slice(buf);
            loop {
                vec.push(scope);
                match scope_map[scope.0 as usize].into_option() {
                    Some(superscope) => scope = superscope,
                    _ => return &*vec
                }
            }
        }
    }
}

/// Records the lifetime of a local variable as `cx.var_parent`
fn record_var_lifetime(visitor: &mut RegionResolutionVisitor,
                       var_id: ast::NodeId,
                       _sp: Span) {
    match visitor.cx.var_parent {
        ROOT_CODE_EXTENT => {
            // this can happen in extern fn declarations like
            //
            // extern fn isalnum(c: c_int) -> c_int
        }
        parent_scope =>
            visitor.region_maps.record_var_scope(var_id, parent_scope),
    }
}

fn resolve_block(visitor: &mut RegionResolutionVisitor, blk: &hir::Block) {
    debug!("resolve_block(blk.id={:?})", blk.id);

    let prev_cx = visitor.cx;
    let block_extent = visitor.new_node_extent_with_dtor(blk.id);

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
        root_id: prev_cx.root_id,
        var_parent: block_extent,
        parent: block_extent,
    };

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
                let stmt_extent = visitor.new_code_extent(
                    CodeExtentData::Remainder(BlockRemainder {
                        block: blk.id,
                        first_statement_index: i as u32
                    })
                );
                visitor.cx = Context {
                    root_id: prev_cx.root_id,
                    var_parent: stmt_extent,
                    parent: stmt_extent,
                };
            }
            visitor.visit_stmt(statement)
        }
        walk_list!(visitor, visit_expr, &blk.expr);
    }

    visitor.cx = prev_cx;
}

fn resolve_arm(visitor: &mut RegionResolutionVisitor, arm: &hir::Arm) {
    visitor.terminating_scopes.insert(arm.body.id);

    if let Some(ref expr) = arm.guard {
        visitor.terminating_scopes.insert(expr.id);
    }

    intravisit::walk_arm(visitor, arm);
}

fn resolve_pat(visitor: &mut RegionResolutionVisitor, pat: &hir::Pat) {
    visitor.new_node_extent(pat.id);

    // If this is a binding (or maybe a binding, I'm too lazy to check
    // the def map) then record the lifetime of that binding.
    match pat.node {
        hir::PatIdent(..) => {
            record_var_lifetime(visitor, pat.id, pat.span);
        }
        _ => { }
    }

    intravisit::walk_pat(visitor, pat);
}

fn resolve_stmt(visitor: &mut RegionResolutionVisitor, stmt: &hir::Stmt) {
    let stmt_id = stmt_id(stmt);
    debug!("resolve_stmt(stmt.id={:?})", stmt_id);

    // Every statement will clean up the temporaries created during
    // execution of that statement. Therefore each statement has an
    // associated destruction scope that represents the extent of the
    // statement plus its destructors, and thus the extent for which
    // regions referenced by the destructors need to survive.
    visitor.terminating_scopes.insert(stmt_id);
    let stmt_extent = visitor.new_node_extent_with_dtor(stmt_id);

    let prev_parent = visitor.cx.parent;
    visitor.cx.parent = stmt_extent;
    intravisit::walk_stmt(visitor, stmt);
    visitor.cx.parent = prev_parent;
}

fn resolve_expr(visitor: &mut RegionResolutionVisitor, expr: &hir::Expr) {
    debug!("resolve_expr(expr.id={:?})", expr.id);

    let expr_extent = visitor.new_node_extent_with_dtor(expr.id);
    let prev_cx = visitor.cx;
    visitor.cx.parent = expr_extent;

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

            hir::ExprIf(_, ref then, Some(ref otherwise)) => {
                terminating(then.id);
                terminating(otherwise.id);
            }

            hir::ExprIf(ref expr, ref then, None) => {
                terminating(expr.id);
                terminating(then.id);
            }

            hir::ExprLoop(ref body, _) => {
                terminating(body.id);
            }

            hir::ExprWhile(ref expr, ref body, _) => {
                terminating(expr.id);
                terminating(body.id);
            }

            hir::ExprMatch(..) => {
                visitor.cx.var_parent = expr_extent;
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

    intravisit::walk_expr(visitor, expr);
    visitor.cx = prev_cx;
}

fn resolve_local(visitor: &mut RegionResolutionVisitor, local: &hir::Local) {
    debug!("resolve_local(local.id={:?},local.init={:?})",
           local.id,local.init.is_some());

    // For convenience in trans, associate with the local-id the var
    // scope that will be used for any bindings declared in this
    // pattern.
    let blk_scope = visitor.cx.var_parent;
    assert!(blk_scope != ROOT_CODE_EXTENT); // locals must be within a block
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
            hir::PatIdent(hir::BindByRef(_), _, _) => true,

            hir::PatStruct(_, ref field_pats, _) => {
                field_pats.iter().any(|fp| is_binding_pat(&*fp.node.pat))
            }

            hir::PatVec(ref pats1, ref pats2, ref pats3) => {
                pats1.iter().any(|p| is_binding_pat(&**p)) ||
                pats2.iter().any(|p| is_binding_pat(&**p)) ||
                pats3.iter().any(|p| is_binding_pat(&**p))
            }

            hir::PatEnum(_, Some(ref subpats)) |
            hir::PatTup(ref subpats) => {
                subpats.iter().any(|p| is_binding_pat(&**p))
            }

            hir::PatBox(ref subpat) => {
                is_binding_pat(&**subpat)
            }

            _ => false,
        }
    }

    /// True if `ty` is a borrowed pointer type like `&int` or `&[...]`.
    fn is_borrowed_ty(ty: &hir::Ty) -> bool {
        match ty.node {
            hir::TyRptr(..) => true,
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
                                          expr: &hir::Expr,
                                          blk_id: CodeExtent) {
        match expr.node {
            hir::ExprAddrOf(_, ref subexpr) => {
                record_rvalue_scope_if_borrow_expr(visitor, &**subexpr, blk_id);
                record_rvalue_scope(visitor, &**subexpr, blk_id);
            }
            hir::ExprStruct(_, ref fields, _) => {
                for field in fields {
                    record_rvalue_scope_if_borrow_expr(
                        visitor, &*field.expr, blk_id);
                }
            }
            hir::ExprVec(ref subexprs) |
            hir::ExprTup(ref subexprs) => {
                for subexpr in subexprs {
                    record_rvalue_scope_if_borrow_expr(
                        visitor, &**subexpr, blk_id);
                }
            }
            hir::ExprCast(ref subexpr, _) => {
                record_rvalue_scope_if_borrow_expr(visitor, &**subexpr, blk_id)
            }
            hir::ExprBlock(ref block) => {
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
                               expr: &'a hir::Expr,
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
                    expr = &**subexpr;
                }
                _ => {
                    return;
                }
            }
        }
    }
}

fn resolve_item(visitor: &mut RegionResolutionVisitor, item: &hir::Item) {
    // Items create a new outer block scope as far as we're concerned.
    let prev_cx = visitor.cx;
    let prev_ts = mem::replace(&mut visitor.terminating_scopes, NodeSet());
    visitor.cx = Context {
        root_id: None,
        var_parent: ROOT_CODE_EXTENT,
        parent: ROOT_CODE_EXTENT
    };
    intravisit::walk_item(visitor, item);
    visitor.create_item_scope_if_needed(item.id);
    visitor.cx = prev_cx;
    visitor.terminating_scopes = prev_ts;
}

fn resolve_fn(visitor: &mut RegionResolutionVisitor,
              kind: FnKind,
              decl: &hir::FnDecl,
              body: &hir::Block,
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

    visitor.cx.parent = visitor.new_code_extent(
        CodeExtentData::CallSiteScope { fn_id: id, body_id: body.id });

    let fn_decl_scope = visitor.new_code_extent(
        CodeExtentData::ParameterScope { fn_id: id, body_id: body.id });

    if let Some(root_id) = visitor.cx.root_id {
        visitor.region_maps.record_fn_parent(body.id, root_id);
    }

    let outer_cx = visitor.cx;
    let outer_ts = mem::replace(&mut visitor.terminating_scopes, NodeSet());
    visitor.terminating_scopes.insert(body.id);

    // The arguments and `self` are parented to the fn.
    visitor.cx = Context {
        root_id: Some(body.id),
        parent: ROOT_CODE_EXTENT,
        var_parent: fn_decl_scope,
    };

    intravisit::walk_fn_decl(visitor, decl);
    intravisit::walk_fn_kind(visitor, kind);

    // The body of the every fn is a root scope.
    visitor.cx = Context {
        root_id: Some(body.id),
        parent: fn_decl_scope,
        var_parent: fn_decl_scope
    };
    visitor.visit_block(body);

    // Restore context we had at the start.
    visitor.cx = outer_cx;
    visitor.terminating_scopes = outer_ts;
}

impl<'a> RegionResolutionVisitor<'a> {
    /// Records the current parent (if any) as the parent of `child_scope`.
    fn new_code_extent(&mut self, child_scope: CodeExtentData) -> CodeExtent {
        self.region_maps.intern_code_extent(child_scope, self.cx.parent)
    }

    fn new_node_extent(&mut self, child_scope: ast::NodeId) -> CodeExtent {
        self.new_code_extent(CodeExtentData::Misc(child_scope))
    }

    fn new_node_extent_with_dtor(&mut self, id: ast::NodeId) -> CodeExtent {
        // If node was previously marked as a terminating scope during the
        // recursive visit of its parent node in the AST, then we need to
        // account for the destruction scope representing the extent of
        // the destructors that run immediately after it completes.
        if self.terminating_scopes.contains(&id) {
            let ds = self.new_code_extent(
                CodeExtentData::DestructionScope(id));
            self.region_maps.intern_node(id, ds)
        } else {
            self.new_node_extent(id)
        }
    }

    fn create_item_scope_if_needed(&mut self, id: ast::NodeId) {
        // create a region for the destruction scope - this is needed
        // for constructing parameter environments based on the item.
        // functions put their destruction scopes *inside* their parameter
        // scopes.
        let scope = CodeExtentData::DestructionScope(id);
        if !self.region_maps.code_extent_interner.borrow().contains_key(&scope) {
            self.region_maps.intern_code_extent(scope, ROOT_CODE_EXTENT);
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

    fn visit_impl_item(&mut self, ii: &hir::ImplItem) {
        intravisit::walk_impl_item(self, ii);
        self.create_item_scope_if_needed(ii.id);
    }

    fn visit_trait_item(&mut self, ti: &hir::TraitItem) {
        intravisit::walk_trait_item(self, ti);
        self.create_item_scope_if_needed(ti.id);
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

pub fn resolve_crate(sess: &Session, krate: &hir::Crate) -> RegionMaps {
    let maps = RegionMaps {
        code_extents: RefCell::new(vec![]),
        code_extent_interner: RefCell::new(FnvHashMap()),
        scope_map: RefCell::new(vec![]),
        var_map: RefCell::new(NodeMap()),
        rvalue_scopes: RefCell::new(NodeMap()),
        fn_tree: RefCell::new(NodeMap()),
    };
    let root_extent = maps.bogus_code_extent(
        CodeExtentData::DestructionScope(ast::DUMMY_NODE_ID));
    assert_eq!(root_extent, ROOT_CODE_EXTENT);
    let bogus_extent = maps.bogus_code_extent(
        CodeExtentData::Misc(ast::DUMMY_NODE_ID));
    assert_eq!(bogus_extent, DUMMY_CODE_EXTENT);
    {
        let mut visitor = RegionResolutionVisitor {
            sess: sess,
            region_maps: &maps,
            cx: Context {
                root_id: None,
                parent: ROOT_CODE_EXTENT,
                var_parent: ROOT_CODE_EXTENT
            },
            terminating_scopes: NodeSet()
        };
        krate.visit_all_items(&mut visitor);
    }
    return maps;
}

pub fn resolve_inlined_item(sess: &Session,
                            region_maps: &RegionMaps,
                            item: &InlinedItem) {
    let mut visitor = RegionResolutionVisitor {
        sess: sess,
        region_maps: region_maps,
        cx: Context {
            root_id: None,
            parent: ROOT_CODE_EXTENT,
            var_parent: ROOT_CODE_EXTENT
        },
        terminating_scopes: NodeSet()
    };
    item.visit(&mut visitor);
}
