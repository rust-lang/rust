//! This file builds up the `ScopeTree`, which describes
//! the parent links in the region hierarchy.
//!
//! For more information about how MIR-based region-checking works,
//! see the [rustc guide].
//!
//! [rustc guide]: https://rust-lang.github.io/rustc-guide/mir/borrowck.html

use crate::ich::{StableHashingContext, NodeIdHashingMode};
use crate::util::nodemap::{FxHashMap, FxHashSet};
use crate::ty;

use std::mem;
use std::fmt;
use rustc_macros::HashStable;
use syntax::source_map;
use syntax_pos::{Span, DUMMY_SP};
use crate::ty::{DefIdTree, TyCtxt};
use crate::ty::query::Providers;

use crate::hir;
use crate::hir::Node;
use crate::hir::def_id::DefId;
use crate::hir::intravisit::{self, Visitor, NestedVisitorMap};
use crate::hir::{Block, Arm, Pat, PatKind, Stmt, Expr, Local};
use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher,
                                           StableHasherResult};

/// Scope represents a statically-describable scope that can be
/// used to bound the lifetime/region for values.
///
/// `Node(node_id)`: Any AST node that has any scope at all has the
/// `Node(node_id)` scope. Other variants represent special cases not
/// immediately derivable from the abstract syntax tree structure.
///
/// `DestructionScope(node_id)` represents the scope of destructors
/// implicitly-attached to `node_id` that run immediately after the
/// expression for `node_id` itself. Not every AST node carries a
/// `DestructionScope`, but those that are `terminating_scopes` do;
/// see discussion with `ScopeTree`.
///
/// `Remainder { block, statement_index }` represents
/// the scope of user code running immediately after the initializer
/// expression for the indexed statement, until the end of the block.
///
/// So: the following code can be broken down into the scopes beneath:
///
/// ```text
/// let a = f().g( 'b: { let x = d(); let y = d(); x.h(y)  }   ) ;
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
///  (M1.): Node scope of the whole `let a = ...;` statement.
///  (M2.): Node scope of the `f()` expression.
///  (M3.): Node scope of the `f().g(..)` expression.
///  (M4.): Node scope of the block labeled `'b:`.
///  (M5.): Node scope of the `let x = d();` statement
///  (D6.): DestructionScope for temporaries created during M5.
///  (R7.): Remainder scope for block `'b:`, stmt 0 (let x = ...).
///  (M8.): Node scope of the `let y = d();` statement.
///  (D9.): DestructionScope for temporaries created during M8.
/// (R10.): Remainder scope for block `'b:`, stmt 1 (let y = ...).
/// (D11.): DestructionScope for temporaries and bindings from block `'b:`.
/// (D12.): DestructionScope for temporaries created during M1 (e.g., f()).
/// ```
///
/// Note that while the above picture shows the destruction scopes
/// as following their corresponding node scopes, in the internal
/// data structures of the compiler the destruction scopes are
/// represented as enclosing parents. This is sound because we use the
/// enclosing parent relationship just to ensure that referenced
/// values live long enough; phrased another way, the starting point
/// of each range is not really the important thing in the above
/// picture, but rather the ending point.
//
// FIXME(pnkfelix): this currently derives `PartialOrd` and `Ord` to
// placate the same deriving in `ty::FreeRegion`, but we may want to
// actually attach a more meaningful ordering to scopes than the one
// generated via deriving here.
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Copy,
         RustcEncodable, RustcDecodable, HashStable)]
pub struct Scope {
    pub id: hir::ItemLocalId,
    pub data: ScopeData,
}

impl fmt::Debug for Scope {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.data {
            ScopeData::Node => write!(fmt, "Node({:?})", self.id),
            ScopeData::CallSite => write!(fmt, "CallSite({:?})", self.id),
            ScopeData::Arguments => write!(fmt, "Arguments({:?})", self.id),
            ScopeData::Destruction => write!(fmt, "Destruction({:?})", self.id),
            ScopeData::Remainder(fsi) => write!(
                fmt,
                "Remainder {{ block: {:?}, first_statement_index: {}}}",
                self.id,
                fsi.as_u32(),
            ),
        }
    }
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Debug, Copy,
         RustcEncodable, RustcDecodable, HashStable)]
pub enum ScopeData {
    Node,

    /// Scope of the call-site for a function or closure
    /// (outlives the arguments as well as the body).
    CallSite,

    /// Scope of arguments passed to a function or closure
    /// (they outlive its body).
    Arguments,

    /// Scope of destructors for temporaries of node-id.
    Destruction,

    /// Scope following a `let id = expr;` binding in a block.
    Remainder(FirstStatementIndex)
}

newtype_index! {
    /// Represents a subscope of `block` for a binding that is introduced
    /// by `block.stmts[first_statement_index]`. Such subscopes represent
    /// a suffix of the block. Note that each subscope does not include
    /// the initializer expression, if any, for the statement indexed by
    /// `first_statement_index`.
    ///
    /// For example, given `{ let (a, b) = EXPR_1; let c = EXPR_2; ... }`:
    ///
    /// * The subscope with `first_statement_index == 0` is scope of both
    ///   `a` and `b`; it does not include EXPR_1, but does include
    ///   everything after that first `let`. (If you want a scope that
    ///   includes EXPR_1 as well, then do not use `Scope::Remainder`,
    ///   but instead another `Scope` that encompasses the whole block,
    ///   e.g., `Scope::Node`.
    ///
    /// * The subscope with `first_statement_index == 1` is scope of `c`,
    ///   and thus does not include EXPR_2, but covers the `...`.
    pub struct FirstStatementIndex {
        derive [HashStable]
    }
}

// compilation error if size of `ScopeData` is not the same as a `u32`
static_assert_size!(ScopeData, 4);

impl Scope {
    /// Returns a item-local ID associated with this scope.
    ///
    /// N.B., likely to be replaced as API is refined; e.g., pnkfelix
    /// anticipates `fn entry_node_id` and `fn each_exit_node_id`.
    pub fn item_local_id(&self) -> hir::ItemLocalId {
        self.id
    }

    pub fn hir_id(&self, scope_tree: &ScopeTree) -> hir::HirId {
        match scope_tree.root_body {
            Some(hir_id) => {
                hir::HirId {
                    owner: hir_id.owner,
                    local_id: self.item_local_id()
                }
            }
            None => hir::DUMMY_HIR_ID
        }
    }

    /// Returns the span of this `Scope`. Note that in general the
    /// returned span may not correspond to the span of any `NodeId` in
    /// the AST.
    pub fn span(&self, tcx: TyCtxt<'_>, scope_tree: &ScopeTree) -> Span {
        let hir_id = self.hir_id(scope_tree);
        if hir_id == hir::DUMMY_HIR_ID {
            return DUMMY_SP;
        }
        let span = tcx.hir().span(hir_id);
        if let ScopeData::Remainder(first_statement_index) = self.data {
            if let Node::Block(ref blk) = tcx.hir().get(hir_id) {
                // Want span for scope starting after the
                // indexed statement and ending at end of
                // `blk`; reuse span of `blk` and shift `lo`
                // forward to end of indexed statement.
                //
                // (This is the special case aluded to in the
                // doc-comment for this method)

                let stmt_span = blk.stmts[first_statement_index.index()].span;

                // To avoid issues with macro-generated spans, the span
                // of the statement must be nested in that of the block.
                if span.lo() <= stmt_span.lo() && stmt_span.lo() <= span.hi() {
                    return Span::new(stmt_span.lo(), span.hi(), span.ctxt());
                }
            }
         }
         span
    }
}

pub type ScopeDepth = u32;

/// The region scope tree encodes information about region relationships.
#[derive(Default, Debug)]
pub struct ScopeTree {
    /// If not empty, this body is the root of this region hierarchy.
    root_body: Option<hir::HirId>,

    /// The parent of the root body owner, if the latter is an
    /// an associated const or method, as impls/traits can also
    /// have lifetime parameters free in this body.
    root_parent: Option<hir::HirId>,

    /// `parent_map` maps from a scope ID to the enclosing scope id;
    /// this is usually corresponding to the lexical nesting, though
    /// in the case of closures the parent scope is the innermost
    /// conditional expression or repeating block. (Note that the
    /// enclosing scope ID for the block associated with a closure is
    /// the closure itself.)
    parent_map: FxHashMap<Scope, (Scope, ScopeDepth)>,

    /// `var_map` maps from a variable or binding ID to the block in
    /// which that variable is declared.
    var_map: FxHashMap<hir::ItemLocalId, Scope>,

    /// maps from a `NodeId` to the associated destruction scope (if any)
    destruction_scopes: FxHashMap<hir::ItemLocalId, Scope>,

    /// `rvalue_scopes` includes entries for those expressions whose cleanup scope is
    /// larger than the default. The map goes from the expression id
    /// to the cleanup scope id. For rvalues not present in this
    /// table, the appropriate cleanup scope is the innermost
    /// enclosing statement, conditional expression, or repeating
    /// block (see `terminating_scopes`).
    /// In constants, None is used to indicate that certain expressions
    /// escape into 'static and should have no local cleanup scope.
    rvalue_scopes: FxHashMap<hir::ItemLocalId, Option<Scope>>,

    /// Encodes the hierarchy of fn bodies. Every fn body (including
    /// closures) forms its own distinct region hierarchy, rooted in
    /// the block that is the fn body. This map points from the ID of
    /// that root block to the ID of the root block for the enclosing
    /// fn, if any. Thus the map structures the fn bodies into a
    /// hierarchy based on their lexical mapping. This is used to
    /// handle the relationships between regions in a fn and in a
    /// closure defined by that fn. See the "Modeling closures"
    /// section of the README in infer::region_constraints for
    /// more details.
    closure_tree: FxHashMap<hir::ItemLocalId, hir::ItemLocalId>,

    /// If there are any `yield` nested within a scope, this map
    /// stores the `Span` of the last one and its index in the
    /// postorder of the Visitor traversal on the HIR.
    ///
    /// HIR Visitor postorder indexes might seem like a peculiar
    /// thing to care about. but it turns out that HIR bindings
    /// and the temporary results of HIR expressions are never
    /// storage-live at the end of HIR nodes with postorder indexes
    /// lower than theirs, and therefore don't need to be suspended
    /// at yield-points at these indexes.
    ///
    /// For an example, suppose we have some code such as:
    /// ```rust,ignore (example)
    ///     foo(f(), yield y, bar(g()))
    /// ```
    ///
    /// With the HIR tree (calls numbered for expository purposes)
    /// ```
    ///     Call#0(foo, [Call#1(f), Yield(y), Call#2(bar, Call#3(g))])
    /// ```
    ///
    /// Obviously, the result of `f()` was created before the yield
    /// (and therefore needs to be kept valid over the yield) while
    /// the result of `g()` occurs after the yield (and therefore
    /// doesn't). If we want to infer that, we can look at the
    /// postorder traversal:
    /// ```plain,ignore
    ///     `foo` `f` Call#1 `y` Yield `bar` `g` Call#3 Call#2 Call#0
    /// ```
    ///
    /// In which we can easily see that `Call#1` occurs before the yield,
    /// and `Call#3` after it.
    ///
    /// To see that this method works, consider:
    ///
    /// Let `D` be our binding/temporary and `U` be our other HIR node, with
    /// `HIR-postorder(U) < HIR-postorder(D)` (in our example, U would be
    /// the yield and D would be one of the calls). Let's show that
    /// `D` is storage-dead at `U`.
    ///
    /// Remember that storage-live/storage-dead refers to the state of
    /// the *storage*, and does not consider moves/drop flags.
    ///
    /// Then:
    ///     1. From the ordering guarantee of HIR visitors (see
    ///     `rustc::hir::intravisit`), `D` does not dominate `U`.
    ///     2. Therefore, `D` is *potentially* storage-dead at `U` (because
    ///     we might visit `U` without ever getting to `D`).
    ///     3. However, we guarantee that at each HIR point, each
    ///     binding/temporary is always either always storage-live
    ///     or always storage-dead. This is what is being guaranteed
    ///     by `terminating_scopes` including all blocks where the
    ///     count of executions is not guaranteed.
    ///     4. By `2.` and `3.`, `D` is *statically* storage-dead at `U`,
    ///     QED.
    ///
    /// I don't think this property relies on `3.` in an essential way - it
    /// is probably still correct even if we have "unrestricted" terminating
    /// scopes. However, why use the complicated proof when a simple one
    /// works?
    ///
    /// A subtle thing: `box` expressions, such as `box (&x, yield 2, &y)`. It
    /// might seem that a `box` expression creates a `Box<T>` temporary
    /// when it *starts* executing, at `HIR-preorder(BOX-EXPR)`. That might
    /// be true in the MIR desugaring, but it is not important in the semantics.
    ///
    /// The reason is that semantically, until the `box` expression returns,
    /// the values are still owned by their containing expressions. So
    /// we'll see that `&x`.
    yield_in_scope: FxHashMap<Scope, YieldData>,

    /// The number of visit_expr and visit_pat calls done in the body.
    /// Used to sanity check visit_expr/visit_pat call count when
    /// calculating generator interiors.
    body_expr_count: FxHashMap<hir::BodyId, usize>,
}

#[derive(Debug, Copy, Clone, RustcEncodable, RustcDecodable, HashStable)]
pub struct YieldData {
    /// `Span` of the yield.
    pub span: Span,
    /// The number of expressions and patterns appearing before the `yield` in the body + 1.
    pub expr_and_pat_count: usize,
    pub source: hir::YieldSource,
}

#[derive(Debug, Copy, Clone)]
pub struct Context {
    /// the root of the current region tree. This is typically the id
    /// of the innermost fn body. Each fn forms its own disjoint tree
    /// in the region hierarchy. These fn bodies are themselves
    /// arranged into a tree. See the "Modeling closures" section of
    /// the README in infer::region_constraints for more
    /// details.
    root_id: Option<hir::ItemLocalId>,

    /// The scope that contains any new variables declared, plus its depth in
    /// the scope tree.
    var_parent: Option<(Scope, ScopeDepth)>,

    /// Region parent of expressions, etc., plus its depth in the scope tree.
    parent: Option<(Scope, ScopeDepth)>,
}

struct RegionResolutionVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,

    // The number of expressions and patterns visited in the current body
    expr_and_pat_count: usize,
    // When this is `true`, we record the `Scopes` we encounter
    // when processing a Yield expression. This allows us to fix
    // up their indices.
    pessimistic_yield: bool,
    // Stores scopes when pessimistic_yield is true.
    fixup_scopes: Vec<Scope>,
    // Generated scope tree:
    scope_tree: ScopeTree,

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
    /// temporaries we would have to cleanup. Therefore, we ensure that
    /// the temporaries never outlast the conditional/repeating
    /// expression, preventing the need for dynamic checks and/or
    /// arbitrary amounts of stack space. Terminating scopes end
    /// up being contained in a DestructionScope that contains the
    /// destructor's execution.
    terminating_scopes: FxHashSet<hir::ItemLocalId>,
}

struct ExprLocatorVisitor {
    hir_id: hir::HirId,
    result: Option<usize>,
    expr_and_pat_count: usize,
}

// This visitor has to have the same visit_expr calls as RegionResolutionVisitor
// since `expr_count` is compared against the results there.
impl<'tcx> Visitor<'tcx> for ExprLocatorVisitor {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_pat(&mut self, pat: &'tcx Pat) {
        intravisit::walk_pat(self, pat);

        self.expr_and_pat_count += 1;

        if pat.hir_id == self.hir_id {
            self.result = Some(self.expr_and_pat_count);
        }
    }

    fn visit_expr(&mut self, expr: &'tcx Expr) {
        debug!("ExprLocatorVisitor - pre-increment {} expr = {:?}",
               self.expr_and_pat_count,
               expr);

        intravisit::walk_expr(self, expr);

        self.expr_and_pat_count += 1;

        debug!("ExprLocatorVisitor - post-increment {} expr = {:?}",
               self.expr_and_pat_count,
               expr);

        if expr.hir_id == self.hir_id {
            self.result = Some(self.expr_and_pat_count);
        }
    }
}

impl<'tcx> ScopeTree {
    pub fn record_scope_parent(&mut self, child: Scope, parent: Option<(Scope, ScopeDepth)>) {
        debug!("{:?}.parent = {:?}", child, parent);

        if let Some(p) = parent {
            let prev = self.parent_map.insert(child, p);
            assert!(prev.is_none());
        }

        // record the destruction scopes for later so we can query them
        if let ScopeData::Destruction = child.data {
            self.destruction_scopes.insert(child.item_local_id(), child);
        }
    }

    pub fn each_encl_scope<E>(&self, mut e: E) where E: FnMut(Scope, Scope) {
        for (&child, &parent) in &self.parent_map {
            e(child, parent.0)
        }
    }

    pub fn each_var_scope<E>(&self, mut e: E) where E: FnMut(&hir::ItemLocalId, Scope) {
        for (child, &parent) in self.var_map.iter() {
            e(child, parent)
        }
    }

    pub fn opt_destruction_scope(&self, n: hir::ItemLocalId) -> Option<Scope> {
        self.destruction_scopes.get(&n).cloned()
    }

    /// Records that `sub_closure` is defined within `sup_closure`. These ids
    /// should be the ID of the block that is the fn body, which is
    /// also the root of the region hierarchy for that fn.
    fn record_closure_parent(&mut self,
                             sub_closure: hir::ItemLocalId,
                             sup_closure: hir::ItemLocalId) {
        debug!("record_closure_parent(sub_closure={:?}, sup_closure={:?})",
               sub_closure, sup_closure);
        assert!(sub_closure != sup_closure);
        let previous = self.closure_tree.insert(sub_closure, sup_closure);
        assert!(previous.is_none());
    }

    fn record_var_scope(&mut self, var: hir::ItemLocalId, lifetime: Scope) {
        debug!("record_var_scope(sub={:?}, sup={:?})", var, lifetime);
        assert!(var != lifetime.item_local_id());
        self.var_map.insert(var, lifetime);
    }

    fn record_rvalue_scope(&mut self, var: hir::ItemLocalId, lifetime: Option<Scope>) {
        debug!("record_rvalue_scope(sub={:?}, sup={:?})", var, lifetime);
        if let Some(lifetime) = lifetime {
            assert!(var != lifetime.item_local_id());
        }
        self.rvalue_scopes.insert(var, lifetime);
    }

    pub fn opt_encl_scope(&self, id: Scope) -> Option<Scope> {
        //! Returns the narrowest scope that encloses `id`, if any.
        self.parent_map.get(&id).cloned().map(|(p, _)| p)
    }

    #[allow(dead_code)] // used in cfg
    pub fn encl_scope(&self, id: Scope) -> Scope {
        //! Returns the narrowest scope that encloses `id`, if any.
        self.opt_encl_scope(id).unwrap()
    }

    /// Returns the lifetime of the local variable `var_id`
    pub fn var_scope(&self, var_id: hir::ItemLocalId) -> Scope {
        self.var_map.get(&var_id).cloned().unwrap_or_else(||
            bug!("no enclosing scope for id {:?}", var_id))
    }

    pub fn temporary_scope(&self, expr_id: hir::ItemLocalId) -> Option<Scope> {
        //! Returns the scope when temp created by expr_id will be cleaned up

        // check for a designated rvalue scope
        if let Some(&s) = self.rvalue_scopes.get(&expr_id) {
            debug!("temporary_scope({:?}) = {:?} [custom]", expr_id, s);
            return s;
        }

        // else, locate the innermost terminating scope
        // if there's one. Static items, for instance, won't
        // have an enclosing scope, hence no scope will be
        // returned.
        let mut id = Scope { id: expr_id, data: ScopeData::Node };

        while let Some(&(p, _)) = self.parent_map.get(&id) {
            match p.data {
                ScopeData::Destruction => {
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

    pub fn var_region(&self, id: hir::ItemLocalId) -> ty::RegionKind {
        //! Returns the lifetime of the variable `id`.

        let scope = ty::ReScope(self.var_scope(id));
        debug!("var_region({:?}) = {:?}", id, scope);
        scope
    }

    pub fn scopes_intersect(&self, scope1: Scope, scope2: Scope) -> bool {
        self.is_subscope_of(scope1, scope2) ||
        self.is_subscope_of(scope2, scope1)
    }

    /// Returns `true` if `subscope` is equal to or is lexically nested inside `superscope`, and
    /// `false` otherwise.
    pub fn is_subscope_of(&self,
                          subscope: Scope,
                          superscope: Scope)
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

        debug!("is_subscope_of({:?}, {:?})=true", subscope, superscope);

        return true;
    }

    /// Returns the ID of the innermost containing body
    pub fn containing_body(&self, mut scope: Scope) -> Option<hir::ItemLocalId> {
        loop {
            if let ScopeData::CallSite = scope.data {
                return Some(scope.item_local_id());
            }

            scope = self.opt_encl_scope(scope)?;
        }
    }

    /// Finds the nearest common ancestor of two scopes. That is, finds the
    /// smallest scope which is greater than or equal to both `scope_a` and
    /// `scope_b`.
    pub fn nearest_common_ancestor(&self, scope_a: Scope, scope_b: Scope) -> Scope {
        if scope_a == scope_b { return scope_a; }

        let mut a = scope_a;
        let mut b = scope_b;

        // Get the depth of each scope's parent. If either scope has no parent,
        // it must be the root, which means we can stop immediately because the
        // root must be the nearest common ancestor. (In practice, this is
        // moderately common.)
        let (parent_a, parent_a_depth) = match self.parent_map.get(&a) {
            Some(pd) => *pd,
            None => return a,
        };
        let (parent_b, parent_b_depth) = match self.parent_map.get(&b) {
            Some(pd) => *pd,
            None => return b,
        };

        if parent_a_depth > parent_b_depth {
            // `a` is lower than `b`. Move `a` up until it's at the same depth
            // as `b`. The first move up is trivial because we already found
            // `parent_a` above; the loop does the remaining N-1 moves.
            a = parent_a;
            for _ in 0..(parent_a_depth - parent_b_depth - 1) {
                a = self.parent_map.get(&a).unwrap().0;
            }
        } else if parent_b_depth > parent_a_depth {
            // `b` is lower than `a`.
            b = parent_b;
            for _ in 0..(parent_b_depth - parent_a_depth - 1) {
                b = self.parent_map.get(&b).unwrap().0;
            }
        } else {
            // Both scopes are at the same depth, and we know they're not equal
            // because that case was tested for at the top of this function. So
            // we can trivially move them both up one level now.
            assert!(parent_a_depth != 0);
            a = parent_a;
            b = parent_b;
        }

        // Now both scopes are at the same level. We move upwards in lockstep
        // until they match. In practice, this loop is almost always executed
        // zero times because `a` is almost always a direct ancestor of `b` or
        // vice versa.
        while a != b {
            a = self.parent_map.get(&a).unwrap().0;
            b = self.parent_map.get(&b).unwrap().0;
        };

        a
    }

    /// Assuming that the provided region was defined within this `ScopeTree`,
    /// returns the outermost `Scope` that the region outlives.
    pub fn early_free_scope(&self, tcx: TyCtxt<'tcx>, br: &ty::EarlyBoundRegion) -> Scope {
        let param_owner = tcx.parent(br.def_id).unwrap();

        let param_owner_id = tcx.hir().as_local_hir_id(param_owner).unwrap();
        let scope = tcx.hir().maybe_body_owned_by(param_owner_id).map(|body_id| {
            tcx.hir().body(body_id).value.hir_id.local_id
        }).unwrap_or_else(|| {
            // The lifetime was defined on node that doesn't own a body,
            // which in practice can only mean a trait or an impl, that
            // is the parent of a method, and that is enforced below.
            if Some(param_owner_id) != self.root_parent {
                tcx.sess.delay_span_bug(
                    DUMMY_SP,
                    &format!("free_scope: {:?} not recognized by the \
                              region scope tree for {:?} / {:?}",
                             param_owner,
                             self.root_parent.map(|id| tcx.hir().local_def_id_from_hir_id(id)),
                             self.root_body.map(|hir_id| DefId::local(hir_id.owner))));
            }

            // The trait/impl lifetime is in scope for the method's body.
            self.root_body.unwrap().local_id
        });

        Scope { id: scope, data: ScopeData::CallSite }
    }

    /// Assuming that the provided region was defined within this `ScopeTree`,
    /// returns the outermost `Scope` that the region outlives.
    pub fn free_scope(&self, tcx: TyCtxt<'tcx>, fr: &ty::FreeRegion) -> Scope {
        let param_owner = match fr.bound_region {
            ty::BoundRegion::BrNamed(def_id, _) => {
                tcx.parent(def_id).unwrap()
            }
            _ => fr.scope
        };

        // Ensure that the named late-bound lifetimes were defined
        // on the same function that they ended up being freed in.
        assert_eq!(param_owner, fr.scope);

        let param_owner_id = tcx.hir().as_local_hir_id(param_owner).unwrap();
        let body_id = tcx.hir().body_owned_by(param_owner_id);
        Scope { id: tcx.hir().body(body_id).value.hir_id.local_id, data: ScopeData::CallSite }
    }

    /// Checks whether the given scope contains a `yield`. If so,
    /// returns `Some((span, expr_count))` with the span of a yield we found and
    /// the number of expressions and patterns appearing before the `yield` in the body + 1.
    /// If there a are multiple yields in a scope, the one with the highest number is returned.
    pub fn yield_in_scope(&self, scope: Scope) -> Option<YieldData> {
        self.yield_in_scope.get(&scope).cloned()
    }

    /// Checks whether the given scope contains a `yield` and if that yield could execute
    /// after `expr`. If so, it returns the span of that `yield`.
    /// `scope` must be inside the body.
    pub fn yield_in_scope_for_expr(&self,
                                   scope: Scope,
                                   expr_hir_id: hir::HirId,
                                   body: &'tcx hir::Body) -> Option<Span> {
        self.yield_in_scope(scope).and_then(|YieldData { span, expr_and_pat_count, .. }| {
            let mut visitor = ExprLocatorVisitor {
                hir_id: expr_hir_id,
                result: None,
                expr_and_pat_count: 0,
            };
            visitor.visit_body(body);
            if expr_and_pat_count >= visitor.result.unwrap() {
                Some(span)
            } else {
                None
            }
        })
    }

    /// Gives the number of expressions visited in a body.
    /// Used to sanity check visit_expr call count when
    /// calculating generator interiors.
    pub fn body_expr_count(&self, body_id: hir::BodyId) -> Option<usize> {
        self.body_expr_count.get(&body_id).map(|r| *r)
    }
}

/// Records the lifetime of a local variable as `cx.var_parent`
fn record_var_lifetime(
    visitor: &mut RegionResolutionVisitor<'_>,
    var_id: hir::ItemLocalId,
    _sp: Span,
) {
    match visitor.cx.var_parent {
        None => {
            // this can happen in extern fn declarations like
            //
            // extern fn isalnum(c: c_int) -> c_int
        }
        Some((parent_scope, _)) =>
            visitor.scope_tree.record_var_scope(var_id, parent_scope),
    }
}

fn resolve_block<'tcx>(visitor: &mut RegionResolutionVisitor<'tcx>, blk: &'tcx hir::Block) {
    debug!("resolve_block(blk.hir_id={:?})", blk.hir_id);

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
    // scope, and thus a temporary (e.g., the result of calling
    // `bar()` in the initializer expression for `let inner = ...;`)
    // will be cleaned up immediately after its corresponding
    // statement (i.e., `let inner = ...;`) executes.
    //
    // On the other hand, temporaries associated with evaluating the
    // tail expression for the block are assigned lifetimes so that
    // they will be cleaned up as part of the terminating scope
    // *surrounding* the block expression. Here, the terminating
    // scope for the block expression is the `quux(..)` call; so
    // those temporaries will only be cleaned up *after* both
    // `other_argument()` has run and also the call to `quux(..)`
    // itself has returned.

    visitor.enter_node_scope_with_dtor(blk.hir_id.local_id);
    visitor.cx.var_parent = visitor.cx.parent;

    {
        // This block should be kept approximately in sync with
        // `intravisit::walk_block`. (We manually walk the block, rather
        // than call `walk_block`, in order to maintain precise
        // index information.)

        for (i, statement) in blk.stmts.iter().enumerate() {
            match statement.node {
                hir::StmtKind::Local(..) |
                hir::StmtKind::Item(..) => {
                    // Each declaration introduces a subscope for bindings
                    // introduced by the declaration; this subscope covers a
                    // suffix of the block. Each subscope in a block has the
                    // previous subscope in the block as a parent, except for
                    // the first such subscope, which has the block itself as a
                    // parent.
                    visitor.enter_scope(
                        Scope {
                            id: blk.hir_id.local_id,
                            data: ScopeData::Remainder(FirstStatementIndex::new(i))
                        }
                    );
                    visitor.cx.var_parent = visitor.cx.parent;
                }
                hir::StmtKind::Expr(..) |
                hir::StmtKind::Semi(..) => {}
            }
            visitor.visit_stmt(statement)
        }
        walk_list!(visitor, visit_expr, &blk.expr);
    }

    visitor.cx = prev_cx;
}

fn resolve_arm<'tcx>(visitor: &mut RegionResolutionVisitor<'tcx>, arm: &'tcx hir::Arm) {
    let prev_cx = visitor.cx;

    visitor.enter_scope(
        Scope {
            id: arm.hir_id.local_id,
            data: ScopeData::Node,
        }
    );
    visitor.cx.var_parent = visitor.cx.parent;

    visitor.terminating_scopes.insert(arm.body.hir_id.local_id);

    if let Some(hir::Guard::If(ref expr)) = arm.guard {
        visitor.terminating_scopes.insert(expr.hir_id.local_id);
    }

    intravisit::walk_arm(visitor, arm);

    visitor.cx = prev_cx;
}

fn resolve_pat<'tcx>(visitor: &mut RegionResolutionVisitor<'tcx>, pat: &'tcx hir::Pat) {
    visitor.record_child_scope(Scope { id: pat.hir_id.local_id, data: ScopeData::Node });

    // If this is a binding then record the lifetime of that binding.
    if let PatKind::Binding(..) = pat.node {
        record_var_lifetime(visitor, pat.hir_id.local_id, pat.span);
    }

    debug!("resolve_pat - pre-increment {} pat = {:?}", visitor.expr_and_pat_count, pat);

    intravisit::walk_pat(visitor, pat);

    visitor.expr_and_pat_count += 1;

    debug!("resolve_pat - post-increment {} pat = {:?}", visitor.expr_and_pat_count, pat);
}

fn resolve_stmt<'tcx>(visitor: &mut RegionResolutionVisitor<'tcx>, stmt: &'tcx hir::Stmt) {
    let stmt_id = stmt.hir_id.local_id;
    debug!("resolve_stmt(stmt.id={:?})", stmt_id);

    // Every statement will clean up the temporaries created during
    // execution of that statement. Therefore each statement has an
    // associated destruction scope that represents the scope of the
    // statement plus its destructors, and thus the scope for which
    // regions referenced by the destructors need to survive.
    visitor.terminating_scopes.insert(stmt_id);

    let prev_parent = visitor.cx.parent;
    visitor.enter_node_scope_with_dtor(stmt_id);

    intravisit::walk_stmt(visitor, stmt);

    visitor.cx.parent = prev_parent;
}

fn resolve_expr<'tcx>(visitor: &mut RegionResolutionVisitor<'tcx>, expr: &'tcx hir::Expr) {
    debug!("resolve_expr - pre-increment {} expr = {:?}", visitor.expr_and_pat_count, expr);

    let prev_cx = visitor.cx;
    visitor.enter_node_scope_with_dtor(expr.hir_id.local_id);

    {
        let terminating_scopes = &mut visitor.terminating_scopes;
        let mut terminating = |id: hir::ItemLocalId| {
            terminating_scopes.insert(id);
        };
        match expr.node {
            // Conditional or repeating scopes are always terminating
            // scopes, meaning that temporaries cannot outlive them.
            // This ensures fixed size stacks.

            hir::ExprKind::Binary(
                source_map::Spanned { node: hir::BinOpKind::And, .. }, _, ref r) |
            hir::ExprKind::Binary(
                source_map::Spanned { node: hir::BinOpKind::Or, .. }, _, ref r) => {
                    // For shortcircuiting operators, mark the RHS as a terminating
                    // scope since it only executes conditionally.
                    terminating(r.hir_id.local_id);
            }

            hir::ExprKind::Loop(ref body, _, _) => {
                terminating(body.hir_id.local_id);
            }

            hir::ExprKind::While(ref expr, ref body, _) => {
                terminating(expr.hir_id.local_id);
                terminating(body.hir_id.local_id);
            }

            hir::ExprKind::DropTemps(ref expr) => {
                // `DropTemps(expr)` does not denote a conditional scope.
                // Rather, we want to achieve the same behavior as `{ let _t = expr; _t }`.
                terminating(expr.hir_id.local_id);
            }

            hir::ExprKind::AssignOp(..) | hir::ExprKind::Index(..) |
            hir::ExprKind::Unary(..) | hir::ExprKind::Call(..) | hir::ExprKind::MethodCall(..) => {
                // FIXME(https://github.com/rust-lang/rfcs/issues/811) Nested method calls
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

    let prev_pessimistic = visitor.pessimistic_yield;

    // Ordinarily, we can rely on the visit order of HIR intravisit
    // to correspond to the actual execution order of statements.
    // However, there's a weird corner case with compund assignment
    // operators (e.g. `a += b`). The evaluation order depends on whether
    // or not the operator is overloaded (e.g. whether or not a trait
    // like AddAssign is implemented).

    // For primitive types (which, despite having a trait impl, don't actually
    // end up calling it), the evluation order is right-to-left. For example,
    // the following code snippet:
    //
    //    let y = &mut 0;
    //    *{println!("LHS!"); y} += {println!("RHS!"); 1};
    //
    // will print:
    //
    // RHS!
    // LHS!
    //
    // However, if the operator is used on a non-primitive type,
    // the evaluation order will be left-to-right, since the operator
    // actually get desugared to a method call. For example, this
    // nearly identical code snippet:
    //
    //     let y = &mut String::new();
    //    *{println!("LHS String"); y} += {println!("RHS String"); "hi"};
    //
    // will print:
    // LHS String
    // RHS String
    //
    // To determine the actual execution order, we need to perform
    // trait resolution. Unfortunately, we need to be able to compute
    // yield_in_scope before type checking is even done, as it gets
    // used by AST borrowcheck.
    //
    // Fortunately, we don't need to know the actual execution order.
    // It suffices to know the 'worst case' order with respect to yields.
    // Specifically, we need to know the highest 'expr_and_pat_count'
    // that we could assign to the yield expression. To do this,
    // we pick the greater of the two values from the left-hand
    // and right-hand expressions. This makes us overly conservative
    // about what types could possibly live across yield points,
    // but we will never fail to detect that a type does actually
    // live across a yield point. The latter part is critical -
    // we're already overly conservative about what types will live
    // across yield points, as the generated MIR will determine
    // when things are actually live. However, for typecheck to work
    // properly, we can't miss any types.


    match expr.node {
        // Manually recurse over closures, because they are the only
        // case of nested bodies that share the parent environment.
        hir::ExprKind::Closure(.., body, _, _) => {
            let body = visitor.tcx.hir().body(body);
            visitor.visit_body(body);
        },
        hir::ExprKind::AssignOp(_, ref left_expr, ref right_expr) => {
            debug!("resolve_expr - enabling pessimistic_yield, was previously {}",
                   prev_pessimistic);

            let start_point = visitor.fixup_scopes.len();
            visitor.pessimistic_yield = true;

            // If the actual execution order turns out to be right-to-left,
            // then we're fine. However, if the actual execution order is left-to-right,
            // then we'll assign too low a count to any `yield` expressions
            // we encounter in 'right_expression' - they should really occur after all of the
            // expressions in 'left_expression'.
            visitor.visit_expr(&right_expr);
            visitor.pessimistic_yield = prev_pessimistic;

            debug!("resolve_expr - restoring pessimistic_yield to {}", prev_pessimistic);
            visitor.visit_expr(&left_expr);
            debug!("resolve_expr - fixing up counts to {}", visitor.expr_and_pat_count);

            // Remove and process any scopes pushed by the visitor
            let target_scopes = visitor.fixup_scopes.drain(start_point..);

            for scope in target_scopes {
                let mut yield_data = visitor.scope_tree.yield_in_scope.get_mut(&scope).unwrap();
                let count = yield_data.expr_and_pat_count;
                let span = yield_data.span;

                // expr_and_pat_count never decreases. Since we recorded counts in yield_in_scope
                // before walking the left-hand side, it should be impossible for the recorded
                // count to be greater than the left-hand side count.
                if count > visitor.expr_and_pat_count {
                    bug!("Encountered greater count {} at span {:?} - expected no greater than {}",
                         count, span, visitor.expr_and_pat_count);
                }
                let new_count = visitor.expr_and_pat_count;
                debug!("resolve_expr - increasing count for scope {:?} from {} to {} at span {:?}",
                       scope, count, new_count, span);

                yield_data.expr_and_pat_count = new_count;
            }

        }

        _ => intravisit::walk_expr(visitor, expr)
    }

    visitor.expr_and_pat_count += 1;

    debug!("resolve_expr post-increment {}, expr = {:?}", visitor.expr_and_pat_count, expr);

    if let hir::ExprKind::Yield(_, source) = &expr.node {
        // Mark this expr's scope and all parent scopes as containing `yield`.
        let mut scope = Scope { id: expr.hir_id.local_id, data: ScopeData::Node };
        loop {
            let data = YieldData {
                span: expr.span,
                expr_and_pat_count: visitor.expr_and_pat_count,
                source: *source,
            };
            visitor.scope_tree.yield_in_scope.insert(scope, data);
            if visitor.pessimistic_yield {
                debug!("resolve_expr in pessimistic_yield - marking scope {:?} for fixup", scope);
                visitor.fixup_scopes.push(scope);
            }

            // Keep traversing up while we can.
            match visitor.scope_tree.parent_map.get(&scope) {
                // Don't cross from closure bodies to their parent.
                Some(&(superscope, _)) => match superscope.data {
                    ScopeData::CallSite => break,
                    _ => scope = superscope
                },
                None => break
            }
        }
    }

    visitor.cx = prev_cx;
}

fn resolve_local<'tcx>(
    visitor: &mut RegionResolutionVisitor<'tcx>,
    pat: Option<&'tcx hir::Pat>,
    init: Option<&'tcx hir::Expr>,
) {
    debug!("resolve_local(pat={:?}, init={:?})", pat, init);

    let blk_scope = visitor.cx.var_parent.map(|(p, _)| p);

    // As an exception to the normal rules governing temporary
    // lifetimes, initializers in a let have a temporary lifetime
    // of the enclosing block. This means that e.g., a program
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
    // 3. `ET`, which matches both rvalues like `foo()` as well as places
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

    if let Some(expr) = init {
        record_rvalue_scope_if_borrow_expr(visitor, &expr, blk_scope);

        if let Some(pat) = pat {
            if is_binding_pat(pat) {
                record_rvalue_scope(visitor, &expr, blk_scope);
            }
        }
    }

    // Make sure we visit the initializer first, so expr_and_pat_count remains correct
    if let Some(expr) = init {
        visitor.visit_expr(expr);
    }
    if let Some(pat) = pat {
        visitor.visit_pat(pat);
    }

    /// Returns `true` if `pat` match the `P&` non-terminal.
    ///
    ///     P& = ref X
    ///        | StructName { ..., P&, ... }
    ///        | VariantName(..., P&, ...)
    ///        | [ ..., P&, ... ]
    ///        | ( ..., P&, ... )
    ///        | box P&
    fn is_binding_pat(pat: &hir::Pat) -> bool {
        // Note that the code below looks for *explicit* refs only, that is, it won't
        // know about *implicit* refs as introduced in #42640.
        //
        // This is not a problem. For example, consider
        //
        //      let (ref x, ref y) = (Foo { .. }, Bar { .. });
        //
        // Due to the explicit refs on the left hand side, the below code would signal
        // that the temporary value on the right hand side should live until the end of
        // the enclosing block (as opposed to being dropped after the let is complete).
        //
        // To create an implicit ref, however, you must have a borrowed value on the RHS
        // already, as in this example (which won't compile before #42640):
        //
        //      let Foo { x, .. } = &Foo { x: ..., ... };
        //
        // in place of
        //
        //      let Foo { ref x, .. } = Foo { ... };
        //
        // In the former case (the implicit ref version), the temporary is created by the
        // & expression, and its lifetime would be extended to the end of the block (due
        // to a different rule, not the below code).
        match pat.node {
            PatKind::Binding(hir::BindingAnnotation::Ref, ..) |
            PatKind::Binding(hir::BindingAnnotation::RefMut, ..) => true,

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
    fn record_rvalue_scope_if_borrow_expr<'tcx>(
        visitor: &mut RegionResolutionVisitor<'tcx>,
        expr: &hir::Expr,
        blk_id: Option<Scope>,
    ) {
        match expr.node {
            hir::ExprKind::AddrOf(_, ref subexpr) => {
                record_rvalue_scope_if_borrow_expr(visitor, &subexpr, blk_id);
                record_rvalue_scope(visitor, &subexpr, blk_id);
            }
            hir::ExprKind::Struct(_, ref fields, _) => {
                for field in fields {
                    record_rvalue_scope_if_borrow_expr(
                        visitor, &field.expr, blk_id);
                }
            }
            hir::ExprKind::Array(ref subexprs) |
            hir::ExprKind::Tup(ref subexprs) => {
                for subexpr in subexprs {
                    record_rvalue_scope_if_borrow_expr(
                        visitor, &subexpr, blk_id);
                }
            }
            hir::ExprKind::Cast(ref subexpr, _) => {
                record_rvalue_scope_if_borrow_expr(visitor, &subexpr, blk_id)
            }
            hir::ExprKind::Block(ref block, _) => {
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
    /// Note: ET is intended to match "rvalues or places based on rvalues".
    fn record_rvalue_scope<'tcx>(
        visitor: &mut RegionResolutionVisitor<'tcx>,
        expr: &hir::Expr,
        blk_scope: Option<Scope>,
    ) {
        let mut expr = expr;
        loop {
            // Note: give all the expressions matching `ET` with the
            // extended temporary lifetime, not just the innermost rvalue,
            // because in codegen if we must compile e.g., `*rvalue()`
            // into a temporary, we request the temporary scope of the
            // outer expression.
            visitor.scope_tree.record_rvalue_scope(expr.hir_id.local_id, blk_scope);

            match expr.node {
                hir::ExprKind::AddrOf(_, ref subexpr) |
                hir::ExprKind::Unary(hir::UnDeref, ref subexpr) |
                hir::ExprKind::Field(ref subexpr, _) |
                hir::ExprKind::Index(ref subexpr, _) => {
                    expr = &subexpr;
                }
                _ => {
                    return;
                }
            }
        }
    }
}

impl<'tcx> RegionResolutionVisitor<'tcx> {
    /// Records the current parent (if any) as the parent of `child_scope`.
    /// Returns the depth of `child_scope`.
    fn record_child_scope(&mut self, child_scope: Scope) -> ScopeDepth {
        let parent = self.cx.parent;
        self.scope_tree.record_scope_parent(child_scope, parent);
        // If `child_scope` has no parent, it must be the root node, and so has
        // a depth of 1. Otherwise, its depth is one more than its parent's.
        parent.map_or(1, |(_p, d)| d + 1)
    }

    /// Records the current parent (if any) as the parent of `child_scope`,
    /// and sets `child_scope` as the new current parent.
    fn enter_scope(&mut self, child_scope: Scope) {
        let child_depth = self.record_child_scope(child_scope);
        self.cx.parent = Some((child_scope, child_depth));
    }

    fn enter_node_scope_with_dtor(&mut self, id: hir::ItemLocalId) {
        // If node was previously marked as a terminating scope during the
        // recursive visit of its parent node in the AST, then we need to
        // account for the destruction scope representing the scope of
        // the destructors that run immediately after it completes.
        if self.terminating_scopes.contains(&id) {
            self.enter_scope(Scope { id, data: ScopeData::Destruction });
        }
        self.enter_scope(Scope { id, data: ScopeData::Node });
    }
}

impl<'tcx> Visitor<'tcx> for RegionResolutionVisitor<'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_block(&mut self, b: &'tcx Block) {
        resolve_block(self, b);
    }

    fn visit_body(&mut self, body: &'tcx hir::Body) {
        let body_id = body.id();
        let owner_id = self.tcx.hir().body_owner(body_id);

        debug!("visit_body(id={:?}, span={:?}, body.id={:?}, cx.parent={:?})",
               owner_id,
               self.tcx.sess.source_map().span_to_string(body.value.span),
               body_id,
               self.cx.parent);

        let outer_ec = mem::replace(&mut self.expr_and_pat_count, 0);
        let outer_cx = self.cx;
        let outer_ts = mem::take(&mut self.terminating_scopes);
        self.terminating_scopes.insert(body.value.hir_id.local_id);

        if let Some(root_id) = self.cx.root_id {
            self.scope_tree.record_closure_parent(body.value.hir_id.local_id, root_id);
        }
        self.cx.root_id = Some(body.value.hir_id.local_id);

        self.enter_scope(Scope { id: body.value.hir_id.local_id, data: ScopeData::CallSite });
        self.enter_scope(Scope { id: body.value.hir_id.local_id, data: ScopeData::Arguments });

        // The arguments and `self` are parented to the fn.
        self.cx.var_parent = self.cx.parent.take();
        for argument in &body.arguments {
            self.visit_pat(&argument.pat);
        }

        // The body of the every fn is a root scope.
        self.cx.parent = self.cx.var_parent;
        if self.tcx.hir().body_owner_kind(owner_id).is_fn_or_closure() {
            self.visit_expr(&body.value)
        } else {
            // Only functions have an outer terminating (drop) scope, while
            // temporaries in constant initializers may be 'static, but only
            // according to rvalue lifetime semantics, using the same
            // syntactical rules used for let initializers.
            //
            // e.g., in `let x = &f();`, the temporary holding the result from
            // the `f()` call lives for the entirety of the surrounding block.
            //
            // Similarly, `const X: ... = &f();` would have the result of `f()`
            // live for `'static`, implying (if Drop restrictions on constants
            // ever get lifted) that the value *could* have a destructor, but
            // it'd get leaked instead of the destructor running during the
            // evaluation of `X` (if at all allowed by CTFE).
            //
            // However, `const Y: ... = g(&f());`, like `let y = g(&f());`,
            // would *not* let the `f()` temporary escape into an outer scope
            // (i.e., `'static`), which means that after `g` returns, it drops,
            // and all the associated destruction scope rules apply.
            self.cx.var_parent = None;
            resolve_local(self, None, Some(&body.value));
        }

        if body.generator_kind.is_some() {
            self.scope_tree.body_expr_count.insert(body_id, self.expr_and_pat_count);
        }

        // Restore context we had at the start.
        self.expr_and_pat_count = outer_ec;
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
        resolve_local(self, Some(&l.pat), l.init.as_ref().map(|e| &**e));
    }
}

fn region_scope_tree(tcx: TyCtxt<'_>, def_id: DefId) -> &ScopeTree {
    let closure_base_def_id = tcx.closure_base_def_id(def_id);
    if closure_base_def_id != def_id {
        return tcx.region_scope_tree(closure_base_def_id);
    }

    let id = tcx.hir().as_local_hir_id(def_id).unwrap();
    let scope_tree = if let Some(body_id) = tcx.hir().maybe_body_owned_by(id) {
        let mut visitor = RegionResolutionVisitor {
            tcx,
            scope_tree: ScopeTree::default(),
            expr_and_pat_count: 0,
            cx: Context {
                root_id: None,
                parent: None,
                var_parent: None,
            },
            terminating_scopes: Default::default(),
            pessimistic_yield: false,
            fixup_scopes: vec![],
        };

        let body = tcx.hir().body(body_id);
        visitor.scope_tree.root_body = Some(body.value.hir_id);

        // If the item is an associated const or a method,
        // record its impl/trait parent, as it can also have
        // lifetime parameters free in this body.
        match tcx.hir().get(id) {
            Node::ImplItem(_) |
            Node::TraitItem(_) => {
                visitor.scope_tree.root_parent = Some(tcx.hir().get_parent_item(id));
            }
            _ => {}
        }

        visitor.visit_body(body);

        visitor.scope_tree
    } else {
        ScopeTree::default()
    };

    tcx.arena.alloc(scope_tree)
}

pub fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        region_scope_tree,
        ..*providers
    };
}

impl<'a> HashStable<StableHashingContext<'a>> for ScopeTree {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let ScopeTree {
            root_body,
            root_parent,
            ref body_expr_count,
            ref parent_map,
            ref var_map,
            ref destruction_scopes,
            ref rvalue_scopes,
            ref closure_tree,
            ref yield_in_scope,
        } = *self;

        hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
            root_body.hash_stable(hcx, hasher);
            root_parent.hash_stable(hcx, hasher);
        });

        body_expr_count.hash_stable(hcx, hasher);
        parent_map.hash_stable(hcx, hasher);
        var_map.hash_stable(hcx, hasher);
        destruction_scopes.hash_stable(hcx, hasher);
        rvalue_scopes.hash_stable(hcx, hasher);
        closure_tree.hash_stable(hcx, hasher);
        yield_in_scope.hash_stable(hcx, hasher);
    }
}
