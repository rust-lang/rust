//! This file declares the `ScopeTree` type, which describes
//! the parent links in the region hierarchy.
//!
//! For more information about how MIR-based region-checking works,
//! see the [rustc dev guide].
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/borrow_check.html

use std::fmt;
use std::ops::Deref;

use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::unord::UnordMap;
use rustc_hir as hir;
use rustc_hir::{HirId, HirIdMap, Node};
use rustc_macros::{HashStable, TyDecodable, TyEncodable};
use rustc_span::{DUMMY_SP, Span};
use tracing::debug;

use crate::ty::TyCtxt;

/// Represents a statically-describable scope that can be used to
/// bound the lifetime/region for values.
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
// placate the same deriving in `ty::LateParamRegion`, but we may want to
// actually attach a more meaningful ordering to scopes than the one
// generated via deriving here.
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Copy, TyEncodable, TyDecodable)]
#[derive(HashStable)]
pub struct Scope {
    pub local_id: hir::ItemLocalId,
    pub data: ScopeData,
}

impl fmt::Debug for Scope {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.data {
            ScopeData::Node => write!(fmt, "Node({:?})", self.local_id),
            ScopeData::CallSite => write!(fmt, "CallSite({:?})", self.local_id),
            ScopeData::Arguments => write!(fmt, "Arguments({:?})", self.local_id),
            ScopeData::Destruction => write!(fmt, "Destruction({:?})", self.local_id),
            ScopeData::IfThen => write!(fmt, "IfThen({:?})", self.local_id),
            ScopeData::IfThenRescope => write!(fmt, "IfThen[edition2024]({:?})", self.local_id),
            ScopeData::Remainder(fsi) => write!(
                fmt,
                "Remainder {{ block: {:?}, first_statement_index: {}}}",
                self.local_id,
                fsi.as_u32(),
            ),
        }
    }
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Debug, Copy, TyEncodable, TyDecodable)]
#[derive(HashStable)]
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

    /// Scope of the condition and then block of an if expression
    /// Used for variables introduced in an if-let expression.
    IfThen,

    /// Scope of the condition and then block of an if expression
    /// Used for variables introduced in an if-let expression,
    /// whose lifetimes do not cross beyond this scope.
    IfThenRescope,

    /// Scope following a `let id = expr;` binding in a block.
    Remainder(FirstStatementIndex),
}

rustc_index::newtype_index! {
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
    #[derive(HashStable)]
    #[encodable]
    #[orderable]
    pub struct FirstStatementIndex {}
}

// compilation error if size of `ScopeData` is not the same as a `u32`
rustc_data_structures::static_assert_size!(ScopeData, 4);

impl Scope {
    pub fn hir_id(&self, scope_tree: &ScopeTree) -> Option<HirId> {
        scope_tree.root_body.map(|hir_id| HirId { owner: hir_id.owner, local_id: self.local_id })
    }

    /// Returns the span of this `Scope`. Note that in general the
    /// returned span may not correspond to the span of any `NodeId` in
    /// the AST.
    pub fn span(&self, tcx: TyCtxt<'_>, scope_tree: &ScopeTree) -> Span {
        let Some(hir_id) = self.hir_id(scope_tree) else {
            return DUMMY_SP;
        };
        let span = tcx.hir_span(hir_id);
        if let ScopeData::Remainder(first_statement_index) = self.data {
            if let Node::Block(blk) = tcx.hir_node(hir_id) {
                // Want span for scope starting after the
                // indexed statement and ending at end of
                // `blk`; reuse span of `blk` and shift `lo`
                // forward to end of indexed statement.
                //
                // (This is the special case alluded to in the
                // doc-comment for this method)

                let stmt_span = blk.stmts[first_statement_index.index()].span;

                // To avoid issues with macro-generated spans, the span
                // of the statement must be nested in that of the block.
                if span.lo() <= stmt_span.lo() && stmt_span.lo() <= span.hi() {
                    return span.with_lo(stmt_span.lo());
                }
            }
        }
        span
    }
}

/// The region scope tree encodes information about region relationships.
#[derive(Default, Debug, HashStable)]
pub struct ScopeTree {
    /// If not empty, this body is the root of this region hierarchy.
    pub root_body: Option<HirId>,

    /// Maps from a scope ID to the enclosing scope id;
    /// this is usually corresponding to the lexical nesting, though
    /// in the case of closures the parent scope is the innermost
    /// conditional expression or repeating block. (Note that the
    /// enclosing scope ID for the block associated with a closure is
    /// the closure itself.)
    pub parent_map: FxIndexMap<Scope, Scope>,

    /// Maps from a variable or binding ID to the block in which that
    /// variable is declared.
    var_map: FxIndexMap<hir::ItemLocalId, Scope>,

    /// Identifies expressions which, if captured into a temporary, ought to
    /// have a temporary whose lifetime extends to the end of the enclosing *block*,
    /// and not the enclosing *statement*. Expressions that are not present in this
    /// table are not rvalue candidates. The set of rvalue candidates is computed
    /// during type check based on a traversal of the AST.
    pub rvalue_candidates: HirIdMap<RvalueCandidate>,

    /// Backwards incompatible scoping that will be introduced in future editions.
    /// This information is used later for linting to identify locals and
    /// temporary values that will receive backwards-incompatible drop orders.
    pub backwards_incompatible_scope: UnordMap<hir::ItemLocalId, Scope>,

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
    ///
    /// ```text
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
    /// `HIR-postorder(U) < HIR-postorder(D)`. Suppose, as in our example,
    /// U is the yield and D is one of the calls.
    /// Let's show that `D` is storage-dead at `U`.
    ///
    /// Remember that storage-live/storage-dead refers to the state of
    /// the *storage*, and does not consider moves/drop flags.
    ///
    /// Then:
    ///
    ///   1. From the ordering guarantee of HIR visitors (see
    ///   `rustc_hir::intravisit`), `D` does not dominate `U`.
    ///
    ///   2. Therefore, `D` is *potentially* storage-dead at `U` (because
    ///   we might visit `U` without ever getting to `D`).
    ///
    ///   3. However, we guarantee that at each HIR point, each
    ///   binding/temporary is always either always storage-live
    ///   or always storage-dead. This is what is being guaranteed
    ///   by `terminating_scopes` including all blocks where the
    ///   count of executions is not guaranteed.
    ///
    ///   4. By `2.` and `3.`, `D` is *statically* storage-dead at `U`,
    ///   QED.
    ///
    /// This property ought to not on (3) in an essential way -- it
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
    pub yield_in_scope: UnordMap<Scope, Vec<YieldData>>,
}

/// See the `rvalue_candidates` field for more information on rvalue
/// candidates in general.
/// The `lifetime` field is None to indicate that certain expressions escape
/// into 'static and should have no local cleanup scope.
#[derive(Debug, Copy, Clone, HashStable)]
pub struct RvalueCandidate {
    pub target: hir::ItemLocalId,
    pub lifetime: Option<Scope>,
}

#[derive(Debug, Copy, Clone, HashStable)]
pub struct YieldData {
    /// The `Span` of the yield.
    pub span: Span,
    /// The number of expressions and patterns appearing before the `yield` in the body, plus one.
    pub expr_and_pat_count: usize,
    pub source: hir::YieldSource,
}

impl ScopeTree {
    pub fn record_scope_parent(&mut self, child: Scope, parent: Option<Scope>) {
        debug!("{:?}.parent = {:?}", child, parent);

        if let Some(p) = parent {
            let prev = self.parent_map.insert(child, p);
            assert!(prev.is_none());
        }
    }

    pub fn record_var_scope(&mut self, var: hir::ItemLocalId, lifetime: Scope) {
        debug!("record_var_scope(sub={:?}, sup={:?})", var, lifetime);
        assert!(var != lifetime.local_id);
        self.var_map.insert(var, lifetime);
    }

    pub fn record_rvalue_candidate(&mut self, var: HirId, candidate: RvalueCandidate) {
        debug!("record_rvalue_candidate(var={var:?}, candidate={candidate:?})");
        if let Some(lifetime) = &candidate.lifetime {
            assert!(var.local_id != lifetime.local_id)
        }
        self.rvalue_candidates.insert(var, candidate);
    }

    /// Returns the narrowest scope that encloses `id`, if any.
    pub fn opt_encl_scope(&self, id: Scope) -> Option<Scope> {
        self.parent_map.get(&id).cloned()
    }

    /// Returns the lifetime of the local variable `var_id`, if any.
    pub fn var_scope(&self, var_id: hir::ItemLocalId) -> Option<Scope> {
        self.var_map.get(&var_id).cloned()
    }

    /// Returns `true` if `subscope` is equal to or is lexically nested inside `superscope`, and
    /// `false` otherwise.
    ///
    /// Used by clippy.
    pub fn is_subscope_of(&self, subscope: Scope, superscope: Scope) -> bool {
        let mut s = subscope;
        debug!("is_subscope_of({:?}, {:?})", subscope, superscope);
        while superscope != s {
            match self.opt_encl_scope(s) {
                None => {
                    debug!("is_subscope_of({:?}, {:?}, s={:?})=false", subscope, superscope, s);
                    return false;
                }
                Some(scope) => s = scope,
            }
        }

        debug!("is_subscope_of({:?}, {:?})=true", subscope, superscope);

        true
    }

    /// Checks whether the given scope contains a `yield`. If so,
    /// returns `Some(YieldData)`. If not, returns `None`.
    pub fn yield_in_scope(&self, scope: Scope) -> Option<&[YieldData]> {
        self.yield_in_scope.get(&scope).map(Deref::deref)
    }
}
