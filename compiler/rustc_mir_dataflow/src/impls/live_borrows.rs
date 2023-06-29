//! This module defines a liveness analysis for `Local`s that are live due to outstanding references
//! or raw pointers. We can, however, not solely rely on tracking references and pointers due to soundness
//! issues. Exposed raw pointers (i.e. those cast to `usize`) and function calls would make this simple
//! analysis unsound, so we have to handle them as follows:
//!
//!     * exposed pointers (i.e. a cast of a raw pointer to `usize`)
//!         => These count towards the liveness of the `Local` that is behind the raw pointer, e.g. for
//!            `_5 = _4 as usize` (where `_4 = AddressOf(_3)`), we keep `_3` alive on any use site of _5.
//!            The same holds for any re-assignments of the variable containing the exposed pointer, e.g.
//!            for `_6 = _5` or `_6.0 = _5` we keep `_3` alive for any use site of `_6`.
//!            Note: we cover `Local`s corresponding to exposed pointers or re-assignments of those exposed pointers
//!            under the concept of a `LocalWithRefs`. `LocalWithRefs` are `Local`s that might contain references or
//!            pointers and that need to be kept alive if they're behind a reference or pointer.
//!            E.g. for `struct Foo<'a> {p: &'a Bar}` and `_4 = Foo { p: _3 }`, where `_3 = Ref(_2)`, any reference or
//!            pointer of `_4` requires both `_3` and `4` to be kept alive. `LocalWithRefs` are also used for `Local`s
//!            corresponding to destination places of function calls or yield statements, since they could refer to a
//!            reference, pointer or another `LocalWithRefs` that was returned from the function call or yield statement.
//!     * function calls and yield statements could allow for any `Operand` corresponding to a borrow or `LocalWithRefs`
//!       to be moved into the destination place.
//!         => destination places count towards the liveness of the `Local` that could be behind any ref/ptr or `LocalWithRefs`
//!            corresponding to an `Operand`, e.g. for `TerminatorKind::Call { func, args: [_3, _4], destination: _5, .. }`,
//!            where `_3` has type `Ref` or `RawPtr` or is a `LocalWithRefs` we keep any `Local` which `_3` could refer
//!            to live on any use site of `_5`.
//!     * function calls could move refs/ptrs or `LocalWithRefs` in arguments into `Local`s behind mutable references or shared references
//!       that allow for interior mutability.
//!         => E.g. let `Op1` and `Op2` be `Operand`s of a function call and suppose that at least one corresponds to either a mutable
//!            references or shared references that allow for interior mutability and the other is a shared reference or a `LocalWithRefs`.
//!            Let `Locals2` be the set of `Local`s that need to be kept alive due to the borrows corresponding to `Op2`
//!            Then we need to start tracking all `Local`s in `Locals2` for the borrow corresponding to `Op1`, since the `Place`
//!            corresponding to `Op2` might be moved into `Op1` in then call.
//!
//! As an example for what this analysis does:
//!
//! ```ignore(rust)
//! use std::ops::Generator;
//!
//! struct Bar {}
//! struct Foo {
//!     p: *const Bar,
//! }
//!
//! fn takes_and_returns_ref<'a, T>(arg: &'a T) -> &'a T {
//!     arg
//! }
//!
//! fn takes_ptr<T>(arg: *const T) {}
//!
//! fn gen() -> impl Generator<Yield = u32, Return = ()> {
//!     static move || {
//!         let bar = Bar {};   // `Local1`
//!         let bar2 = Bar {};  // `Local2`
//!         let bar_ptr = &bar as *const Bar;   // live locals: [`Local1`]
//!
//!         yield 1;    // live locals: [`Local1`]
//!
//!         let foo = Foo { p: bar_ptr };   // `Local3` is a `LocalWithRefs`, live_locals: [`Local1`]
//!         let bar2_ref = takes_and_returns_ref(&bar2);    // live locals: [`Local1`, `Local2`, `Local3`]
//!         let bar_ptr = foo.p;    // Assignment of `LocalWithRefs` (need to be conservative and keep those alive)
//!                                 // live locals: [`Local1`, `Local3`]
//!         takes_ptr(bar_ptr);     // live locals: [`Local1`, `Local3`]
//!
//!         yield 2;   // live locals: []
//!
//!         let bar3 = Bar {};  // `Local4`
//!         let bar3_ref = &bar3;   // live locals: [`Local4`]
//!
//!         yield 3;    // live locals: [`Local4`]
//!
//!         takes_and_returns_ref(bar3_ref);    // live locals: [`Local4`]
//!     }     
//! }
//! ```
//!
//! Following is a description of the algorithm:
//!     1. build a dependency graph that relates `Local`s based on their borrowing relationships.
//!        As an example if we have something like this (in a simplified MIR representation):
//!
//!        ```ignore
//!         _4 = Bar {}
//!         _5 = Ref(_4)
//!         ...
//!         _10 = f(_5)
//!        ```
//!
//!         Then we add edges from `_5` to `_4` and `_4` to `_5` (for an explanation of why the edge
//!         from `_4` to `_5` is necessary see the comment in `handle_ravlue_or_ptr`) and from
//!         `_10` to `_5`.
//!     2. perform a liveness analysis for borrowed `Local`s.
//!        Continuing our example from step 1, `_5` is live from its definition to the function call and
//!        `_10` is live (it's a `LocalWithRefs`, since it might refer to `_5`) only at the statement
//!        corresponding to the call.
//!     3. Combine the two analyses from step 1 and 2. For any `Local` that corresponds
//!        to a borrow (`_5` in our example), we want to keep the `Local` (`_4`), which is actually
//!        borrowed through it, live over the range at which the borrow is live. Hence for any point
//!        in that range we traverse our dependency graph and look for nodes that correspond to borrowed
//!        `Local`s ("leaf nodes"). In our example we would find an edge from `_5` to `_4`, which is a leaf
//!        node and hence we keep `_4` live over that range.
//!
//! There are some corner cases we need to look out for to make this analysis sound. Let's look
//! at each of the three steps in more detail and elaborate how these steps deal with these corner
//! cases.
//!
//! 1. Dependency Graph
//!
//! The `Node`s in the dependency graph include data values of type `NodeKind`. `NodeKind` has
//! three variants: `Local`, `Borrow` and `LocalWithRefs`.
//!     * `NodeKind::Local` is used for `Local`s that are borrowed somewhere (`_4` in our example), but aren't
//!        themselves references or pointers.
//!     * `NodeKind::Borrow` is used for `Local`s that correspond to borrows (`_5` in our example). We equate
//!        re-borrows with the `Node` that corresponds to the original borrow.
//!     * `NodeKind::LocalWithRefs` is used for `Local`s that aren't themselves refs/ptrs, but contain
//!       `Local`s that correspond to refs/ptrs or other `Local`s with `Node`s of kind `NodeKind::LocalWithRef`s.
//!       `LocalWithRefs` is also used for exposed pointers.
//!       Let's look at an example:
//!
//!        ```ignore
//!         _4 = Bar {}
//!         _5 = Ref(_4)
//!         _6 = Foo(..)(move _5)
//!         ...
//!         _7 = (_6.0)
//!         ```
//!
//!         In this example `_6` would be given `NodeKind::LocalWithRefs` and our graph would look
//!         as follows:
//!
//!         `_7` (NodeKind::Borrow) <-> `_6` (NodeKind::LocalWithRefs) <-> `_5` (NodeKind::Borrow) <-> `_4` (NodeKind::Local)
//!
//!         On the one hand we need to treat `Local`s with `Node`s of kind `NodeKind::LocalWithRefs` similarly
//!         to how we treat `Local`s with `Node`s of kind `NodeKind::Local`, in the sense that if they are
//!         borrowed we want to keep them live over the live range of the borrow. But on the other hand we
//!         want to also treat them like `Local`s with `Node`s of kind `NodeKind::Borrow` as they ultimately
//!         could also contain references or pointers that refer to other `Local`s. So we want a
//!         path in the graph from a `NodeKind::LocalWithRef`s node to the `NodeKind::Local` nodes, whose borrows
//!         they might contain.
//!
//!         Additionally `NodeKind::LocalWithRefs` is also used for raw pointers that are cast to
//!         `usize`:
//!
//!         ```ignore
//!         _4 = Bar {}
//!         _5 = AddressOf(_4)
//!         _6 = _5 as usize
//!         _7 = Aggregate(..) (move _6)
//!         _8 = (_7.0)
//!         ```
//!
//!         In this example our graph would have the following edges:
//!             * `_5` (Borrow) <-> `_4` (Local)
//!             * `_6` (LocalWithRefs) <-> `_5` (Borrow)
//!             * `_7` (LocalWithRefs) <-> `_6` (LocalWithRefs)
//!             * `_8` (LocalWithRefs) <-> `_7` (LocalWithRefs)
//!
//!         We also have to be careful about dealing with `Terminator`s. Whenever we pass references,
//!         pointers or `Local`s with `NodeKind::LocalWithRefs` to a `TerminatorKind::Call` or
//!         `TerminatorKind::Yield`, the destination `Place` or resume place, resp., might contain
//!         these references, pointers or `LocalWithRefs`, hence we have to be conservative
//!         and keep the `destination` `Local` and `resume_arg` `Local` live.
//!
//! 2. Liveness analysis for borrows
//!
//! We perform a standard liveness analysis on any outstanding references, pointers or `LocalWithRefs`
//! So we `gen` at any use site, which are either direct uses of these `Local`s or projections that contain
//! these `Local`s. So e.g.:
//!
//! ```ignore
//! 1. _3 = Foo {}
//! 2. _4 = Bar {}
//! 3. _5 = Ref(_3)
//! 4. _6 = Ref(_4)
//! 5. _7 = Aggregate(..)(move _5)
//! 6. _8 = Call(..)(move _6)
//! 7. _9 = (_8.0)
//! 8. _10 = const 5
//! 9. (_7.0) = move _10
//! ```
//!
//! * `_5` is live from stmt 3 to stmt 9
//! * `_6` is live from stmt 4 to stmt 7
//! * `_7` is a `Local` of kind `LocalWithRefs` so needs to be taken into account in the
//!   analyis. It's live from stmt 5 to stmt 9
//! * `_8` is a `Local` of kind `LocalWithRefs`. It's live from 6. to 7.
//! * `_9` is a `Local` of kind `LocalWithRefs`. It's live at 7.
//!
//! 3. Determining which `Local`s are borrowed
//!
//! Let's use our last example again. The dependency graph for that example looks as follows:
//!
//! `_5` (Borrow) <-> `_3` (Local)
//! `_6` (Borrow) <-> `_4` (Local)
//! `_7` (LocalWithRef) <-> `_5` (Borrow)
//! `_8` (LocalWithRef) -> `_6` (Borrow)
//! `_9` (LocalWithRef) <-> `_8` (LocalWithRef)
//! `_7` (LocalWithRef) <-> `_10` (Local)
//!
//! We then construct a strongly connected components graph from the dependency graph, yielding:
//!
//! SCC1: [_3, _5, _7, _10]
//! SCC2: [_4, _6, _8, _9]
//!
//! Now for each statement in the `Body` we check which refs/ptrs or `LocalWithRefs` are live at that statement
//! and then perform a depth-first search in the scc graph, collecting all `Local`s that need to be kept alive
//! (`Local`s that have `Node`s in the graph of either `NodeKind::Local` or `NodeKind::LocalWithRefs`).
//! So at each of those statements we have the following `Local`s that are live due to borrows:
//!
//! 1. {}
//! 2. {}
//! 3. {_3}
//! 4. {_3, _4}
//! 5. {_3, _4, _7}
//! 6. {_3, _4, _7, _8}
//! 7. {_3, _4, _7, _8, _9}
//! 8. {_3, _7}
//! 9. {_3, _7, _10}
//!
//! Ensuring soundness in all cases requires us to be more conservative (i.e. keeping more `Local`s alive) than necessary
//! in most situations. To eliminate all the unnecessary `Local`s we use the fact that the analysis performed by
//! `MaybeBorrowedLocals` functions as an upper bound for which `Local`s need to be kept alive. Hence we take the intersection
//! of the two analyses at each statement. The results of `MaybeBorrowedLocals` for our example are:
//!
//! 1. {}
//! 2. {}
//! 3. {_3}
//! 4. {_3, _4}
//! 5. {_3, _4}
//! 6. {_3, _4}
//! 7. {_3, _4,}
//! 8. {_3, _4}
//! 9. {_3, _4}
//!
//! Taking the intersection hence yields:
//!
//! 1. {}
//! 2. {}
//! 3. {_3}
//! 4. {_3, _4}
//! 5. {_3, _4}
//! 6. {_3, _4}
//! 7. {_3, _4}
//! 8. {_3}
//! 9. {_3}

use super::*;

use crate::framework::{Analysis, Results, ResultsCursor};
use crate::impls::MaybeBorrowedLocals;
use crate::{
    AnalysisDomain, Backward, CallReturnPlaces, GenKill, GenKillAnalysis, ResultsRefCursor,
};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::graph;
use rustc_data_structures::graph::implementation::{Graph, NodeIndex};
use rustc_data_structures::graph::scc::Sccs;
use rustc_middle::mir::visit::PlaceContext;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty::TypeVisitable;
use rustc_middle::ty::{self, Ty, TypeSuperVisitable};

use std::cell::RefCell;
use std::ops::{ControlFlow, Deref, DerefMut};

// FIXME Properly determine a reasonable value
const DEPTH_LEVEL: u8 = 5;

/// Checks whether a given type allows for interior mutability
struct MaybeContainsInteriorMutabilityVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    has_interior_mut: bool,
    reached_depth_limit: bool,
    current_depth_level: u8,
}

impl<'tcx> MaybeContainsInteriorMutabilityVisitor<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        MaybeContainsInteriorMutabilityVisitor {
            tcx,
            has_interior_mut: false,
            reached_depth_limit: false,
            current_depth_level: 0,
        }
    }
}

impl<'tcx> ty::TypeVisitor<TyCtxt<'tcx>> for MaybeContainsInteriorMutabilityVisitor<'tcx> {
    type BreakTy = ();

    #[instrument(skip(self), level = "debug")]
    fn visit_ty(
        &mut self,
        t: <TyCtxt<'tcx> as ty::Interner>::Ty,
    ) -> std::ops::ControlFlow<Self::BreakTy>
    where
        <TyCtxt<'tcx> as ty::Interner>::Ty: ty::TypeSuperVisitable<TyCtxt<'tcx>>,
    {
        self.current_depth_level += 1;
        debug!(?self.current_depth_level);
        if self.current_depth_level >= DEPTH_LEVEL {
            self.reached_depth_limit = true;
            return ControlFlow::Break(());
        }

        let control_flow = match t.kind() {
            ty::Param(..) => {
                // Need to be conservative here
                self.has_interior_mut = true;
                return ControlFlow::Break(());
            }
            ty::Adt(adt_def, substs) => {
                if adt_def.is_unsafe_cell() {
                    self.has_interior_mut = true;
                    return ControlFlow::Break(());
                }

                let mut control_flow = ControlFlow::Continue(());
                for field in adt_def.all_fields() {
                    let field_ty = field.ty(self.tcx, substs);
                    control_flow = field_ty.visit_with(self);

                    if control_flow == ControlFlow::Break(()) {
                        return ControlFlow::Break(());
                    }
                }

                control_flow
            }
            _ => t.super_visit_with(self),
        };

        self.current_depth_level -= 1;
        control_flow
    }
}

#[derive(Copy, Clone, Debug)]
enum NodeKind {
    // An node corresponding to the place on the lhs of an assignment like `_3 = Ref(_, _, _4)`.
    Borrow(Local),

    // Nodes corresponding to the place on the lhs of a statement like
    // `_2 = Aggregate(Adt(..), _, _, _, _), [move _3, move _6])`,
    // where _3 and _6 are places corresponding to references or raw pointers.
    LocalWithRefs(Local),

    // Nodes corresponding to the borrowed place of an assignment like `_2 = Ref(_, _, borrowed_place)`,
    // if `borrowed_place` is a non-ref or non-ptr value.
    Local(Local),
}

impl NodeKind {
    fn get_local(&self) -> Local {
        match self {
            Self::Borrow(local) => *local,
            Self::LocalWithRefs(local) => *local,
            Self::Local(local) => *local,
        }
    }
}

struct BorrowDepGraph {
    graph: Graph<NodeKind, ()>,

    // Maps `Local`s, for which we have nodes in the graph, to the `NodeIndex`es of those nodes.
    locals_to_node_indexes: FxHashMap<Local, NodeIndex>,
}

impl Deref for BorrowDepGraph {
    type Target = Graph<NodeKind, ()>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for BorrowDepGraph {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

impl graph::DirectedGraph for BorrowDepGraph {
    type Node = NodeIndex;
}

impl graph::WithNumNodes for BorrowDepGraph {
    fn num_nodes(&self) -> usize {
        return self.len_nodes();
    }
}

impl graph::WithSuccessors for BorrowDepGraph {
    fn successors(&self, node: Self::Node) -> <Self as graph::GraphSuccessors<'_>>::Iter {
        Box::new(self.successor_nodes(node))
    }
}

impl<'a> graph::GraphSuccessors<'a> for BorrowDepGraph {
    type Item = NodeIndex;
    type Iter = Box<dyn Iterator<Item = NodeIndex> + 'a>;
}

/// Used to build a dependency graph between borrows/pointers and the `Local`s that
/// they reference.
/// We add edges to the graph in following situations:
///     * direct assignment of reference or raw pointer (e.g. `_4 = Ref(_, _ , borrowed_place)` or
///       `_4 = AddressOf(_, borrowed_place)`). For this case we create a `Node` of kind
///       `NodeKind::Borrow` for the `Local` being assigned to and an edge to either an existing
///       `Node` or if none exists yet to a new `Node` of type `NodeKind::Local` corresponding to
///       a non-ref/ptr `Local`.
///     * assignments to non-reference or non-pointer `Local`s, which themselves might contain
///       references or pointers (e.g. `_2 = Aggregate(Adt(..), _, _, _, _), [move _3, move _6])`,
///       where `_3` and `_6` are places corresponding to references or raw pointers). In this case
///       we create a `Node` of kind `NodeKind::LocalWithRefs` for `_2`. Since `_3` and `_6` are
///       `Local`s that correspond to references, pointers or composite types that might contain
///       references or pointers (`NodeKind::LocalWithRefs`), there already exist `Node`s for these
///       `Local`s. We then add edges from the `Node` for `_2` to both the `Node` for `_3` and the
///       `Node` for `_6`.
///     * `destination` places for `TerminatorKind::Call` and the `resume_arg` places for
///        `TerminatorKind::Yield` if we pass in any references, pointers or composite values that
///        might correspond to references, pointers or exposed pointers (`NodeKind::LocalWithRef`s).
///        The rationale for this is that the return values of both of these terminators might themselves
///        contain any of the references or pointers passed as arguments.
struct BorrowDependencies<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    local_decls: &'a LocalDecls<'tcx>,

    // Tracks the dependencies of places and the references/pointers they may contain,
    // e.g. if we have `_3 = Ref(_, _, _2)` we add an edge from _3 to _2. We later use
    // this graph to allow us to infer which locals need to be kept live in the
    // liveness analysis.
    dep_graph: BorrowDepGraph,

    // Contains the `Local` to which we're currently assigning.
    current_local: Option<Local>,

    // Maps locals that correspond to re-borrows to the borrow that was re-borrowed,
    // so for `_4 = &(*_3)` we include `_4 -> _3` in `reborrows_map`.
    reborrows_map: FxHashMap<Local, Local>,
}

impl<'a, 'tcx> BorrowDependencies<'a, 'tcx> {
    #[instrument(skip(local_decls, tcx), level = "debug")]
    fn new(local_decls: &'a LocalDecls<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        let num_nodes = local_decls.len();
        let approx_num_edges = 3 * num_nodes;

        BorrowDependencies {
            tcx,
            local_decls,
            dep_graph: BorrowDepGraph {
                locals_to_node_indexes: Default::default(),
                graph: Graph::with_capacity(num_nodes, approx_num_edges),
            },
            current_local: None,
            reborrows_map: Default::default(),
        }
    }

    fn maybe_create_node(&mut self, node_kind: NodeKind) -> NodeIndex {
        let local = node_kind.get_local();
        if let Some(node_idx) = self.dep_graph.locals_to_node_indexes.get(&local) {
            *node_idx
        } else {
            if let Some(reborrowed_local) = self.reborrows_map.get(&local) {
                let Some(node_idx) = self.dep_graph.locals_to_node_indexes.get(&reborrowed_local) else {
                    bug!("reborrowed local should have a node ({:?})", reborrowed_local);
                };

                *node_idx
            } else {
                let node_idx = self.dep_graph.add_node(node_kind);
                debug!(
                    "inserting {:?} into locals_to_node_indexes, node idx: {:?}",
                    local, node_idx
                );
                self.dep_graph.locals_to_node_indexes.insert(local, node_idx);
                node_idx
            }
        }
    }

    fn add_edge(&mut self, from_node_idx: NodeIndex, to_node_idx: NodeIndex) {
        debug!(
            "adding edge from {:?}({:?}) -> {:?}({:?})",
            from_node_idx,
            self.dep_graph.node(from_node_idx).data,
            to_node_idx,
            self.dep_graph.node(to_node_idx).data,
        );

        self.dep_graph.add_edge(from_node_idx, to_node_idx, ());
    }

    /// Panics if `local` doesn't have a `Node` in `self.dep_graph`.
    fn get_node_idx_for_local(&self, local: Local) -> NodeIndex {
        if let Some(reborrowed_local) = self.reborrows_map.get(&local) {
            *self.dep_graph.locals_to_node_indexes.get(reborrowed_local).unwrap_or_else(|| {
                bug!("should have created a Node for re-borrowed Local {:?}", reborrowed_local)
            })
        } else {
            *self
                .dep_graph
                .locals_to_node_indexes
                .get(&local)
                .unwrap_or_else(|| bug!("expected {:?} to have a Node", local))
        }
    }

    #[inline]
    fn place_is_mut_ref_or_has_interior_mut(&self, place: &Place<'tcx>) -> bool {
        let ty = place.ty(self.local_decls, self.tcx).ty;
        match ty.kind() {
            ty::Ref(_, inner_ty, mutbl) | ty::RawPtr(ty::TypeAndMut { ty: inner_ty, mutbl }) => {
                *mutbl == Mutability::Mut || self.type_has_interior_mutability(*inner_ty)
            }
            _ => false,
        }
    }

    #[instrument(skip(self), level = "debug")]
    /// Checks whether ty type of `place` is a ref/ptr or whether it's local is a `LocalWithRefs`.
    fn operand_maybe_needs_edge(&self, place: &Place<'tcx>) -> bool {
        let ty = place.ty(self.local_decls, self.tcx).ty;

        // Also account for `LocalWithRef`s
        let is_local_with_refs =
            if let Some(node_idx) = self.dep_graph.locals_to_node_indexes.get(&place.local) {
                let node_for_place = self.dep_graph.node(*node_idx);
                debug!(?node_for_place.data);
                matches!(node_for_place.data, NodeKind::LocalWithRefs(_))
            } else {
                false
            };

        ty.is_ref() || ty.is_unsafe_ptr() || is_local_with_refs
    }

    #[instrument(skip(self), level = "debug")]
    fn maybe_create_edges_for_operands(&mut self, args: &Vec<Operand<'tcx>>) {
        for i in 0..args.len() {
            let outer_operand = &args[i];
            debug!(?outer_operand);
            match outer_operand {
                Operand::Copy(outer_place) | Operand::Move(outer_place) => {
                    if self.operand_maybe_needs_edge(outer_place) {
                        for j in i + 1..args.len() {
                            let inner_operand = &args[j];
                            debug!(?inner_operand);
                            match inner_operand {
                                Operand::Copy(inner_place) | Operand::Move(inner_place) => {
                                    if self.operand_maybe_needs_edge(inner_place) {
                                        let node_idx_outer =
                                            self.get_node_idx_for_local(outer_place.local);
                                        let node_idx_inner =
                                            self.get_node_idx_for_local(inner_place.local);

                                        if self.place_is_mut_ref_or_has_interior_mut(outer_place) {
                                            self.add_edge(node_idx_outer, node_idx_inner);
                                        }

                                        if self.place_is_mut_ref_or_has_interior_mut(inner_place) {
                                            self.add_edge(node_idx_inner, node_idx_outer);
                                        }
                                    }
                                }
                                Operand::Constant(_) => {}
                            }
                        }
                    }
                }
                Operand::Constant(_) => {}
            }
        }
    }

    fn type_has_interior_mutability(&self, ty: Ty<'tcx>) -> bool {
        // FIXME maybe cache those?
        let mut visitor = MaybeContainsInteriorMutabilityVisitor::new(self.tcx);
        ty.visit_with(&mut visitor);

        visitor.has_interior_mut || visitor.reached_depth_limit
    }

    fn handle_rvalue_ref_or_ptr(&mut self, borrowed_place: &Place<'tcx>, mutbl: Mutability) {
        let Some(src_local) = self.current_local else {
                    bug!("Expected self.current_local to be set with Rvalue::Ref|Rvalue::AddressOf");
                };

        let src_node_idx = if borrowed_place.is_indirect() {
            // Don't introduce new nodes for re-borrows.
            if let Some(_) = self.dep_graph.locals_to_node_indexes.get(&borrowed_place.local) {
                self.reborrows_map.insert(src_local, borrowed_place.local);

                return;
            } else {
                // we're in a statement like `_4 = Ref(..)` or `_4 = AddressOf(..)`, hence NodeKind::Borrow for `_4`
                let src_node_kind = NodeKind::Borrow(src_local);
                self.maybe_create_node(src_node_kind)
            }
        } else {
            // we're in a statement like `_4 = Ref(..)`, hence NodeKind::Borrow for `_4`
            let src_node_kind = NodeKind::Borrow(src_local);
            self.maybe_create_node(src_node_kind)
        };

        // If we haven't previously added a node for `borrowed_place.local` then it can be neither
        // `NodeKind::Borrow` nor `NodeKind::LocalsWithRefs`.
        let borrowed_node_kind = NodeKind::Local(borrowed_place.local);
        let node_idx = self.maybe_create_node(borrowed_node_kind);

        // We need to reach the borrowed local through this borrow
        self.add_edge(src_node_idx, node_idx);

        // If this is a mutable borrow or the type might contain interior mutatability
        // we need to also have a path from any other borrows of the locals to which
        // `borrowed_place` refers and this borrow. As an example:
        //
        //  struct WithInteriorMut {
        //      a: RefCell<usize>,
        //  }
        //
        //  fn switch_interior_mut<'a>(raw_ref_exposed: usize, int_mut: &'a WithInteriorMut) {
        //      let mut mut_ref = int_mut.a.borrow_mut();
        //      *mut_ref = raw_ref_exposed;
        //  }
        //
        //  fn gen() -> impl Generator<Yield = u32, Return = ()> {
        //     static move || {
        //         let x = Foo { a: 11 };
        //         let p = &x as *const Foo;
        //         let exposed_p = p as usize;
        //         let int_mut = WithInteriorMut { a: RefCell::new(exposed_p) };
        //         let x2 = Foo { a: 13 };
        //         let p2 = &x2 as *const Foo;
        //         let exposed_p2 = p2 as usize;
        //
        //         yield 12;
        //
        //         switch_interior_mut(exposed_p2, &int_mut);
        //
        //         yield 15;
        //
        //         let int_mut_back = int_mut.a.borrow();
        //         let ref_to_foo2 = unsafe { &*(*int_mut_back as *const Foo) };
        //     }
        // }
        //
        // with MIR that looks something like this (simplified):
        //
        // _3 = Foo { a: const 11_usize },
        // _5 = &_3,
        // _4 = &raw const (*_5),
        // _7 = _4
        // _6 = move _7 as usize (PointerExposeAddress)
        // _10 = _6
        // Terminator(Call, kind: _9 = RefCell::<usize>::new(move _10))
        // _8 = WithInteriorMut { a: move _9 }
        // _11 = Foo { a: const 13_usize}
        // _13 = &_11
        // _12 = &raw const (*13)
        // _15 = _12
        // _14 = move _15 as usize (PointerExposeAddress)
        // Terminator(Yield, _16 = yield(const 12_u32))
        // _18 = _14
        // _20 = &8
        // _19 = &(*_20)
        // Terminator(Call, _17 = switch_interior_mut(move _18, move _19))
        // Terminator(Yield, _21 = yield(const 15_u32))
        // _23 = &(_8.0)
        // _22 = RefCell::<usize>::borrow(move _23)
        // _28 = &_22
        // Terminator(Call, _27 = <Ref<usize> as Deref>::deref(move _28))
        // _26 = (*_27)
        // _25 = move _26 as *const Foo (PointerFromExposedAddress)
        // _24 = &(*_25)
        //
        // Note how we introduce an immutable ref `_20` for `_8`, which we then
        // mutate through the `switch_interior_mut` call by putting the exposed
        // pointer that is contained in `_18` into `_8.0`. We therefore have the
        // following edges in our graph:
        //
        // _20 -> _8
        // _8 <- _20
        // _18 -> _20 (_19 is the same as _20 since it's a reborrow)
        // _20 <- _18
        // (ignoring _17 here since it's unused)
        //
        // We then take a reference of `8.0` via `_23` (remember `8.0` now contains
        // an exposed pointer to `_11`) which we later cast/convert to a reference of
        // `_11` again. So we need to also keep `_11` alive up to this point.
        // If we only had an edge from `_20` to `_8` there would be no way to reach the
        // node for `_18` from `_24`, so we wouldn't be able to keep `_11` alive.
        let borrow_place_ty = borrowed_place.ty(self.local_decls, self.tcx).ty;
        if matches!(mutbl, Mutability::Mut) || self.type_has_interior_mutability(borrow_place_ty) {
            self.add_edge(node_idx, src_node_idx);
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for BorrowDependencies<'a, 'tcx> {
    #[instrument(skip(self, body), level = "debug")]
    fn visit_body(&mut self, body: &Body<'tcx>) {
        // Add nodes for the arguments
        for i in 1..=body.arg_count {
            let local = Local::from_usize(i);
            let node_kind = NodeKind::Local(local);
            let node_idx = self.dep_graph.add_node(node_kind);
            self.dep_graph.locals_to_node_indexes.insert(local, node_idx);
        }

        self.super_body(body)
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match &statement.kind {
            StatementKind::Assign(assign) => {
                let assign_place = assign.0;
                debug!("assign_place_ty: {:?}", assign_place.ty(self.local_decls, self.tcx).ty);
                let assign_local = assign_place.local;
                self.current_local = Some(assign_local);
                debug!("set current_local to {:?}", self.current_local);

                // Do not call `visit_place` here as this might introduce a self-edge, which our liveness analysis
                // assumes not to exist.
                self.visit_rvalue(&assign.1, location)
            }
            StatementKind::FakeRead(..)
            | StatementKind::StorageDead(_)
            | StatementKind::StorageLive(_)
            | StatementKind::AscribeUserType(..)
            | StatementKind::Deinit(_)
            | StatementKind::Coverage(_) => {}
            _ => {
                self.super_statement(statement, location);
            }
        }

        self.current_local = None;
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        debug!("self.current_local: {:?}", self.current_local);
        match rvalue {
            Rvalue::Use(Operand::Move(place) | Operand::Copy(place)) => {
                if matches!(
                    place.ty(self.local_decls, self.tcx).ty.kind(),
                    ty::Ref(..) | ty::RawPtr(..)
                ) {
                    let Some(src_local) = self.current_local else {
                                bug!("Expected self.current_local to be set when encountering Rvalue");
                            };

                    // These are just moves of refs/ptrs, hence `NodeKind::Borrow`.
                    let src_node_kind = NodeKind::Borrow(src_local);
                    let src_node_idx = self.maybe_create_node(src_node_kind);
                    let node_kind = NodeKind::Borrow(place.local);
                    let node_idx = self.maybe_create_node(node_kind);

                    self.add_edge(src_node_idx, node_idx);
                    self.add_edge(node_idx, src_node_idx);
                } else {
                    // Don't introduce edges for moved/copied `Local`s that correspond to `NodeKind::Local` or of
                    // `Local`s that don't have a `Node` in the graph already.
                    if let Some(node_idx) = self.dep_graph.locals_to_node_indexes.get(&place.local)
                    {
                        if matches!(self.dep_graph.node(*node_idx).data, NodeKind::Local(_)) {
                            return;
                        }
                    } else {
                        return;
                    }

                    self.super_rvalue(rvalue, location);
                }
            }
            Rvalue::Ref(_, borrow_kind, borrowed_place) => {
                let mutbl = match borrow_kind {
                    BorrowKind::Mut { .. } => Mutability::Mut,
                    BorrowKind::Shallow | BorrowKind::Shared => Mutability::Not,
                };

                self.handle_rvalue_ref_or_ptr(borrowed_place, mutbl)
            }
            Rvalue::AddressOf(mutbl, borrowed_place) => {
                self.handle_rvalue_ref_or_ptr(borrowed_place, *mutbl);
            }
            _ => self.super_rvalue(rvalue, location),
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        // Add edges for places that correspond to references/raw pointers or nodes of kind `LocalWithRefs`
        let place_ty = place.ty(self.local_decls, self.tcx).ty;
        debug!(?place_ty);
        debug!("current local: {:?}", self.current_local);

        match self.current_local {
            Some(src_local) => {
                match place_ty.kind() {
                    ty::Ref(..) | ty::RawPtr(..) => {
                        // If we haven't created a node for this before, then this must be a
                        // `NodeKind::LocalWithRefs` as we would have handled the
                        // other possible assignment case (`NodeKind::Local`) previously in
                        // `visit_rvalue`.
                        let src_node_kind = NodeKind::LocalWithRefs(src_local);
                        let src_node_idx = self.maybe_create_node(src_node_kind);

                        let borrowed_node_kind = NodeKind::Borrow(place.local);
                        let node_idx = self.maybe_create_node(borrowed_node_kind);

                        self.add_edge(src_node_idx, node_idx);

                        // FIXME don't think this edge is necessary in all cases. We should be able to do something
                        // similar to what is done in `handle_rvalue_ref_or_ptr`, i.e. only add this edge in a mutable
                        // context or if interior mutability is used.
                        self.add_edge(node_idx, src_node_idx);
                    }
                    _ => {
                        if let Some(node_idx) =
                            self.dep_graph.locals_to_node_indexes.get(&place.local)
                        {
                            // LocalsWithRefs = LocalWithRefs/Local assigment

                            let node_idx = *node_idx;
                            let src_node_kind = NodeKind::LocalWithRefs(src_local);
                            let src_node_idx = self.maybe_create_node(src_node_kind);

                            self.add_edge(src_node_idx, node_idx);

                            // FIXME not sure this edge is necessary (this edge is not necessary for moves, so we might
                            // want to handle this case in `visit_rvalue` instead).
                            // For copies does dest prop eliminate the possibility of using both the copy and the copied
                            // value after the assignment?
                            self.add_edge(node_idx, src_node_idx);
                        }
                    }
                }
            }
            _ => {}
        }

        self.super_place(place, context, location)
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        match &terminator.kind {
            TerminatorKind::Call { destination, func, args, .. } => {
                let dest_ty = destination.ty(self.local_decls, self.tcx).ty;
                debug!(?dest_ty);

                // To ensure safety we need to add `destination` to the graph as a `Node` with `NodeKind::LocalWithRefs`
                // if we pass in any refs/ptrs or `Local`s corresponding to `NodeKind::LocalWithRefs`. The reason for this
                // is that the function could include those refs/ptrs in its return value. It's not sufficient
                // to look for the existence of `ty::Ref` or `ty::RawPtr` in the type of the return type, since the
                // function could also cast pointers to integers e.g. .
                // FIMXE: I don't think we really have to create edges from the destination place
                // to the operands. Interior mutability isn't obviously a problem here.
                self.current_local = Some(destination.local);

                self.visit_operand(func, location);
                for arg in args {
                    self.visit_operand(arg, location);
                }

                // Additionally we have to introduce edges between borrowed operands, since we could
                // mutate those in the call (either through mutable references or interior mutability)
                self.maybe_create_edges_for_operands(args);

                self.current_local = None;
            }
            TerminatorKind::Yield { resume_arg, value, .. } => {
                let resume_arg_ty = resume_arg.ty(self.local_decls, self.tcx).ty;
                debug!(?resume_arg_ty);

                // We may need to add edges from `destination`, see the comment for this statement
                // in `TerminatorKind::Call` for the rationale behind this.
                self.current_local = Some(resume_arg.local);

                self.super_operand(value, location);
            }
            TerminatorKind::InlineAsm { .. } => {
                // TO-DO
            }
            _ => self.super_terminator(terminator, location),
        }
    }
}

pub struct BorrowedLocalsResults<'a, 'mir, 'tcx> {
    /// the results of the liveness analysis of `LiveBorrows`
    borrows_analysis_results: Results<'tcx, LiveBorrows<'mir, 'tcx>>,

    /// Maps each `Local` that corresponds to a reference, pointer or a node of kind
    /// `NodeKind::LocalWithRefs` (i.e. `Local`s which either correspond to refs, pointers or
    /// exposed pointers or a composite value that might include refs, pointers or exposed pointers)
    /// to the set of `Local`s that are borrowed through those references, pointers or composite values.
    borrowed_local_to_locals_to_keep_alive: FxHashMap<Local, FxHashSet<Local>>,

    /// The results cursor of the `MaybeBorrowedLocals` analysis. Needed as an upper bound, since
    /// to ensure soundness the `LiveBorrows` analysis would keep more `Local`s alive than
    /// strictly necessary.
    maybe_borrowed_locals_results_cursor: RefCell<
        ResultsCursor<'mir, 'tcx, MaybeBorrowedLocals, &'a Results<'tcx, MaybeBorrowedLocals>>,
    >,
}

impl<'a, 'mir, 'tcx> BorrowedLocalsResults<'a, 'mir, 'tcx>
where
    'tcx: 'mir,
    'tcx: 'a,
{
    fn new(
        borrows_analysis_results: Results<'tcx, LiveBorrows<'mir, 'tcx>>,
        maybe_borrowed_locals_results_cursor: ResultsCursor<
            'mir,
            'tcx,
            MaybeBorrowedLocals,
            &'a Results<'tcx, MaybeBorrowedLocals>,
        >,
        dep_graph: BorrowDepGraph,
    ) -> Self {
        let borrowed_local_to_locals_to_keep_alive = Self::get_locals_to_keep_alive_map(dep_graph);
        Self {
            borrows_analysis_results,
            borrowed_local_to_locals_to_keep_alive,
            maybe_borrowed_locals_results_cursor: RefCell::new(
                maybe_borrowed_locals_results_cursor,
            ),
        }
    }

    /// Uses the dependency graph to find all locals that we need to keep live for a given
    /// `Node` (or more specically the `Local` corresponding to that `Node`).
    #[instrument(skip(dep_graph), level = "debug")]
    fn get_locals_to_keep_alive_map(
        dep_graph: BorrowDepGraph,
    ) -> FxHashMap<Local, FxHashSet<Local>> {
        let mut borrows_to_locals: FxHashMap<Local, FxHashSet<Local>> = Default::default();
        let mut memoization_map: FxHashMap<NodeIndex, FxHashSet<Local>> = Default::default();

        // create SCCs for dependency graph and map each local to its SCC.
        let sccs: Sccs<NodeIndex, NodeIndex> = Sccs::new(&dep_graph);

        // Contains the Locals to keep alive for each scc.
        let mut scc_to_locals_to_keep_alive: FxHashMap<NodeIndex, FxHashSet<Local>> =
            Default::default();

        // Maps each `Local` that has a node in the dependency graph to its SCC index.
        let mut local_to_component: FxHashMap<Local, NodeIndex> = Default::default();

        for (node_idx, scc_idx) in sccs.scc_indices().iter().enumerate() {
            debug!(?node_idx, ?scc_idx);
            let node = dep_graph.node(NodeIndex(node_idx));
            let local = node.data.get_local();

            if matches!(node.data, NodeKind::LocalWithRefs(_) | NodeKind::Local(_)) {
                scc_to_locals_to_keep_alive.entry(*scc_idx).or_default().insert(local);
            }

            local_to_component.insert(local, *scc_idx);
        }

        debug!("scc_to_locals_to_keep_alive: {:#?}", scc_to_locals_to_keep_alive);

        for scc in sccs.all_sccs() {
            let succs = sccs.successors(scc);
            debug!("succs of {:?}: {:?}", scc, succs);
        }

        debug!("local_to_component: {:#?}", local_to_component);

        for (_, node) in dep_graph.enumerated_nodes() {
            if matches!(node.data, NodeKind::Borrow(_) | NodeKind::LocalWithRefs(_)) {
                let current_local = node.data.get_local();
                let scc = local_to_component
                    .get(&current_local)
                    .unwrap_or_else(|| bug!("{:?} should have a component", current_local));

                let locals_to_keep_alive = Self::dfs_for_local(
                    current_local,
                    *scc,
                    &sccs,
                    &scc_to_locals_to_keep_alive,
                    &local_to_component,
                    &mut memoization_map,
                );

                borrows_to_locals.insert(current_local, locals_to_keep_alive);
            }
        }

        debug!("borrows_to_locals: {:#?}", borrows_to_locals);
        borrows_to_locals
    }

    #[instrument(skip(sccs, memoization_map), level = "debug")]
    /// Performs a depth-first search on the singular components graph starting from the `Node` corresponding to
    /// `local` (this is a `Local` corresponding to a `Ref`, `RawPtr` or `LocalWithRefs`).
    /// Collects all `Local`s that need to be kept alive for the borrow corresponding to `local`.
    fn dfs_for_local(
        local: Local,
        current_scc_idx: NodeIndex,
        sccs: &Sccs<NodeIndex, NodeIndex>,
        scc_to_locals_to_keep_alive: &FxHashMap<NodeIndex, FxHashSet<Local>>,
        local_to_scc: &FxHashMap<Local, NodeIndex>,
        memoization_map: &mut FxHashMap<NodeIndex, FxHashSet<Local>>,
    ) -> FxHashSet<Local> {
        if let Some(locals_to_keep_alive) = memoization_map.get(&current_scc_idx) {
            return locals_to_keep_alive.clone();
        }

        let mut locals_to_keep_alive: FxHashSet<Local> = Default::default();
        if let Some(locals_to_keep_alive_for_scc) =
            scc_to_locals_to_keep_alive.get(&current_scc_idx)
        {
            #[allow(rustc::potential_query_instability)]
            // Order is irrelevant here.
            for local_to_keep_alive in locals_to_keep_alive_for_scc {
                locals_to_keep_alive.insert(*local_to_keep_alive);
            }
        }

        let succs = sccs.successors(current_scc_idx);
        debug!(?succs);
        for succ_scc_idx in succs {
            let locals_to_keep_alive_for_succ = Self::dfs_for_local(
                local,
                *succ_scc_idx,
                sccs,
                scc_to_locals_to_keep_alive,
                local_to_scc,
                memoization_map,
            );

            #[allow(rustc::potential_query_instability)]
            // Order is irrelevant here.
            for local_to_keep_alive in locals_to_keep_alive_for_succ {
                locals_to_keep_alive.insert(local_to_keep_alive);
            }
        }

        memoization_map.insert(current_scc_idx, locals_to_keep_alive.clone());

        locals_to_keep_alive
    }
}

/// The function gets the results of the borrowed locals analysis in this module. See the module
/// doc-comment for information on what exactly this analysis does.
#[instrument(skip(tcx, maybe_borrowed_locals_cursor, body), level = "debug")]
pub fn get_borrowed_locals_results<'a, 'mir, 'tcx>(
    body: &'mir Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    maybe_borrowed_locals_cursor: ResultsCursor<
        'mir,
        'tcx,
        MaybeBorrowedLocals,
        &'a Results<'tcx, MaybeBorrowedLocals>,
    >,
) -> BorrowedLocalsResults<'a, 'mir, 'tcx> {
    debug!("body: {:#?}", body);

    let mut borrow_deps = BorrowDependencies::new(body.local_decls(), tcx);
    borrow_deps.visit_body(body);

    if cfg!(debug_assertions) {
        let dep_graph = &borrow_deps.dep_graph;

        debug!(
            "nodes: {:#?}",
            dep_graph
                .all_nodes()
                .clone()
                .into_iter()
                .enumerate()
                .map(|(i, node)| (i, node.data))
                .collect::<Vec<_>>()
        );

        debug!("edges:");
        for edge in dep_graph.all_edges() {
            let src_node_idx = edge.source();
            let src_node = dep_graph.node(src_node_idx);
            let target_node_idx = edge.target();
            let target_node = dep_graph.node(target_node_idx);
            debug!(
                "{:?}({:?}) -> {:?}({:?}) ({:?})",
                src_node_idx, src_node.data, target_node_idx, target_node.data, edge.data
            )
        }
    }

    let reborrows_map = borrow_deps.reborrows_map;
    let local_to_nodekind = borrow_deps
        .dep_graph
        .all_nodes()
        .iter()
        .map(|node| {
            let nodekind = node.data;
            let local = nodekind.get_local();

            (local, nodekind)
        })
        .collect::<FxHashMap<Local, NodeKind>>();

    let live_borrows = LiveBorrows::new(body, tcx, local_to_nodekind, reborrows_map);
    let live_borrows_results =
        live_borrows.into_engine(tcx, body).pass_name("borrowed_locals").iterate_to_fixpoint();

    BorrowedLocalsResults::new(
        live_borrows_results,
        maybe_borrowed_locals_cursor,
        borrow_deps.dep_graph,
    )
}

/// The `ResultsCursor` equivalent for the borrowed locals analysis. Since this analysis doesn't
/// require convergence, we expose the set of borrowed `Local`s for a `Location` directly via
/// the `get` method without the need for any prior 'seek' calls.
pub struct BorrowedLocalsResultsCursor<'a, 'mir, 'tcx> {
    // The cursor for the liveness analysis performed by `LiveBorrows`
    borrows_analysis_cursor: ResultsRefCursor<'a, 'mir, 'tcx, LiveBorrows<'mir, 'tcx>>,

    // Maps each `Local` corresponding to a reference or pointer to the set of `Local`s
    // that are borrowed through the ref/ptr. Additionally contains entries for `Local`s
    // corresponding to `NodeKind::LocalWithRefs` since they might contain refs, ptrs or
    // exposed pointers and need to be treated equivalently to refs/ptrs
    borrowed_local_to_locals_to_keep_alive: &'a FxHashMap<Local, FxHashSet<Local>>,

    // the cursor of the conservative borrowed locals analysis
    maybe_borrowed_locals_results_cursor: &'a RefCell<
        ResultsCursor<'mir, 'tcx, MaybeBorrowedLocals, &'a Results<'tcx, MaybeBorrowedLocals>>,
    >,
}

impl<'a, 'mir, 'tcx> BorrowedLocalsResultsCursor<'a, 'mir, 'tcx> {
    pub fn new(body: &'mir Body<'tcx>, results: &'a BorrowedLocalsResults<'a, 'mir, 'tcx>) -> Self {
        let mut cursor = ResultsCursor::new(body, &results.borrows_analysis_results);

        // We don't care about the order of the blocks, only about the result at a given location.
        // This statement is necessary since we're performing a backward analysis in `LiveBorrows`,
        // but want `Self::get` to be usable in forward analyses as well.
        cursor.allow_unreachable();

        Self {
            borrows_analysis_cursor: cursor,
            borrowed_local_to_locals_to_keep_alive: &results.borrowed_local_to_locals_to_keep_alive,
            maybe_borrowed_locals_results_cursor: &results.maybe_borrowed_locals_results_cursor,
        }
    }

    /// Returns all `Local`s that need to be live at the given `Location` because of
    /// outstanding references or raw pointers.
    pub fn get(&mut self, loc: Location) -> BitSet<Local> {
        self.borrows_analysis_cursor.seek_before_primary_effect(loc);
        let live_borrows_at_loc = self.borrows_analysis_cursor.get();
        debug!(?live_borrows_at_loc);

        let mut borrowed_locals = BitSet::new_empty(live_borrows_at_loc.domain_size());
        for borrowed_local in live_borrows_at_loc.iter() {
            debug!(?borrowed_local);
            if let Some(locals_to_keep_alive) =
                self.borrowed_local_to_locals_to_keep_alive.get(&borrowed_local)
            {
                debug!(?locals_to_keep_alive);
                #[allow(rustc::potential_query_instability)]
                // Order is irrelevant here.
                for local in locals_to_keep_alive.iter() {
                    borrowed_locals.insert(*local);
                }
            }
        }

        // use results of conservative analysis as an "upper bound" on the borrowed locals. This
        // is necessary since to guarantee soundness for this analysis we would have to keep
        // more `Local`s alive than strictly necessary.
        let mut maybe_borrowed_locals_cursor =
            self.maybe_borrowed_locals_results_cursor.borrow_mut();
        maybe_borrowed_locals_cursor.allow_unreachable();
        maybe_borrowed_locals_cursor.seek_before_primary_effect(loc);
        let upper_bound_borrowed_locals = maybe_borrowed_locals_cursor.get();
        borrowed_locals.intersect(upper_bound_borrowed_locals);

        debug!(?borrowed_locals);
        borrowed_locals
    }
}

/// Performs a liveness analysis for borrows and raw pointers. This analysis also tracks `Local`s
/// corresponding to `Node`s of kind `NodeKind::LocalWithRefs`, as these could potentially refer to
/// or include references, pointers or exposed pointers.
#[derive(Clone)]
pub struct LiveBorrows<'mir, 'tcx> {
    body: &'mir Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    local_to_nodekind: FxHashMap<Local, NodeKind>,
    reborrows_map: FxHashMap<Local, Local>,
}

impl<'mir, 'tcx> LiveBorrows<'mir, 'tcx> {
    fn new(
        body: &'mir Body<'tcx>,
        tcx: TyCtxt<'tcx>,
        local_to_nodekind: FxHashMap<Local, NodeKind>,
        reborrows_map: FxHashMap<Local, Local>,
    ) -> Self {
        LiveBorrows { body, tcx, local_to_nodekind, reborrows_map }
    }

    fn transfer_function<'b, T>(
        &self,
        trans: &'b mut T,
    ) -> TransferFunction<'mir, 'b, '_, 'tcx, T> {
        TransferFunction {
            body: self.body,
            tcx: self.tcx,
            _trans: trans,
            local_to_nodekind: &self.local_to_nodekind,
            reborrows_map: &self.reborrows_map,
        }
    }
}

impl<'mir, 'tcx> crate::CloneAnalysis for LiveBorrows<'mir, 'tcx> {
    fn clone_analysis(&self) -> Self {
        self.clone()
    }
}

impl<'a, 'tcx> AnalysisDomain<'tcx> for LiveBorrows<'a, 'tcx> {
    type Domain = BitSet<Local>;
    type Direction = Backward;
    const NAME: &'static str = "live_borrows";

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = unborrowed
        BitSet::new_empty(body.local_decls().len())
    }

    fn initialize_start_block(&self, _: &mir::Body<'tcx>, _: &mut Self::Domain) {
        // Not supported for backward analyses
    }
}

impl<'a, 'tcx> GenKillAnalysis<'tcx> for LiveBorrows<'a, 'tcx> {
    type Idx = Local;

    #[instrument(skip(self, trans), level = "debug")]
    fn statement_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        self.transfer_function(trans).visit_statement(statement, location);
    }

    #[instrument(skip(self, trans), level = "debug")]
    fn terminator_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        self.transfer_function(trans).visit_terminator(terminator, location);
    }

    fn call_return_effect(
        &self,
        _trans: &mut impl GenKill<Self::Idx>,
        _block: mir::BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
    }
}

/// A `Visitor` that defines the transfer function for `LiveBorrows`.
struct TransferFunction<'a, 'b, 'c, 'tcx, T> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    _trans: &'b mut T,
    local_to_nodekind: &'c FxHashMap<Local, NodeKind>,
    reborrows_map: &'c FxHashMap<Local, Local>,
}

impl<'a, 'b, 'c, 'tcx, T> TransferFunction<'a, 'b, 'c, 'tcx, T>
where
    T: GenKill<Local>,
{
    fn gen(&mut self, local: Local) {
        debug!("gen {:?}", local);
        if let Some(reborrowed_local) = self.reborrows_map.get(&local) {
            self._trans.gen(*reborrowed_local);
        } else {
            if self.local_to_nodekind.get(&local).is_some() {
                self._trans.gen(local)
            }
        }
    }

    fn kill(&mut self, local: Local) {
        debug!("killing {:?}", local);
        if let Some(reborrowed_local) = self.reborrows_map.get(&local) {
            self._trans.kill(*reborrowed_local);
        } else {
            if self.local_to_nodekind.get(&local).is_some() {
                self._trans.kill(local)
            }
        }
    }
}

impl<'a, 'tcx, T> Visitor<'tcx> for TransferFunction<'a, '_, '_, 'tcx, T>
where
    T: GenKill<Local>,
{
    #[instrument(skip(self), level = "debug")]
    fn visit_statement(&mut self, stmt: &Statement<'tcx>, location: Location) {
        match &stmt.kind {
            StatementKind::Assign(assign) => {
                let lhs_place = assign.0;
                let projection = lhs_place.projection;
                let lhs_place_ty = lhs_place.ty(self.body.local_decls(), self.tcx).ty;
                debug!(?lhs_place, ?lhs_place_ty);

                match projection.as_slice() {
                    &[] | &[ProjectionElem::OpaqueCast(_)] => {
                        // If there aren't any projections or just an OpaqueCast we need to
                        // kill the local.
                        match lhs_place_ty.kind() {
                            ty::Ref(..) | ty::RawPtr(..) => {
                                self.kill(lhs_place.local);

                                self.visit_rvalue(&assign.1, location);
                            }
                            _ => {
                                if let Some(NodeKind::LocalWithRefs(_)) =
                                    self.local_to_nodekind.get(&lhs_place.local)
                                {
                                    self.kill(lhs_place.local);
                                }
                                self.super_assign(&assign.0, &assign.1, location);
                            }
                        }
                    }
                    _ => {
                        // With any other projection elements, a projection of a local (of type ref/ptr)
                        // is actually a use-site, but we handle this in the call to `visit_place`.
                        self.super_assign(&assign.0, &assign.1, location);
                    }
                }
            }
            StatementKind::FakeRead(..)
            | StatementKind::StorageDead(_)
            | StatementKind::StorageLive(_) => {}
            _ => self.super_statement(stmt, location),
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_place(
        &mut self,
        place: &mir::Place<'tcx>,
        context: PlaceContext,
        location: mir::Location,
    ) {
        let local = place.local;
        let local_ty = self.body.local_decls()[local].ty;
        debug!(?local_ty);

        match local_ty.kind() {
            ty::Ref(..) | ty::RawPtr(..) => {
                self.gen(local);
            }
            _ => {
                if let Some(NodeKind::LocalWithRefs(_)) = self.local_to_nodekind.get(&local) {
                    self.gen(local);
                }
            }
        }

        self.super_place(place, context, location);
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_operand(&mut self, operand: &mir::Operand<'tcx>, location: mir::Location) {
        self.super_operand(operand, location);
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        match &terminator.kind {
            TerminatorKind::Call { destination, args, .. } => {
                match destination.projection.as_slice() {
                    &[] | &[ProjectionElem::OpaqueCast(_)] => {
                        self.kill(destination.local);

                        for arg in args {
                            match arg {
                                Operand::Copy(place) | Operand::Move(place) => {
                                    if let Some(NodeKind::LocalWithRefs(_)) =
                                        self.local_to_nodekind.get(&place.local)
                                    {
                                        // these are `Local`s that contain references/pointers or are raw pointers
                                        // that were assigned to raw pointers, which were cast to usize. Since the
                                        // function call is free to use these in any form, we need to gen them here.
                                        self.gen(place.local);
                                    } else {
                                        self.super_operand(arg, location)
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => self.super_terminator(terminator, location),
                }
            }
            TerminatorKind::Yield { resume_arg, value, .. } => {
                match resume_arg.projection.as_slice() {
                    &[] | &[ProjectionElem::OpaqueCast(_)] => {
                        self.kill(resume_arg.local);

                        match value {
                            Operand::Copy(place) | Operand::Move(place) => {
                                if let Some(NodeKind::LocalWithRefs(_)) =
                                    self.local_to_nodekind.get(&place.local)
                                {
                                    // these are `Local`s that contain references/pointers or are raw pointers
                                    // that were assigned to raw pointers, which were cast to usize. Since the
                                    // function call is free to use these in any form, we need to gen them here.
                                    self.gen(place.local);
                                } else {
                                    self.super_operand(value, location)
                                }
                            }
                            _ => {}
                        }
                        self.visit_operand(value, location)
                    }
                    _ => self.super_terminator(terminator, location),
                }
            }
            TerminatorKind::Drop { place: dropped_place, .. } => {
                // Drop terminators may call custom drop glue (`Drop::drop`), which takes `&mut
                // self` as a parameter. In the general case, a drop impl could launder that
                // reference into the surrounding environment through a raw pointer, thus creating
                // a valid `*mut` pointing to the dropped local. We are not yet willing to declare
                // this particular case UB, so we must treat all dropped locals as mutably borrowed
                // for now. See discussion on [#61069].
                //
                // [#61069]: https://github.com/rust-lang/rust/pull/61069
                if !dropped_place.is_indirect() {
                    self.gen(dropped_place.local);
                }
            }
            _ => self.super_terminator(terminator, location),
        }
    }
}
