use rustc_data_structures::fx::FxHashMap;
use std::cell::RefCell;
use std::hash::Hash;
use std::marker::PhantomData;
use crate::util::common::MemoizationMap;

use super::{DepKind, DepNodeIndex, DepGraph};

/// A DepTrackingMap offers a subset of the `Map` API and ensures that
/// we make calls to `read` and `write` as appropriate. We key the
/// maps with a unique type for brevity.
pub struct DepTrackingMap<M: DepTrackingMapConfig> {
    phantom: PhantomData<M>,
    graph: DepGraph,
    map: FxHashMap<M::Key, (M::Value, DepNodeIndex)>,
}

pub trait DepTrackingMapConfig {
    type Key: Eq + Hash + Clone;
    type Value: Clone;
    fn to_dep_kind() -> DepKind;
}

impl<M: DepTrackingMapConfig> DepTrackingMap<M> {
    pub fn new(graph: DepGraph) -> DepTrackingMap<M> {
        DepTrackingMap {
            phantom: PhantomData,
            graph,
            map: Default::default(),
        }
    }
}

impl<M: DepTrackingMapConfig> MemoizationMap for RefCell<DepTrackingMap<M>> {
    type Key = M::Key;
    type Value = M::Value;

    /// Memoizes an entry in the dep-tracking-map. If the entry is not
    /// already present, then `op` will be executed to compute its value.
    /// The resulting dependency graph looks like this:
    ///
    ///     [op] -> Map(key) -> CurrentTask
    ///
    /// Here, `[op]` represents whatever nodes `op` reads in the
    /// course of execution; `Map(key)` represents the node for this
    /// map, and `CurrentTask` represents the current task when
    /// `memoize` is invoked.
    ///
    /// **Important:** when `op` is invoked, the current task will be
    /// switched to `Map(key)`. Therefore, if `op` makes use of any
    /// HIR nodes or shared state accessed through its closure
    /// environment, it must explicitly register a read of that
    /// state. As an example, see `type_of_item` in `collect`,
    /// which looks something like this:
    ///
    /// ```
    /// fn type_of_item(..., item: &hir::Item) -> Ty<'tcx> {
    ///     let item_def_id = ccx.tcx.hir().local_def_id(it.id);
    ///     ccx.tcx.item_types.memoized(item_def_id, || {
    ///         ccx.tcx.dep_graph.read(DepNode::Hir(item_def_id)); // (*)
    ///         compute_type_of_item(ccx, item)
    ///     });
    /// }
    /// ```
    ///
    /// The key is the line marked `(*)`: the closure implicitly
    /// accesses the body of the item `item`, so we register a read
    /// from `Hir(item_def_id)`.
    fn memoize<OP>(&self, key: M::Key, op: OP) -> M::Value
        where OP: FnOnce() -> M::Value
    {
        let graph;
        {
            let this = self.borrow();
            if let Some(&(ref result, dep_node)) = this.map.get(&key) {
                this.graph.read_index(dep_node);
                return result.clone();
            }
            graph = this.graph.clone();
        }

        let (result, dep_node) = graph.with_anon_task(M::to_dep_kind(), op);
        self.borrow_mut().map.insert(key, (result.clone(), dep_node));
        graph.read_index(dep_node);
        result
    }
}
