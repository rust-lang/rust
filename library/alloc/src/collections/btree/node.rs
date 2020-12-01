// This is an attempt at an implementation following the ideal
//
// ```
// struct BTreeMap<K, V> {
//     height: usize,
//     root: Option<Box<Node<K, V, height>>>
// }
//
// struct Node<K, V, height: usize> {
//     keys: [K; 2 * B - 1],
//     vals: [V; 2 * B - 1],
//     edges: [if height > 0 { Box<Node<K, V, height - 1>> } else { () }; 2 * B],
//     parent: Option<(NonNull<Node<K, V, height + 1>>, u16)>,
//     len: u16,
// }
// ```
//
// Since Rust doesn't actually have dependent types and polymorphic recursion,
// we make do with lots of unsafety.

// A major goal of this module is to avoid complexity by treating the tree as a generic (if
// weirdly shaped) container and avoiding dealing with most of the B-Tree invariants. As such,
// this module doesn't care whether the entries are sorted, which nodes can be underfull, or
// even what underfull means. However, we do rely on a few invariants:
//
// - Trees must have uniform depth/height. This means that every path down to a leaf from a
//   given node has exactly the same length.
// - A node of length `n` has `n` keys, `n` values, and `n + 1` edges.
//   This implies that even an empty node has at least one edge.
//   For a leaf node, "having an edge" only means we can identify a position in the node,
//   since leaf edges are empty and need no data representation. In an internal node,
//   an edge both identifies a position and contains a pointer to a child node.

use core::cmp::Ordering;
use core::marker::PhantomData;
use core::mem::{self, MaybeUninit};
use core::ptr::{self, NonNull};

use crate::alloc::{AllocRef, Global, Layout};
use crate::boxed::Box;

const B: usize = 6;
pub const CAPACITY: usize = 2 * B - 1;
pub const MIN_LEN_AFTER_SPLIT: usize = B - 1;
const KV_IDX_CENTER: usize = B - 1;
const EDGE_IDX_LEFT_OF_CENTER: usize = B - 1;
const EDGE_IDX_RIGHT_OF_CENTER: usize = B;

/// The underlying representation of leaf nodes and part of the representation of internal nodes.
struct LeafNode<K, V> {
    /// We want to be covariant in `K` and `V`.
    parent: Option<NonNull<InternalNode<K, V>>>,

    /// This node's index into the parent node's `edges` array.
    /// `*node.parent.edges[node.parent_idx]` should be the same thing as `node`.
    /// This is only guaranteed to be initialized when `parent` is non-null.
    parent_idx: MaybeUninit<u16>,

    /// The number of keys and values this node stores.
    len: u16,

    /// The arrays storing the actual data of the node. Only the first `len` elements of each
    /// array are initialized and valid.
    keys: [MaybeUninit<K>; CAPACITY],
    vals: [MaybeUninit<V>; CAPACITY],
}

impl<K, V> LeafNode<K, V> {
    /// Creates a new `LeafNode`. Unsafe because all nodes should really be hidden behind
    /// `BoxedNode`, preventing accidental dropping of uninitialized keys and values.
    unsafe fn new() -> Self {
        LeafNode {
            // As a general policy, we leave fields uninitialized if they can be, as this should
            // be both slightly faster and easier to track in Valgrind.
            keys: MaybeUninit::uninit_array(),
            vals: MaybeUninit::uninit_array(),
            parent: None,
            parent_idx: MaybeUninit::uninit(),
            len: 0,
        }
    }
}

/// The underlying representation of internal nodes. As with `LeafNode`s, these should be hidden
/// behind `BoxedNode`s to prevent dropping uninitialized keys and values. Any pointer to an
/// `InternalNode` can be directly casted to a pointer to the underlying `LeafNode` portion of the
/// node, allowing code to act on leaf and internal nodes generically without having to even check
/// which of the two a pointer is pointing at. This property is enabled by the use of `repr(C)`.
#[repr(C)]
// gdb_providers.py uses this type name for introspection.
struct InternalNode<K, V> {
    data: LeafNode<K, V>,

    /// The pointers to the children of this node. `len + 1` of these are considered
    /// initialized and valid. Although during the process of `into_iter` or `drop`,
    /// some pointers are dangling while others still need to be traversed.
    edges: [MaybeUninit<BoxedNode<K, V>>; 2 * B],
}

impl<K, V> InternalNode<K, V> {
    /// Creates a new `InternalNode`.
    ///
    /// This is unsafe for two reasons. First, it returns an `InternalNode` by value, risking
    /// dropping of uninitialized fields. Second, an invariant of internal nodes is that `len + 1`
    /// edges are initialized and valid, meaning that even when the node is empty (having a
    /// `len` of 0), there must be one initialized and valid edge. This function does not set up
    /// such an edge.
    unsafe fn new() -> Self {
        InternalNode { data: unsafe { LeafNode::new() }, edges: MaybeUninit::uninit_array() }
    }
}

/// A managed, non-null pointer to a node. This is either an owned pointer to
/// `LeafNode<K, V>` or an owned pointer to `InternalNode<K, V>`.
///
/// However, `BoxedNode` contains no information as to which of the two types
/// of nodes it actually contains, and, partially due to this lack of information,
/// is not a separate type and has no destructor.
type BoxedNode<K, V> = NonNull<LeafNode<K, V>>;

/// An owned tree.
///
/// Note that this does not have a destructor, and must be cleaned up manually.
pub type Root<K, V> = NodeRef<marker::Owned, K, V, marker::LeafOrInternal>;

impl<K, V> Root<K, V> {
    /// Returns a new owned tree, with its own root node that is initially empty.
    pub fn new() -> Self {
        NodeRef::new_leaf().forget_type()
    }
}

impl<K, V> NodeRef<marker::Owned, K, V, marker::Leaf> {
    fn new_leaf() -> Self {
        Self::from_new_leaf(Box::new(unsafe { LeafNode::new() }))
    }

    fn from_new_leaf(leaf: Box<LeafNode<K, V>>) -> Self {
        NodeRef { height: 0, node: NonNull::from(Box::leak(leaf)), _marker: PhantomData }
    }
}

impl<K, V> NodeRef<marker::Owned, K, V, marker::Internal> {
    fn from_new_internal(internal: Box<InternalNode<K, V>>, height: usize) -> Self {
        NodeRef { height, node: NonNull::from(Box::leak(internal)).cast(), _marker: PhantomData }
    }
}

impl<K, V, Type> NodeRef<marker::Owned, K, V, Type> {
    /// Mutably borrows the owned node. Unlike `reborrow_mut`, this is safe,
    /// because the return value cannot be used to destroy the node itself,
    /// and there cannot be other references to the tree (except during the
    /// process of `into_iter` or `drop`, but that is a horrific already).
    pub fn borrow_mut(&mut self) -> NodeRef<marker::Mut<'_>, K, V, Type> {
        NodeRef { height: self.height, node: self.node, _marker: PhantomData }
    }

    /// Slightly mutably borrows the owned node.
    pub fn borrow_valmut(&mut self) -> NodeRef<marker::ValMut<'_>, K, V, Type> {
        NodeRef { height: self.height, node: self.node, _marker: PhantomData }
    }
}

impl<K, V> NodeRef<marker::Owned, K, V, marker::LeafOrInternal> {
    /// Adds a new internal node with a single edge pointing to the previous root node,
    /// make that new node the root node, and return it. This increases the height by 1
    /// and is the opposite of `pop_internal_level`.
    pub fn push_internal_level(&mut self) -> NodeRef<marker::Mut<'_>, K, V, marker::Internal> {
        let mut new_node = Box::new(unsafe { InternalNode::new() });
        new_node.edges[0].write(self.node);
        let mut new_root = NodeRef::from_new_internal(new_node, self.height + 1);
        new_root.borrow_mut().first_edge().correct_parent_link();
        *self = new_root.forget_type();

        // `self.borrow_mut()`, except that we just forgot we're internal now:
        NodeRef { height: self.height, node: self.node, _marker: PhantomData }
    }

    /// Removes the internal root node, using its first child as the new root node.
    /// As it is intended only to be called when the root node has only one child,
    /// no cleanup is done on any of the other children.
    /// This decreases the height by 1 and is the opposite of `push_internal_level`.
    ///
    /// Requires exclusive access to the `Root` object but not to the root node;
    /// it will not invalidate other handles or references to the root node.
    ///
    /// Panics if there is no internal level, i.e., if the root node is a leaf.
    pub fn pop_internal_level(&mut self) {
        assert!(self.height > 0);

        let top = self.node;

        let internal_node = NodeRef { height: self.height, node: top, _marker: PhantomData };
        *self = internal_node.first_edge().descend();
        self.borrow_mut().clear_parent_link();

        unsafe {
            Global.dealloc(top.cast(), Layout::new::<InternalNode<K, V>>());
        }
    }
}

// N.B. `NodeRef` is always covariant in `K` and `V`, even when the `BorrowType`
// is `Mut`. This is technically wrong, but cannot result in any unsafety due to
// internal use of `NodeRef` because we stay completely generic over `K` and `V`.
// However, whenever a public type wraps `NodeRef`, make sure that it has the
// correct variance.
///
/// A reference to a node.
///
/// This type has a number of parameters that controls how it acts:
/// - `BorrowType`: A dummy type that describes the kind of borrow and carries a lifetime.
///    - When this is `Immut<'a>`, the `NodeRef` acts roughly like `&'a Node`.
///    - When this is `ValMut<'a>`, the `NodeRef` acts roughly like `&'a Node`
///      with respect to keys and tree structure, but also allows many
///      mutable references to values throughout the tree to coexist.
///    - When this is `Mut<'a>`, the `NodeRef` acts roughly like `&'a mut Node`,
///      although insert methods allow a mutable pointer to a value to coexist.
///    - When this is `Owned`, the `NodeRef` acts roughly like `Box<Node>`,
///      but does not have a destructor, and must be cleaned up manually.
///   Since any `NodeRef` allows navigating through the tree, `BorrowType`
///   effectively applies to the entire tree, not just the node itself.
/// - `K` and `V`: These are the types of keys and values stored in the nodes.
/// - `Type`: This can be `Leaf`, `Internal`, or `LeafOrInternal`. When this is
///   `Leaf`, the `NodeRef` points to a leaf node, when this is `Internal` the
///   `NodeRef` points to an internal node, and when this is `LeafOrInternal` the
///   `NodeRef` could be pointing to either type of node.
///   `Type` is named `NodeType` when used outside `NodeRef`.
///
/// Both `BorrowType` and `NodeType` restrict what methods we implement, to
/// exploit static type safety. There are limitations in the way we can apply
/// such restrictions:
/// - For each type parameter, we can only define a method either generically
///   or for one particular type. For example, we cannot define a method like
///   `key_at` generically for all `BorrowType`, because we want it to return
///   `&'a K` for most choices of `BorrowType`, but plain `K` for `Owned`.
///   We cannot define `key_at` once for all types that carry a lifetime.
///   Therefore, we define it only for the least powerful type `Immut<'a>`.
/// - We cannot get implicit coercion from say `Mut<'a>` to `Immut<'a>`.
///   Therefore, we have to explicitly call `reborrow` on a more powerfull
///   `NodeRef` in order to reach a method like `key_at`.
/// - All methods on `NodeRef` that return some kind of reference, except
///   `reborrow` and `reborrow_mut`, take `self` by value and not by reference.
///   This avoids silently returning a second reference somewhere in the tree.
///   That is irrelevant when `BorrowType` is `Immut<'a>`, but the rule does
///   no harm because we make those `NodeRef` implicitly `Copy`.
///   The rule also avoids implicitly returning the lifetime of `&self`,
///   instead of the lifetime carried by `BorrowType`.
///   An exception to this rule are the insert functions.
/// - Given the above, we need a `reborrow_mut` to explicitly copy a `Mut<'a>`
///   `NodeRef` whenever we want to invoke a method returning an extra reference
///   somewhere in the tree.
pub struct NodeRef<BorrowType, K, V, Type> {
    /// The number of levels that the node and the level of leaves are apart, a
    /// constant of the node that cannot be entirely described by `Type`, and that
    /// the node itself does not store. We only need to store the height of the root
    /// node, and derive every other node's height from it.
    /// Must be zero if `Type` is `Leaf` and non-zero if `Type` is `Internal`.
    height: usize,
    /// The pointer to the leaf or internal node. The definition of `InternalNode`
    /// ensures that the pointer is valid either way.
    node: NonNull<LeafNode<K, V>>,
    _marker: PhantomData<(BorrowType, Type)>,
}

impl<'a, K: 'a, V: 'a, Type> Copy for NodeRef<marker::Immut<'a>, K, V, Type> {}
impl<'a, K: 'a, V: 'a, Type> Clone for NodeRef<marker::Immut<'a>, K, V, Type> {
    fn clone(&self) -> Self {
        *self
    }
}

unsafe impl<BorrowType, K: Sync, V: Sync, Type> Sync for NodeRef<BorrowType, K, V, Type> {}

unsafe impl<'a, K: Sync + 'a, V: Sync + 'a, Type> Send for NodeRef<marker::Immut<'a>, K, V, Type> {}
unsafe impl<'a, K: Send + 'a, V: Send + 'a, Type> Send for NodeRef<marker::Mut<'a>, K, V, Type> {}
unsafe impl<'a, K: Send + 'a, V: Send + 'a, Type> Send for NodeRef<marker::ValMut<'a>, K, V, Type> {}
unsafe impl<K: Send, V: Send, Type> Send for NodeRef<marker::Owned, K, V, Type> {}

impl<BorrowType, K, V> NodeRef<BorrowType, K, V, marker::Internal> {
    /// Unpack a node reference that was packed as `NodeRef::parent`.
    fn from_internal(node: NonNull<InternalNode<K, V>>, height: usize) -> Self {
        debug_assert!(height > 0);
        NodeRef { height, node: node.cast(), _marker: PhantomData }
    }
}

impl<BorrowType, K, V> NodeRef<BorrowType, K, V, marker::Internal> {
    /// Exposes the data of an internal node.
    ///
    /// Returns a raw ptr to avoid invalidating other references to this node.
    fn as_internal_ptr(this: &Self) -> *mut InternalNode<K, V> {
        // SAFETY: the static node type is `Internal`.
        this.node.as_ptr() as *mut InternalNode<K, V>
    }
}

impl<'a, K, V> NodeRef<marker::Immut<'a>, K, V, marker::Internal> {
    /// Exposes the data of an internal node in an immutable tree.
    fn as_internal(this: &Self) -> &'a InternalNode<K, V> {
        let ptr = Self::as_internal_ptr(this);
        // SAFETY: there can be no mutable references into this tree borrowed as `Immut`.
        unsafe { &*ptr }
    }
}

impl<'a, K, V> NodeRef<marker::Mut<'a>, K, V, marker::Internal> {
    /// Offers exclusive access to the data of an internal node.
    fn as_internal_mut(this: &mut Self) -> &'a mut InternalNode<K, V> {
        let ptr = Self::as_internal_ptr(this);
        unsafe { &mut *ptr }
    }
}

impl<BorrowType, K, V, Type> NodeRef<BorrowType, K, V, Type> {
    /// Finds the length of the node. This is the number of keys or values.
    /// The number of edges is `len() + 1`.
    /// Note that, despite being safe, calling this function can have the side effect
    /// of invalidating mutable references that unsafe code has created.
    pub fn len(&self) -> usize {
        // Crucially, we only access the `len` field here. If BorrowType is marker::ValMut,
        // there might be outstanding mutable references to values that we must not invalidate.
        unsafe { usize::from((*Self::as_leaf_ptr(self)).len) }
    }

    /// Returns the number of levels that the node and leaves are apart. Zero
    /// height means the node is a leaf itself. If you picture trees with the
    /// root on top, the number says at which elevation the node appears.
    /// If you picture trees with leaves on top, the number says how high
    /// the tree extends above the node.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Temporarily takes out another, immutable reference to the same node.
    pub fn reborrow(&self) -> NodeRef<marker::Immut<'_>, K, V, Type> {
        NodeRef { height: self.height, node: self.node, _marker: PhantomData }
    }

    /// Exposes the leaf portion of any leaf or internal node.
    ///
    /// Returns a raw ptr to avoid invalidating other references to this node.
    fn as_leaf_ptr(this: &Self) -> *mut LeafNode<K, V> {
        // The node must be valid for at least the LeafNode portion.
        // This is not a reference in the NodeRef type because we don't know if
        // it should be unique or shared.
        this.node.as_ptr()
    }
}

impl<'a, K: 'a, V: 'a, Type> NodeRef<marker::Immut<'a>, K, V, Type> {
    /// Exposes one of the keys stored in the node.
    ///
    /// # Safety
    /// The node has more than `idx` initialized elements.
    pub unsafe fn key_at(self, idx: usize) -> &'a K {
        debug_assert!(idx < self.len());
        unsafe { Self::as_leaf(&self).keys.get_unchecked(idx).assume_init_ref() }
    }

    /// Exposes one of the values stored in the node.
    ///
    /// # Safety
    /// The node has more than `idx` initialized elements.
    unsafe fn val_at(self, idx: usize) -> &'a V {
        debug_assert!(idx < self.len());
        unsafe { Self::as_leaf(&self).vals.get_unchecked(idx).assume_init_ref() }
    }
}

impl<'a, K, V> NodeRef<marker::Immut<'a>, K, V, marker::Internal> {
    /// Exposes the contents of one of the edges in the node.
    ///
    /// # Safety
    /// The node has more than `idx` initialized elements.
    unsafe fn edge_at(self, idx: usize) -> &'a BoxedNode<K, V> {
        debug_assert!(idx <= self.len());
        unsafe { Self::as_internal(&self).edges.get_unchecked(idx).assume_init_ref() }
    }
}

impl<BorrowType, K, V, Type> NodeRef<BorrowType, K, V, Type> {
    /// Finds the parent of the current node. Returns `Ok(handle)` if the current
    /// node actually has a parent, where `handle` points to the edge of the parent
    /// that points to the current node. Returns `Err(self)` if the current node has
    /// no parent, giving back the original `NodeRef`.
    ///
    /// The method name assumes you picture trees with the root node on top.
    ///
    /// `edge.descend().ascend().unwrap()` and `node.ascend().unwrap().descend()` should
    /// both, upon success, do nothing.
    pub fn ascend(
        self,
    ) -> Result<Handle<NodeRef<BorrowType, K, V, marker::Internal>, marker::Edge>, Self> {
        // We need to use raw pointers to nodes because, if BorrowType is marker::ValMut,
        // there might be outstanding mutable references to values that we must not invalidate.
        let leaf_ptr: *const _ = Self::as_leaf_ptr(&self);
        unsafe { (*leaf_ptr).parent }
            .as_ref()
            .map(|parent| Handle {
                node: NodeRef::from_internal(*parent, self.height + 1),
                idx: unsafe { usize::from((*leaf_ptr).parent_idx.assume_init()) },
                _marker: PhantomData,
            })
            .ok_or(self)
    }

    pub fn first_edge(self) -> Handle<Self, marker::Edge> {
        unsafe { Handle::new_edge(self, 0) }
    }

    pub fn last_edge(self) -> Handle<Self, marker::Edge> {
        let len = self.len();
        unsafe { Handle::new_edge(self, len) }
    }

    /// Note that `self` must be nonempty.
    pub fn first_kv(self) -> Handle<Self, marker::KV> {
        let len = self.len();
        assert!(len > 0);
        unsafe { Handle::new_kv(self, 0) }
    }

    /// Note that `self` must be nonempty.
    pub fn last_kv(self) -> Handle<Self, marker::KV> {
        let len = self.len();
        assert!(len > 0);
        unsafe { Handle::new_kv(self, len - 1) }
    }
}

impl<'a, K: 'a, V: 'a, Type> NodeRef<marker::Immut<'a>, K, V, Type> {
    /// Exposes the leaf portion of any leaf or internal node in an immutable tree.
    fn as_leaf(this: &Self) -> &'a LeafNode<K, V> {
        let ptr = Self::as_leaf_ptr(this);
        // SAFETY: there can be no mutable references into this tree borrowed as `Immut`.
        unsafe { &*ptr }
    }
}

impl<K, V> NodeRef<marker::Owned, K, V, marker::LeafOrInternal> {
    /// Similar to `ascend`, gets a reference to a node's parent node, but also
    /// deallocate the current node in the process. This is unsafe because the
    /// current node will still be accessible despite being deallocated.
    pub unsafe fn deallocate_and_ascend(
        self,
    ) -> Option<Handle<NodeRef<marker::Owned, K, V, marker::Internal>, marker::Edge>> {
        let height = self.height;
        let node = self.node;
        let ret = self.ascend().ok();
        unsafe {
            Global.dealloc(
                node.cast(),
                if height > 0 {
                    Layout::new::<InternalNode<K, V>>()
                } else {
                    Layout::new::<LeafNode<K, V>>()
                },
            );
        }
        ret
    }
}

impl<'a, K, V, Type> NodeRef<marker::Mut<'a>, K, V, Type> {
    /// Unsafely asserts to the compiler the static information that this node is a `Leaf`.
    unsafe fn cast_to_leaf_unchecked(self) -> NodeRef<marker::Mut<'a>, K, V, marker::Leaf> {
        debug_assert!(self.height == 0);
        NodeRef { height: self.height, node: self.node, _marker: PhantomData }
    }

    /// Unsafely asserts to the compiler the static information that this node is an `Internal`.
    unsafe fn cast_to_internal_unchecked(self) -> NodeRef<marker::Mut<'a>, K, V, marker::Internal> {
        debug_assert!(self.height > 0);
        NodeRef { height: self.height, node: self.node, _marker: PhantomData }
    }

    /// Temporarily takes out another, mutable reference to the same node. Beware, as
    /// this method is very dangerous, doubly so since it may not immediately appear
    /// dangerous.
    ///
    /// Because mutable pointers can roam anywhere around the tree, the returned
    /// pointer can easily be used to make the original pointer dangling, out of
    /// bounds, or invalid under stacked borrow rules.
    // FIXME(@gereeter) consider adding yet another type parameter to `NodeRef`
    // that restricts the use of navigation methods on reborrowed pointers,
    // preventing this unsafety.
    unsafe fn reborrow_mut(&mut self) -> NodeRef<marker::Mut<'_>, K, V, Type> {
        NodeRef { height: self.height, node: self.node, _marker: PhantomData }
    }

    /// Offers exclusive access to the leaf portion of any leaf or internal node.
    fn as_leaf_mut(this: &mut Self) -> &'a mut LeafNode<K, V> {
        let ptr = Self::as_leaf_ptr(this);
        // SAFETY: we have exclusive access to the entire node.
        unsafe { &mut *ptr }
    }
}

impl<'a, K: 'a, V: 'a, Type> NodeRef<marker::Mut<'a>, K, V, Type> {
    /// Offers exclusive access to a part of the key storage area.
    ///
    /// # Safety
    /// The node has more than `idx` initialized elements.
    unsafe fn into_key_area_mut_at(mut self, idx: usize) -> &'a mut MaybeUninit<K> {
        debug_assert!(idx < self.len());
        unsafe { Self::as_leaf_mut(&mut self).keys.get_unchecked_mut(idx) }
    }

    /// Offers exclusive access to a part of the value storage area.
    ///
    /// # Safety
    /// The node has more than `idx` initialized elements.
    unsafe fn into_val_area_mut_at(mut self, idx: usize) -> &'a mut MaybeUninit<V> {
        debug_assert!(idx < self.len());
        unsafe { Self::as_leaf_mut(&mut self).vals.get_unchecked_mut(idx) }
    }
}

impl<'a, K: 'a, V: 'a> NodeRef<marker::Mut<'a>, K, V, marker::Internal> {
    /// Offers exclusive access to a part of the storage area for edge contents.
    ///
    /// # Safety
    /// The node has at least `idx` initialized elements.
    unsafe fn into_edge_area_mut_at(mut self, idx: usize) -> &'a mut MaybeUninit<BoxedNode<K, V>> {
        debug_assert!(idx <= self.len());
        unsafe { Self::as_internal_mut(&mut self).edges.get_unchecked_mut(idx) }
    }
}

impl<'a, K: 'a, V: 'a, Type> NodeRef<marker::Immut<'a>, K, V, Type> {
    /// Exposes the entire key storage area in the node,
    /// regardless of the node's current length,
    /// having exclusive access to the entire node.
    unsafe fn key_area(self) -> &'a [MaybeUninit<K>] {
        Self::as_leaf(&self).keys.as_slice()
    }

    /// Exposes the entire value storage area in the node,
    /// regardless of the node's current length,
    /// having exclusive access to the entire node.
    unsafe fn val_area(self) -> &'a [MaybeUninit<V>] {
        Self::as_leaf(&self).vals.as_slice()
    }
}

impl<'a, K: 'a, V: 'a> NodeRef<marker::Immut<'a>, K, V, marker::Internal> {
    /// Exposes the entire storage area for edge contents in the node,
    /// regardless of the node's current length,
    /// having exclusive access to the entire node.
    unsafe fn edge_area(self) -> &'a [MaybeUninit<BoxedNode<K, V>>] {
        Self::as_internal(&self).edges.as_slice()
    }
}

impl<'a, K: 'a, V: 'a, Type> NodeRef<marker::Mut<'a>, K, V, Type> {
    /// Offers exclusive access to a sized slice of key storage area in the node.
    unsafe fn into_key_area_slice(mut self) -> &'a mut [MaybeUninit<K>] {
        let len = self.len();
        // SAFETY: the caller will not be able to call further methods on self
        // until the key slice reference is dropped, as we have unique access
        // for the lifetime of the borrow.
        unsafe { Self::as_leaf_mut(&mut self).keys.get_unchecked_mut(..len) }
    }

    /// Offers exclusive access to a sized slice of value storage area in the node.
    unsafe fn into_val_area_slice(mut self) -> &'a mut [MaybeUninit<V>] {
        let len = self.len();
        // SAFETY: the caller will not be able to call further methods on self
        // until the value slice reference is dropped, as we have unique access
        // for the lifetime of the borrow.
        unsafe { Self::as_leaf_mut(&mut self).vals.get_unchecked_mut(..len) }
    }
}

impl<'a, K: 'a, V: 'a> NodeRef<marker::Mut<'a>, K, V, marker::Internal> {
    /// Offers exclusive access to a sized slice of storage area for edge contents in the node.
    unsafe fn into_edge_area_slice(mut self) -> &'a mut [MaybeUninit<BoxedNode<K, V>>] {
        let len = self.len();
        // SAFETY: the caller will not be able to call further methods on self
        // until the edge slice reference is dropped, as we have unique access
        // for the lifetime of the borrow.
        unsafe { Self::as_internal_mut(&mut self).edges.get_unchecked_mut(..len + 1) }
    }
}

impl<'a, K, V, Type> NodeRef<marker::ValMut<'a>, K, V, Type> {
    /// # Safety
    /// - The node has more than `idx` initialized elements.
    unsafe fn into_key_val_mut_at(mut self, idx: usize) -> (&'a K, &'a mut V) {
        // We only create a reference to the one element we are interested in,
        // to avoid aliasing with outstanding references to other elements,
        // in particular, those returned to the caller in earlier iterations.
        let leaf = Self::as_leaf_ptr(&mut self);
        let keys = unsafe { &raw const (*leaf).keys };
        let vals = unsafe { &raw mut (*leaf).vals };
        // We must coerce to unsized array pointers because of Rust issue #74679.
        let keys: *const [_] = keys;
        let vals: *mut [_] = vals;
        let key = unsafe { (&*keys.get_unchecked(idx)).assume_init_ref() };
        let val = unsafe { (&mut *vals.get_unchecked_mut(idx)).assume_init_mut() };
        (key, val)
    }
}

impl<'a, K: 'a, V: 'a, Type> NodeRef<marker::Mut<'a>, K, V, Type> {
    /// Exposes exclusive access to the length of the node.
    pub fn into_len_mut(mut self) -> &'a mut u16 {
        &mut (*Self::as_leaf_mut(&mut self)).len
    }
}

impl<'a, K: 'a, V: 'a> NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal> {
    /// Set or clear the node's link to its parent edge,
    /// without invalidating other references to the node.
    fn set_parent_link(&mut self, parent: NonNull<InternalNode<K, V>>, parent_idx: usize) {
        let leaf = Self::as_leaf_ptr(self);
        unsafe { (*leaf).parent = Some(parent) };
        unsafe { (*leaf).parent_idx.write(parent_idx as u16) };
    }

    /// Clear the node's link to its parent edge.
    /// This only makes sense when there are no other references to the node.
    fn clear_parent_link(&mut self) {
        let leaf = Self::as_leaf_mut(self);
        leaf.parent = None;
    }
}

impl<'a, K: 'a, V: 'a> NodeRef<marker::Mut<'a>, K, V, marker::Leaf> {
    /// Adds a key-value pair to the end of the node.
    pub fn push(&mut self, key: K, val: V) {
        let len = unsafe { self.reborrow_mut().into_len_mut() };
        let idx = usize::from(*len);
        assert!(idx < CAPACITY);
        *len += 1;
        unsafe {
            self.reborrow_mut().into_key_area_mut_at(idx).write(key);
            self.reborrow_mut().into_val_area_mut_at(idx).write(val);
        }
    }

    /// Adds a key-value pair to the beginning of the node.
    fn push_front(&mut self, key: K, val: V) {
        assert!(self.len() < CAPACITY);

        unsafe {
            *self.reborrow_mut().into_len_mut() += 1;
            slice_insert(self.reborrow_mut().into_key_area_slice(), 0, key);
            slice_insert(self.reborrow_mut().into_val_area_slice(), 0, val);
        }
    }
}

impl<'a, K, V> NodeRef<marker::Mut<'a>, K, V, marker::Internal> {
    /// # Safety
    /// Every item returned by `range` is a valid edge index for the node.
    unsafe fn correct_childrens_parent_links<R: Iterator<Item = usize>>(&mut self, range: R) {
        for i in range {
            debug_assert!(i <= self.len());
            unsafe { Handle::new_edge(self.reborrow_mut(), i) }.correct_parent_link();
        }
    }

    fn correct_all_childrens_parent_links(&mut self) {
        let len = self.len();
        unsafe { self.correct_childrens_parent_links(0..=len) };
    }
}

impl<'a, K: 'a, V: 'a> NodeRef<marker::Mut<'a>, K, V, marker::Internal> {
    /// Adds a key-value pair, and an edge to go to the right of that pair,
    /// to the end of the node.
    pub fn push(&mut self, key: K, val: V, edge: Root<K, V>) {
        assert!(edge.height == self.height - 1);

        let len = unsafe { self.reborrow_mut().into_len_mut() };
        let idx = usize::from(*len);
        assert!(idx < CAPACITY);
        *len += 1;
        unsafe {
            self.reborrow_mut().into_key_area_mut_at(idx).write(key);
            self.reborrow_mut().into_val_area_mut_at(idx).write(val);
            self.reborrow_mut().into_edge_area_mut_at(idx + 1).write(edge.node);
            Handle::new_edge(self.reborrow_mut(), idx + 1).correct_parent_link();
        }
    }

    /// Adds a key-value pair, and an edge to go to the left of that pair,
    /// to the beginning of the node.
    fn push_front(&mut self, key: K, val: V, edge: Root<K, V>) {
        assert!(edge.height == self.height - 1);
        assert!(self.len() < CAPACITY);

        unsafe {
            *self.reborrow_mut().into_len_mut() += 1;
            slice_insert(self.reborrow_mut().into_key_area_slice(), 0, key);
            slice_insert(self.reborrow_mut().into_val_area_slice(), 0, val);
            slice_insert(self.reborrow_mut().into_edge_area_slice(), 0, edge.node);
        }

        self.correct_all_childrens_parent_links();
    }
}

impl<'a, K: 'a, V: 'a> NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal> {
    /// Removes a key-value pair from the end of the node and returns the pair.
    /// Also removes the edge that was to the right of that pair and, if the node
    /// is internal, returns the orphaned subtree that this edge owned.
    fn pop(&mut self) -> (K, V, Option<Root<K, V>>) {
        debug_assert!(self.len() > 0);

        let idx = self.len() - 1;

        unsafe {
            let key = ptr::read(self.reborrow().key_at(idx));
            let val = ptr::read(self.reborrow().val_at(idx));
            let edge = match self.reborrow_mut().force() {
                ForceResult::Leaf(_) => None,
                ForceResult::Internal(internal) => {
                    let node = ptr::read(internal.reborrow().edge_at(idx + 1));
                    let mut edge = Root { node, height: internal.height - 1, _marker: PhantomData };
                    // In practice, clearing the parent is a waste of time, because we will
                    // insert the node elsewhere and set its parent link again.
                    edge.borrow_mut().clear_parent_link();
                    Some(edge)
                }
            };

            *self.reborrow_mut().into_len_mut() -= 1;
            (key, val, edge)
        }
    }

    /// Removes a key-value pair from the beginning of the node and returns the pair.
    /// Also removes the edge that was to the left of that pair and, if the node is
    /// internal, returns the orphaned subtree that this edge owned.
    fn pop_front(&mut self) -> (K, V, Option<Root<K, V>>) {
        debug_assert!(self.len() > 0);

        let old_len = self.len();

        unsafe {
            let key = slice_remove(self.reborrow_mut().into_key_area_slice(), 0);
            let val = slice_remove(self.reborrow_mut().into_val_area_slice(), 0);
            let edge = match self.reborrow_mut().force() {
                ForceResult::Leaf(_) => None,
                ForceResult::Internal(mut internal) => {
                    let node = slice_remove(internal.reborrow_mut().into_edge_area_slice(), 0);
                    let mut edge = Root { node, height: internal.height - 1, _marker: PhantomData };
                    // In practice, clearing the parent is a waste of time, because we will
                    // insert the node elsewhere and set its parent link again.
                    edge.borrow_mut().clear_parent_link();

                    internal.correct_childrens_parent_links(0..old_len);

                    Some(edge)
                }
            };

            *self.reborrow_mut().into_len_mut() -= 1;

            (key, val, edge)
        }
    }

    fn into_kv_pointers_mut(mut self) -> (*mut K, *mut V) {
        let leaf = Self::as_leaf_mut(&mut self);
        let keys = MaybeUninit::slice_as_mut_ptr(&mut leaf.keys);
        let vals = MaybeUninit::slice_as_mut_ptr(&mut leaf.vals);
        (keys, vals)
    }
}

impl<BorrowType, K, V> NodeRef<BorrowType, K, V, marker::LeafOrInternal> {
    /// Checks whether a node is an `Internal` node or a `Leaf` node.
    pub fn force(
        self,
    ) -> ForceResult<
        NodeRef<BorrowType, K, V, marker::Leaf>,
        NodeRef<BorrowType, K, V, marker::Internal>,
    > {
        if self.height == 0 {
            ForceResult::Leaf(NodeRef {
                height: self.height,
                node: self.node,
                _marker: PhantomData,
            })
        } else {
            ForceResult::Internal(NodeRef {
                height: self.height,
                node: self.node,
                _marker: PhantomData,
            })
        }
    }
}

/// A reference to a specific key-value pair or edge within a node. The `Node` parameter
/// must be a `NodeRef`, while the `Type` can either be `KV` (signifying a handle on a key-value
/// pair) or `Edge` (signifying a handle on an edge).
///
/// Note that even `Leaf` nodes can have `Edge` handles. Instead of representing a pointer to
/// a child node, these represent the spaces where child pointers would go between the key-value
/// pairs. For example, in a node with length 2, there would be 3 possible edge locations - one
/// to the left of the node, one between the two pairs, and one at the right of the node.
pub struct Handle<Node, Type> {
    node: Node,
    idx: usize,
    _marker: PhantomData<Type>,
}

impl<Node: Copy, Type> Copy for Handle<Node, Type> {}
// We don't need the full generality of `#[derive(Clone)]`, as the only time `Node` will be
// `Clone`able is when it is an immutable reference and therefore `Copy`.
impl<Node: Copy, Type> Clone for Handle<Node, Type> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Node, Type> Handle<Node, Type> {
    /// Retrieves the node that contains the edge or key-value pair this handle points to.
    pub fn into_node(self) -> Node {
        self.node
    }

    /// Returns the position of this handle in the node.
    pub fn idx(&self) -> usize {
        self.idx
    }
}

impl<BorrowType, K, V, NodeType> Handle<NodeRef<BorrowType, K, V, NodeType>, marker::KV> {
    /// Creates a new handle to a key-value pair in `node`.
    /// Unsafe because the caller must ensure that `idx < node.len()`.
    pub unsafe fn new_kv(node: NodeRef<BorrowType, K, V, NodeType>, idx: usize) -> Self {
        debug_assert!(idx < node.len());

        Handle { node, idx, _marker: PhantomData }
    }

    pub fn left_edge(self) -> Handle<NodeRef<BorrowType, K, V, NodeType>, marker::Edge> {
        unsafe { Handle::new_edge(self.node, self.idx) }
    }

    pub fn right_edge(self) -> Handle<NodeRef<BorrowType, K, V, NodeType>, marker::Edge> {
        unsafe { Handle::new_edge(self.node, self.idx + 1) }
    }
}

impl<BorrowType, K, V, NodeType> NodeRef<BorrowType, K, V, NodeType> {
    /// Could be a public implementation of PartialEq, but only used in this module.
    fn eq(&self, other: &Self) -> bool {
        let Self { node, height, _marker } = self;
        if node.eq(&other.node) {
            debug_assert_eq!(*height, other.height);
            true
        } else {
            false
        }
    }
}

impl<BorrowType, K, V, NodeType, HandleType> PartialEq
    for Handle<NodeRef<BorrowType, K, V, NodeType>, HandleType>
{
    fn eq(&self, other: &Self) -> bool {
        let Self { node, idx, _marker } = self;
        node.eq(&other.node) && *idx == other.idx
    }
}

impl<BorrowType, K, V, NodeType, HandleType> PartialOrd
    for Handle<NodeRef<BorrowType, K, V, NodeType>, HandleType>
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let Self { node, idx, _marker } = self;
        if node.eq(&other.node) { Some(idx.cmp(&other.idx)) } else { None }
    }
}

impl<BorrowType, K, V, NodeType, HandleType>
    Handle<NodeRef<BorrowType, K, V, NodeType>, HandleType>
{
    /// Temporarily takes out another, immutable handle on the same location.
    pub fn reborrow(&self) -> Handle<NodeRef<marker::Immut<'_>, K, V, NodeType>, HandleType> {
        // We can't use Handle::new_kv or Handle::new_edge because we don't know our type
        Handle { node: self.node.reborrow(), idx: self.idx, _marker: PhantomData }
    }
}

impl<'a, K, V, NodeType, HandleType> Handle<NodeRef<marker::Mut<'a>, K, V, NodeType>, HandleType> {
    /// Unsafely asserts to the compiler the static information that the handle's node is a `Leaf`.
    pub unsafe fn cast_to_leaf_unchecked(
        self,
    ) -> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, HandleType> {
        let node = unsafe { self.node.cast_to_leaf_unchecked() };
        Handle { node, idx: self.idx, _marker: PhantomData }
    }

    /// Temporarily takes out another, mutable handle on the same location. Beware, as
    /// this method is very dangerous, doubly so since it may not immediately appear
    /// dangerous.
    ///
    /// For details, see `NodeRef::reborrow_mut`.
    pub unsafe fn reborrow_mut(
        &mut self,
    ) -> Handle<NodeRef<marker::Mut<'_>, K, V, NodeType>, HandleType> {
        // We can't use Handle::new_kv or Handle::new_edge because we don't know our type
        Handle { node: unsafe { self.node.reborrow_mut() }, idx: self.idx, _marker: PhantomData }
    }
}

impl<BorrowType, K, V, NodeType> Handle<NodeRef<BorrowType, K, V, NodeType>, marker::Edge> {
    /// Creates a new handle to an edge in `node`.
    /// Unsafe because the caller must ensure that `idx <= node.len()`.
    pub unsafe fn new_edge(node: NodeRef<BorrowType, K, V, NodeType>, idx: usize) -> Self {
        debug_assert!(idx <= node.len());

        Handle { node, idx, _marker: PhantomData }
    }

    pub fn left_kv(self) -> Result<Handle<NodeRef<BorrowType, K, V, NodeType>, marker::KV>, Self> {
        if self.idx > 0 {
            Ok(unsafe { Handle::new_kv(self.node, self.idx - 1) })
        } else {
            Err(self)
        }
    }

    pub fn right_kv(self) -> Result<Handle<NodeRef<BorrowType, K, V, NodeType>, marker::KV>, Self> {
        if self.idx < self.node.len() {
            Ok(unsafe { Handle::new_kv(self.node, self.idx) })
        } else {
            Err(self)
        }
    }
}

pub enum LeftOrRight<T> {
    Left(T),
    Right(T),
}

/// Given an edge index where we want to insert into a node filled to capacity,
/// computes a sensible KV index of a split point and where to perform the insertion.
/// The goal of the split point is for its key and value to end up in a parent node;
/// the keys, values and edges to the left of the split point become the left child;
/// the keys, values and edges to the right of the split point become the right child.
fn splitpoint(edge_idx: usize) -> (usize, LeftOrRight<usize>) {
    debug_assert!(edge_idx <= CAPACITY);
    // Rust issue #74834 tries to explain these symmetric rules.
    match edge_idx {
        0..EDGE_IDX_LEFT_OF_CENTER => (KV_IDX_CENTER - 1, LeftOrRight::Left(edge_idx)),
        EDGE_IDX_LEFT_OF_CENTER => (KV_IDX_CENTER, LeftOrRight::Left(edge_idx)),
        EDGE_IDX_RIGHT_OF_CENTER => (KV_IDX_CENTER, LeftOrRight::Right(0)),
        _ => (KV_IDX_CENTER + 1, LeftOrRight::Right(edge_idx - (KV_IDX_CENTER + 1 + 1))),
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge> {
    /// Inserts a new key-value pair between the key-value pairs to the right and left of
    /// this edge. This method assumes that there is enough space in the node for the new
    /// pair to fit.
    ///
    /// The returned pointer points to the inserted value.
    fn insert_fit(&mut self, key: K, val: V) -> *mut V {
        debug_assert!(self.node.len() < CAPACITY);

        unsafe {
            *self.node.reborrow_mut().into_len_mut() += 1;
            slice_insert(self.node.reborrow_mut().into_key_area_slice(), self.idx, key);
            slice_insert(self.node.reborrow_mut().into_val_area_slice(), self.idx, val);

            self.node.reborrow_mut().into_val_area_mut_at(self.idx).assume_init_mut()
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge> {
    /// Inserts a new key-value pair between the key-value pairs to the right and left of
    /// this edge. This method splits the node if there isn't enough room.
    ///
    /// The returned pointer points to the inserted value.
    fn insert(mut self, key: K, val: V) -> (InsertResult<'a, K, V, marker::Leaf>, *mut V) {
        if self.node.len() < CAPACITY {
            let val_ptr = self.insert_fit(key, val);
            let kv = unsafe { Handle::new_kv(self.node, self.idx) };
            (InsertResult::Fit(kv), val_ptr)
        } else {
            let (middle_kv_idx, insertion) = splitpoint(self.idx);
            let middle = unsafe { Handle::new_kv(self.node, middle_kv_idx) };
            let mut result = middle.split();
            let mut insertion_edge = match insertion {
                LeftOrRight::Left(insert_idx) => unsafe {
                    Handle::new_edge(result.left.reborrow_mut(), insert_idx)
                },
                LeftOrRight::Right(insert_idx) => unsafe {
                    Handle::new_edge(result.right.borrow_mut(), insert_idx)
                },
            };
            let val_ptr = insertion_edge.insert_fit(key, val);
            (InsertResult::Split(result), val_ptr)
        }
    }
}

impl<'a, K, V> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::Edge> {
    /// Fixes the parent pointer and index in the child node that this edge
    /// links to. This is useful when the ordering of edges has been changed,
    fn correct_parent_link(self) {
        // Create backpointer without invalidating other references to the node.
        let ptr = unsafe { NonNull::new_unchecked(NodeRef::as_internal_ptr(&self.node)) };
        let idx = self.idx;
        let mut child = self.descend();
        child.set_parent_link(ptr, idx);
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::Edge> {
    /// Inserts a new key-value pair and an edge that will go to the right of that new pair
    /// between this edge and the key-value pair to the right of this edge. This method assumes
    /// that there is enough space in the node for the new pair to fit.
    fn insert_fit(&mut self, key: K, val: V, edge: Root<K, V>) {
        debug_assert!(self.node.len() < CAPACITY);
        debug_assert!(edge.height == self.node.height - 1);

        unsafe {
            *self.node.reborrow_mut().into_len_mut() += 1;
            slice_insert(self.node.reborrow_mut().into_key_area_slice(), self.idx, key);
            slice_insert(self.node.reborrow_mut().into_val_area_slice(), self.idx, val);
            slice_insert(self.node.reborrow_mut().into_edge_area_slice(), self.idx + 1, edge.node);

            self.node.correct_childrens_parent_links((self.idx + 1)..=self.node.len());
        }
    }

    /// Inserts a new key-value pair and an edge that will go to the right of that new pair
    /// between this edge and the key-value pair to the right of this edge. This method splits
    /// the node if there isn't enough room.
    fn insert(
        mut self,
        key: K,
        val: V,
        edge: Root<K, V>,
    ) -> InsertResult<'a, K, V, marker::Internal> {
        assert!(edge.height == self.node.height - 1);

        if self.node.len() < CAPACITY {
            self.insert_fit(key, val, edge);
            let kv = unsafe { Handle::new_kv(self.node, self.idx) };
            InsertResult::Fit(kv)
        } else {
            let (middle_kv_idx, insertion) = splitpoint(self.idx);
            let middle = unsafe { Handle::new_kv(self.node, middle_kv_idx) };
            let mut result = middle.split();
            let mut insertion_edge = match insertion {
                LeftOrRight::Left(insert_idx) => unsafe {
                    Handle::new_edge(result.left.reborrow_mut(), insert_idx)
                },
                LeftOrRight::Right(insert_idx) => unsafe {
                    Handle::new_edge(result.right.borrow_mut(), insert_idx)
                },
            };
            insertion_edge.insert_fit(key, val, edge);
            InsertResult::Split(result)
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge> {
    /// Inserts a new key-value pair between the key-value pairs to the right and left of
    /// this edge. This method splits the node if there isn't enough room, and tries to
    /// insert the split off portion into the parent node recursively, until the root is reached.
    ///
    /// If the returned result is a `Fit`, its handle's node can be this edge's node or an ancestor.
    /// If the returned result is a `Split`, the `left` field will be the root node.
    /// The returned pointer points to the inserted value.
    pub fn insert_recursing(
        self,
        key: K,
        value: V,
    ) -> (InsertResult<'a, K, V, marker::LeafOrInternal>, *mut V) {
        let (mut split, val_ptr) = match self.insert(key, value) {
            (InsertResult::Fit(handle), ptr) => {
                return (InsertResult::Fit(handle.forget_node_type()), ptr);
            }
            (InsertResult::Split(split), val_ptr) => (split.forget_node_type(), val_ptr),
        };

        loop {
            split = match split.left.ascend() {
                Ok(parent) => match parent.insert(split.kv.0, split.kv.1, split.right) {
                    InsertResult::Fit(handle) => {
                        return (InsertResult::Fit(handle.forget_node_type()), val_ptr);
                    }
                    InsertResult::Split(split) => split.forget_node_type(),
                },
                Err(root) => {
                    return (InsertResult::Split(SplitResult { left: root, ..split }), val_ptr);
                }
            };
        }
    }
}

impl<BorrowType, K, V> Handle<NodeRef<BorrowType, K, V, marker::Internal>, marker::Edge> {
    /// Finds the node pointed to by this edge.
    ///
    /// The method name assumes you picture trees with the root node on top.
    ///
    /// `edge.descend().ascend().unwrap()` and `node.ascend().unwrap().descend()` should
    /// both, upon success, do nothing.
    pub fn descend(self) -> NodeRef<BorrowType, K, V, marker::LeafOrInternal> {
        // We need to use raw pointers to nodes because, if BorrowType is
        // marker::ValMut, there might be outstanding mutable references to
        // values that we must not invalidate. There's no worry accessing the
        // height field because that value is copied. Beware that, once the
        // node pointer is dereferenced, we access the edges array with a
        // reference (Rust issue #73987) and invalidate any other references
        // to or inside the array, should any be around.
        let parent_ptr = NodeRef::as_internal_ptr(&self.node);
        let node = unsafe { (*parent_ptr).edges.get_unchecked(self.idx).assume_init_read() };
        NodeRef { node, height: self.node.height - 1, _marker: PhantomData }
    }
}

impl<'a, K: 'a, V: 'a, NodeType> Handle<NodeRef<marker::Immut<'a>, K, V, NodeType>, marker::KV> {
    pub fn into_kv(self) -> (&'a K, &'a V) {
        (unsafe { self.node.key_at(self.idx) }, unsafe { self.node.val_at(self.idx) })
    }
}

impl<'a, K: 'a, V: 'a, NodeType> Handle<NodeRef<marker::Mut<'a>, K, V, NodeType>, marker::KV> {
    pub fn into_key_mut(self) -> &'a mut K {
        unsafe { self.node.into_key_area_mut_at(self.idx).assume_init_mut() }
    }

    pub fn into_val_mut(self) -> &'a mut V {
        unsafe { self.node.into_val_area_mut_at(self.idx).assume_init_mut() }
    }
}

impl<'a, K, V, NodeType> Handle<NodeRef<marker::ValMut<'a>, K, V, NodeType>, marker::KV> {
    pub fn into_kv_valmut(self) -> (&'a K, &'a mut V) {
        unsafe { self.node.into_key_val_mut_at(self.idx) }
    }
}

impl<'a, K: 'a, V: 'a, NodeType> Handle<NodeRef<marker::Mut<'a>, K, V, NodeType>, marker::KV> {
    pub fn kv_mut(&mut self) -> (&mut K, &mut V) {
        // We cannot call separate key and value methods, because calling the second one
        // invalidates the reference returned by the first.
        unsafe {
            let leaf = NodeRef::as_leaf_mut(&mut self.node.reborrow_mut());
            let key = leaf.keys.get_unchecked_mut(self.idx).assume_init_mut();
            let val = leaf.vals.get_unchecked_mut(self.idx).assume_init_mut();
            (key, val)
        }
    }
}

impl<'a, K: 'a, V: 'a, NodeType> Handle<NodeRef<marker::Mut<'a>, K, V, NodeType>, marker::KV> {
    /// Helps implementations of `split` for a particular `NodeType`,
    /// by calculating the length of the new node.
    fn split_new_node_len(&self) -> usize {
        debug_assert!(self.idx < self.node.len());
        self.node.len() - self.idx - 1
    }

    /// Helps implementations of `split` for a particular `NodeType`,
    /// by taking care of leaf data.
    fn split_leaf_data(&mut self, new_node: &mut LeafNode<K, V>) -> (K, V) {
        let new_len = self.split_new_node_len();
        new_node.len = new_len as u16;
        unsafe {
            let k = ptr::read(self.node.reborrow().key_at(self.idx));
            let v = ptr::read(self.node.reborrow().val_at(self.idx));

            ptr::copy_nonoverlapping(
                self.node.reborrow().key_area().as_ptr().add(self.idx + 1),
                new_node.keys.as_mut_ptr(),
                new_len,
            );
            ptr::copy_nonoverlapping(
                self.node.reborrow().val_area().as_ptr().add(self.idx + 1),
                new_node.vals.as_mut_ptr(),
                new_len,
            );

            *self.node.reborrow_mut().into_len_mut() = self.idx as u16;
            (k, v)
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::KV> {
    /// Splits the underlying node into three parts:
    ///
    /// - The node is truncated to only contain the key-value pairs to the left of
    ///   this handle.
    /// - The key and value pointed to by this handle are extracted.
    /// - All the key-value pairs to the right of this handle are put into a newly
    ///   allocated node.
    pub fn split(mut self) -> SplitResult<'a, K, V, marker::Leaf> {
        unsafe {
            let mut new_node = Box::new(LeafNode::new());

            let kv = self.split_leaf_data(&mut new_node);

            let right = NodeRef::from_new_leaf(new_node);
            SplitResult { left: self.node, kv, right }
        }
    }

    /// Removes the key-value pair pointed to by this handle and returns it, along with the edge
    /// that the key-value pair collapsed into.
    pub fn remove(
        mut self,
    ) -> ((K, V), Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>) {
        unsafe {
            let k = slice_remove(self.node.reborrow_mut().into_key_area_slice(), self.idx);
            let v = slice_remove(self.node.reborrow_mut().into_val_area_slice(), self.idx);
            *self.node.reborrow_mut().into_len_mut() -= 1;
            ((k, v), self.left_edge())
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::KV> {
    /// Splits the underlying node into three parts:
    ///
    /// - The node is truncated to only contain the edges and key-value pairs to the
    ///   left of this handle.
    /// - The key and value pointed to by this handle are extracted.
    /// - All the edges and key-value pairs to the right of this handle are put into
    ///   a newly allocated node.
    pub fn split(mut self) -> SplitResult<'a, K, V, marker::Internal> {
        unsafe {
            let mut new_node = Box::new(InternalNode::new());
            let new_len = self.split_new_node_len();
            // Move edges out before reducing length:
            ptr::copy_nonoverlapping(
                self.node.reborrow().edge_area().as_ptr().add(self.idx + 1),
                new_node.edges.as_mut_ptr(),
                new_len + 1,
            );
            let kv = self.split_leaf_data(&mut new_node.data);

            let height = self.node.height;
            let mut right = NodeRef::from_new_internal(new_node, height);

            right.borrow_mut().correct_childrens_parent_links(0..=new_len);

            SplitResult { left: self.node, kv, right }
        }
    }
}

/// Represents a session for evaluating and performing a balancing operation
/// around an internal key-value pair.
pub struct BalancingContext<'a, K, V> {
    parent: Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::KV>,
    left_child: NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>,
    right_child: NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>,
}

impl<'a, K, V> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::KV> {
    pub fn consider_for_balancing(self) -> BalancingContext<'a, K, V> {
        let self1 = unsafe { ptr::read(&self) };
        let self2 = unsafe { ptr::read(&self) };
        BalancingContext {
            parent: self,
            left_child: self1.left_edge().descend(),
            right_child: self2.right_edge().descend(),
        }
    }
}

impl<'a, K, V> NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal> {
    /// Chooses a balancing context involving the node as a child, thus between
    /// the KV immediately to the left or to the right in the parent node.
    /// Returns an `Err` if there is no parent.
    ///
    /// This method optimizes for a node that has fewer elements than its left
    /// and right siblings, if they exist, by preferring the left parent KV.
    /// Merging with the left sibling is faster, since we only need to move
    /// the node's N elements, instead of shifting them to the right and moving
    /// more than N elements in front. Stealing from the left sibling is also
    /// typically faster, since we only need to shift the node's N elements to
    /// the right, instead of shifting at least N of the sibling's elements to
    /// the left.
    pub fn choose_parent_kv(self) -> Result<LeftOrRight<BalancingContext<'a, K, V>>, Self> {
        match unsafe { ptr::read(&self) }.ascend() {
            Ok(parent) => match parent.left_kv() {
                Ok(left_parent_kv) => Ok(LeftOrRight::Left(BalancingContext {
                    parent: unsafe { ptr::read(&left_parent_kv) },
                    left_child: left_parent_kv.left_edge().descend(),
                    right_child: self,
                })),
                Err(parent) => match parent.right_kv() {
                    Ok(right_parent_kv) => Ok(LeftOrRight::Right(BalancingContext {
                        parent: unsafe { ptr::read(&right_parent_kv) },
                        left_child: self,
                        right_child: right_parent_kv.right_edge().descend(),
                    })),
                    Err(_) => unreachable!("empty non-root node"),
                },
            },
            Err(root) => Err(root),
        }
    }
}

impl<'a, K, V> BalancingContext<'a, K, V> {
    pub fn left_child_len(&self) -> usize {
        self.left_child.len()
    }

    pub fn right_child_len(&self) -> usize {
        self.right_child.len()
    }

    pub fn into_left_child(self) -> NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal> {
        self.left_child
    }

    pub fn into_right_child(self) -> NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal> {
        self.right_child
    }

    /// Returns `true` if it is valid to call `.merge()` in the balancing context,
    /// i.e., whether there is enough room in a node to hold the combination of
    /// both adjacent child nodes, along with the key-value pair in the parent.
    pub fn can_merge(&self) -> bool {
        self.left_child.len() + 1 + self.right_child.len() <= CAPACITY
    }
}

impl<'a, K: 'a, V: 'a> BalancingContext<'a, K, V> {
    /// Merges the parent's key-value pair and both adjacent child nodes into
    /// the left node and returns an edge handle in that expanded left node.
    /// If `track_edge_idx` is given some value, the returned edge corresponds
    /// to where the edge in that child node ended up,
    ///
    /// Panics unless we `.can_merge()`.
    pub fn merge(
        mut self,
        track_edge_idx: Option<LeftOrRight<usize>>,
    ) -> Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::Edge> {
        let mut left_node = self.left_child;
        let left_len = left_node.len();
        let right_node = self.right_child;
        let right_len = right_node.len();

        assert!(left_len + right_len < CAPACITY);
        assert!(match track_edge_idx {
            None => true,
            Some(LeftOrRight::Left(idx)) => idx <= left_len,
            Some(LeftOrRight::Right(idx)) => idx <= right_len,
        });

        unsafe {
            *left_node.reborrow_mut().into_len_mut() += right_len as u16 + 1;

            let parent_key = slice_remove(
                self.parent.node.reborrow_mut().into_key_area_slice(),
                self.parent.idx,
            );
            left_node.reborrow_mut().into_key_area_mut_at(left_len).write(parent_key);
            ptr::copy_nonoverlapping(
                right_node.reborrow().key_area().as_ptr(),
                left_node.reborrow_mut().into_key_area_slice().as_mut_ptr().add(left_len + 1),
                right_len,
            );

            let parent_val = slice_remove(
                self.parent.node.reborrow_mut().into_val_area_slice(),
                self.parent.idx,
            );
            left_node.reborrow_mut().into_val_area_mut_at(left_len).write(parent_val);
            ptr::copy_nonoverlapping(
                right_node.reborrow().val_area().as_ptr(),
                left_node.reborrow_mut().into_val_area_slice().as_mut_ptr().add(left_len + 1),
                right_len,
            );

            slice_remove(
                &mut self.parent.node.reborrow_mut().into_edge_area_slice(),
                self.parent.idx + 1,
            );
            let parent_old_len = self.parent.node.len();
            self.parent.node.correct_childrens_parent_links(self.parent.idx + 1..parent_old_len);
            *self.parent.node.reborrow_mut().into_len_mut() -= 1;

            if self.parent.node.height > 1 {
                // SAFETY: the height of the nodes being merged is one below the height
                // of the node of this edge, thus above zero, so they are internal.
                let mut left_node = left_node.reborrow_mut().cast_to_internal_unchecked();
                let right_node = right_node.cast_to_internal_unchecked();
                ptr::copy_nonoverlapping(
                    right_node.reborrow().edge_area().as_ptr(),
                    left_node.reborrow_mut().into_edge_area_slice().as_mut_ptr().add(left_len + 1),
                    right_len + 1,
                );

                left_node.correct_childrens_parent_links(left_len + 1..=left_len + 1 + right_len);

                Global.dealloc(right_node.node.cast(), Layout::new::<InternalNode<K, V>>());
            } else {
                Global.dealloc(right_node.node.cast(), Layout::new::<LeafNode<K, V>>());
            }

            let new_idx = match track_edge_idx {
                None => 0,
                Some(LeftOrRight::Left(idx)) => idx,
                Some(LeftOrRight::Right(idx)) => left_len + 1 + idx,
            };
            Handle::new_edge(left_node, new_idx)
        }
    }

    /// Removes a key-value pair from the left child and places it in the key-value storage
    /// of the parent, while pushing the old parent key-value pair into the right child.
    /// Returns a handle to the edge in the right child corresponding to where the original
    /// edge specified by `track_right_edge_idx` ended up.
    pub fn steal_left(
        mut self,
        track_right_edge_idx: usize,
    ) -> Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::Edge> {
        unsafe {
            let (k, v, edge) = self.left_child.pop();

            let k = mem::replace(self.parent.kv_mut().0, k);
            let v = mem::replace(self.parent.kv_mut().1, v);

            match self.right_child.reborrow_mut().force() {
                ForceResult::Leaf(mut leaf) => leaf.push_front(k, v),
                ForceResult::Internal(mut internal) => internal.push_front(k, v, edge.unwrap()),
            }

            Handle::new_edge(self.right_child, 1 + track_right_edge_idx)
        }
    }

    /// Removes a key-value pair from the right child and places it in the key-value storage
    /// of the parent, while pushing the old parent key-value pair onto the left child.
    /// Returns a handle to the edge in the left child specified by `track_left_edge_idx`,
    /// which didn't move.
    pub fn steal_right(
        mut self,
        track_left_edge_idx: usize,
    ) -> Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::Edge> {
        unsafe {
            let (k, v, edge) = self.right_child.pop_front();

            let k = mem::replace(self.parent.kv_mut().0, k);
            let v = mem::replace(self.parent.kv_mut().1, v);

            match self.left_child.reborrow_mut().force() {
                ForceResult::Leaf(mut leaf) => leaf.push(k, v),
                ForceResult::Internal(mut internal) => internal.push(k, v, edge.unwrap()),
            }

            Handle::new_edge(self.left_child, track_left_edge_idx)
        }
    }

    /// This does stealing similar to `steal_left` but steals multiple elements at once.
    pub fn bulk_steal_left(&mut self, count: usize) {
        unsafe {
            let left_node = &mut self.left_child;
            let old_left_len = left_node.len();
            let right_node = &mut self.right_child;
            let old_right_len = right_node.len();

            // Make sure that we may steal safely.
            assert!(old_right_len + count <= CAPACITY);
            assert!(old_left_len >= count);

            let new_left_len = old_left_len - count;

            // Move leaf data.
            {
                let left_kv = left_node.reborrow_mut().into_kv_pointers_mut();
                let right_kv = right_node.reborrow_mut().into_kv_pointers_mut();
                let parent_kv = {
                    let kv = self.parent.kv_mut();
                    (kv.0 as *mut K, kv.1 as *mut V)
                };

                // Make room for stolen elements in the right child.
                ptr::copy(right_kv.0, right_kv.0.add(count), old_right_len);
                ptr::copy(right_kv.1, right_kv.1.add(count), old_right_len);

                // Move elements from the left child to the right one.
                move_kv(left_kv, new_left_len + 1, right_kv, 0, count - 1);

                // Move parent's key-value pair to the right child.
                move_kv(parent_kv, 0, right_kv, count - 1, 1);

                // Move the left-most stolen pair to the parent.
                move_kv(left_kv, new_left_len, parent_kv, 0, 1);
            }

            *left_node.reborrow_mut().into_len_mut() -= count as u16;
            *right_node.reborrow_mut().into_len_mut() += count as u16;

            match (left_node.reborrow_mut().force(), right_node.reborrow_mut().force()) {
                (ForceResult::Internal(left), ForceResult::Internal(mut right)) => {
                    // Make room for stolen edges.
                    let left = left.reborrow();
                    let right_edges = right.reborrow_mut().into_edge_area_slice().as_mut_ptr();
                    ptr::copy(right_edges, right_edges.add(count), old_right_len + 1);
                    right.correct_childrens_parent_links(count..count + old_right_len + 1);

                    // Steal edges.
                    move_edges(left, new_left_len + 1, right, 0, count);
                }
                (ForceResult::Leaf(_), ForceResult::Leaf(_)) => {}
                _ => unreachable!(),
            }
        }
    }

    /// The symmetric clone of `bulk_steal_left`.
    pub fn bulk_steal_right(&mut self, count: usize) {
        unsafe {
            let left_node = &mut self.left_child;
            let old_left_len = left_node.len();
            let right_node = &mut self.right_child;
            let old_right_len = right_node.len();

            // Make sure that we may steal safely.
            assert!(old_left_len + count <= CAPACITY);
            assert!(old_right_len >= count);

            let new_right_len = old_right_len - count;

            // Move leaf data.
            {
                let left_kv = left_node.reborrow_mut().into_kv_pointers_mut();
                let right_kv = right_node.reborrow_mut().into_kv_pointers_mut();
                let parent_kv = {
                    let kv = self.parent.kv_mut();
                    (kv.0 as *mut K, kv.1 as *mut V)
                };

                // Move parent's key-value pair to the left child.
                move_kv(parent_kv, 0, left_kv, old_left_len, 1);

                // Move elements from the right child to the left one.
                move_kv(right_kv, 0, left_kv, old_left_len + 1, count - 1);

                // Move the right-most stolen pair to the parent.
                move_kv(right_kv, count - 1, parent_kv, 0, 1);

                // Fill gap where stolen elements used to be.
                ptr::copy(right_kv.0.add(count), right_kv.0, new_right_len);
                ptr::copy(right_kv.1.add(count), right_kv.1, new_right_len);
            }

            *left_node.reborrow_mut().into_len_mut() += count as u16;
            *right_node.reborrow_mut().into_len_mut() -= count as u16;

            match (left_node.reborrow_mut().force(), right_node.reborrow_mut().force()) {
                (ForceResult::Internal(left), ForceResult::Internal(mut right)) => {
                    // Steal edges.
                    move_edges(right.reborrow(), 0, left, old_left_len + 1, count);

                    // Fill gap where stolen edges used to be.
                    let right_edges = right.reborrow_mut().into_edge_area_slice().as_mut_ptr();
                    ptr::copy(right_edges.add(count), right_edges, new_right_len + 1);
                    right.correct_childrens_parent_links(0..=new_right_len);
                }
                (ForceResult::Leaf(_), ForceResult::Leaf(_)) => {}
                _ => unreachable!(),
            }
        }
    }
}

unsafe fn move_kv<K, V>(
    source: (*mut K, *mut V),
    source_offset: usize,
    dest: (*mut K, *mut V),
    dest_offset: usize,
    count: usize,
) {
    unsafe {
        ptr::copy_nonoverlapping(source.0.add(source_offset), dest.0.add(dest_offset), count);
        ptr::copy_nonoverlapping(source.1.add(source_offset), dest.1.add(dest_offset), count);
    }
}

// Source and destination must have the same height.
unsafe fn move_edges<'a, K: 'a, V: 'a>(
    source: NodeRef<marker::Immut<'a>, K, V, marker::Internal>,
    source_offset: usize,
    mut dest: NodeRef<marker::Mut<'a>, K, V, marker::Internal>,
    dest_offset: usize,
    count: usize,
) {
    unsafe {
        let source_ptr = source.edge_area().as_ptr();
        let dest_ptr = dest.reborrow_mut().into_edge_area_slice().as_mut_ptr();
        ptr::copy_nonoverlapping(source_ptr.add(source_offset), dest_ptr.add(dest_offset), count);
        dest.correct_childrens_parent_links(dest_offset..dest_offset + count);
    }
}

impl<BorrowType, K, V> NodeRef<BorrowType, K, V, marker::Leaf> {
    /// Removes any static information asserting that this node is a `Leaf` node.
    pub fn forget_type(self) -> NodeRef<BorrowType, K, V, marker::LeafOrInternal> {
        NodeRef { height: self.height, node: self.node, _marker: PhantomData }
    }
}

impl<BorrowType, K, V> NodeRef<BorrowType, K, V, marker::Internal> {
    /// Removes any static information asserting that this node is an `Internal` node.
    pub fn forget_type(self) -> NodeRef<BorrowType, K, V, marker::LeafOrInternal> {
        NodeRef { height: self.height, node: self.node, _marker: PhantomData }
    }
}

impl<BorrowType, K, V> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::Edge> {
    pub fn forget_node_type(
        self,
    ) -> Handle<NodeRef<BorrowType, K, V, marker::LeafOrInternal>, marker::Edge> {
        unsafe { Handle::new_edge(self.node.forget_type(), self.idx) }
    }
}

impl<BorrowType, K, V> Handle<NodeRef<BorrowType, K, V, marker::Internal>, marker::Edge> {
    pub fn forget_node_type(
        self,
    ) -> Handle<NodeRef<BorrowType, K, V, marker::LeafOrInternal>, marker::Edge> {
        unsafe { Handle::new_edge(self.node.forget_type(), self.idx) }
    }
}

impl<BorrowType, K, V> Handle<NodeRef<BorrowType, K, V, marker::Leaf>, marker::KV> {
    pub fn forget_node_type(
        self,
    ) -> Handle<NodeRef<BorrowType, K, V, marker::LeafOrInternal>, marker::KV> {
        unsafe { Handle::new_kv(self.node.forget_type(), self.idx) }
    }
}

impl<BorrowType, K, V> Handle<NodeRef<BorrowType, K, V, marker::Internal>, marker::KV> {
    pub fn forget_node_type(
        self,
    ) -> Handle<NodeRef<BorrowType, K, V, marker::LeafOrInternal>, marker::KV> {
        unsafe { Handle::new_kv(self.node.forget_type(), self.idx) }
    }
}

impl<BorrowType, K, V, HandleType>
    Handle<NodeRef<BorrowType, K, V, marker::LeafOrInternal>, HandleType>
{
    /// Checks whether the underlying node is an `Internal` node or a `Leaf` node.
    pub fn force(
        self,
    ) -> ForceResult<
        Handle<NodeRef<BorrowType, K, V, marker::Leaf>, HandleType>,
        Handle<NodeRef<BorrowType, K, V, marker::Internal>, HandleType>,
    > {
        match self.node.force() {
            ForceResult::Leaf(node) => {
                ForceResult::Leaf(Handle { node, idx: self.idx, _marker: PhantomData })
            }
            ForceResult::Internal(node) => {
                ForceResult::Internal(Handle { node, idx: self.idx, _marker: PhantomData })
            }
        }
    }
}

impl<'a, K, V> Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::Edge> {
    /// Move the suffix after `self` from one node to another one. `right` must be empty.
    /// The first edge of `right` remains unchanged.
    pub fn move_suffix(
        &mut self,
        right: &mut NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>,
    ) {
        unsafe {
            let new_left_len = self.idx;
            let mut left_node = self.reborrow_mut().into_node();

            let new_right_len = left_node.len() - new_left_len;
            let mut right_node = right.reborrow_mut();

            assert!(right_node.len() == 0);
            assert!(left_node.height == right_node.height);

            if new_right_len > 0 {
                let left_kv = left_node.reborrow_mut().into_kv_pointers_mut();
                let right_kv = right_node.reborrow_mut().into_kv_pointers_mut();

                move_kv(left_kv, new_left_len, right_kv, 0, new_right_len);

                *left_node.reborrow_mut().into_len_mut() = new_left_len as u16;
                *right_node.reborrow_mut().into_len_mut() = new_right_len as u16;

                match (left_node.force(), right_node.force()) {
                    (ForceResult::Internal(left), ForceResult::Internal(right)) => {
                        let left = left.reborrow();
                        move_edges(left, new_left_len + 1, right, 1, new_right_len);
                    }
                    (ForceResult::Leaf(_), ForceResult::Leaf(_)) => {}
                    _ => unreachable!(),
                }
            }
        }
    }
}

pub enum ForceResult<Leaf, Internal> {
    Leaf(Leaf),
    Internal(Internal),
}

/// Result of insertion, when a node needed to expand beyond its capacity.
pub struct SplitResult<'a, K, V, NodeType> {
    // Altered node in existing tree with elements and edges that belong to the left of `kv`.
    pub left: NodeRef<marker::Mut<'a>, K, V, NodeType>,
    // Some key and value split off, to be inserted elsewhere.
    pub kv: (K, V),
    // Owned, unattached, new node with elements and edges that belong to the right of `kv`.
    pub right: NodeRef<marker::Owned, K, V, NodeType>,
}

impl<'a, K, V> SplitResult<'a, K, V, marker::Leaf> {
    pub fn forget_node_type(self) -> SplitResult<'a, K, V, marker::LeafOrInternal> {
        SplitResult { left: self.left.forget_type(), kv: self.kv, right: self.right.forget_type() }
    }
}

impl<'a, K, V> SplitResult<'a, K, V, marker::Internal> {
    pub fn forget_node_type(self) -> SplitResult<'a, K, V, marker::LeafOrInternal> {
        SplitResult { left: self.left.forget_type(), kv: self.kv, right: self.right.forget_type() }
    }
}

pub enum InsertResult<'a, K, V, NodeType> {
    Fit(Handle<NodeRef<marker::Mut<'a>, K, V, NodeType>, marker::KV>),
    Split(SplitResult<'a, K, V, NodeType>),
}

pub mod marker {
    use core::marker::PhantomData;

    pub enum Leaf {}
    pub enum Internal {}
    pub enum LeafOrInternal {}

    pub enum Owned {}
    pub struct Immut<'a>(PhantomData<&'a ()>);
    pub struct Mut<'a>(PhantomData<&'a mut ()>);
    pub struct ValMut<'a>(PhantomData<&'a mut ()>);

    pub enum KV {}
    pub enum Edge {}
}

/// Inserts a value into a slice of initialized elements followed by one uninitialized element.
///
/// # Safety
/// The slice has more than `idx` elements.
unsafe fn slice_insert<T>(slice: &mut [MaybeUninit<T>], idx: usize, val: T) {
    unsafe {
        let len = slice.len();
        debug_assert!(len > idx);
        let slice_ptr = slice.as_mut_ptr();
        if len > idx + 1 {
            ptr::copy(slice_ptr.add(idx), slice_ptr.add(idx + 1), len - idx - 1);
        }
        (*slice_ptr.add(idx)).write(val);
    }
}

/// Removes and returns a value from a slice of all initialized elements, leaving behind one
/// trailing uninitialized element.
///
/// # Safety
/// The slice has more than `idx` elements.
unsafe fn slice_remove<T>(slice: &mut [MaybeUninit<T>], idx: usize) -> T {
    unsafe {
        let len = slice.len();
        debug_assert!(idx < len);
        let slice_ptr = slice.as_mut_ptr();
        let ret = (*slice_ptr.add(idx)).assume_init_read();
        ptr::copy(slice_ptr.add(idx + 1), slice_ptr.add(idx), len - idx - 1);
        ret
    }
}

#[cfg(test)]
mod tests;
