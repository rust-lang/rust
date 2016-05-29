// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
//     edges: if height > 0 {
//         [Box<Node<K, V, height - 1>>; 2 * B]
//     } else { () },
//     parent: *const Node<K, V, height + 1>,
//     parent_idx: u16,
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
// - A node of length `n` has `n` keys, `n` values, and (in an internal node) `n + 1` edges.
//   This implies that even an empty internal node has at least one edge.

use alloc::heap;
use core::marker::PhantomData;
use core::mem;
use core::nonzero::NonZero;
use core::ptr::{self, Unique};
use core::slice;

use boxed::Box;

const B: usize = 6;
pub const MIN_LEN: usize = B - 1;
pub const CAPACITY: usize = 2 * B - 1;

/// The underlying representation of leaf nodes. Note that it is often unsafe to actually store
/// these, since only the first `len` keys and values are assumed to be initialized. As such,
/// these should always be put behind pointers, and specifically behind `BoxedNode` in the owned
/// case.
///
/// See also rust-lang/rfcs#197, which would make this structure significantly more safe by
/// avoiding accidentally dropping unused and uninitialized keys and values.
struct LeafNode<K, V> {
    /// The arrays storing the actual data of the node. Only the first `len` elements of each
    /// array are initialized and valid.
    keys: [K; CAPACITY],
    vals: [V; CAPACITY],

    /// We use `*const` as opposed to `*mut` so as to be covariant in `K` and `V`.
    /// This either points to an actual node or is null.
    parent: *const InternalNode<K, V>,

    /// This node's index into the parent node's `edges` array.
    /// `*node.parent.edges[node.parent_idx]` should be the same thing as `node`.
    /// This is only guaranteed to be initialized when `parent` is nonnull.
    parent_idx: u16,

    /// The number of keys and values this node stores.
    ///
    /// This is at the end of the node's representation and next to `parent_idx` to encourage
    /// the compiler to join `len` and `parent_idx` into the same 32-bit word, reducing space
    /// overhead.
    len: u16,
}

impl<K, V> LeafNode<K, V> {
    /// Creates a new `LeafNode`. Unsafe because all nodes should really be hidden behind
    /// `BoxedNode`, preventing accidental dropping of uninitialized keys and values.
    unsafe fn new() -> Self {
        LeafNode {
            // As a general policy, we leave fields uninitialized if they can be, as this should
            // be both slightly faster and easier to track in Valgrind.
            keys: mem::uninitialized(),
            vals: mem::uninitialized(),
            parent: ptr::null(),
            parent_idx: mem::uninitialized(),
            len: 0
        }
    }
}

/// The underlying representation of internal nodes. As with `LeafNode`s, these should be hidden
/// behind `BoxedNode`s to prevent dropping uninitialized keys and values. Any pointer to an
/// `InternalNode` can be directly casted to a pointer to the underlying `LeafNode` portion of the
/// node, allowing code to act on leaf and internal nodes generically without having to even check
/// which of the two a pointer is pointing at. This property is enabled by the use of `repr(C)`.
#[repr(C)]
struct InternalNode<K, V> {
    data: LeafNode<K, V>,

    /// The pointers to the children of this node. `len + 1` of these are considered
    /// initialized and valid.
    edges: [BoxedNode<K, V>; 2 * B],
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
        InternalNode {
            data: LeafNode::new(),
            edges: mem::uninitialized()
        }
    }
}

/// An owned pointer to a node. This basically is either `Box<LeafNode<K, V>>` or
/// `Box<InternalNode<K, V>>`. However, it contains no information as to which of the two types
/// of nodes is acutally behind the box, and, partially due to this lack of information, has no
/// destructor.
struct BoxedNode<K, V> {
    ptr: Unique<LeafNode<K, V>>
}

impl<K, V> BoxedNode<K, V> {
    fn from_leaf(node: Box<LeafNode<K, V>>) -> Self {
        unsafe {
            BoxedNode { ptr: Unique::new(Box::into_raw(node)) }
        }
    }

    fn from_internal(node: Box<InternalNode<K, V>>) -> Self {
        unsafe {
            BoxedNode { ptr: Unique::new(Box::into_raw(node) as *mut LeafNode<K, V>) }
        }
    }

    unsafe fn from_ptr(ptr: NonZero<*const LeafNode<K, V>>) -> Self {
        BoxedNode { ptr: Unique::new(*ptr as *mut LeafNode<K, V>) }
    }

    fn as_ptr(&self) -> NonZero<*const LeafNode<K, V>> {
        unsafe {
            NonZero::new(*self.ptr as *const LeafNode<K, V>)
        }
    }
}

/// An owned tree. Note that despite being owned, this does not have a destructor,
/// and must be cleaned up manually.
pub struct Root<K, V> {
    node: BoxedNode<K, V>,
    height: usize
}

unsafe impl<K: Sync, V: Sync> Sync for Root<K, V> { }
unsafe impl<K: Send, V: Send> Send for Root<K, V> { }

impl<K, V> Root<K, V> {
    pub fn new_leaf() -> Self {
        Root {
            node: BoxedNode::from_leaf(Box::new(unsafe { LeafNode::new() })),
            height: 0
        }
    }

    pub fn as_ref(&self)
            -> NodeRef<marker::Immut, K, V, marker::LeafOrInternal> {
        NodeRef {
            height: self.height,
            node: self.node.as_ptr(),
            root: self as *const _ as *mut _,
            _marker: PhantomData,
        }
    }

    pub fn as_mut(&mut self)
            -> NodeRef<marker::Mut, K, V, marker::LeafOrInternal> {
        NodeRef {
            height: self.height,
            node: self.node.as_ptr(),
            root: self as *mut _,
            _marker: PhantomData,
        }
    }

    pub fn into_ref(self)
            -> NodeRef<marker::Owned, K, V, marker::LeafOrInternal> {
        NodeRef {
            height: self.height,
            node: self.node.as_ptr(),
            root: ptr::null_mut(), // FIXME: Is there anything better to do here?
            _marker: PhantomData,
        }
    }

    /// Adds a new internal node with a single edge, pointing to the previous root, and make that
    /// new node the root. This increases the height by 1 and is the opposite of `pop_level`.
    pub fn push_level(&mut self)
            -> NodeRef<marker::Mut, K, V, marker::Internal> {
        let mut new_node = Box::new(unsafe { InternalNode::new() });
        new_node.edges[0] = unsafe { BoxedNode::from_ptr(self.node.as_ptr()) };

        self.node = BoxedNode::from_internal(new_node);
        self.height += 1;

        let mut ret = NodeRef {
            height: self.height,
            node: self.node.as_ptr(),
            root: self as *mut _,
            _marker: PhantomData
        };

        unsafe {
            ret.reborrow_mut().first_edge().correct_parent_link();
        }

        ret
    }

    /// Removes the root node, using its first child as the new root. This cannot be called when
    /// the tree consists only of a leaf node. As it is intended only to be called when the root
    /// has only one edge, no cleanup is done on any of the other children are elements of the root.
    /// This decreases the height by 1 and is the opposite of `push_level`.
    pub fn pop_level(&mut self) {
        debug_assert!(self.height > 0);

        let top = *self.node.ptr as *mut u8;

        self.node = unsafe {
            BoxedNode::from_ptr(self.as_mut()
                                    .cast_unchecked::<marker::Internal>()
                                    .first_edge()
                                    .descend()
                                    .node)
        };
        self.height -= 1;
        self.as_mut().as_leaf_mut().parent = ptr::null();

        unsafe {
            heap::deallocate(
                top,
                mem::size_of::<InternalNode<K, V>>(),
                mem::align_of::<InternalNode<K, V>>()
            );
        }
    }
}

// N.B. `NodeRef` is always covariant in `K` and `V`, even when the `BorrowType`
// is `Mut`. This is technically wrong, but cannot result in any unsafety due to
// internal use of `NodeRef` because we stay completely generic over `K` and `V`.
// However, whenever a public type wraps `NodeRef`, make sure that it has the
// correct variance.
/// A reference to a node.
///
/// This type has a number of paramaters that controls how it acts:
/// - `BorrowType`: This can be `Immut<'a>` or `Mut<'a>` for some `'a` or `Owned`.
///    When this is `Immut<'a>`, the `NodeRef` acts roughly like `&'a Node`,
///    when this is `Mut<'a>`, the `NodeRef` acts roughly like `&'a mut Node`,
///    and when this is `Owned`, the `NodeRef` acts roughly like `Box<Node>`.
/// - `K` and `V`: These control what types of things are stored in the nodes.
/// - `Type`: This can be `Leaf`, `Internal`, or `LeafOrInternal`. When this is
///   `Leaf`, the `NodeRef` points to a leaf node, when this is `Internal` the
///   `NodeRef` points to an internal node, and when this is `LeafOrInternal` the
///   `NodeRef` could be pointing to either type of node.
pub struct NodeRef<BorrowType, K, V, Type> {
    height: usize,
    node: NonZero<*const LeafNode<K, V>>,
    // This is null unless the borrow type is `Mut`
    root: *const Root<K, V>,
    _marker: PhantomData<(BorrowType, Type)>
}

impl<'a, K: 'a, V: 'a, Type> Copy for NodeRef<marker::Immut<'a>, K, V, Type> { }
impl<'a, K: 'a, V: 'a, Type> Clone for NodeRef<marker::Immut<'a>, K, V, Type> {
    fn clone(&self) -> Self {
        *self
    }
}

unsafe impl<BorrowType, K: Sync, V: Sync, Type> Sync
    for NodeRef<BorrowType, K, V, Type> { }

unsafe impl<'a, K: Sync + 'a, V: Sync + 'a, Type> Send
   for NodeRef<marker::Immut<'a>, K, V, Type> { }
unsafe impl<'a, K: Send + 'a, V: Send + 'a, Type> Send
   for NodeRef<marker::Mut<'a>, K, V, Type> { }
unsafe impl<K: Send, V: Send, Type> Send
   for NodeRef<marker::Owned, K, V, Type> { }

impl<BorrowType, K, V> NodeRef<BorrowType, K, V, marker::Internal> {
    fn as_internal(&self) -> &InternalNode<K, V> {
        unsafe {
            &*(*self.node as *const InternalNode<K, V>)
        }
    }
}

impl<'a, K, V> NodeRef<marker::Mut<'a>, K, V, marker::Internal> {
    fn as_internal_mut(&mut self) -> &mut InternalNode<K, V> {
        unsafe {
            &mut *(*self.node as *mut InternalNode<K, V>)
        }
    }
}


impl<BorrowType, K, V, Type> NodeRef<BorrowType, K, V, Type> {
    /// Finds the length of the node. This is the number of keys or values. In an
    /// internal node, the number of edges is `len() + 1`.
    pub fn len(&self) -> usize {
        self.as_leaf().len as usize
    }

    /// Returns the height of this node in the whole tree. Zero height denotes the
    /// leaf level.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Removes any static information about whether this node is a `Leaf` or an
    /// `Internal` node.
    pub fn forget_type(self) -> NodeRef<BorrowType, K, V, marker::LeafOrInternal> {
        NodeRef {
            height: self.height,
            node: self.node,
            root: self.root,
            _marker: PhantomData
        }
    }

    /// Temporarily takes out another, immutable reference to the same node.
    fn reborrow<'a>(&'a self) -> NodeRef<marker::Immut<'a>, K, V, Type> {
        NodeRef {
            height: self.height,
            node: self.node,
            root: self.root,
            _marker: PhantomData
        }
    }

    fn as_leaf(&self) -> &LeafNode<K, V> {
        unsafe {
            &**self.node
        }
    }

    pub fn keys(&self) -> &[K] {
        self.reborrow().into_slices().0
    }

    pub fn vals(&self) -> &[V] {
        self.reborrow().into_slices().1
    }

    /// Finds the parent of the current node. Returns `Ok(handle)` if the current
    /// node actually has a parent, where `handle` points to the edge of the parent
    /// that points to the current node. Returns `Err(self)` if the current node has
    /// no parent, giving back the original `NodeRef`.
    ///
    /// `edge.descend().ascend().unwrap()` and `node.ascend().unwrap().descend()` should
    /// both, upon success, do nothing.
    pub fn ascend(self) -> Result<
        Handle<
            NodeRef<
                BorrowType,
                K, V,
                marker::Internal
            >,
            marker::Edge
        >,
        Self
    > {
        if self.as_leaf().parent.is_null() {
            Err(self)
        } else {
            Ok(Handle {
                node: NodeRef {
                    height: self.height + 1,
                    node: unsafe {
                        NonZero::new(self.as_leaf().parent as *mut LeafNode<K, V>)
                    },
                    root: self.root,
                    _marker: PhantomData
                },
                idx: self.as_leaf().parent_idx as usize,
                _marker: PhantomData
            })
        }
    }

    pub fn first_edge(self) -> Handle<Self, marker::Edge> {
        Handle::new_edge(self, 0)
    }

    pub fn last_edge(self) -> Handle<Self, marker::Edge> {
        let len = self.len();
        Handle::new_edge(self, len)
    }

    /// Note that `self` must be nonempty.
    pub fn first_kv(self) -> Handle<Self, marker::KV> {
        debug_assert!(self.len() > 0);
        Handle::new_kv(self, 0)
    }

    /// Note that `self` must be nonempty.
    pub fn last_kv(self) -> Handle<Self, marker::KV> {
        let len = self.len();
        debug_assert!(len > 0);
        Handle::new_kv(self, len - 1)
    }
}

impl<K, V> NodeRef<marker::Owned, K, V, marker::Leaf> {
    /// Similar to `ascend`, gets a reference to a node's parent node, but also
    /// deallocate the current node in the process. This is unsafe because the
    /// current node will still be accessible despite being deallocated.
    pub unsafe fn deallocate_and_ascend(self) -> Option<
        Handle<
            NodeRef<
                marker::Owned,
                K, V,
                marker::Internal
            >,
            marker::Edge
        >
    > {
        let ptr = self.as_leaf() as *const LeafNode<K, V> as *const u8 as *mut u8;
        let ret = self.ascend().ok();
        heap::deallocate(ptr, mem::size_of::<LeafNode<K, V>>(), mem::align_of::<LeafNode<K, V>>());
        ret
    }
}

impl<K, V> NodeRef<marker::Owned, K, V, marker::Internal> {
    /// Similar to `ascend`, gets a reference to a node's parent node, but also
    /// deallocate the current node in the process. This is unsafe because the
    /// current node will still be accessible despite being deallocated.
    pub unsafe fn deallocate_and_ascend(self) -> Option<
        Handle<
            NodeRef<
                marker::Owned,
                K, V,
                marker::Internal
            >,
            marker::Edge
        >
    > {
        let ptr = self.as_internal() as *const InternalNode<K, V> as *const u8 as *mut u8;
        let ret = self.ascend().ok();
        heap::deallocate(
            ptr,
            mem::size_of::<InternalNode<K, V>>(),
            mem::align_of::<InternalNode<K, V>>()
        );
        ret
    }
}

impl<'a, K, V, Type> NodeRef<marker::Mut<'a>, K, V, Type> {
    /// Unsafely asserts to the compiler some static information about whether this
    /// node is a `Leaf`.
    unsafe fn cast_unchecked<NewType>(&mut self)
            -> NodeRef<marker::Mut, K, V, NewType> {

        NodeRef {
            height: self.height,
            node: self.node,
            root: self.root,
            _marker: PhantomData
        }
    }

    /// Temporarily takes out another, mutable reference to the same node. Beware, as
    /// this method is very dangerous, doubly so since it may not immediately appear
    /// dangerous.
    ///
    /// Because mutable pointers can roam anywhere around the tree and can even (through
    /// `into_root_mut`) mess with the root of the tree, the result of `reborrow_mut`
    /// can easily be used to make the original mutable pointer dangling, or, in the case
    /// of a reborrowed handle, out of bounds.
    // FIXME(@gereeter) consider adding yet another type parameter to `NodeRef` that restricts
    // the use of `ascend` and `into_root_mut` on reborrowed pointers, preventing this unsafety.
    unsafe fn reborrow_mut(&mut self) -> NodeRef<marker::Mut, K, V, Type> {
        NodeRef {
            height: self.height,
            node: self.node,
            root: self.root,
            _marker: PhantomData
        }
    }

    fn as_leaf_mut(&mut self) -> &mut LeafNode<K, V> {
        unsafe {
            &mut *(*self.node as *mut LeafNode<K, V>)
        }
    }

    pub fn keys_mut(&mut self) -> &mut [K] {
        unsafe { self.reborrow_mut().into_slices_mut().0 }
    }

    pub fn vals_mut(&mut self) -> &mut [V] {
        unsafe { self.reborrow_mut().into_slices_mut().1 }
    }
}

impl<'a, K: 'a, V: 'a, Type> NodeRef<marker::Immut<'a>, K, V, Type> {
    pub fn into_slices(self) -> (&'a [K], &'a [V]) {
        unsafe {
            (
                slice::from_raw_parts(
                    self.as_leaf().keys.as_ptr(),
                    self.len()
                ),
                slice::from_raw_parts(
                    self.as_leaf().vals.as_ptr(),
                    self.len()
                )
            )
        }
    }
}

impl<'a, K: 'a, V: 'a, Type> NodeRef<marker::Mut<'a>, K, V, Type> {
    /// Gets a mutable reference to the root itself. This is useful primarily when the
    /// height of the tree needs to be adjusted. Never call this on a reborrowed pointer.
    pub fn into_root_mut(self) -> &'a mut Root<K, V> {
        unsafe {
            &mut *(self.root as *mut Root<K, V>)
        }
    }

    pub fn into_slices_mut(mut self) -> (&'a mut [K], &'a mut [V]) {
        unsafe {
            (
                slice::from_raw_parts_mut(
                    &mut self.as_leaf_mut().keys as *mut [K] as *mut K,
                    self.len()
                ),
                slice::from_raw_parts_mut(
                    &mut self.as_leaf_mut().vals as *mut [V] as *mut V,
                    self.len()
                )
            )
        }
    }
}

impl<'a, K, V> NodeRef<marker::Mut<'a>, K, V, marker::Leaf> {
    /// Adds a key/value pair the end of the node.
    pub fn push(&mut self, key: K, val: V) {
        // Necessary for correctness, but this is an internal module
        debug_assert!(self.len() < CAPACITY);

        let idx = self.len();

        unsafe {
            ptr::write(self.keys_mut().get_unchecked_mut(idx), key);
            ptr::write(self.vals_mut().get_unchecked_mut(idx), val);
        }

        self.as_leaf_mut().len += 1;
    }

    /// Adds a key/value pair to the beginning of the node.
    pub fn push_front(&mut self, key: K, val: V) {
        // Necessary for correctness, but this is an internal module
        debug_assert!(self.len() < CAPACITY);

        unsafe {
            slice_insert(self.keys_mut(), 0, key);
            slice_insert(self.vals_mut(), 0, val);
        }

        self.as_leaf_mut().len += 1;
    }
}

impl<'a, K, V> NodeRef<marker::Mut<'a>, K, V, marker::Internal> {
    /// Adds a key/value pair and an edge to go to the right of that pair to
    /// the end of the node.
    pub fn push(&mut self, key: K, val: V, edge: Root<K, V>) {
        // Necessary for correctness, but this is an internal module
        debug_assert!(edge.height == self.height - 1);
        debug_assert!(self.len() < CAPACITY);

        let idx = self.len();

        unsafe {
            ptr::write(self.keys_mut().get_unchecked_mut(idx), key);
            ptr::write(self.vals_mut().get_unchecked_mut(idx), val);
            ptr::write(self.as_internal_mut().edges.get_unchecked_mut(idx + 1), edge.node);

            self.as_leaf_mut().len += 1;

            Handle::new_edge(self.reborrow_mut(), idx + 1).correct_parent_link();
        }
    }

    fn correct_childrens_parent_links(&mut self, first: usize, after_last: usize) {
        for i in first..after_last {
            Handle::new_edge(unsafe { self.reborrow_mut() }, i).correct_parent_link();
        }
    }

    fn correct_all_childrens_parent_links(&mut self) {
        let len = self.len();
        self.correct_childrens_parent_links(0, len + 1);
    }

    /// Adds a key/value pair and an edge to go to the left of that pair to
    /// the beginning of the node.
    pub fn push_front(&mut self, key: K, val: V, edge: Root<K, V>) {
        // Necessary for correctness, but this is an internal module
        debug_assert!(edge.height == self.height - 1);
        debug_assert!(self.len() < CAPACITY);

        unsafe {
            slice_insert(self.keys_mut(), 0, key);
            slice_insert(self.vals_mut(), 0, val);
            slice_insert(
                slice::from_raw_parts_mut(
                    self.as_internal_mut().edges.as_mut_ptr(),
                    self.len()+1
                ),
                0,
                edge.node
            );

            self.as_leaf_mut().len += 1;

            self.correct_all_childrens_parent_links();
        }
    }
}

impl<'a, K, V> NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal> {
    /// Removes a key/value pair from the end of this node. If this is an internal node,
    /// also removes the edge that was to the right of that pair.
    pub fn pop(&mut self) -> (K, V, Option<Root<K, V>>) {
        // Necessary for correctness, but this is an internal module
        debug_assert!(self.len() > 0);

        let idx = self.len() - 1;

        unsafe {
            let key = ptr::read(self.keys().get_unchecked(idx));
            let val = ptr::read(self.vals().get_unchecked(idx));
            let edge = match self.reborrow_mut().force() {
                ForceResult::Leaf(_) => None,
                ForceResult::Internal(internal) => {
                    let edge = ptr::read(internal.as_internal().edges.get_unchecked(idx + 1));
                    let mut new_root = Root { node: edge, height: internal.height - 1 };
                    new_root.as_mut().as_leaf_mut().parent = ptr::null();
                    Some(new_root)
                }
            };

            self.as_leaf_mut().len -= 1;
            (key, val, edge)
        }
    }

    /// Removes a key/value pair from the beginning of this node. If this is an internal node,
    /// also removes the edge that was to the left of that pair.
    pub fn pop_front(&mut self) -> (K, V, Option<Root<K, V>>) {
        // Necessary for correctness, but this is an internal module
        debug_assert!(self.len() > 0);

        let old_len = self.len();

        unsafe {
            let key = slice_remove(self.keys_mut(), 0);
            let val = slice_remove(self.vals_mut(), 0);
            let edge = match self.reborrow_mut().force() {
                ForceResult::Leaf(_) => None,
                ForceResult::Internal(mut internal) => {
                    let edge = slice_remove(
                        slice::from_raw_parts_mut(
                            internal.as_internal_mut().edges.as_mut_ptr(),
                            old_len+1
                        ),
                        0
                    );

                    let mut new_root = Root { node: edge, height: internal.height - 1 };
                    new_root.as_mut().as_leaf_mut().parent = ptr::null();

                    for i in 0..old_len {
                        Handle::new_edge(internal.reborrow_mut(), i).correct_parent_link();
                    }

                    Some(new_root)
                }
            };

            self.as_leaf_mut().len -= 1;

            (key, val, edge)
        }
    }

    fn into_kv_pointers_mut(mut self) -> (*mut K, *mut V) {
        (
            self.keys_mut().as_mut_ptr(),
            self.vals_mut().as_mut_ptr()
        )
    }
}

impl<BorrowType, K, V> NodeRef<BorrowType, K, V, marker::LeafOrInternal> {
    /// Checks whether a node is an `Internal` node or a `Leaf` node.
    pub fn force(self) -> ForceResult<
        NodeRef<BorrowType, K, V, marker::Leaf>,
        NodeRef<BorrowType, K, V, marker::Internal>
    > {
        if self.height == 0 {
            ForceResult::Leaf(NodeRef {
                height: self.height,
                node: self.node,
                root: self.root,
                _marker: PhantomData
            })
        } else {
            ForceResult::Internal(NodeRef {
                height: self.height,
                node: self.node,
                root: self.root,
                _marker: PhantomData
            })
        }
    }
}

/// A reference to a specific key/value pair or edge within a node. The `Node` parameter
/// must be a `NodeRef`, while the `Type` can either be `KV` (signifying a handle on a key/value
/// pair) or `Edge` (signifying a handle on an edge).
///
/// Note that even `Leaf` nodes can have `Edge` handles. Instead of representing a pointer to
/// a child node, these represent the spaces where child pointers would go between the key/value
/// pairs. For example, in a node with length 2, there would be 3 possible edge locations - one
/// to the left of the node, one between the two pairs, and one at the right of the node.
pub struct Handle<Node, Type> {
    node: Node,
    idx: usize,
    _marker: PhantomData<Type>
}

impl<Node: Copy, Type> Copy for Handle<Node, Type> { }
// We don't need the full generality of `#[derive(Clone)]`, as the only time `Node` will be
// `Clone`able is when it is an immutable reference and therefore `Copy`.
impl<Node: Copy, Type> Clone for Handle<Node, Type> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Node, Type> Handle<Node, Type> {
    /// Retrieves the node that contains the edge of key/value pair this handle pointes to.
    pub fn into_node(self) -> Node {
        self.node
    }
}

impl<BorrowType, K, V, NodeType> Handle<NodeRef<BorrowType, K, V, NodeType>, marker::KV> {
    /// Creates a new handle to a key/value pair in `node`. `idx` must be less than `node.len()`.
    pub fn new_kv(node: NodeRef<BorrowType, K, V, NodeType>, idx: usize) -> Self {
        // Necessary for correctness, but in a private module
        debug_assert!(idx < node.len());

        Handle {
            node: node,
            idx: idx,
            _marker: PhantomData
        }
    }

    pub fn left_edge(self) -> Handle<NodeRef<BorrowType, K, V, NodeType>, marker::Edge> {
        Handle::new_edge(self.node, self.idx)
    }

    pub fn right_edge(self) -> Handle<NodeRef<BorrowType, K, V, NodeType>, marker::Edge> {
        Handle::new_edge(self.node, self.idx + 1)
    }
}

impl<BorrowType, K, V, NodeType, HandleType> PartialEq
        for Handle<NodeRef<BorrowType, K, V, NodeType>, HandleType> {

    fn eq(&self, other: &Self) -> bool {
        self.node.node == other.node.node && self.idx == other.idx
    }
}

impl<BorrowType, K, V, NodeType, HandleType>
        Handle<NodeRef<BorrowType, K, V, NodeType>, HandleType> {

    /// Temporarily takes out another, immutable handle on the same location.
    pub fn reborrow(&self)
            -> Handle<NodeRef<marker::Immut, K, V, NodeType>, HandleType> {

        // We can't use Handle::new_kv or Handle::new_edge because we don't know our type
        Handle {
            node: self.node.reborrow(),
            idx: self.idx,
            _marker: PhantomData
        }
    }
}

impl<'a, K, V, NodeType, HandleType>
        Handle<NodeRef<marker::Mut<'a>, K, V, NodeType>, HandleType> {

    /// Temporarily takes out another, mutable handle on the same location. Beware, as
    /// this method is very dangerous, doubly so since it may not immediately appear
    /// dangerous.
    ///
    /// Because mutable pointers can roam anywhere around the tree and can even (through
    /// `into_root_mut`) mess with the root of the tree, the result of `reborrow_mut`
    /// can easily be used to make the original mutable pointer dangling, or, in the case
    /// of a reborrowed handle, out of bounds.
    // FIXME(@gereeter) consider adding yet another type parameter to `NodeRef` that restricts
    // the use of `ascend` and `into_root_mut` on reborrowed pointers, preventing this unsafety.
    pub unsafe fn reborrow_mut(&mut self)
            -> Handle<NodeRef<marker::Mut, K, V, NodeType>, HandleType> {

        // We can't use Handle::new_kv or Handle::new_edge because we don't know our type
        Handle {
            node: self.node.reborrow_mut(),
            idx: self.idx,
            _marker: PhantomData
        }
    }
}

impl<BorrowType, K, V, NodeType>
        Handle<NodeRef<BorrowType, K, V, NodeType>, marker::Edge> {

    /// Creates a new handle to an edge in `node`. `idx` must be less than or equal to
    /// `node.len()`.
    pub fn new_edge(node: NodeRef<BorrowType, K, V, NodeType>, idx: usize) -> Self {
        // Necessary for correctness, but in a private module
        debug_assert!(idx <= node.len());

        Handle {
            node: node,
            idx: idx,
            _marker: PhantomData
        }
    }

    pub fn left_kv(self)
            -> Result<Handle<NodeRef<BorrowType, K, V, NodeType>, marker::KV>, Self> {

        if self.idx > 0 {
            Ok(Handle::new_kv(self.node, self.idx - 1))
        } else {
            Err(self)
        }
    }

    pub fn right_kv(self)
            -> Result<Handle<NodeRef<BorrowType, K, V, NodeType>, marker::KV>, Self> {

        if self.idx < self.node.len() {
            Ok(Handle::new_kv(self.node, self.idx))
        } else {
            Err(self)
        }
    }
}

impl<'a, K, V> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge> {
    /// Inserts a new key/value pair between the key/value pairs to the right and left of
    /// this edge. This method assumes that there is enough space in the node for the new
    /// pair to fit.
    ///
    /// The returned pointer points to the inserted value.
    fn insert_fit(&mut self, key: K, val: V) -> *mut V {
        // Necessary for correctness, but in a private module
        debug_assert!(self.node.len() < CAPACITY);

        unsafe {
            slice_insert(self.node.keys_mut(), self.idx, key);
            slice_insert(self.node.vals_mut(), self.idx, val);

            self.node.as_leaf_mut().len += 1;

            self.node.vals_mut().get_unchecked_mut(self.idx)
        }
    }

    /// Inserts a new key/value pair between the key/value pairs to the right and left of
    /// this edge. This method splits the node if there isn't enough room.
    ///
    /// The returned pointer points to the inserted value.
    pub fn insert(mut self, key: K, val: V)
            -> (InsertResult<'a, K, V, marker::Leaf>, *mut V) {

        if self.node.len() < CAPACITY {
            let ptr = self.insert_fit(key, val);
            (InsertResult::Fit(Handle::new_kv(self.node, self.idx)), ptr)
        } else {
            let middle = Handle::new_kv(self.node, B);
            let (mut left, k, v, mut right) = middle.split();
            let ptr = if self.idx <= B {
                unsafe {
                    Handle::new_edge(left.reborrow_mut(), self.idx).insert_fit(key, val)
                }
            } else {
                unsafe {
                    Handle::new_edge(
                        right.as_mut().cast_unchecked::<marker::Leaf>(),
                        self.idx - (B + 1)
                    ).insert_fit(key, val)
                }
            };
            (InsertResult::Split(left, k, v, right), ptr)
        }
    }
}

impl<'a, K, V> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::Edge> {
    /// Fixes the parent pointer and index in the child node below this edge. This is useful
    /// when the ordering of edges has been changed, such as in the various `insert` methods.
    fn correct_parent_link(mut self) {
        let idx = self.idx as u16;
        let ptr = self.node.as_internal_mut() as *mut _;
        let mut child = self.descend();
        child.as_leaf_mut().parent = ptr;
        child.as_leaf_mut().parent_idx = idx;
    }

    /// Unsafely asserts to the compiler some static information about whether the underlying
    /// node of this handle is a `Leaf`.
    unsafe fn cast_unchecked<NewType>(&mut self)
            -> Handle<NodeRef<marker::Mut, K, V, NewType>, marker::Edge> {

        Handle::new_edge(self.node.cast_unchecked(), self.idx)
    }

    /// Inserts a new key/value pair and an edge that will go to the right of that new pair
    /// between this edge and the key/value pair to the right of this edge. This method assumes
    /// that there is enough space in the node for the new pair to fit.
    fn insert_fit(&mut self, key: K, val: V, edge: Root<K, V>) {
        // Necessary for correctness, but in an internal module
        debug_assert!(self.node.len() < CAPACITY);
        debug_assert!(edge.height == self.node.height - 1);

        unsafe {
            // This cast is a lie, but it allows us to reuse the key/value insertion logic.
            self.cast_unchecked::<marker::Leaf>().insert_fit(key, val);

            slice_insert(
                slice::from_raw_parts_mut(
                    self.node.as_internal_mut().edges.as_mut_ptr(),
                    self.node.len()
                ),
                self.idx + 1,
                edge.node
            );

            for i in (self.idx+1)..(self.node.len()+1) {
                Handle::new_edge(self.node.reborrow_mut(), i).correct_parent_link();
            }
        }
    }

    /// Inserts a new key/value pair and an edge that will go to the right of that new pair
    /// between this edge and the key/value pair to the right of this edge. This method splits
    /// the node if there isn't enough room.
    pub fn insert(mut self, key: K, val: V, edge: Root<K, V>)
            -> InsertResult<'a, K, V, marker::Internal> {

        // Necessary for correctness, but this is an internal module
        debug_assert!(edge.height == self.node.height - 1);

        if self.node.len() < CAPACITY {
            self.insert_fit(key, val, edge);
            InsertResult::Fit(Handle::new_kv(self.node, self.idx))
        } else {
            let middle = Handle::new_kv(self.node, B);
            let (mut left, k, v, mut right) = middle.split();
            if self.idx <= B {
                unsafe {
                    Handle::new_edge(left.reborrow_mut(), self.idx).insert_fit(key, val, edge);
                }
            } else {
                unsafe {
                    Handle::new_edge(
                        right.as_mut().cast_unchecked::<marker::Internal>(),
                        self.idx - (B + 1)
                    ).insert_fit(key, val, edge);
                }
            }
            InsertResult::Split(left, k, v, right)
        }
    }
}

impl<BorrowType, K, V>
        Handle<NodeRef<BorrowType, K, V, marker::Internal>, marker::Edge> {

    /// Finds the node pointed to by this edge.
    ///
    /// `edge.descend().ascend().unwrap()` and `node.ascend().unwrap().descend()` should
    /// both, upon success, do nothing.
    pub fn descend(self) -> NodeRef<BorrowType, K, V, marker::LeafOrInternal> {
        NodeRef {
            height: self.node.height - 1,
            node: unsafe { self.node.as_internal().edges.get_unchecked(self.idx).as_ptr() },
            root: self.node.root,
            _marker: PhantomData
        }
    }
}

impl<'a, K: 'a, V: 'a, NodeType>
        Handle<NodeRef<marker::Immut<'a>, K, V, NodeType>, marker::KV> {

    pub fn into_kv(self) -> (&'a K, &'a V) {
        let (keys, vals) = self.node.into_slices();
        unsafe {
            (keys.get_unchecked(self.idx), vals.get_unchecked(self.idx))
        }
    }
}

impl<'a, K: 'a, V: 'a, NodeType>
        Handle<NodeRef<marker::Mut<'a>, K, V, NodeType>, marker::KV> {

    pub fn into_kv_mut(self) -> (&'a mut K, &'a mut V) {
        let (mut keys, mut vals) = self.node.into_slices_mut();
        unsafe {
            (keys.get_unchecked_mut(self.idx), vals.get_unchecked_mut(self.idx))
        }
    }
}

impl<'a, K, V, NodeType> Handle<NodeRef<marker::Mut<'a>, K, V, NodeType>, marker::KV> {
    pub fn kv_mut(&mut self) -> (&mut K, &mut V) {
        unsafe {
            let (mut keys, mut vals) = self.node.reborrow_mut().into_slices_mut();
            (keys.get_unchecked_mut(self.idx), vals.get_unchecked_mut(self.idx))
        }
    }
}

impl<'a, K, V> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::KV> {
    /// Splits the underlying node into three parts:
    ///
    /// - The node is truncated to only contain the key/value pairs to the right of
    ///   this handle.
    /// - The key and value pointed to by this handle and extracted.
    /// - All the key/value pairs to the right of this handle are put into a newly
    ///   allocated node.
    pub fn split(mut self)
            -> (NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, K, V, Root<K, V>) {
        unsafe {
            let mut new_node = Box::new(LeafNode::new());

            let k = ptr::read(self.node.keys().get_unchecked(self.idx));
            let v = ptr::read(self.node.vals().get_unchecked(self.idx));

            let new_len = self.node.len() - self.idx - 1;

            ptr::copy_nonoverlapping(
                self.node.keys().as_ptr().offset(self.idx as isize + 1),
                new_node.keys.as_mut_ptr(),
                new_len
            );
            ptr::copy_nonoverlapping(
                self.node.vals().as_ptr().offset(self.idx as isize + 1),
                new_node.vals.as_mut_ptr(),
                new_len
            );

            self.node.as_leaf_mut().len = self.idx as u16;
            new_node.len = new_len as u16;

            (
                self.node,
                k, v,
                Root {
                    node: BoxedNode::from_leaf(new_node),
                    height: 0
                }
            )
        }
    }

    /// Removes the key/value pair pointed to by this handle, returning the edge between the
    /// now adjacent key/value pairs to the left and right of this handle.
    pub fn remove(mut self)
            -> (Handle<NodeRef<marker::Mut<'a>, K, V, marker::Leaf>, marker::Edge>, K, V) {
        unsafe {
            let k = slice_remove(self.node.keys_mut(), self.idx);
            let v = slice_remove(self.node.vals_mut(), self.idx);
            self.node.as_leaf_mut().len -= 1;
            (self.left_edge(), k, v)
        }
    }
}

impl<'a, K, V> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::KV> {
    /// Splits the underlying node into three parts:
    ///
    /// - The node is truncated to only contain the edges and key/value pairs to the
    ///   right of this handle.
    /// - The key and value pointed to by this handle and extracted.
    /// - All the edges and key/value pairs to the right of this handle are put into
    ///   a newly allocated node.
    pub fn split(mut self)
            -> (NodeRef<marker::Mut<'a>, K, V, marker::Internal>, K, V, Root<K, V>) {
        unsafe {
            let mut new_node = Box::new(InternalNode::new());

            let k = ptr::read(self.node.keys().get_unchecked(self.idx));
            let v = ptr::read(self.node.vals().get_unchecked(self.idx));

            let height = self.node.height;
            let new_len = self.node.len() - self.idx - 1;

            ptr::copy_nonoverlapping(
                self.node.keys().as_ptr().offset(self.idx as isize + 1),
                new_node.data.keys.as_mut_ptr(),
                new_len
            );
            ptr::copy_nonoverlapping(
                self.node.vals().as_ptr().offset(self.idx as isize + 1),
                new_node.data.vals.as_mut_ptr(),
                new_len
            );
            ptr::copy_nonoverlapping(
                self.node.as_internal().edges.as_ptr().offset(self.idx as isize + 1),
                new_node.edges.as_mut_ptr(),
                new_len + 1
            );

            self.node.as_leaf_mut().len = self.idx as u16;
            new_node.data.len = new_len as u16;

            let mut new_root = Root {
                node: BoxedNode::from_internal(new_node),
                height: height
            };

            for i in 0..(new_len+1) {
                Handle::new_edge(new_root.as_mut().cast_unchecked(), i).correct_parent_link();
            }

            (
                self.node,
                k, v,
                new_root
            )
        }
    }

    /// Returns whether it is valid to call `.merge()`, i.e., whether there is enough room in
    /// a node to hold the combination of the nodes to the left and right of this handle along
    /// with the key/value pair at this handle.
    pub fn can_merge(&self) -> bool {
        (
            self.reborrow()
                .left_edge()
                .descend()
                .len()
          + self.reborrow()
                .right_edge()
                .descend()
                .len()
          + 1
        ) <= CAPACITY
    }

    /// Combines the node immediately to the left of this handle, the key/value pair pointed
    /// to by this handle, and the node immediately to the right of this handle into one new
    /// child of the underlying node, returning an edge referencing that new child.
    ///
    /// Assumes that this edge `.can_merge()`.
    pub fn merge(mut self)
            -> Handle<NodeRef<marker::Mut<'a>, K, V, marker::Internal>, marker::Edge> {
        let self1 = unsafe { ptr::read(&self) };
        let self2 = unsafe { ptr::read(&self) };
        let mut left_node = self1.left_edge().descend();
        let left_len = left_node.len();
        let mut right_node = self2.right_edge().descend();
        let right_len = right_node.len();

        // necessary for correctness, but in a private module
        debug_assert!(left_len + right_len + 1 <= CAPACITY);

        unsafe {
            ptr::write(left_node.keys_mut().get_unchecked_mut(left_len),
                       slice_remove(self.node.keys_mut(), self.idx));
            ptr::copy_nonoverlapping(
                right_node.keys().as_ptr(),
                left_node.keys_mut().as_mut_ptr().offset(left_len as isize + 1),
                right_len
            );
            ptr::write(left_node.vals_mut().get_unchecked_mut(left_len),
                       slice_remove(self.node.vals_mut(), self.idx));
            ptr::copy_nonoverlapping(
                right_node.vals().as_ptr(),
                left_node.vals_mut().as_mut_ptr().offset(left_len as isize + 1),
                right_len
            );

            slice_remove(&mut self.node.as_internal_mut().edges, self.idx + 1);
            for i in self.idx+1..self.node.len() {
                Handle::new_edge(self.node.reborrow_mut(), i).correct_parent_link();
            }
            self.node.as_leaf_mut().len -= 1;

            left_node.as_leaf_mut().len += right_len as u16 + 1;

            if self.node.height > 1 {
                ptr::copy_nonoverlapping(
                    right_node.cast_unchecked().as_internal().edges.as_ptr(),
                    left_node.cast_unchecked()
                             .as_internal_mut()
                             .edges
                             .as_mut_ptr()
                             .offset(left_len as isize + 1),
                    right_len + 1
                );

                for i in left_len+1..left_len+right_len+2 {
                    Handle::new_edge(
                        left_node.cast_unchecked().reborrow_mut(),
                        i
                    ).correct_parent_link();
                }

                heap::deallocate(
                    *right_node.node as *mut u8,
                    mem::size_of::<InternalNode<K, V>>(),
                    mem::align_of::<InternalNode<K, V>>()
                );
            } else {
                heap::deallocate(
                    *right_node.node as *mut u8,
                    mem::size_of::<LeafNode<K, V>>(),
                    mem::align_of::<LeafNode<K, V>>()
                );
            }

            Handle::new_edge(self.node, self.idx)
        }
    }

    /// This removes a key/value pair from the left child and replaces it with the key/value pair
    /// pointed to by this handle while pushing the old key/value pair of this handle into the right
    /// child.
    pub fn steal_left(&mut self) {
        unsafe {
            let (k, v, edge) = self.reborrow_mut().left_edge().descend().pop();

            let k = mem::replace(self.reborrow_mut().into_kv_mut().0, k);
            let v = mem::replace(self.reborrow_mut().into_kv_mut().1, v);

            match self.reborrow_mut().right_edge().descend().force() {
                ForceResult::Leaf(mut leaf) => leaf.push_front(k, v),
                ForceResult::Internal(mut internal) => internal.push_front(k, v, edge.unwrap())
            }
        }
    }

    /// This removes a key/value pair from the right child and replaces it with the key/value pair
    /// pointed to by this handle while pushing the old key/value pair of this handle into the left
    /// child.
    pub fn steal_right(&mut self) {
        unsafe {
            let (k, v, edge) = self.reborrow_mut().right_edge().descend().pop_front();

            let k = mem::replace(self.reborrow_mut().into_kv_mut().0, k);
            let v = mem::replace(self.reborrow_mut().into_kv_mut().1, v);

            match self.reborrow_mut().left_edge().descend().force() {
                ForceResult::Leaf(mut leaf) => leaf.push(k, v),
                ForceResult::Internal(mut internal) => internal.push(k, v, edge.unwrap())
            }
        }
    }

    /// This does stealing similar to `steal_left` but steals multiple elements at once.
    pub fn bulk_steal_left(&mut self, count: usize) {
        unsafe {
            let mut left_node = ptr::read(self).left_edge().descend();
            let left_len = left_node.len();
            let mut right_node = ptr::read(self).right_edge().descend();
            let right_len = right_node.len();

            // Make sure that we may steal safely.
            debug_assert!(right_len + count <= CAPACITY);
            debug_assert!(left_len >= count);

            let new_left_len = left_len - count;

            // Move data.
            {
                let left_kv = left_node.reborrow_mut().into_kv_pointers_mut();
                let right_kv = right_node.reborrow_mut().into_kv_pointers_mut();
                let parent_kv = {
                    let kv = self.reborrow_mut().into_kv_mut();
                    (kv.0 as *mut K, kv.1 as *mut V)
                };

                // Make room for stolen elements in the right child.
                ptr::copy(right_kv.0,
                          right_kv.0.offset(count as isize),
                          right_len);
                ptr::copy(right_kv.1,
                          right_kv.1.offset(count as isize),
                          right_len);

                // Move elements from the left child to the right one.
                move_kv(left_kv, new_left_len + 1, right_kv, 0, count - 1);

                // Move parent's key/value pair to the right child.
                move_kv(parent_kv, 0, right_kv, count - 1, 1);

                // Move the left-most stolen pair to the parent.
                move_kv(left_kv, new_left_len, parent_kv, 0, 1);
            }

            left_node.reborrow_mut().as_leaf_mut().len -= count as u16;
            right_node.reborrow_mut().as_leaf_mut().len += count as u16;

            match (left_node.force(), right_node.force()) {
                (ForceResult::Internal(left), ForceResult::Internal(mut right)) => {
                    // Make room for stolen edges.
                    let right_edges = right.reborrow_mut().as_internal_mut().edges.as_mut_ptr();
                    ptr::copy(right_edges,
                              right_edges.offset(count as isize),
                              right_len + 1);
                    right.correct_childrens_parent_links(count, count + right_len + 1);

                    move_edges(left, new_left_len + 1, right, 0, count);
                },
                (ForceResult::Leaf(_), ForceResult::Leaf(_)) => { }
                _ => { unreachable!(); }
            }
        }
    }

    /// The symmetric clone of `bulk_steal_left`.
    pub fn bulk_steal_right(&mut self, count: usize) {
        unsafe {
            let mut left_node = ptr::read(self).left_edge().descend();
            let left_len = left_node.len();
            let mut right_node = ptr::read(self).right_edge().descend();
            let right_len = right_node.len();

            // Make sure that we may steal safely.
            debug_assert!(left_len + count <= CAPACITY);
            debug_assert!(right_len >= count);

            let new_right_len = right_len - count;

            // Move data.
            {
                let left_kv = left_node.reborrow_mut().into_kv_pointers_mut();
                let right_kv = right_node.reborrow_mut().into_kv_pointers_mut();
                let parent_kv = {
                    let kv = self.reborrow_mut().into_kv_mut();
                    (kv.0 as *mut K, kv.1 as *mut V)
                };

                // Move parent's key/value pair to the left child.
                move_kv(parent_kv, 0, left_kv, left_len, 1);

                // Move elements from the right child to the left one.
                move_kv(right_kv, 0, left_kv, left_len + 1, count - 1);

                // Move the right-most stolen pair to the parent.
                move_kv(right_kv, count - 1, parent_kv, 0, 1);

                // Fix right indexing
                ptr::copy(right_kv.0.offset(count as isize),
                          right_kv.0,
                          new_right_len);
                ptr::copy(right_kv.1.offset(count as isize),
                          right_kv.1,
                          new_right_len);
            }

            left_node.reborrow_mut().as_leaf_mut().len += count as u16;
            right_node.reborrow_mut().as_leaf_mut().len -= count as u16;

            match (left_node.force(), right_node.force()) {
                (ForceResult::Internal(left), ForceResult::Internal(mut right)) => {
                    move_edges(right.reborrow_mut(), 0, left, left_len + 1, count);

                    // Fix right indexing.
                    let right_edges = right.reborrow_mut().as_internal_mut().edges.as_mut_ptr();
                    ptr::copy(right_edges.offset(count as isize),
                              right_edges,
                              new_right_len + 1);
                    right.correct_childrens_parent_links(0, new_right_len + 1);
                },
                (ForceResult::Leaf(_), ForceResult::Leaf(_)) => { }
                _ => { unreachable!(); }
            }
        }
    }
}

unsafe fn move_kv<K, V>(
    source: (*mut K, *mut V), source_offset: usize,
    dest: (*mut K, *mut V), dest_offset: usize,
    count: usize)
{
    ptr::copy_nonoverlapping(source.0.offset(source_offset as isize),
                             dest.0.offset(dest_offset as isize),
                             count);
    ptr::copy_nonoverlapping(source.1.offset(source_offset as isize),
                             dest.1.offset(dest_offset as isize),
                             count);
}

// Source and destination must have the same height.
unsafe fn move_edges<K, V>(
    mut source: NodeRef<marker::Mut, K, V, marker::Internal>, source_offset: usize,
    mut dest: NodeRef<marker::Mut, K, V, marker::Internal>, dest_offset: usize,
    count: usize)
{
    let source_ptr = source.as_internal_mut().edges.as_mut_ptr();
    let dest_ptr = dest.as_internal_mut().edges.as_mut_ptr();
    ptr::copy_nonoverlapping(source_ptr.offset(source_offset as isize),
                             dest_ptr.offset(dest_offset as isize),
                             count);
    dest.correct_childrens_parent_links(dest_offset, dest_offset + count);
}

impl<BorrowType, K, V, HandleType>
        Handle<NodeRef<BorrowType, K, V, marker::LeafOrInternal>, HandleType> {

    /// Check whether the underlying node is an `Internal` node or a `Leaf` node.
    pub fn force(self) -> ForceResult<
        Handle<NodeRef<BorrowType, K, V, marker::Leaf>, HandleType>,
        Handle<NodeRef<BorrowType, K, V, marker::Internal>, HandleType>
    > {
        match self.node.force() {
            ForceResult::Leaf(node) => ForceResult::Leaf(Handle {
                node: node,
                idx: self.idx,
                _marker: PhantomData
            }),
            ForceResult::Internal(node) => ForceResult::Internal(Handle {
                node: node,
                idx: self.idx,
                _marker: PhantomData
            })
        }
    }
}

impl<'a, K, V> Handle<NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>, marker::Edge> {
    /// Move the suffix after `self` from one node to another one. `right` must be empty.
    /// The first edge of `right` remains unchanged.
    pub fn move_suffix(&mut self,
            right: &mut NodeRef<marker::Mut<'a>, K, V, marker::LeafOrInternal>) {
        unsafe {
            let left_new_len = self.idx;
            let mut left_node = self.reborrow_mut().into_node();

            let right_new_len = left_node.len() - left_new_len;
            let mut right_node = right.reborrow_mut();

            debug_assert!(right_node.len() == 0);
            debug_assert!(left_node.height == right_node.height);

            let left_kv = left_node.reborrow_mut().into_kv_pointers_mut();
            let right_kv = right_node.reborrow_mut().into_kv_pointers_mut();


            move_kv(left_kv, left_new_len, right_kv, 0, right_new_len);

            left_node.reborrow_mut().as_leaf_mut().len = left_new_len as u16;
            right_node.reborrow_mut().as_leaf_mut().len = right_new_len as u16;

            match (left_node.force(), right_node.force()) {
                (ForceResult::Internal(left), ForceResult::Internal(right)) => {
                    move_edges(left, left_new_len + 1, right, 1, right_new_len);
                },
                (ForceResult::Leaf(_), ForceResult::Leaf(_)) => { }
                _ => { unreachable!(); }
            }
        }
    }
}

pub enum ForceResult<Leaf, Internal> {
    Leaf(Leaf),
    Internal(Internal)
}

pub enum InsertResult<'a, K, V, Type> {
    Fit(Handle<NodeRef<marker::Mut<'a>, K, V, Type>, marker::KV>),
    Split(NodeRef<marker::Mut<'a>, K, V, Type>, K, V, Root<K, V>)
}

pub mod marker {
    use core::marker::PhantomData;

    pub enum Leaf { }
    pub enum Internal { }
    pub enum LeafOrInternal { }

    pub enum Owned { }
    pub struct Immut<'a>(PhantomData<&'a ()>);
    pub struct Mut<'a>(PhantomData<&'a mut ()>);

    pub enum KV { }
    pub enum Edge { }
}

unsafe fn slice_insert<T>(slice: &mut [T], idx: usize, val: T) {
    ptr::copy(
        slice.as_ptr().offset(idx as isize),
        slice.as_mut_ptr().offset(idx as isize + 1),
        slice.len() - idx
    );
    ptr::write(slice.get_unchecked_mut(idx), val);
}

unsafe fn slice_remove<T>(slice: &mut [T], idx: usize) -> T {
    let ret = ptr::read(slice.get_unchecked(idx));
    ptr::copy(
        slice.as_ptr().offset(idx as isize + 1),
        slice.as_mut_ptr().offset(idx as isize),
        slice.len() - idx - 1
    );
    ret
}
