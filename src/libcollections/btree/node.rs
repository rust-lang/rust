// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This module represents all the internal representation and logic for a B-Tree's node
// with a safe interface, so that BTreeMap itself does not depend on any of these details.

pub use self::InsertionResult::*;
pub use self::SearchResult::*;
pub use self::ForceResult::*;
pub use self::TraversalItem::*;

use core::prelude::*;

use core::{slice, mem, ptr, cmp, num, raw};
use core::iter::Zip;
use core::borrow::BorrowFrom;
use alloc::heap;

/// Represents the result of an Insertion: either the item fit, or the node had to split
pub enum InsertionResult<K, V> {
    /// The inserted element fit
    Fit,
    /// The inserted element did not fit, so the node was split
    Split(K, V, Node<K, V>),
}

/// Represents the result of a search for a key in a single node
pub enum SearchResult<NodeRef> {
    /// The element was found at the given index
    Found(Handle<NodeRef, handle::KV, handle::LeafOrInternal>),
    /// The element wasn't found, but if it's anywhere, it must be beyond this edge
    GoDown(Handle<NodeRef, handle::Edge, handle::LeafOrInternal>),
}

/// A B-Tree Node. We keep keys/edges/values separate to optimize searching for keys.
#[unsafe_no_drop_flag]
pub struct Node<K, V> {
    // To avoid the need for multiple allocations, we allocate a single buffer with enough space
    // for `capacity` keys, `capacity` values, and (in internal nodes) `capacity + 1` edges.
    // Despite this, we store three separate pointers to the three "chunks" of the buffer because
    // the performance drops significantly if the locations of the vals and edges need to be
    // recalculated upon access.
    //
    // These will never be null during normal usage of a `Node`. However, to avoid the need for a
    // drop flag, `Node::drop` zeroes `keys`, signaling that the `Node` has already been cleaned
    // up.
    keys: *mut K,
    vals: *mut V,

    // In leaf nodes, this will be null, and no space will be allocated for edges.
    edges: *mut Node<K, V>,

    // At any given time, there will be `_len` keys, `_len` values, and (in an internal node)
    // `_len + 1` edges. In a leaf node, there will never be any edges.
    //
    // Note: instead of accessing this field directly, please call the `len()` method, which should
    // be more stable in the face of representation changes.
    _len: uint,

    // FIXME(gereeter) It shouldn't be necessary to store the capacity in every node, as it should
    // be constant throughout the tree. Once a solution to this is found, it might be possible to
    // also pass down the offsets into the buffer that vals and edges are stored at, removing the
    // need for those two pointers.
    //
    // Note: instead of accessing this field directly, please call the `capacity()` method, which
    // should be more stable in the face of representation changes.
    _capacity: uint,
}

/// Rounds up to a multiple of a power of two. Returns the closest multiple
/// of `target_alignment` that is higher or equal to `unrounded`.
///
/// # Panics
///
/// Fails if `target_alignment` is not a power of two.
#[inline]
fn round_up_to_next(unrounded: uint, target_alignment: uint) -> uint {
    assert!(num::UnsignedInt::is_power_of_two(target_alignment));
    (unrounded + target_alignment - 1) & !(target_alignment - 1)
}

#[test]
fn test_rounding() {
    assert_eq!(round_up_to_next(0, 4), 0);
    assert_eq!(round_up_to_next(1, 4), 4);
    assert_eq!(round_up_to_next(2, 4), 4);
    assert_eq!(round_up_to_next(3, 4), 4);
    assert_eq!(round_up_to_next(4, 4), 4);
    assert_eq!(round_up_to_next(5, 4), 8);
}

// Returns a tuple of (val_offset, edge_offset),
// from the start of a mallocated array.
#[inline]
fn calculate_offsets(keys_size: uint,
                     vals_size: uint, vals_align: uint,
                     edges_align: uint)
                     -> (uint, uint) {
    let vals_offset = round_up_to_next(keys_size, vals_align);
    let end_of_vals = vals_offset + vals_size;

    let edges_offset = round_up_to_next(end_of_vals, edges_align);

    (vals_offset, edges_offset)
}

// Returns a tuple of (minimum required alignment, array_size),
// from the start of a mallocated array.
#[inline]
fn calculate_allocation(keys_size: uint, keys_align: uint,
                        vals_size: uint, vals_align: uint,
                        edges_size: uint, edges_align: uint)
                        -> (uint, uint) {
    let (_, edges_offset) = calculate_offsets(keys_size,
                                              vals_size, vals_align,
                                                         edges_align);
    let end_of_edges = edges_offset + edges_size;

    let min_align = cmp::max(keys_align, cmp::max(vals_align, edges_align));

    (min_align, end_of_edges)
}

#[test]
fn test_offset_calculation() {
    assert_eq!(calculate_allocation(128, 8, 15, 1, 4, 4), (8, 148));
    assert_eq!(calculate_allocation(3, 1, 2, 1, 1, 1), (1, 6));
    assert_eq!(calculate_allocation(6, 2, 12, 4, 24, 8), (8, 48));
    assert_eq!(calculate_offsets(128, 15, 1, 4), (128, 144));
    assert_eq!(calculate_offsets(3, 2, 1, 1), (3, 5));
    assert_eq!(calculate_offsets(6, 12, 4, 8), (8, 24));
}

fn calculate_allocation_generic<K, V>(capacity: uint, is_leaf: bool) -> (uint, uint) {
    let (keys_size, keys_align) = (capacity * mem::size_of::<K>(), mem::min_align_of::<K>());
    let (vals_size, vals_align) = (capacity * mem::size_of::<V>(), mem::min_align_of::<V>());
    let (edges_size, edges_align) = if is_leaf {
        (0, 1)
    } else {
        ((capacity + 1) * mem::size_of::<Node<K, V>>(), mem::min_align_of::<Node<K, V>>())
    };

    calculate_allocation(
            keys_size, keys_align,
            vals_size, vals_align,
            edges_size, edges_align
    )
}

fn calculate_offsets_generic<K, V>(capacity: uint, is_leaf: bool) -> (uint, uint) {
    let keys_size = capacity * mem::size_of::<K>();
    let vals_size = capacity * mem::size_of::<V>();
    let vals_align = mem::min_align_of::<V>();
    let edges_align = if is_leaf {
        1
    } else {
        mem::min_align_of::<Node<K, V>>()
    };

    calculate_offsets(
            keys_size,
            vals_size, vals_align,
                       edges_align
    )
}

/// An iterator over a slice that owns the elements of the slice but not the allocation.
struct RawItems<T> {
    head: *const T,
    tail: *const T,
}

impl<T> RawItems<T> {
    unsafe fn from_slice(slice: &[T]) -> RawItems<T> {
        RawItems::from_parts(slice.as_ptr(), slice.len())
    }

    unsafe fn from_parts(ptr: *const T, len: uint) -> RawItems<T> {
        if mem::size_of::<T>() == 0 {
            RawItems {
                head: ptr,
                tail: (ptr as uint + len) as *const T,
            }
        } else {
            RawItems {
                head: ptr,
                tail: ptr.offset(len as int),
            }
        }
    }

    unsafe fn push(&mut self, val: T) {
        ptr::write(self.tail as *mut T, val);

        if mem::size_of::<T>() == 0 {
            self.tail = (self.tail as uint + 1) as *const T;
        } else {
            self.tail = self.tail.offset(1);
        }
    }
}

impl<T> Iterator<T> for RawItems<T> {
    fn next(&mut self) -> Option<T> {
        if self.head == self.tail {
            None
        } else {
            unsafe {
                let ret = Some(ptr::read(self.head));

                if mem::size_of::<T>() == 0 {
                    self.head = (self.head as uint + 1) as *const T;
                } else {
                    self.head = self.head.offset(1);
                }

                ret
            }
        }
    }
}

impl<T> DoubleEndedIterator<T> for RawItems<T> {
    fn next_back(&mut self) -> Option<T> {
        if self.head == self.tail {
            None
        } else {
            unsafe {
                if mem::size_of::<T>() == 0 {
                    self.tail = (self.tail as uint - 1) as *const T;
                } else {
                    self.tail = self.tail.offset(-1);
                }

                Some(ptr::read(self.tail))
            }
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for RawItems<T> {
    fn drop(&mut self) {
        for _ in *self {}
    }
}

#[unsafe_destructor]
impl<K, V> Drop for Node<K, V> {
    fn drop(&mut self) {
        if self.keys.is_null() {
            // We have already cleaned up this node.
            return;
        }

        // Do the actual cleanup.
        unsafe {
            drop(RawItems::from_slice(self.keys()));
            drop(RawItems::from_slice(self.vals()));
            drop(RawItems::from_slice(self.edges()));

            self.destroy();
        }

        self.keys = ptr::null_mut();
    }
}

impl<K, V> Node<K, V> {
    /// Make a new internal node. The caller must initialize the result to fix the invariant that
    /// there are `len() + 1` edges.
    unsafe fn new_internal(capacity: uint) -> Node<K, V> {
        let (alignment, size) = calculate_allocation_generic::<K, V>(capacity, false);

        let buffer = heap::allocate(size, alignment);
        if buffer.is_null() { ::alloc::oom(); }

        let (vals_offset, edges_offset) = calculate_offsets_generic::<K, V>(capacity, false);

        Node {
            keys: buffer as *mut K,
            vals: buffer.offset(vals_offset as int) as *mut V,
            edges: buffer.offset(edges_offset as int) as *mut Node<K, V>,
            _len: 0,
            _capacity: capacity,
        }
    }

    /// Make a new leaf node
    fn new_leaf(capacity: uint) -> Node<K, V> {
        let (alignment, size) = calculate_allocation_generic::<K, V>(capacity, true);

        let buffer = unsafe { heap::allocate(size, alignment) };
        if buffer.is_null() { ::alloc::oom(); }

        let (vals_offset, _) = calculate_offsets_generic::<K, V>(capacity, true);

        Node {
            keys: buffer as *mut K,
            vals: unsafe { buffer.offset(vals_offset as int) as *mut V },
            edges: ptr::null_mut(),
            _len: 0,
            _capacity: capacity,
        }
    }

    unsafe fn destroy(&mut self) {
        let (alignment, size) =
                calculate_allocation_generic::<K, V>(self.capacity(), self.is_leaf());
        heap::deallocate(self.keys as *mut u8, size, alignment);
    }

    #[inline]
    pub fn as_slices<'a>(&'a self) -> (&'a [K], &'a [V]) {
        unsafe {(
            mem::transmute(raw::Slice {
                data: self.keys as *const K,
                len: self.len()
            }),
            mem::transmute(raw::Slice {
                data: self.vals as *const V,
                len: self.len()
            })
        )}
    }

    #[inline]
    pub fn as_slices_mut<'a>(&'a mut self) -> (&'a mut [K], &'a mut [V]) {
        unsafe { mem::transmute(self.as_slices()) }
    }

    #[inline]
    pub fn as_slices_internal<'a>(&'a self) -> (&'a [K], &'a [V], &'a [Node<K, V>]) {
        let (keys, vals) = self.as_slices();
        let edges: &[_] = if self.is_leaf() {
            &[]
        } else {
            unsafe {
                mem::transmute(raw::Slice {
                    data: self.edges as *const Node<K, V>,
                    len: self.len() + 1
                })
            }
        };
        (keys, vals, edges)
    }

    #[inline]
    pub fn as_slices_internal_mut<'a>(&'a mut self) -> (&'a mut [K], &'a mut [V],
                                                        &'a mut [Node<K, V>]) {
        unsafe { mem::transmute(self.as_slices_internal()) }
    }

    #[inline]
    pub fn keys<'a>(&'a self) -> &'a [K] {
        self.as_slices().0
    }

    #[inline]
    pub fn keys_mut<'a>(&'a mut self) -> &'a mut [K] {
        self.as_slices_mut().0
    }

    #[inline]
    pub fn vals<'a>(&'a self) -> &'a [V] {
        self.as_slices().1
    }

    #[inline]
    pub fn vals_mut<'a>(&'a mut self) -> &'a mut [V] {
        self.as_slices_mut().1
    }

    #[inline]
    pub fn edges<'a>(&'a self) -> &'a [Node<K, V>] {
        self.as_slices_internal().2
    }

    #[inline]
    pub fn edges_mut<'a>(&'a mut self) -> &'a mut [Node<K, V>] {
        self.as_slices_internal_mut().2
    }
}

// FIXME(gereeter) Write an efficient clone_from
impl<K: Clone, V: Clone> Clone for Node<K, V> {
    fn clone(&self) -> Node<K, V> {
        let mut ret = if self.is_leaf() {
            Node::new_leaf(self.capacity())
        } else {
            unsafe { Node::new_internal(self.capacity()) }
        };

        unsafe {
            // For failure safety
            let mut keys = RawItems::from_parts(ret.keys().as_ptr(), 0);
            let mut vals = RawItems::from_parts(ret.vals().as_ptr(), 0);
            let mut edges = RawItems::from_parts(ret.edges().as_ptr(), 0);

            for key in self.keys().iter() {
                keys.push(key.clone())
            }
            for val in self.vals().iter() {
                vals.push(val.clone())
            }
            for edge in self.edges().iter() {
                edges.push(edge.clone())
            }

            mem::forget(keys);
            mem::forget(vals);
            mem::forget(edges);

            ret._len = self.len();
        }

        ret
    }
}

/// A reference to something in the middle of a `Node`. There are two `Type`s of `Handle`s,
/// namely `KV` handles, which point to key/value pairs, and `Edge` handles, which point to edges
/// before or after key/value pairs. Methods are provided for removing pairs, inserting into edges,
/// accessing the stored values, and moving around the `Node`.
///
/// This handle is generic, and can take any sort of reference to a `Node`. The reason for this is
/// two-fold. First of all, it reduces the amount of repetitive code, implementing functions that
/// don't need mutability on both mutable and immutable references. Secondly and more importantly,
/// this allows users of the `Handle` API to associate metadata with the reference. This is used in
/// `BTreeMap` to give `Node`s temporary "IDs" that persist to when the `Node` is used in a
/// `Handle`.
///
/// # A note on safety
///
/// Unfortunately, the extra power afforded by being generic also means that safety can technically
/// be broken. For sensible implementations of `Deref` and `DerefMut`, these handles are perfectly
/// safe. As long as repeatedly calling `.deref()` results in the same Node being returned each
/// time, everything should work fine. However, if the `Deref` implementation swaps in multiple
/// different nodes, then the indices that are assumed to be in bounds suddenly stop being so. For
/// example:
///
/// ```rust,ignore
/// struct Nasty<'a> {
///     first: &'a Node<uint, uint>,
///     second: &'a Node<uint, uint>,
///     flag: &'a Cell<bool>,
/// }
///
/// impl<'a> Deref<Node<uint, uint>> for Nasty<'a> {
///     fn deref(&self) -> &Node<uint, uint> {
///         if self.flag.get() {
///             &*self.second
///         } else {
///             &*self.first
///         }
///     }
/// }
///
/// fn main() {
///     let flag = Cell::new(false);
///     let mut small_node = Node::make_leaf_root(3);
///     let mut large_node = Node::make_leaf_root(100);
///
///     for i in range(0, 100) {
///         // Insert to the end
///         large_node.edge_handle(i).insert_as_leaf(i, i);
///     }
///
///     let nasty = Nasty {
///         first: &large_node,
///         second: &small_node,
///         flag: &flag
///     }
///
///     // The handle points at index 75.
///     let handle = Node::search(nasty, 75);
///
///     // Now the handle still points at index 75, but on the small node, which has no index 75.
///     flag.set(true);
///
///     println!("Uninitialized memory: {}", handle.into_kv());
/// }
/// ```
#[deriving(Copy)]
pub struct Handle<NodeRef, Type, NodeType> {
    node: NodeRef,
    index: uint
}

pub mod handle {
    // Handle types.
    pub enum KV {}
    pub enum Edge {}

    // Handle node types.
    pub enum LeafOrInternal {}
    pub enum Leaf {}
    pub enum Internal {}
}

impl<K: Ord, V> Node<K, V> {
    /// Searches for the given key in the node. If it finds an exact match,
    /// `Found` will be yielded with the matching index. If it doesn't find an exact match,
    /// `GoDown` will be yielded with the index of the subtree the key must lie in.
    pub fn search<Sized? Q, NodeRef: Deref<Node<K, V>>>(node: NodeRef, key: &Q)
                  -> SearchResult<NodeRef> where Q: BorrowFrom<K> + Ord {
        // FIXME(Gankro): Tune when to search linear or binary based on B (and maybe K/V).
        // For the B configured as of this writing (B = 6), binary search was *significantly*
        // worse for uints.
        let (found, index) = node.search_linear(key);
        if found {
            Found(Handle {
                node: node,
                index: index
            })
        } else {
            GoDown(Handle {
                node: node,
                index: index
            })
        }
    }

    fn search_linear<Sized? Q>(&self, key: &Q) -> (bool, uint) where Q: BorrowFrom<K> + Ord {
        for (i, k) in self.keys().iter().enumerate() {
            match key.cmp(BorrowFrom::borrow_from(k)) {
                Greater => {},
                Equal => return (true, i),
                Less => return (false, i),
            }
        }
        (false, self.len())
    }
}

// Public interface
impl <K, V> Node<K, V> {
    /// Make a leaf root from scratch
    pub fn make_leaf_root(b: uint) -> Node<K, V> {
        Node::new_leaf(capacity_from_b(b))
    }

    /// Make an internal root and swap it with an old root
    pub fn make_internal_root(left_and_out: &mut Node<K,V>, b: uint, key: K, value: V,
            right: Node<K,V>) {
        let node = mem::replace(left_and_out, unsafe { Node::new_internal(capacity_from_b(b)) });
        left_and_out._len = 1;
        unsafe {
            ptr::write(left_and_out.keys_mut().unsafe_mut(0), key);
            ptr::write(left_and_out.vals_mut().unsafe_mut(0), value);
            ptr::write(left_and_out.edges_mut().unsafe_mut(0), node);
            ptr::write(left_and_out.edges_mut().unsafe_mut(1), right);
        }
    }

    /// How many key-value pairs the node contains
    pub fn len(&self) -> uint {
        self._len
    }

    /// How many key-value pairs the node can fit
    pub fn capacity(&self) -> uint {
        self._capacity
    }

    /// If the node has any children
    pub fn is_leaf(&self) -> bool {
        self.edges.is_null()
    }

    /// if the node has too few elements
    pub fn is_underfull(&self) -> bool {
        self.len() < min_load_from_capacity(self.capacity())
    }

    /// if the node cannot fit any more elements
    pub fn is_full(&self) -> bool {
        self.len() == self.capacity()
    }
}

impl<K, V, NodeRef: Deref<Node<K, V>>, Type, NodeType> Handle<NodeRef, Type, NodeType> {
    /// Returns a reference to the node that contains the pointed-to edge or key/value pair. This
    /// is very different from `edge` and `edge_mut` because those return children of the node
    /// returned by `node`.
    pub fn node(&self) -> &Node<K, V> {
        &*self.node
    }
}

impl<K, V, NodeRef: DerefMut<Node<K, V>>, Type, NodeType> Handle<NodeRef, Type, NodeType> {
    /// Converts a handle into one that stores the same information using a raw pointer. This can
    /// be useful in conjunction with `from_raw` when the type system is insufficient for
    /// determining the lifetimes of the nodes.
    pub fn as_raw(&mut self) -> Handle<*mut Node<K, V>, Type, NodeType> {
        Handle {
            node: &mut *self.node as *mut _,
            index: self.index
        }
    }
}

impl<K, V, Type, NodeType> Handle<*mut Node<K, V>, Type, NodeType> {
    /// Converts from a handle stored with a raw pointer, which isn't directly usable, to a handle
    /// stored with a reference. This is an unsafe inverse of `as_raw`, and together they allow
    /// unsafely extending the lifetime of the reference to the `Node`.
    pub unsafe fn from_raw<'a>(&'a self) -> Handle<&'a Node<K, V>, Type, NodeType> {
        Handle {
            node: &*self.node,
            index: self.index
        }
    }

    /// Converts from a handle stored with a raw pointer, which isn't directly usable, to a handle
    /// stored with a mutable reference. This is an unsafe inverse of `as_raw`, and together they
    /// allow unsafely extending the lifetime of the reference to the `Node`.
    pub unsafe fn from_raw_mut<'a>(&'a mut self) -> Handle<&'a mut Node<K, V>, Type, NodeType> {
        Handle {
            node: &mut *self.node,
            index: self.index
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<&'a Node<K, V>, handle::Edge, handle::Internal> {
    /// Turns the handle into a reference to the edge it points at. This is necessary because the
    /// returned pointer has a larger lifetime than what would be returned by `edge` or `edge_mut`,
    /// making it more suitable for moving down a chain of nodes.
    pub fn into_edge(self) -> &'a Node<K, V> {
        unsafe {
            self.node.edges().unsafe_get(self.index)
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<&'a mut Node<K, V>, handle::Edge, handle::Internal> {
    /// Turns the handle into a mutable reference to the edge it points at. This is necessary
    /// because the returned pointer has a larger lifetime than what would be returned by
    /// `edge_mut`, making it more suitable for moving down a chain of nodes.
    pub fn into_edge_mut(self) -> &'a mut Node<K, V> {
        unsafe {
            self.node.edges_mut().unsafe_mut(self.index)
        }
    }
}

impl<K, V, NodeRef: Deref<Node<K, V>>> Handle<NodeRef, handle::Edge, handle::Internal> {
    // This doesn't exist because there are no uses for it,
    // but is fine to add, analagous to edge_mut.
    //
    // /// Returns a reference to the edge pointed-to by this handle. This should not be
    // /// confused with `node`, which references the parent node of what is returned here.
    // pub fn edge(&self) -> &Node<K, V>
}

pub enum ForceResult<NodeRef, Type> {
    Leaf(Handle<NodeRef, Type, handle::Leaf>),
    Internal(Handle<NodeRef, Type, handle::Internal>)
}

impl<K, V, NodeRef: Deref<Node<K, V>>, Type> Handle<NodeRef, Type, handle::LeafOrInternal> {
    /// Figure out whether this handle is pointing to something in a leaf node or to something in
    /// an internal node, clarifying the type according to the result.
    pub fn force(self) -> ForceResult<NodeRef, Type> {
        if self.node.is_leaf() {
            Leaf(Handle {
                node: self.node,
                index: self.index
            })
        } else {
            Internal(Handle {
                node: self.node,
                index: self.index
            })
        }
    }
}

impl<K, V, NodeRef: DerefMut<Node<K, V>>> Handle<NodeRef, handle::Edge, handle::Leaf> {
    /// Tries to insert this key-value pair at the given index in this leaf node
    /// If the node is full, we have to split it.
    ///
    /// Returns a *mut V to the inserted value, because the caller may want this when
    /// they're done mutating the tree, but we don't want to borrow anything for now.
    pub fn insert_as_leaf(mut self, key: K, value: V) ->
            (InsertionResult<K, V>, *mut V) {
        if !self.node.is_full() {
            // The element can fit, just insert it
            (Fit, unsafe { self.node.insert_kv(self.index, key, value) as *mut _ })
        } else {
            // The element can't fit, this node is full. Split it into two nodes.
            let (new_key, new_val, mut new_right) = self.node.split();
            let left_len = self.node.len();

            let ptr = unsafe {
                if self.index <= left_len {
                    self.node.insert_kv(self.index, key, value)
                } else {
                    // We need to subtract 1 because in splitting we took out new_key and new_val.
                    // Just being in the right node means we are past left_len k/v pairs in the
                    // left node and 1 k/v pair in the parent node.
                    new_right.insert_kv(self.index - left_len - 1, key, value)
                }
            } as *mut _;

            (Split(new_key, new_val, new_right), ptr)
        }
    }
}

impl<K, V, NodeRef: DerefMut<Node<K, V>>> Handle<NodeRef, handle::Edge, handle::Internal> {
    /// Returns a mutable reference to the edge pointed-to by this handle. This should not be
    /// confused with `node`, which references the parent node of what is returned here.
    pub fn edge_mut(&mut self) -> &mut Node<K, V> {
        unsafe {
            self.node.edges_mut().unsafe_mut(self.index)
        }
    }

    /// Tries to insert this key-value pair at the given index in this internal node
    /// If the node is full, we have to split it.
    pub fn insert_as_internal(mut self, key: K, value: V, right: Node<K, V>)
            -> InsertionResult<K, V> {
        if !self.node.is_full() {
            // The element can fit, just insert it
            unsafe {
                self.node.insert_kv(self.index, key, value);
                self.node.insert_edge(self.index + 1, right); // +1 to insert to the right
            }
            Fit
        } else {
            // The element can't fit, this node is full. Split it into two nodes.
            let (new_key, new_val, mut new_right) = self.node.split();
            let left_len = self.node.len();

            if self.index <= left_len {
                unsafe {
                    self.node.insert_kv(self.index, key, value);
                    self.node.insert_edge(self.index + 1, right); // +1 to insert to the right
                }
            } else {
                unsafe {
                    // The -1 here is for the same reason as in insert_as_internal - because we
                    // split, there are actually left_len + 1 k/v pairs before the right node, with
                    // the extra 1 being put in the parent.
                    new_right.insert_kv(self.index - left_len - 1, key, value);
                    new_right.insert_edge(self.index - left_len, right);
                }
            }

            Split(new_key, new_val, new_right)
        }
    }

    /// Handle an underflow in this node's child. We favour handling "to the left" because we know
    /// we're empty, but our neighbour can be full. Handling to the left means when we choose to
    /// steal, we pop off the end of our neighbour (always fast) and "unshift" ourselves
    /// (always slow, but at least faster since we know we're half-empty).
    /// Handling "to the right" reverses these roles. Of course, we merge whenever possible
    /// because we want dense nodes, and merging is about equal work regardless of direction.
    pub fn handle_underflow(mut self) {
        unsafe {
            if self.index > 0 {
                self.handle_underflow_to_left();
            } else {
                self.handle_underflow_to_right();
            }
        }
    }

    /// Right is underflowed. Tries to steal from left,
    /// but merges left and right if left is low too.
    unsafe fn handle_underflow_to_left(&mut self) {
        let left_len = self.node.edges()[self.index - 1].len();
        if left_len > min_load_from_capacity(self.node.capacity()) {
            self.left_kv().steal_rightward();
        } else {
            self.left_kv().merge_children();
        }
    }

    /// Left is underflowed. Tries to steal from the right,
    /// but merges left and right if right is low too.
    unsafe fn handle_underflow_to_right(&mut self) {
        let right_len = self.node.edges()[self.index + 1].len();
        if right_len > min_load_from_capacity(self.node.capacity()) {
            self.right_kv().steal_leftward();
        } else {
            self.right_kv().merge_children();
        }
    }
}

impl<K, V, NodeRef: DerefMut<Node<K, V>>, NodeType> Handle<NodeRef, handle::Edge, NodeType> {
    /// Gets the handle pointing to the key/value pair just to the left of the pointed-to edge.
    /// This is unsafe because the handle might point to the first edge in the node, which has no
    /// pair to its left.
    unsafe fn left_kv<'a>(&'a mut self) -> Handle<&'a mut Node<K, V>, handle::KV, NodeType> {
        Handle {
            node: &mut *self.node,
            index: self.index - 1
        }
    }

    /// Gets the handle pointing to the key/value pair just to the right of the pointed-to edge.
    /// This is unsafe because the handle might point to the last edge in the node, which has no
    /// pair to its right.
    unsafe fn right_kv<'a>(&'a mut self) -> Handle<&'a mut Node<K, V>, handle::KV, NodeType> {
        Handle {
            node: &mut *self.node,
            index: self.index
        }
    }
}

impl<'a, K: 'a, V: 'a, NodeType> Handle<&'a Node<K, V>, handle::KV, NodeType> {
    /// Turns the handle into references to the key and value it points at. This is necessary
    /// because the returned pointers have larger lifetimes than what would be returned by `key`
    /// or `val`.
    pub fn into_kv(self) -> (&'a K, &'a V) {
        let (keys, vals) = self.node.as_slices();
        unsafe {
            (
                keys.unsafe_get(self.index),
                vals.unsafe_get(self.index)
            )
        }
    }
}

impl<'a, K: 'a, V: 'a, NodeType> Handle<&'a mut Node<K, V>, handle::KV, NodeType> {
    /// Turns the handle into mutable references to the key and value it points at. This is
    /// necessary because the returned pointers have larger lifetimes than what would be returned
    /// by `key_mut` or `val_mut`.
    pub fn into_kv_mut(self) -> (&'a mut K, &'a mut V) {
        let (keys, vals) = self.node.as_slices_mut();
        unsafe {
            (
                keys.unsafe_mut(self.index),
                vals.unsafe_mut(self.index)
            )
        }
    }

    /// Convert this handle into one pointing at the edge immediately to the left of the key/value
    /// pair pointed-to by this handle. This is useful because it returns a reference with larger
    /// lifetime than `left_edge`.
    pub fn into_left_edge(self) -> Handle<&'a mut Node<K, V>, handle::Edge, NodeType> {
        Handle {
            node: &mut *self.node,
            index: self.index
        }
    }
}

impl<'a, K: 'a, V: 'a, NodeRef: Deref<Node<K, V>> + 'a, NodeType> Handle<NodeRef, handle::KV,
                                                                         NodeType> {
    // These are fine to include, but are currently unneeded.
    //
    // /// Returns a reference to the key pointed-to by this handle. This doesn't return a
    // /// reference with a lifetime as large as `into_kv_mut`, but it also does not consume the
    // /// handle.
    // pub fn key(&'a self) -> &'a K {
    //     unsafe { self.node.keys().unsafe_get(self.index) }
    // }
    //
    // /// Returns a reference to the value pointed-to by this handle. This doesn't return a
    // /// reference with a lifetime as large as `into_kv_mut`, but it also does not consume the
    // /// handle.
    // pub fn val(&'a self) -> &'a V {
    //     unsafe { self.node.vals().unsafe_get(self.index) }
    // }
}

impl<'a, K: 'a, V: 'a, NodeRef: DerefMut<Node<K, V>> + 'a, NodeType> Handle<NodeRef, handle::KV,
                                                                            NodeType> {
    /// Returns a mutable reference to the key pointed-to by this handle. This doesn't return a
    /// reference with a lifetime as large as `into_kv_mut`, but it also does not consume the
    /// handle.
    pub fn key_mut(&'a mut self) -> &'a mut K {
        unsafe { self.node.keys_mut().unsafe_mut(self.index) }
    }

    /// Returns a mutable reference to the value pointed-to by this handle. This doesn't return a
    /// reference with a lifetime as large as `into_kv_mut`, but it also does not consume the
    /// handle.
    pub fn val_mut(&'a mut self) -> &'a mut V {
        unsafe { self.node.vals_mut().unsafe_mut(self.index) }
    }
}

impl<K, V, NodeRef: DerefMut<Node<K, V>>, NodeType> Handle<NodeRef, handle::KV, NodeType> {
    /// Gets the handle pointing to the edge immediately to the left of the key/value pair pointed
    /// to by this handle.
    pub fn left_edge<'a>(&'a mut self) -> Handle<&'a mut Node<K, V>, handle::Edge, NodeType> {
        Handle {
            node: &mut *self.node,
            index: self.index
        }
    }

    /// Gets the handle pointing to the edge immediately to the right of the key/value pair pointed
    /// to by this handle.
    pub fn right_edge<'a>(&'a mut self) -> Handle<&'a mut Node<K, V>, handle::Edge, NodeType> {
        Handle {
            node: &mut *self.node,
            index: self.index + 1
        }
    }
}

impl<K, V, NodeRef: DerefMut<Node<K, V>>> Handle<NodeRef, handle::KV, handle::Leaf> {
    /// Removes the key/value pair at the handle's location.
    ///
    /// # Panics (in debug build)
    ///
    /// Panics if the node containing the pair is not a leaf node.
    pub fn remove_as_leaf(mut self) -> (K, V) {
        unsafe { self.node.remove_kv(self.index) }
    }
}

impl<K, V, NodeRef: DerefMut<Node<K, V>>> Handle<NodeRef, handle::KV, handle::Internal> {
    /// Steal! Stealing is roughly analogous to a binary tree rotation.
    /// In this case, we're "rotating" right.
    unsafe fn steal_rightward(&mut self) {
        // Take the biggest stuff off left
        let (mut key, mut val, edge) = {
            let mut left_handle = self.left_edge();
            let left = left_handle.edge_mut();
            let (key, val) = left.pop_kv();
            let edge = if left.is_leaf() {
                None
            } else {
                Some(left.pop_edge())
            };

            (key, val, edge)
        };

        // Swap the parent's separating key-value pair with left's
        mem::swap(&mut key, self.key_mut());
        mem::swap(&mut val, self.val_mut());

        // Put them at the start of right
        let mut right_handle = self.right_edge();
        let right = right_handle.edge_mut();
        right.insert_kv(0, key, val);
        match edge {
            Some(edge) => right.insert_edge(0, edge),
            None => {}
        }
    }

    /// Steal! Stealing is roughly analogous to a binary tree rotation.
    /// In this case, we're "rotating" left.
    unsafe fn steal_leftward(&mut self) {
        // Take the smallest stuff off right
        let (mut key, mut val, edge) = {
            let mut right_handle = self.right_edge();
            let right = right_handle.edge_mut();
            let (key, val) = right.remove_kv(0);
            let edge = if right.is_leaf() {
                None
            } else {
                Some(right.remove_edge(0))
            };

            (key, val, edge)
        };

        // Swap the parent's separating key-value pair with right's
        mem::swap(&mut key, self.key_mut());
        mem::swap(&mut val, self.val_mut());

        // Put them at the end of left
        let mut left_handle = self.left_edge();
        let left = left_handle.edge_mut();
        left.push_kv(key, val);
        match edge {
            Some(edge) => left.push_edge(edge),
            None => {}
        }
    }

    /// Merge! Smooshes left and right into one node, along with the key-value
    /// pair that separated them in their parent.
    unsafe fn merge_children(mut self) {
        // Permanently remove right's index, and the key-value pair that separates
        // left and right
        let (key, val) = self.node.remove_kv(self.index);
        let right = self.node.remove_edge(self.index + 1);

        // Give left right's stuff.
        self.left_edge().edge_mut()
            .absorb(key, val, right);
    }
}

impl<K, V> Node<K, V> {
    /// Returns the mutable handle pointing to the key/value pair at a given index.
    ///
    /// # Panics (in debug build)
    ///
    /// Panics if the given index is out of bounds.
    pub fn kv_handle(&mut self, index: uint) -> Handle<&mut Node<K, V>, handle::KV,
                                                       handle::LeafOrInternal> {
        // Necessary for correctness, but in a private module
        debug_assert!(index < self.len(), "kv_handle index out of bounds");
        Handle {
            node: self,
            index: index
        }
    }

    pub fn iter<'a>(&'a self) -> Traversal<'a, K, V> {
        let is_leaf = self.is_leaf();
        let (keys, vals, edges) = self.as_slices_internal();
        Traversal {
            inner: ElemsAndEdges(
                keys.iter().zip(vals.iter()),
                edges.iter()
            ),
            head_is_edge: true,
            tail_is_edge: true,
            has_edges: !is_leaf,
        }
    }

    pub fn iter_mut<'a>(&'a mut self) -> MutTraversal<'a, K, V> {
        let is_leaf = self.is_leaf();
        let (keys, vals, edges) = self.as_slices_internal_mut();
        MutTraversal {
            inner: ElemsAndEdges(
                keys.iter().zip(vals.iter_mut()),
                edges.iter_mut()
            ),
            head_is_edge: true,
            tail_is_edge: true,
            has_edges: !is_leaf,
        }
    }

    pub fn into_iter(self) -> MoveTraversal<K, V> {
        unsafe {
            let ret = MoveTraversal {
                inner: MoveTraversalImpl {
                    keys: RawItems::from_slice(self.keys()),
                    vals: RawItems::from_slice(self.vals()),
                    edges: RawItems::from_slice(self.edges()),

                    ptr: self.keys as *mut u8,
                    capacity: self.capacity(),
                    is_leaf: self.is_leaf()
                },
                head_is_edge: true,
                tail_is_edge: true,
                has_edges: !self.is_leaf(),
            };
            mem::forget(self);
            ret
        }
    }

    /// When a node has no keys or values and only a single edge, extract that edge.
    pub fn hoist_lone_child(&mut self) {
        // Necessary for correctness, but in a private module
        debug_assert!(self.len() == 0);
        debug_assert!(!self.is_leaf());

        unsafe {
            let ret = ptr::read(self.edges().unsafe_get(0));
            self.destroy();
            ptr::write(self, ret);
        }
    }
}

// Vector functions (all unchecked)
impl<K, V> Node<K, V> {
    // This must be followed by push_edge on an internal node.
    #[inline]
    unsafe fn push_kv(&mut self, key: K, val: V) {
        let len = self.len();

        ptr::write(self.keys_mut().unsafe_mut(len), key);
        ptr::write(self.vals_mut().unsafe_mut(len), val);

        self._len += 1;
    }

    // This can only be called immediately after a call to push_kv.
    #[inline]
    unsafe fn push_edge(&mut self, edge: Node<K, V>) {
        let len = self.len();

        ptr::write(self.edges_mut().unsafe_mut(len), edge);
    }

    // This must be followed by insert_edge on an internal node.
    #[inline]
    unsafe fn insert_kv(&mut self, index: uint, key: K, val: V) -> &mut V {
        ptr::copy_memory(
            self.keys_mut().as_mut_ptr().offset(index as int + 1),
            self.keys().as_ptr().offset(index as int),
            self.len() - index
        );
        ptr::copy_memory(
            self.vals_mut().as_mut_ptr().offset(index as int + 1),
            self.vals().as_ptr().offset(index as int),
            self.len() - index
        );

        ptr::write(self.keys_mut().unsafe_mut(index), key);
        ptr::write(self.vals_mut().unsafe_mut(index), val);

        self._len += 1;

        self.vals_mut().unsafe_mut(index)
    }

    // This can only be called immediately after a call to insert_kv.
    #[inline]
    unsafe fn insert_edge(&mut self, index: uint, edge: Node<K, V>) {
        ptr::copy_memory(
            self.edges_mut().as_mut_ptr().offset(index as int + 1),
            self.edges().as_ptr().offset(index as int),
            self.len() - index
        );
        ptr::write(self.edges_mut().unsafe_mut(index), edge);
    }

    // This must be followed by pop_edge on an internal node.
    #[inline]
    unsafe fn pop_kv(&mut self) -> (K, V) {
        let key = ptr::read(self.keys().unsafe_get(self.len() - 1));
        let val = ptr::read(self.vals().unsafe_get(self.len() - 1));

        self._len -= 1;

        (key, val)
    }

    // This can only be called immediately after a call to pop_kv.
    #[inline]
    unsafe fn pop_edge(&mut self) -> Node<K, V> {
        let edge = ptr::read(self.edges().unsafe_get(self.len() + 1));

        edge
    }

    // This must be followed by remove_edge on an internal node.
    #[inline]
    unsafe fn remove_kv(&mut self, index: uint) -> (K, V) {
        let key = ptr::read(self.keys().unsafe_get(index));
        let val = ptr::read(self.vals().unsafe_get(index));

        ptr::copy_memory(
            self.keys_mut().as_mut_ptr().offset(index as int),
            self.keys().as_ptr().offset(index as int + 1),
            self.len() - index - 1
        );
        ptr::copy_memory(
            self.vals_mut().as_mut_ptr().offset(index as int),
            self.vals().as_ptr().offset(index as int + 1),
            self.len() - index - 1
        );

        self._len -= 1;

        (key, val)
    }

    // This can only be called immediately after a call to remove_kv.
    #[inline]
    unsafe fn remove_edge(&mut self, index: uint) -> Node<K, V> {
        let edge = ptr::read(self.edges().unsafe_get(index));

        ptr::copy_memory(
            self.edges_mut().as_mut_ptr().offset(index as int),
            self.edges().as_ptr().offset(index as int + 1),
            self.len() - index + 1
        );

        edge
    }
}

// Private implementation details
impl<K, V> Node<K, V> {
    /// Node is full, so split it into two nodes, and yield the middle-most key-value pair
    /// because we have one too many, and our parent now has one too few
    fn split(&mut self) -> (K, V, Node<K, V>) {
        // Necessary for correctness, but in a private funtion
        debug_assert!(self.len() > 0);

        let mut right = if self.is_leaf() {
            Node::new_leaf(self.capacity())
        } else {
            unsafe { Node::new_internal(self.capacity()) }
        };

        unsafe {
            right._len = self.len() / 2;
            let right_offset = self.len() - right.len();
            ptr::copy_nonoverlapping_memory(
                right.keys_mut().as_mut_ptr(),
                self.keys().as_ptr().offset(right_offset as int),
                right.len()
            );
            ptr::copy_nonoverlapping_memory(
                right.vals_mut().as_mut_ptr(),
                self.vals().as_ptr().offset(right_offset as int),
                right.len()
            );
            if !self.is_leaf() {
                ptr::copy_nonoverlapping_memory(
                    right.edges_mut().as_mut_ptr(),
                    self.edges().as_ptr().offset(right_offset as int),
                    right.len() + 1
                );
            }

            let key = ptr::read(self.keys().unsafe_get(right_offset - 1));
            let val = ptr::read(self.vals().unsafe_get(right_offset - 1));

            self._len = right_offset - 1;

            (key, val, right)
        }
    }

    /// Take all the values from right, seperated by the given key and value
    fn absorb(&mut self, key: K, val: V, mut right: Node<K, V>) {
        // Necessary for correctness, but in a private function
        // Just as a sanity check, make sure we can fit this guy in
        debug_assert!(self.len() + right.len() <= self.capacity());
        debug_assert!(self.is_leaf() == right.is_leaf());

        unsafe {
            let old_len = self.len();
            self._len += right.len() + 1;

            ptr::write(self.keys_mut().unsafe_mut(old_len), key);
            ptr::write(self.vals_mut().unsafe_mut(old_len), val);

            ptr::copy_nonoverlapping_memory(
                self.keys_mut().as_mut_ptr().offset(old_len as int + 1),
                right.keys().as_ptr(),
                right.len()
            );
            ptr::copy_nonoverlapping_memory(
                self.vals_mut().as_mut_ptr().offset(old_len as int + 1),
                right.vals().as_ptr(),
                right.len()
            );
            if !self.is_leaf() {
                ptr::copy_nonoverlapping_memory(
                    self.edges_mut().as_mut_ptr().offset(old_len as int + 1),
                    right.edges().as_ptr(),
                    right.len() + 1
                );
            }

            right.destroy();
            mem::forget(right);
        }
    }
}

/// Get the capacity of a node from the order of the parent B-Tree
fn capacity_from_b(b: uint) -> uint {
    2 * b - 1
}

/// Get the minimum load of a node from its capacity
fn min_load_from_capacity(cap: uint) -> uint {
    // B - 1
    cap / 2
}

/// A trait for pairs of `Iterator`s, one over edges and the other over key/value pairs. This is
/// necessary, as the `MoveTraversalImpl` needs to have a destructor that deallocates the `Node`,
/// and a pair of `Iterator`s would require two independent destructors.
trait TraversalImpl<K, V, E> {
    fn next_kv(&mut self) -> Option<(K, V)>;
    fn next_kv_back(&mut self) -> Option<(K, V)>;

    fn next_edge(&mut self) -> Option<E>;
    fn next_edge_back(&mut self) -> Option<E>;
}

/// A `TraversalImpl` that actually is backed by two iterators. This works in the non-moving case,
/// as no deallocation needs to be done.
struct ElemsAndEdges<Elems, Edges>(Elems, Edges);

impl<K, V, E, Elems: DoubleEndedIterator<(K, V)>, Edges: DoubleEndedIterator<E>>
        TraversalImpl<K, V, E> for ElemsAndEdges<Elems, Edges> {

    fn next_kv(&mut self) -> Option<(K, V)> { self.0.next() }
    fn next_kv_back(&mut self) -> Option<(K, V)> { self.0.next_back() }

    fn next_edge(&mut self) -> Option<E> { self.1.next() }
    fn next_edge_back(&mut self) -> Option<E> { self.1.next_back() }
}

/// A `TraversalImpl` taking a `Node` by value.
struct MoveTraversalImpl<K, V> {
    keys: RawItems<K>,
    vals: RawItems<V>,
    edges: RawItems<Node<K, V>>,

    // For deallocation when we are done iterating.
    ptr: *mut u8,
    capacity: uint,
    is_leaf: bool
}

impl<K, V> TraversalImpl<K, V, Node<K, V>> for MoveTraversalImpl<K, V> {
    fn next_kv(&mut self) -> Option<(K, V)> {
        match (self.keys.next(), self.vals.next()) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None
        }
    }

    fn next_kv_back(&mut self) -> Option<(K, V)> {
        match (self.keys.next_back(), self.vals.next_back()) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None
        }
    }

    fn next_edge(&mut self) -> Option<Node<K, V>> {
        // Necessary for correctness, but in a private module
        debug_assert!(!self.is_leaf);
        self.edges.next()
    }

    fn next_edge_back(&mut self) -> Option<Node<K, V>> {
        // Necessary for correctness, but in a private module
        debug_assert!(!self.is_leaf);
        self.edges.next_back()
    }
}

#[unsafe_destructor]
impl<K, V> Drop for MoveTraversalImpl<K, V> {
    fn drop(&mut self) {
        // We need to cleanup the stored values manually, as the RawItems destructor would run
        // after our deallocation.
        for _ in self.keys {}
        for _ in self.vals {}
        for _ in self.edges {}

        let (alignment, size) =
                calculate_allocation_generic::<K, V>(self.capacity, self.is_leaf);
        unsafe { heap::deallocate(self.ptr, size, alignment) };
    }
}

/// An abstraction over all the different kinds of traversals a node supports
struct AbsTraversal<Impl> {
    inner: Impl,
    head_is_edge: bool,
    tail_is_edge: bool,
    has_edges: bool,
}

/// A single atomic step in a traversal. Either an element is visited, or an edge is followed
pub enum TraversalItem<K, V, E> {
    Elem(K, V),
    Edge(E),
}

/// A traversal over a node's entries and edges
pub type Traversal<'a, K, V> = AbsTraversal<ElemsAndEdges<Zip<slice::Items<'a, K>,
                                                              slice::Items<'a, V>>,
                                                              slice::Items<'a, Node<K, V>>>>;

/// A mutable traversal over a node's entries and edges
pub type MutTraversal<'a, K, V> = AbsTraversal<ElemsAndEdges<Zip<slice::Items<'a, K>,
                                                                 slice::MutItems<'a, V>>,
                                                                 slice::MutItems<'a, Node<K, V>>>>;

/// An owning traversal over a node's entries and edges
pub type MoveTraversal<K, V> = AbsTraversal<MoveTraversalImpl<K, V>>;


impl<K, V, E, Impl: TraversalImpl<K, V, E>>
        Iterator<TraversalItem<K, V, E>> for AbsTraversal<Impl> {

    fn next(&mut self) -> Option<TraversalItem<K, V, E>> {
        let head_is_edge = self.head_is_edge;
        self.head_is_edge = !head_is_edge;

        if head_is_edge && self.has_edges {
            self.inner.next_edge().map(|node| Edge(node))
        } else {
            self.inner.next_kv().map(|(k, v)| Elem(k, v))
        }
    }
}

impl<K, V, E, Impl: TraversalImpl<K, V, E>>
        DoubleEndedIterator<TraversalItem<K, V, E>> for AbsTraversal<Impl> {

    fn next_back(&mut self) -> Option<TraversalItem<K, V, E>> {
        let tail_is_edge = self.tail_is_edge;
        self.tail_is_edge = !tail_is_edge;

        if tail_is_edge && self.has_edges {
            self.inner.next_edge_back().map(|node| Edge(node))
        } else {
            self.inner.next_kv_back().map(|(k, v)| Elem(k, v))
        }
    }
}
