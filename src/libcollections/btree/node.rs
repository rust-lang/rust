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

use core::cmp::Ordering::{Greater, Less, Equal};
use core::intrinsics::arith_offset;
use core::iter::Zip;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::ptr::Unique;
use core::{slice, mem, ptr, cmp};
use alloc::heap::{self, EMPTY};

use borrow::Borrow;

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
    keys: Unique<K>,
    vals: Unique<V>,

    // In leaf nodes, this will be None, and no space will be allocated for edges.
    edges: Option<Unique<Node<K, V>>>,

    // At any given time, there will be `_len` keys, `_len` values, and (in an internal node)
    // `_len + 1` edges. In a leaf node, there will never be any edges.
    //
    // Note: instead of accessing this field directly, please call the `len()` method, which should
    // be more stable in the face of representation changes.
    _len: usize,

    // FIXME(gereeter) It shouldn't be necessary to store the capacity in every node, as it should
    // be constant throughout the tree. Once a solution to this is found, it might be possible to
    // also pass down the offsets into the buffer that vals and edges are stored at, removing the
    // need for those two pointers.
    //
    // Note: instead of accessing this field directly, please call the `capacity()` method, which
    // should be more stable in the face of representation changes.
    _capacity: usize,
}

struct NodeSlice<'a, K: 'a, V: 'a> {
    keys: &'a [K],
    vals: &'a [V],
    pub edges: &'a [Node<K, V>],
    head_is_edge: bool,
    tail_is_edge: bool,
    has_edges: bool,
}

struct MutNodeSlice<'a, K: 'a, V: 'a> {
    keys: &'a [K],
    vals: &'a mut [V],
    pub edges: &'a mut [Node<K, V>],
    head_is_edge: bool,
    tail_is_edge: bool,
    has_edges: bool,
}

/// Rounds up to a multiple of a power of two. Returns the closest multiple
/// of `target_alignment` that is higher or equal to `unrounded`.
///
/// # Panics
///
/// Fails if `target_alignment` is not a power of two.
#[inline]
fn round_up_to_next(unrounded: usize, target_alignment: usize) -> usize {
    assert!(target_alignment.is_power_of_two());
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
fn calculate_offsets(keys_size: usize,
                     vals_size: usize, vals_align: usize,
                     edges_align: usize)
                     -> (usize, usize) {
    let vals_offset = round_up_to_next(keys_size, vals_align);
    let end_of_vals = vals_offset + vals_size;

    let edges_offset = round_up_to_next(end_of_vals, edges_align);

    (vals_offset, edges_offset)
}

// Returns a tuple of (minimum required alignment, array_size),
// from the start of a mallocated array.
#[inline]
fn calculate_allocation(keys_size: usize, keys_align: usize,
                        vals_size: usize, vals_align: usize,
                        edges_size: usize, edges_align: usize)
                        -> (usize, usize) {
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

fn calculate_allocation_generic<K, V>(capacity: usize, is_leaf: bool) -> (usize, usize) {
    let (keys_size, keys_align) = (capacity * mem::size_of::<K>(), mem::align_of::<K>());
    let (vals_size, vals_align) = (capacity * mem::size_of::<V>(), mem::align_of::<V>());
    let (edges_size, edges_align) = if is_leaf {
        // allocate one edge to ensure that we don't pass size 0 to `heap::allocate`
        if mem::size_of::<K>() == 0 && mem::size_of::<V>() == 0 {
            (1, mem::align_of::<Node<K, V>>())
        } else {
            (0, 1)
        }
    } else {
        ((capacity + 1) * mem::size_of::<Node<K, V>>(), mem::align_of::<Node<K, V>>())
    };

    calculate_allocation(
            keys_size, keys_align,
            vals_size, vals_align,
            edges_size, edges_align
    )
}

fn calculate_offsets_generic<K, V>(capacity: usize, is_leaf: bool) -> (usize, usize) {
    let keys_size = capacity * mem::size_of::<K>();
    let vals_size = capacity * mem::size_of::<V>();
    let vals_align = mem::align_of::<V>();
    let edges_align = if is_leaf {
        1
    } else {
        mem::align_of::<Node<K, V>>()
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

    unsafe fn from_parts(ptr: *const T, len: usize) -> RawItems<T> {
        if mem::size_of::<T>() == 0 {
            RawItems {
                head: ptr,
                tail: arith_offset(ptr as *const i8, len as isize) as *const T,
            }
        } else {
            RawItems {
                head: ptr,
                tail: ptr.offset(len as isize),
            }
        }
    }

    unsafe fn push(&mut self, val: T) {
        ptr::write(self.tail as *mut T, val);

        if mem::size_of::<T>() == 0 {
            self.tail = arith_offset(self.tail as *const i8, 1) as *const T;
        } else {
            self.tail = self.tail.offset(1);
        }
    }
}

impl<T> Iterator for RawItems<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.head == self.tail {
            None
        } else {
            unsafe {
                let ret = Some(ptr::read(self.head));

                if mem::size_of::<T>() == 0 {
                    self.head = arith_offset(self.head as *const i8, 1) as *const T;
                } else {
                    self.head = self.head.offset(1);
                }

                ret
            }
        }
    }
}

impl<T> DoubleEndedIterator for RawItems<T> {
    fn next_back(&mut self) -> Option<T> {
        if self.head == self.tail {
            None
        } else {
            unsafe {
                if mem::size_of::<T>() == 0 {
                    self.tail = arith_offset(self.tail as *const i8, -1) as *const T;
                } else {
                    self.tail = self.tail.offset(-1);
                }

                Some(ptr::read(self.tail))
            }
        }
    }
}

impl<T> Drop for RawItems<T> {
    #[unsafe_destructor_blind_to_params]
    fn drop(&mut self) {
        for _ in self {}
    }
}

impl<K, V> Drop for Node<K, V> {
    #[unsafe_destructor_blind_to_params]
    fn drop(&mut self) {
        if self.keys.is_null() ||
            (unsafe { self.keys.get() as *const K as usize == mem::POST_DROP_USIZE })
        {
            // Since we have #[unsafe_no_drop_flag], we have to watch
            // out for the sentinel value being stored in self.keys. (Using
            // null is technically a violation of the `Unique`
            // requirements, though.)
            return;
        }

        // Do the actual cleanup.
        unsafe {
            drop(RawItems::from_slice(self.keys()));
            drop(RawItems::from_slice(self.vals()));
            drop(RawItems::from_slice(self.edges()));

            self.destroy();
        }

        self.keys = unsafe { Unique::new(ptr::null_mut()) };
    }
}

impl<K, V> Node<K, V> {
    /// Make a new internal node. The caller must initialize the result to fix the invariant that
    /// there are `len() + 1` edges.
    unsafe fn new_internal(capacity: usize) -> Node<K, V> {
        let (alignment, size) = calculate_allocation_generic::<K, V>(capacity, false);

        let buffer = heap::allocate(size, alignment);
        if buffer.is_null() { ::alloc::oom(); }

        let (vals_offset, edges_offset) = calculate_offsets_generic::<K, V>(capacity, false);

        Node {
            keys: Unique::new(buffer as *mut K),
            vals: Unique::new(buffer.offset(vals_offset as isize) as *mut V),
            edges: Some(Unique::new(buffer.offset(edges_offset as isize) as *mut Node<K, V>)),
            _len: 0,
            _capacity: capacity,
        }
    }

    /// Make a new leaf node
    fn new_leaf(capacity: usize) -> Node<K, V> {
        let (alignment, size) = calculate_allocation_generic::<K, V>(capacity, true);

        let buffer = unsafe { heap::allocate(size, alignment) };
        if buffer.is_null() { ::alloc::oom(); }

        let (vals_offset, _) = calculate_offsets_generic::<K, V>(capacity, true);

        Node {
            keys: unsafe { Unique::new(buffer as *mut K) },
            vals: unsafe { Unique::new(buffer.offset(vals_offset as isize) as *mut V) },
            edges: None,
            _len: 0,
            _capacity: capacity,
        }
    }

    unsafe fn destroy(&mut self) {
        let (alignment, size) =
                calculate_allocation_generic::<K, V>(self.capacity(), self.is_leaf());
        heap::deallocate(*self.keys as *mut u8, size, alignment);
    }

    #[inline]
    pub fn as_slices<'a>(&'a self) -> (&'a [K], &'a [V]) {
        unsafe {(
            slice::from_raw_parts(*self.keys, self.len()),
            slice::from_raw_parts(*self.vals, self.len()),
        )}
    }

    #[inline]
    pub fn as_slices_mut<'a>(&'a mut self) -> (&'a mut [K], &'a mut [V]) {
        unsafe {(
            slice::from_raw_parts_mut(*self.keys, self.len()),
            slice::from_raw_parts_mut(*self.vals, self.len()),
        )}
    }

    #[inline]
    pub fn as_slices_internal<'b>(&'b self) -> NodeSlice<'b, K, V> {
        let is_leaf = self.is_leaf();
        let (keys, vals) = self.as_slices();
        let edges: &[_] = if self.is_leaf() {
            &[]
        } else {
            unsafe {
                let data = match self.edges {
                    None => heap::EMPTY as *const Node<K,V>,
                    Some(ref p) => **p as *const Node<K,V>,
                };
                slice::from_raw_parts(data, self.len() + 1)
            }
        };
        NodeSlice {
            keys: keys,
            vals: vals,
            edges: edges,
            head_is_edge: true,
            tail_is_edge: true,
            has_edges: !is_leaf,
        }
    }

    #[inline]
    pub fn as_slices_internal_mut<'b>(&'b mut self) -> MutNodeSlice<'b, K, V> {
        let len = self.len();
        let is_leaf = self.is_leaf();
        let keys = unsafe { slice::from_raw_parts_mut(*self.keys, len) };
        let vals = unsafe { slice::from_raw_parts_mut(*self.vals, len) };
        let edges: &mut [_] = if is_leaf {
            &mut []
        } else {
            unsafe {
                let data = match self.edges {
                    None => heap::EMPTY as *mut Node<K,V>,
                    Some(ref mut p) => **p as *mut Node<K,V>,
                };
                slice::from_raw_parts_mut(data, len + 1)
            }
        };
        MutNodeSlice {
            keys: keys,
            vals: vals,
            edges: edges,
            head_is_edge: true,
            tail_is_edge: true,
            has_edges: !is_leaf,
        }
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
        self.as_slices_internal().edges
    }

    #[inline]
    pub fn edges_mut<'a>(&'a mut self) -> &'a mut [Node<K, V>] {
        self.as_slices_internal_mut().edges
    }
}

// FIXME(gereeter) Write an efficient clone_from
#[stable(feature = "rust1", since = "1.0.0")]
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

            for key in self.keys() {
                keys.push(key.clone())
            }
            for val in self.vals() {
                vals.push(val.clone())
            }
            for edge in self.edges() {
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
///     first: &'a Node<usize, usize>,
///     second: &'a Node<usize, usize>,
///     flag: &'a Cell<bool>,
/// }
///
/// impl<'a> Deref for Nasty<'a> {
///     type Target = Node<usize, usize>;
///
///     fn deref(&self) -> &Node<usize, usize> {
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
///     for i in 0..100 {
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
///     println!("Uninitialized memory: {:?}", handle.into_kv());
/// }
/// ```
#[derive(Copy, Clone)]
pub struct Handle<NodeRef, Type, NodeType> {
    node: NodeRef,
    index: usize,
    marker: PhantomData<(Type, NodeType)>,
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
    pub fn search<Q: ?Sized, NodeRef: Deref<Target=Node<K, V>>>(node: NodeRef, key: &Q)
                  -> SearchResult<NodeRef> where K: Borrow<Q>, Q: Ord {
        // FIXME(Gankro): Tune when to search linear or binary based on B (and maybe K/V).
        // For the B configured as of this writing (B = 6), binary search was *significantly*
        // worse for usizes.
        match node.as_slices_internal().search_linear(key) {
            (index, true) => Found(Handle { node: node, index: index, marker: PhantomData }),
            (index, false) => GoDown(Handle { node: node, index: index, marker: PhantomData }),
        }
    }
}

// Public interface
impl <K, V> Node<K, V> {
    /// Make a leaf root from scratch
    pub fn make_leaf_root(b: usize) -> Node<K, V> {
        Node::new_leaf(capacity_from_b(b))
    }

    /// Make an internal root and swap it with an old root
    pub fn make_internal_root(left_and_out: &mut Node<K,V>, b: usize, key: K, value: V,
            right: Node<K,V>) {
        let node = mem::replace(left_and_out, unsafe { Node::new_internal(capacity_from_b(b)) });
        left_and_out._len = 1;
        unsafe {
            ptr::write(left_and_out.keys_mut().get_unchecked_mut(0), key);
            ptr::write(left_and_out.vals_mut().get_unchecked_mut(0), value);
            ptr::write(left_and_out.edges_mut().get_unchecked_mut(0), node);
            ptr::write(left_and_out.edges_mut().get_unchecked_mut(1), right);
        }
    }

    /// How many key-value pairs the node contains
    pub fn len(&self) -> usize {
        self._len
    }

    /// Does the node not contain any key-value pairs
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// How many key-value pairs the node can fit
    pub fn capacity(&self) -> usize {
        self._capacity
    }

    /// If the node has any children
    pub fn is_leaf(&self) -> bool {
        self.edges.is_none()
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

impl<K, V, NodeRef: Deref<Target=Node<K, V>>, Type, NodeType> Handle<NodeRef, Type, NodeType> {
    /// Returns a reference to the node that contains the pointed-to edge or key/value pair. This
    /// is very different from `edge` and `edge_mut` because those return children of the node
    /// returned by `node`.
    pub fn node(&self) -> &Node<K, V> {
        &*self.node
    }
}

impl<K, V, NodeRef, Type, NodeType> Handle<NodeRef, Type, NodeType> where
    NodeRef: Deref<Target=Node<K, V>> + DerefMut,
{
    /// Converts a handle into one that stores the same information using a raw pointer. This can
    /// be useful in conjunction with `from_raw` when the type system is insufficient for
    /// determining the lifetimes of the nodes.
    pub fn as_raw(&mut self) -> Handle<*mut Node<K, V>, Type, NodeType> {
        Handle {
            node: &mut *self.node as *mut _,
            index: self.index,
            marker: PhantomData,
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
            index: self.index,
            marker: PhantomData,
        }
    }

    /// Converts from a handle stored with a raw pointer, which isn't directly usable, to a handle
    /// stored with a mutable reference. This is an unsafe inverse of `as_raw`, and together they
    /// allow unsafely extending the lifetime of the reference to the `Node`.
    pub unsafe fn from_raw_mut<'a>(&'a mut self) -> Handle<&'a mut Node<K, V>, Type, NodeType> {
        Handle {
            node: &mut *self.node,
            index: self.index,
            marker: PhantomData,
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<&'a Node<K, V>, handle::Edge, handle::Internal> {
    /// Turns the handle into a reference to the edge it points at. This is necessary because the
    /// returned pointer has a larger lifetime than what would be returned by `edge` or `edge_mut`,
    /// making it more suitable for moving down a chain of nodes.
    pub fn into_edge(self) -> &'a Node<K, V> {
        unsafe {
            self.node.edges().get_unchecked(self.index)
        }
    }
}

impl<'a, K: 'a, V: 'a> Handle<&'a mut Node<K, V>, handle::Edge, handle::Internal> {
    /// Turns the handle into a mutable reference to the edge it points at. This is necessary
    /// because the returned pointer has a larger lifetime than what would be returned by
    /// `edge_mut`, making it more suitable for moving down a chain of nodes.
    pub fn into_edge_mut(self) -> &'a mut Node<K, V> {
        unsafe {
            self.node.edges_mut().get_unchecked_mut(self.index)
        }
    }
}

impl<K, V, NodeRef: Deref<Target=Node<K, V>>> Handle<NodeRef, handle::Edge, handle::Internal> {
    // This doesn't exist because there are no uses for it,
    // but is fine to add, analogous to edge_mut.
    //
    // /// Returns a reference to the edge pointed-to by this handle. This should not be
    // /// confused with `node`, which references the parent node of what is returned here.
    // pub fn edge(&self) -> &Node<K, V>
}

pub enum ForceResult<NodeRef, Type> {
    Leaf(Handle<NodeRef, Type, handle::Leaf>),
    Internal(Handle<NodeRef, Type, handle::Internal>)
}

impl<K, V, NodeRef: Deref<Target=Node<K, V>>, Type> Handle<NodeRef, Type, handle::LeafOrInternal> {
    /// Figure out whether this handle is pointing to something in a leaf node or to something in
    /// an internal node, clarifying the type according to the result.
    pub fn force(self) -> ForceResult<NodeRef, Type> {
        if self.node.is_leaf() {
            Leaf(Handle {
                node: self.node,
                index: self.index,
                marker: PhantomData,
            })
        } else {
            Internal(Handle {
                node: self.node,
                index: self.index,
                marker: PhantomData,
            })
        }
    }
}
impl<K, V, NodeRef> Handle<NodeRef, handle::Edge, handle::Leaf> where
    NodeRef: Deref<Target=Node<K, V>> + DerefMut,
{
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

impl<K, V, NodeRef> Handle<NodeRef, handle::Edge, handle::Internal> where
    NodeRef: Deref<Target=Node<K, V>> + DerefMut,
{
    /// Returns a mutable reference to the edge pointed-to by this handle. This should not be
    /// confused with `node`, which references the parent node of what is returned here.
    pub fn edge_mut(&mut self) -> &mut Node<K, V> {
        unsafe {
            self.node.edges_mut().get_unchecked_mut(self.index)
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

    /// Handle an underflow in this node's child. We favor handling "to the left" because we know
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

impl<K, V, NodeRef, NodeType> Handle<NodeRef, handle::Edge, NodeType> where
    NodeRef: Deref<Target=Node<K, V>> + DerefMut,
{
    /// Gets the handle pointing to the key/value pair just to the left of the pointed-to edge.
    /// This is unsafe because the handle might point to the first edge in the node, which has no
    /// pair to its left.
    unsafe fn left_kv<'a>(&'a mut self) -> Handle<&'a mut Node<K, V>, handle::KV, NodeType> {
        Handle {
            node: &mut *self.node,
            index: self.index - 1,
            marker: PhantomData,
        }
    }

    /// Gets the handle pointing to the key/value pair just to the right of the pointed-to edge.
    /// This is unsafe because the handle might point to the last edge in the node, which has no
    /// pair to its right.
    unsafe fn right_kv<'a>(&'a mut self) -> Handle<&'a mut Node<K, V>, handle::KV, NodeType> {
        Handle {
            node: &mut *self.node,
            index: self.index,
            marker: PhantomData,
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
                keys.get_unchecked(self.index),
                vals.get_unchecked(self.index)
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
                keys.get_unchecked_mut(self.index),
                vals.get_unchecked_mut(self.index)
            )
        }
    }

    /// Convert this handle into one pointing at the edge immediately to the left of the key/value
    /// pair pointed-to by this handle. This is useful because it returns a reference with larger
    /// lifetime than `left_edge`.
    pub fn into_left_edge(self) -> Handle<&'a mut Node<K, V>, handle::Edge, NodeType> {
        Handle {
            node: &mut *self.node,
            index: self.index,
            marker: PhantomData,
        }
    }
}

impl<'a, K: 'a, V: 'a, NodeRef: Deref<Target=Node<K, V>> + 'a, NodeType> Handle<NodeRef, handle::KV,
                                                                         NodeType> {
    // These are fine to include, but are currently unneeded.
    //
    // /// Returns a reference to the key pointed-to by this handle. This doesn't return a
    // /// reference with a lifetime as large as `into_kv_mut`, but it also does not consume the
    // /// handle.
    // pub fn key(&'a self) -> &'a K {
    //     unsafe { self.node.keys().get_unchecked(self.index) }
    // }
    //
    // /// Returns a reference to the value pointed-to by this handle. This doesn't return a
    // /// reference with a lifetime as large as `into_kv_mut`, but it also does not consume the
    // /// handle.
    // pub fn val(&'a self) -> &'a V {
    //     unsafe { self.node.vals().get_unchecked(self.index) }
    // }
}

impl<'a, K: 'a, V: 'a, NodeRef, NodeType> Handle<NodeRef, handle::KV, NodeType> where
    NodeRef: 'a + Deref<Target=Node<K, V>> + DerefMut,
{
    /// Returns a mutable reference to the key pointed-to by this handle. This doesn't return a
    /// reference with a lifetime as large as `into_kv_mut`, but it also does not consume the
    /// handle.
    pub fn key_mut(&'a mut self) -> &'a mut K {
        unsafe { self.node.keys_mut().get_unchecked_mut(self.index) }
    }

    /// Returns a mutable reference to the value pointed-to by this handle. This doesn't return a
    /// reference with a lifetime as large as `into_kv_mut`, but it also does not consume the
    /// handle.
    pub fn val_mut(&'a mut self) -> &'a mut V {
        unsafe { self.node.vals_mut().get_unchecked_mut(self.index) }
    }
}

impl<K, V, NodeRef, NodeType> Handle<NodeRef, handle::KV, NodeType> where
    NodeRef: Deref<Target=Node<K, V>> + DerefMut,
{
    /// Gets the handle pointing to the edge immediately to the left of the key/value pair pointed
    /// to by this handle.
    pub fn left_edge<'a>(&'a mut self) -> Handle<&'a mut Node<K, V>, handle::Edge, NodeType> {
        Handle {
            node: &mut *self.node,
            index: self.index,
            marker: PhantomData,
        }
    }

    /// Gets the handle pointing to the edge immediately to the right of the key/value pair pointed
    /// to by this handle.
    pub fn right_edge<'a>(&'a mut self) -> Handle<&'a mut Node<K, V>, handle::Edge, NodeType> {
        Handle {
            node: &mut *self.node,
            index: self.index + 1,
            marker: PhantomData,
        }
    }
}

impl<K, V, NodeRef> Handle<NodeRef, handle::KV, handle::Leaf> where
    NodeRef: Deref<Target=Node<K, V>> + DerefMut,
{
    /// Removes the key/value pair at the handle's location.
    ///
    /// # Panics (in debug build)
    ///
    /// Panics if the node containing the pair is not a leaf node.
    pub fn remove_as_leaf(mut self) -> (K, V) {
        unsafe { self.node.remove_kv(self.index) }
    }
}

impl<K, V, NodeRef> Handle<NodeRef, handle::KV, handle::Internal> where
    NodeRef: Deref<Target=Node<K, V>> + DerefMut
{
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
    pub fn kv_handle(&mut self, index: usize) -> Handle<&mut Node<K, V>, handle::KV,
                                                       handle::LeafOrInternal> {
        // Necessary for correctness, but in a private module
        debug_assert!(index < self.len(), "kv_handle index out of bounds");
        Handle {
            node: self,
            index: index,
            marker: PhantomData,
        }
    }

    pub fn iter<'a>(&'a self) -> Traversal<'a, K, V> {
        self.as_slices_internal().iter()
    }

    pub fn iter_mut<'a>(&'a mut self) -> MutTraversal<'a, K, V> {
        self.as_slices_internal_mut().iter_mut()
    }

    pub fn into_iter(self) -> MoveTraversal<K, V> {
        unsafe {
            let ret = MoveTraversal {
                inner: MoveTraversalImpl {
                    keys: RawItems::from_slice(self.keys()),
                    vals: RawItems::from_slice(self.vals()),
                    edges: RawItems::from_slice(self.edges()),

                    ptr: Unique::new(*self.keys as *mut u8),
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
        debug_assert!(self.is_empty());
        debug_assert!(!self.is_leaf());

        unsafe {
            let ret = ptr::read(self.edges().get_unchecked(0));
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

        ptr::write(self.keys_mut().get_unchecked_mut(len), key);
        ptr::write(self.vals_mut().get_unchecked_mut(len), val);

        self._len += 1;
    }

    // This can only be called immediately after a call to push_kv.
    #[inline]
    unsafe fn push_edge(&mut self, edge: Node<K, V>) {
        let len = self.len();

        ptr::write(self.edges_mut().get_unchecked_mut(len), edge);
    }

    // This must be followed by insert_edge on an internal node.
    #[inline]
    unsafe fn insert_kv(&mut self, index: usize, key: K, val: V) -> &mut V {
        ptr::copy(
            self.keys().as_ptr().offset(index as isize),
            self.keys_mut().as_mut_ptr().offset(index as isize + 1),
            self.len() - index
        );
        ptr::copy(
            self.vals().as_ptr().offset(index as isize),
            self.vals_mut().as_mut_ptr().offset(index as isize + 1),
            self.len() - index
        );

        ptr::write(self.keys_mut().get_unchecked_mut(index), key);
        ptr::write(self.vals_mut().get_unchecked_mut(index), val);

        self._len += 1;

        self.vals_mut().get_unchecked_mut(index)
    }

    // This can only be called immediately after a call to insert_kv.
    #[inline]
    unsafe fn insert_edge(&mut self, index: usize, edge: Node<K, V>) {
        ptr::copy(
            self.edges().as_ptr().offset(index as isize),
            self.edges_mut().as_mut_ptr().offset(index as isize + 1),
            self.len() - index
        );
        ptr::write(self.edges_mut().get_unchecked_mut(index), edge);
    }

    // This must be followed by pop_edge on an internal node.
    #[inline]
    unsafe fn pop_kv(&mut self) -> (K, V) {
        let key = ptr::read(self.keys().get_unchecked(self.len() - 1));
        let val = ptr::read(self.vals().get_unchecked(self.len() - 1));

        self._len -= 1;

        (key, val)
    }

    // This can only be called immediately after a call to pop_kv.
    #[inline]
    unsafe fn pop_edge(&mut self) -> Node<K, V> {
        let edge = ptr::read(self.edges().get_unchecked(self.len() + 1));

        edge
    }

    // This must be followed by remove_edge on an internal node.
    #[inline]
    unsafe fn remove_kv(&mut self, index: usize) -> (K, V) {
        let key = ptr::read(self.keys().get_unchecked(index));
        let val = ptr::read(self.vals().get_unchecked(index));

        ptr::copy(
            self.keys().as_ptr().offset(index as isize + 1),
            self.keys_mut().as_mut_ptr().offset(index as isize),
            self.len() - index - 1
        );
        ptr::copy(
            self.vals().as_ptr().offset(index as isize + 1),
            self.vals_mut().as_mut_ptr().offset(index as isize),
            self.len() - index - 1
        );

        self._len -= 1;

        (key, val)
    }

    // This can only be called immediately after a call to remove_kv.
    #[inline]
    unsafe fn remove_edge(&mut self, index: usize) -> Node<K, V> {
        let edge = ptr::read(self.edges().get_unchecked(index));

        ptr::copy(
            self.edges().as_ptr().offset(index as isize + 1),
            self.edges_mut().as_mut_ptr().offset(index as isize),
            // index can be == len+1, so do the +1 first to avoid underflow.
            (self.len() + 1) - index
        );

        edge
    }
}

// Private implementation details
impl<K, V> Node<K, V> {
    /// Node is full, so split it into two nodes, and yield the middle-most key-value pair
    /// because we have one too many, and our parent now has one too few
    fn split(&mut self) -> (K, V, Node<K, V>) {
        // Necessary for correctness, but in a private function
        debug_assert!(!self.is_empty());

        let mut right = if self.is_leaf() {
            Node::new_leaf(self.capacity())
        } else {
            unsafe { Node::new_internal(self.capacity()) }
        };

        unsafe {
            right._len = self.len() / 2;
            let right_offset = self.len() - right.len();
            ptr::copy_nonoverlapping(
                self.keys().as_ptr().offset(right_offset as isize),
                right.keys_mut().as_mut_ptr(),
                right.len()
            );
            ptr::copy_nonoverlapping(
                self.vals().as_ptr().offset(right_offset as isize),
                right.vals_mut().as_mut_ptr(),
                right.len()
            );
            if !self.is_leaf() {
                ptr::copy_nonoverlapping(
                    self.edges().as_ptr().offset(right_offset as isize),
                    right.edges_mut().as_mut_ptr(),
                    right.len() + 1
                );
            }

            let key = ptr::read(self.keys().get_unchecked(right_offset - 1));
            let val = ptr::read(self.vals().get_unchecked(right_offset - 1));

            self._len = right_offset - 1;

            (key, val, right)
        }
    }

    /// Take all the values from right, separated by the given key and value
    fn absorb(&mut self, key: K, val: V, mut right: Node<K, V>) {
        // Necessary for correctness, but in a private function
        // Just as a sanity check, make sure we can fit this guy in
        debug_assert!(self.len() + right.len() <= self.capacity());
        debug_assert!(self.is_leaf() == right.is_leaf());

        unsafe {
            let old_len = self.len();
            self._len += right.len() + 1;

            ptr::write(self.keys_mut().get_unchecked_mut(old_len), key);
            ptr::write(self.vals_mut().get_unchecked_mut(old_len), val);

            ptr::copy_nonoverlapping(
                right.keys().as_ptr(),
                self.keys_mut().as_mut_ptr().offset(old_len as isize + 1),
                right.len()
            );
            ptr::copy_nonoverlapping(
                right.vals().as_ptr(),
                self.vals_mut().as_mut_ptr().offset(old_len as isize + 1),
                right.len()
            );
            if !self.is_leaf() {
                ptr::copy_nonoverlapping(
                    right.edges().as_ptr(),
                    self.edges_mut().as_mut_ptr().offset(old_len as isize + 1),
                    right.len() + 1
                );
            }

            right.destroy();
            mem::forget(right);
        }
    }
}

/// Get the capacity of a node from the order of the parent B-Tree
fn capacity_from_b(b: usize) -> usize {
    2 * b - 1
}

/// Get the minimum load of a node from its capacity
fn min_load_from_capacity(cap: usize) -> usize {
    // B - 1
    cap / 2
}

/// A trait for pairs of `Iterator`s, one over edges and the other over key/value pairs. This is
/// necessary, as the `MoveTraversalImpl` needs to have a destructor that deallocates the `Node`,
/// and a pair of `Iterator`s would require two independent destructors.
trait TraversalImpl {
    type Item;
    type Edge;

    fn next_kv(&mut self) -> Option<Self::Item>;
    fn next_kv_back(&mut self) -> Option<Self::Item>;

    fn next_edge(&mut self) -> Option<Self::Edge>;
    fn next_edge_back(&mut self) -> Option<Self::Edge>;
}

/// A `TraversalImpl` that actually is backed by two iterators. This works in the non-moving case,
/// as no deallocation needs to be done.
#[derive(Clone)]
struct ElemsAndEdges<Elems, Edges>(Elems, Edges);

impl<K, V, E, Elems: DoubleEndedIterator, Edges: DoubleEndedIterator>
        TraversalImpl for ElemsAndEdges<Elems, Edges>
    where Elems : Iterator<Item=(K, V)>, Edges : Iterator<Item=E>
{
    type Item = (K, V);
    type Edge = E;

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
    ptr: Unique<u8>,
    capacity: usize,
    is_leaf: bool
}

unsafe impl<K: Sync, V: Sync> Sync for MoveTraversalImpl<K, V> {}
unsafe impl<K: Send, V: Send> Send for MoveTraversalImpl<K, V> {}

impl<K, V> TraversalImpl for MoveTraversalImpl<K, V> {
    type Item = (K, V);
    type Edge = Node<K, V>;

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

impl<K, V> Drop for MoveTraversalImpl<K, V> {
    #[unsafe_destructor_blind_to_params]
    fn drop(&mut self) {
        // We need to cleanup the stored values manually, as the RawItems destructor would run
        // after our deallocation.
        for _ in self.keys.by_ref() {}
        for _ in self.vals.by_ref() {}
        for _ in self.edges.by_ref() {}

        let (alignment, size) =
                calculate_allocation_generic::<K, V>(self.capacity, self.is_leaf);
        unsafe { heap::deallocate(*self.ptr, size, alignment) };
    }
}

/// An abstraction over all the different kinds of traversals a node supports
#[derive(Clone)]
struct AbsTraversal<Impl> {
    inner: Impl,
    head_is_edge: bool,
    tail_is_edge: bool,
    has_edges: bool,
}

/// A single atomic step in a traversal.
pub enum TraversalItem<K, V, E> {
    /// An element is visited. This isn't written as `Elem(K, V)` just because `opt.map(Elem)`
    /// requires the function to take a single argument. (Enum constructors are functions.)
    Elem((K, V)),
    /// An edge is followed.
    Edge(E),
}

/// A traversal over a node's entries and edges
pub type Traversal<'a, K, V> = AbsTraversal<ElemsAndEdges<Zip<slice::Iter<'a, K>,
                                                              slice::Iter<'a, V>>,
                                                          slice::Iter<'a, Node<K, V>>>>;

/// A mutable traversal over a node's entries and edges
pub type MutTraversal<'a, K, V> = AbsTraversal<ElemsAndEdges<Zip<slice::Iter<'a, K>,
                                                                 slice::IterMut<'a, V>>,
                                                             slice::IterMut<'a, Node<K, V>>>>;

/// An owning traversal over a node's entries and edges
pub type MoveTraversal<K, V> = AbsTraversal<MoveTraversalImpl<K, V>>;


impl<K, V, E, Impl> Iterator for AbsTraversal<Impl>
        where Impl: TraversalImpl<Item=(K, V), Edge=E> {
    type Item = TraversalItem<K, V, E>;

    fn next(&mut self) -> Option<TraversalItem<K, V, E>> {
        self.next_edge_item().map(Edge).or_else(||
            self.next_kv_item().map(Elem)
        )
    }
}

impl<K, V, E, Impl> DoubleEndedIterator for AbsTraversal<Impl>
        where Impl: TraversalImpl<Item=(K, V), Edge=E> {
    fn next_back(&mut self) -> Option<TraversalItem<K, V, E>> {
        self.next_edge_item_back().map(Edge).or_else(||
            self.next_kv_item_back().map(Elem)
        )
    }
}

impl<K, V, E, Impl> AbsTraversal<Impl>
        where Impl: TraversalImpl<Item=(K, V), Edge=E> {
    /// Advances the iterator and returns the item if it's an edge. Returns None
    /// and does nothing if the first item is not an edge.
    pub fn next_edge_item(&mut self) -> Option<E> {
        // NB. `&& self.has_edges` might be redundant in this condition.
        let edge = if self.head_is_edge && self.has_edges {
            self.inner.next_edge()
        } else {
            None
        };
        self.head_is_edge = false;
        edge
    }

    /// Advances the iterator and returns the item if it's an edge. Returns None
    /// and does nothing if the last item is not an edge.
    pub fn next_edge_item_back(&mut self) -> Option<E> {
        let edge = if self.tail_is_edge && self.has_edges {
            self.inner.next_edge_back()
        } else {
            None
        };
        self.tail_is_edge = false;
        edge
    }

    /// Advances the iterator and returns the item if it's a key-value pair. Returns None
    /// and does nothing if the first item is not a key-value pair.
    pub fn next_kv_item(&mut self) -> Option<(K, V)> {
        if !self.head_is_edge {
            self.head_is_edge = true;
            self.inner.next_kv()
        } else {
            None
        }
    }

    /// Advances the iterator and returns the item if it's a key-value pair. Returns None
    /// and does nothing if the last item is not a key-value pair.
    pub fn next_kv_item_back(&mut self) -> Option<(K, V)> {
        if !self.tail_is_edge {
            self.tail_is_edge = true;
            self.inner.next_kv_back()
        } else {
            None
        }
    }
}

macro_rules! node_slice_impl {
    ($NodeSlice:ident, $Traversal:ident,
     $as_slices_internal:ident, $index:ident, $iter:ident) => {
        impl<'a, K: Ord + 'a, V: 'a> $NodeSlice<'a, K, V> {
            /// Performs linear search in a slice. Returns a tuple of (index, is_exact_match).
            fn search_linear<Q: ?Sized>(&self, key: &Q) -> (usize, bool)
                    where K: Borrow<Q>, Q: Ord {
                for (i, k) in self.keys.iter().enumerate() {
                    match key.cmp(k.borrow()) {
                        Greater => {},
                        Equal => return (i, true),
                        Less => return (i, false),
                    }
                }
                (self.keys.len(), false)
            }

            /// Returns a sub-slice with elements starting with `min_key`.
            pub fn slice_from<Q: ?Sized + Ord>(self, min_key: &Q) -> $NodeSlice<'a, K, V> where
                K: Borrow<Q>,
            {
                //  _______________
                // |_1_|_3_|_5_|_7_|
                // |   |   |   |   |
                // 0 0 1 1 2 2 3 3 4  index
                // |   |   |   |   |
                // \___|___|___|___/  slice_from(&0); pos = 0
                //     \___|___|___/  slice_from(&2); pos = 1
                //     |___|___|___/  slice_from(&3); pos = 1; result.head_is_edge = false
                //         \___|___/  slice_from(&4); pos = 2
                //             \___/  slice_from(&6); pos = 3
                //                \|/ slice_from(&999); pos = 4
                let (pos, pos_is_kv) = self.search_linear(min_key);
                $NodeSlice {
                    has_edges: self.has_edges,
                    edges: if !self.has_edges {
                        self.edges
                    } else {
                        self.edges.$index(pos ..)
                    },
                    keys: &self.keys[pos ..],
                    vals: self.vals.$index(pos ..),
                    head_is_edge: !pos_is_kv,
                    tail_is_edge: self.tail_is_edge,
                }
            }

            /// Returns a sub-slice with elements up to and including `max_key`.
            pub fn slice_to<Q: ?Sized + Ord>(self, max_key: &Q) -> $NodeSlice<'a, K, V> where
                K: Borrow<Q>,
            {
                //  _______________
                // |_1_|_3_|_5_|_7_|
                // |   |   |   |   |
                // 0 0 1 1 2 2 3 3 4  index
                // |   |   |   |   |
                //\|/  |   |   |   |  slice_to(&0); pos = 0
                // \___/   |   |   |  slice_to(&2); pos = 1
                // \___|___|   |   |  slice_to(&3); pos = 1; result.tail_is_edge = false
                // \___|___/   |   |  slice_to(&4); pos = 2
                // \___|___|___/   |  slice_to(&6); pos = 3
                // \___|___|___|___/  slice_to(&999); pos = 4
                let (pos, pos_is_kv) = self.search_linear(max_key);
                let pos = pos + if pos_is_kv { 1 } else { 0 };
                $NodeSlice {
                    has_edges: self.has_edges,
                    edges: if !self.has_edges {
                        self.edges
                    } else {
                        self.edges.$index(.. (pos + 1))
                    },
                    keys: &self.keys[..pos],
                    vals: self.vals.$index(.. pos),
                    head_is_edge: self.head_is_edge,
                    tail_is_edge: !pos_is_kv,
                }
            }
        }

        impl<'a, K: 'a, V: 'a> $NodeSlice<'a, K, V> {
            /// Returns an iterator over key/value pairs and edges in a slice.
            #[inline]
            pub fn $iter(self) -> $Traversal<'a, K, V> {
                let mut edges = self.edges.$iter();
                // Skip edges at both ends, if excluded.
                if !self.head_is_edge { edges.next(); }
                if !self.tail_is_edge { edges.next_back(); }
                // The key iterator is always immutable.
                $Traversal {
                    inner: ElemsAndEdges(
                        self.keys.iter().zip(self.vals.$iter()),
                        edges
                    ),
                    head_is_edge: self.head_is_edge,
                    tail_is_edge: self.tail_is_edge,
                    has_edges: self.has_edges,
                }
            }
        }
    }
}

node_slice_impl!(NodeSlice, Traversal, as_slices_internal, index, iter);
node_slice_impl!(MutNodeSlice, MutTraversal, as_slices_internal_mut, index_mut, iter_mut);
