use super::{InternalNode, LeafNode};
use crate::alloc::{AllocRef, Global, Layout};
use crate::boxed::Box;
use core::ptr::NonNull;

/// A managed, non-null, owned pointer to a node. It points to either an owned
/// `LeafNode<K, V>` or to the leaf portion of an owned `InternalNode<K, V>`.
///
/// However, `BoxedNode` contains no information as to which of the two types
/// of nodes it actually contains, and, partially due to this lack of information,
/// has no destructor.
pub(super) struct BoxedNode<K, V> {
    ptr: NonNull<LeafNode<K, V>>,
}

/// A managed, non-null pointer to a node that knows the node's height. It
/// nodels the type:
/// ```
///     use core::num::NonZeroUsize;
///     use core::ptr::NonNull;
///     # #[allow(dead_code)] struct LeafNode<K, V>(K, V);
///     # #[allow(dead_code)] struct InternalNode<K, V>(K, V);
///     # #[allow(dead_code)]
///     enum UnboxedNode<K, V> {
///         Leaf { ptr: NonNull<LeafNode<K, V>> },
///         Internal { height: NonZeroUsize, ptr: NonNull<InternalNode<K, V>> },
///     }
/// ```
/// by way of a single pointer field and casts, because that is easier for
/// gdb_providers.py, even more so because it is very similar to `BoxedNode`.
///
/// However, `UnboxedNode` contains no information as to whether it owns or
/// shares the node and has no implicit destructor.
pub(super) struct UnboxedNode<K, V> {
    /// The number of levels that the node and leaves are apart, a constant
    /// property of the node that cannot be entirely described by its type.
    /// The node itself does not store its height; we only need to store it
    /// for the root node, and derive every node's height from it.
    height: usize,
    /// The pointer to a leaf (if height is zero) or to the leaf portion of
    /// an internal node (if height is non-zero).
    ptr: NonNull<LeafNode<K, V>>,
}

impl<K, V> UnboxedNode<K, V> {
    pub(super) fn from_new_leaf(leaf: Box<LeafNode<K, V>>) -> Self {
        UnboxedNode { height: 0, ptr: NonNull::from(Box::leak(leaf)) }
    }

    pub(super) fn from_new_internal(internal: Box<InternalNode<K, V>>, height: usize) -> Self {
        debug_assert!(height > 0);
        UnboxedNode { height, ptr: NonNull::from(Box::leak(internal)).cast() }
    }

    /// Create from a parent pointer stored in LeafNode.
    pub(super) fn from_internal(internal_ptr: NonNull<InternalNode<K, V>>, height: usize) -> Self {
        debug_assert!(height > 0);
        UnboxedNode { height, ptr: internal_ptr.cast() }
    }

    /// Unpacks a type-agnostic pointer into a reference aware of type and height.
    pub(super) fn from_boxed_node(boxed_node: BoxedNode<K, V>, height: usize) -> Self {
        UnboxedNode { height, ptr: boxed_node.ptr }
    }

    /// Packs the reference, aware of type and height, into a type-agnostic pointer.
    pub(super) fn into_boxed_node(self) -> BoxedNode<K, V> {
        BoxedNode { ptr: self.ptr }
    }

    /// Returns the number of levels that the node and leaves are apart. Zero
    /// height means the node is a leaf itself.
    pub(super) fn height(&self) -> usize {
        self.height
    }

    /// Implementation of PartialEq, but there's no need to announce that trait.
    pub(super) fn eq(&self, other: &Self) -> bool {
        let Self { height, ptr } = self;
        if *ptr == other.ptr {
            debug_assert_eq!(*height, other.height);
            true
        } else {
            false
        }
    }

    /// Exposes the leaf portion of any leaf or internal node.
    pub(super) fn as_leaf_ptr(this: &Self) -> *mut LeafNode<K, V> {
        this.ptr.as_ptr()
    }

    /// Exposes an internal node.
    /// # Safety
    /// The node must not be a leaf.
    pub(super) unsafe fn as_internal_ptr(this: &Self) -> *mut InternalNode<K, V> {
        debug_assert!(this.height > 0);
        this.ptr.as_ptr() as *mut _
    }

    /// Deallocates the node, assuming its keys, values and edges have been
    /// deallocated or moved elsewhere already.
    pub(super) fn dealloc(self) {
        if self.height == 0 {
            unsafe { self.dealloc_leaf() };
        } else {
            unsafe { self.dealloc_internal() };
        }
    }

    /// # Safety
    /// The node must be a leaf.
    pub(super) unsafe fn dealloc_leaf(self) {
        debug_assert!(self.height == 0);
        unsafe { Global.dealloc(self.ptr.cast(), Layout::new::<LeafNode<K, V>>()) }
    }

    /// # Safety
    /// The node must not be a leaf.
    pub(super) unsafe fn dealloc_internal(self) {
        debug_assert!(self.height > 0);
        unsafe { Global.dealloc(self.ptr.cast(), Layout::new::<InternalNode<K, V>>()) }
    }
}

impl<K, V> Copy for UnboxedNode<K, V> {}
impl<K, V> Clone for UnboxedNode<K, V> {
    fn clone(&self) -> Self {
        *self
    }
}
