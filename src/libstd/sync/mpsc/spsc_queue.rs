//! A single-producer single-consumer concurrent queue
//!
//! This module contains the implementation of an SPSC queue which can be used
//! concurrently between two threads. This data structure is safe to use and
//! enforces the semantics that there is one pusher and one popper.

// http://www.1024cores.net/home/lock-free-algorithms/queues/unbounded-spsc-queue

use core::ptr;
use core::cell::UnsafeCell;

use crate::boxed::Box;
use crate::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use super::cache_aligned::CacheAligned;

// Node within the linked list queue of messages to send
struct Node<T> {
    // FIXME: this could be an uninitialized T if we're careful enough, and
    //      that would reduce memory usage (and be a bit faster).
    //      is it worth it?
    value: Option<T>,           // nullable for re-use of nodes
    cached: bool,               // This node goes into the node cache
    next: AtomicPtr<Node<T>>,   // next node in the queue
}

/// The single-producer single-consumer queue. This structure is not cloneable,
/// but it can be safely shared in an Arc if it is guaranteed that there
/// is only one popper and one pusher touching the queue at any one point in
/// time.
pub struct Queue<T, ProducerAddition=(), ConsumerAddition=()> {
    // consumer fields
    consumer: CacheAligned<Consumer<T, ConsumerAddition>>,

    // producer fields
    producer: CacheAligned<Producer<T, ProducerAddition>>,
}

struct Consumer<T, Addition> {
    tail: UnsafeCell<*mut Node<T>>, // where to pop from
    tail_prev: AtomicPtr<Node<T>>, // where to pop from
    cache_bound: usize, // maximum cache size
    cached_nodes: AtomicUsize, // number of nodes marked as cachable
    addition: Addition,
}

struct Producer<T, Addition> {
    head: UnsafeCell<*mut Node<T>>,      // where to push to
    first: UnsafeCell<*mut Node<T>>,     // where to get new nodes from
    tail_copy: UnsafeCell<*mut Node<T>>, // between first/tail
    addition: Addition,
}

unsafe impl<T: Send, P: Send + Sync, C: Send + Sync> Send for Queue<T, P, C> { }

unsafe impl<T: Send, P: Send + Sync, C: Send + Sync> Sync for Queue<T, P, C> { }

impl<T> Node<T> {
    fn new() -> *mut Node<T> {
        Box::into_raw(box Node {
            value: None,
            cached: false,
            next: AtomicPtr::new(ptr::null_mut::<Node<T>>()),
        })
    }
}

impl<T, ProducerAddition, ConsumerAddition> Queue<T, ProducerAddition, ConsumerAddition> {

    /// Creates a new queue. With given additional elements in the producer and
    /// consumer portions of the queue.
    ///
    /// Due to the performance implications of cache-contention,
    /// we wish to keep fields used mainly by the producer on a separate cache
    /// line than those used by the consumer.
    /// Since cache lines are usually 64 bytes, it is unreasonably expensive to
    /// allocate one for small fields, so we allow users to insert additional
    /// fields into the cache lines already allocated by this for the producer
    /// and consumer.
    ///
    /// This is unsafe as the type system doesn't enforce a single
    /// consumer-producer relationship. It also allows the consumer to `pop`
    /// items while there is a `peek` active due to all methods having a
    /// non-mutable receiver.
    ///
    /// # Arguments
    ///
    ///   * `bound` - This queue implementation is implemented with a linked
    ///               list, and this means that a push is always a malloc. In
    ///               order to amortize this cost, an internal cache of nodes is
    ///               maintained to prevent a malloc from always being
    ///               necessary. This bound is the limit on the size of the
    ///               cache (if desired). If the value is 0, then the cache has
    ///               no bound. Otherwise, the cache will never grow larger than
    ///               `bound` (although the queue itself could be much larger.
    pub unsafe fn with_additions(
        bound: usize,
        producer_addition: ProducerAddition,
        consumer_addition: ConsumerAddition,
    ) -> Self {
        let n1 = Node::new();
        let n2 = Node::new();
        (*n1).next.store(n2, Ordering::Relaxed);
        Queue {
            consumer: CacheAligned::new(Consumer {
                tail: UnsafeCell::new(n2),
                tail_prev: AtomicPtr::new(n1),
                cache_bound: bound,
                cached_nodes: AtomicUsize::new(0),
                addition: consumer_addition
            }),
            producer: CacheAligned::new(Producer {
                head: UnsafeCell::new(n2),
                first: UnsafeCell::new(n1),
                tail_copy: UnsafeCell::new(n1),
                addition: producer_addition
            }),
        }
    }

    /// Pushes a new value onto this queue. Note that to use this function
    /// safely, it must be externally guaranteed that there is only one pusher.
    pub fn push(&self, t: T) {
        unsafe {
            // Acquire a node (which either uses a cached one or allocates a new
            // one), and then append this to the 'head' node.
            let n = self.alloc();
            assert!((*n).value.is_none());
            (*n).value = Some(t);
            (*n).next.store(ptr::null_mut(), Ordering::Relaxed);
            (**self.producer.head.get()).next.store(n, Ordering::Release);
            *(&self.producer.head).get() = n;
        }
    }

    unsafe fn alloc(&self) -> *mut Node<T> {
        // First try to see if we can consume the 'first' node for our uses.
        if *self.producer.first.get() != *self.producer.tail_copy.get() {
            let ret = *self.producer.first.get();
            *self.producer.0.first.get() = (*ret).next.load(Ordering::Relaxed);
            return ret;
        }
        // If the above fails, then update our copy of the tail and try
        // again.
        *self.producer.0.tail_copy.get() =
            self.consumer.tail_prev.load(Ordering::Acquire);
        if *self.producer.first.get() != *self.producer.tail_copy.get() {
            let ret = *self.producer.first.get();
            *self.producer.0.first.get() = (*ret).next.load(Ordering::Relaxed);
            return ret;
        }
        // If all of that fails, then we have to allocate a new node
        // (there's nothing in the node cache).
        Node::new()
    }

    /// Attempts to pop a value from this queue. Remember that to use this type
    /// safely you must ensure that there is only one popper at a time.
    pub fn pop(&self) -> Option<T> {
        unsafe {
            // The `tail` node is not actually a used node, but rather a
            // sentinel from where we should start popping from. Hence, look at
            // tail's next field and see if we can use it. If we do a pop, then
            // the current tail node is a candidate for going into the cache.
            let tail = *self.consumer.tail.get();
            let next = (*tail).next.load(Ordering::Acquire);
            if next.is_null() { return None }
            assert!((*next).value.is_some());
            let ret = (*next).value.take();

            *self.consumer.0.tail.get() = next;
            if self.consumer.cache_bound == 0 {
                self.consumer.tail_prev.store(tail, Ordering::Release);
            } else {
                let cached_nodes = self.consumer.cached_nodes.load(Ordering::Relaxed);
                if cached_nodes < self.consumer.cache_bound && !(*tail).cached {
                    self.consumer.cached_nodes.store(cached_nodes, Ordering::Relaxed);
                    (*tail).cached = true;
                }

                if (*tail).cached {
                    self.consumer.tail_prev.store(tail, Ordering::Release);
                } else {
                    (*self.consumer.tail_prev.load(Ordering::Relaxed))
                        .next.store(next, Ordering::Relaxed);
                    // We have successfully erased all references to 'tail', so
                    // now we can safely drop it.
                    let _: Box<Node<T>> = Box::from_raw(tail);
                }
            }
            ret
        }
    }

    /// Attempts to peek at the head of the queue, returning `None` if the queue
    /// has no data currently
    ///
    /// # Warning
    /// The reference returned is invalid if it is not used before the consumer
    /// pops the value off the queue. If the producer then pushes another value
    /// onto the queue, it will overwrite the value pointed to by the reference.
    pub fn peek(&self) -> Option<&mut T> {
        // This is essentially the same as above with all the popping bits
        // stripped out.
        unsafe {
            let tail = *self.consumer.tail.get();
            let next = (*tail).next.load(Ordering::Acquire);
            if next.is_null() { None } else { (*next).value.as_mut() }
        }
    }

    pub fn producer_addition(&self) -> &ProducerAddition {
        &self.producer.addition
    }

    pub fn consumer_addition(&self) -> &ConsumerAddition {
        &self.consumer.addition
    }
}

impl<T, ProducerAddition, ConsumerAddition> Drop for Queue<T, ProducerAddition, ConsumerAddition> {
    fn drop(&mut self) {
        unsafe {
            let mut cur = *self.producer.first.get();
            while !cur.is_null() {
                let next = (*cur).next.load(Ordering::Relaxed);
                let _n: Box<Node<T>> = Box::from_raw(cur);
                cur = next;
            }
        }
    }
}

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests {
    use super::Queue;
    use crate::sync::Arc;
    use crate::thread;
    use crate::sync::mpsc::channel;

    #[test]
    fn smoke() {
        unsafe {
            let queue = Queue::with_additions(0, (), ());
            queue.push(1);
            queue.push(2);
            assert_eq!(queue.pop(), Some(1));
            assert_eq!(queue.pop(), Some(2));
            assert_eq!(queue.pop(), None);
            queue.push(3);
            queue.push(4);
            assert_eq!(queue.pop(), Some(3));
            assert_eq!(queue.pop(), Some(4));
            assert_eq!(queue.pop(), None);
        }
    }

    #[test]
    fn peek() {
        unsafe {
            let queue = Queue::with_additions(0, (), ());
            queue.push(vec![1]);

            // Ensure the borrowchecker works
            match queue.peek() {
                Some(vec) => {
                    assert_eq!(&*vec, &[1]);
                },
                None => unreachable!()
            }

            match queue.pop() {
                Some(vec) => {
                    assert_eq!(&*vec, &[1]);
                },
                None => unreachable!()
            }
        }
    }

    #[test]
    fn drop_full() {
        unsafe {
            let q: Queue<Box<_>> = Queue::with_additions(0, (), ());
            q.push(box 1);
            q.push(box 2);
        }
    }

    #[test]
    fn smoke_bound() {
        unsafe {
            let q = Queue::with_additions(0, (), ());
            q.push(1);
            q.push(2);
            assert_eq!(q.pop(), Some(1));
            assert_eq!(q.pop(), Some(2));
            assert_eq!(q.pop(), None);
            q.push(3);
            q.push(4);
            assert_eq!(q.pop(), Some(3));
            assert_eq!(q.pop(), Some(4));
            assert_eq!(q.pop(), None);
        }
    }

    #[test]
    fn stress() {
        unsafe {
            stress_bound(0);
            stress_bound(1);
        }

        unsafe fn stress_bound(bound: usize) {
            let q = Arc::new(Queue::with_additions(bound, (), ()));

            let (tx, rx) = channel();
            let q2 = q.clone();
            let _t = thread::spawn(move|| {
                for _ in 0..100000 {
                    loop {
                        match q2.pop() {
                            Some(1) => break,
                            Some(_) => panic!(),
                            None => {}
                        }
                    }
                }
                tx.send(()).unwrap();
            });
            for _ in 0..100000 {
                q.push(1);
            }
            rx.recv().unwrap();
        }
    }
}
