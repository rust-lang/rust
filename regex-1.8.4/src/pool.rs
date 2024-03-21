// This module provides a relatively simple thread-safe pool of reusable
// objects. For the most part, it's implemented by a stack represented by a
// Mutex<Vec<T>>. It has one small trick: because unlocking a mutex is somewhat
// costly, in the case where a pool is accessed by the first thread that tried
// to get a value, we bypass the mutex. Here are some benchmarks showing the
// difference.
//
// 1) misc::anchored_literal_long_non_match    21 (18571 MB/s)
// 2) misc::anchored_literal_long_non_match   107 (3644 MB/s)
// 3) misc::anchored_literal_long_non_match    45 (8666 MB/s)
// 4) misc::anchored_literal_long_non_match    19 (20526 MB/s)
//
// (1) represents our baseline: the master branch at the time of writing when
// using the 'thread_local' crate to implement the pool below.
//
// (2) represents a naive pool implemented completely via Mutex<Vec<T>>. There
// is no special trick for bypassing the mutex.
//
// (3) is the same as (2), except it uses Mutex<Vec<Box<T>>>. It is twice as
// fast because a Box<T> is much smaller than the T we use with a Pool in this
// crate. So pushing and popping a Box<T> from a Vec is quite a bit faster
// than for T.
//
// (4) is the same as (3), but with the trick for bypassing the mutex in the
// case of the first-to-get thread.
//
// Why move off of thread_local? Even though (4) is a hair faster than (1)
// above, this was not the main goal. The main goal was to move off of
// thread_local and find a way to *simply* re-capture some of its speed for
// regex's specific case. So again, why move off of it? The *primary* reason is
// because of memory leaks. See https://github.com/rust-lang/regex/issues/362
// for example. (Why do I want it to be simple? Well, I suppose what I mean is,
// "use as much safe code as possible to minimize risk and be as sure as I can
// be that it is correct.")
//
// My guess is that the thread_local design is probably not appropriate for
// regex since its memory usage scales to the number of active threads that
// have used a regex, where as the pool below scales to the number of threads
// that simultaneously use a regex. While neither case permits contraction,
// since we own the pool data structure below, we can add contraction if a
// clear use case pops up in the wild. More pressingly though, it seems that
// there are at least some use case patterns where one might have many threads
// sitting around that might have used a regex at one point. While thread_local
// does try to reuse space previously used by a thread that has since stopped,
// its maximal memory usage still scales with the total number of active
// threads. In contrast, the pool below scales with the total number of threads
// *simultaneously* using the pool. The hope is that this uses less memory
// overall. And if it doesn't, we can hopefully tune it somehow.
//
// It seems that these sort of conditions happen frequently
// in FFI inside of other more "managed" languages. This was
// mentioned in the issue linked above, and also mentioned here:
// https://github.com/BurntSushi/rure-go/issues/3. And in particular, users
// confirm that disabling the use of thread_local resolves the leak.
//
// There were other weaker reasons for moving off of thread_local as well.
// Namely, at the time, I was looking to reduce dependencies. And for something
// like regex, maintenance can be simpler when we own the full dependency tree.

use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// An atomic counter used to allocate thread IDs.
static COUNTER: AtomicUsize = AtomicUsize::new(1);

thread_local!(
    /// A thread local used to assign an ID to a thread.
    static THREAD_ID: usize = {
        let next = COUNTER.fetch_add(1, Ordering::Relaxed);
        // SAFETY: We cannot permit the reuse of thread IDs since reusing a
        // thread ID might result in more than one thread "owning" a pool,
        // and thus, permit accessing a mutable value from multiple threads
        // simultaneously without synchronization. The intent of this panic is
        // to be a sanity check. It is not expected that the thread ID space
        // will actually be exhausted in practice.
        //
        // This checks that the counter never wraps around, since atomic
        // addition wraps around on overflow.
        if next == 0 {
            panic!("regex: thread ID allocation space exhausted");
        }
        next
    };
);

/// The type of the function used to create values in a pool when the pool is
/// empty and the caller requests one.
type CreateFn<T> =
    Box<dyn Fn() -> T + Send + Sync + UnwindSafe + RefUnwindSafe + 'static>;

/// A simple thread safe pool for reusing values.
///
/// Getting a value out comes with a guard. When that guard is dropped, the
/// value is automatically put back in the pool.
///
/// A Pool<T> impls Sync when T is Send (even if it's not Sync). This means
/// that T can use interior mutability. This is possible because a pool is
/// guaranteed to provide a value to exactly one thread at any time.
///
/// Currently, a pool never contracts in size. Its size is proportional to the
/// number of simultaneous uses.
pub struct Pool<T> {
    /// A stack of T values to hand out. These are used when a Pool is
    /// accessed by a thread that didn't create it.
    stack: Mutex<Vec<Box<T>>>,
    /// A function to create more T values when stack is empty and a caller
    /// has requested a T.
    create: CreateFn<T>,
    /// The ID of the thread that owns this pool. The owner is the thread
    /// that makes the first call to 'get'. When the owner calls 'get', it
    /// gets 'owner_val' directly instead of returning a T from 'stack'.
    /// See comments elsewhere for details, but this is intended to be an
    /// optimization for the common case that makes getting a T faster.
    ///
    /// It is initialized to a value of zero (an impossible thread ID) as a
    /// sentinel to indicate that it is unowned.
    owner: AtomicUsize,
    /// A value to return when the caller is in the same thread that created
    /// the Pool.
    owner_val: T,
}

// SAFETY: Since we want to use a Pool from multiple threads simultaneously
// behind an Arc, we need for it to be Sync. In cases where T is sync, Pool<T>
// would be Sync. However, since we use a Pool to store mutable scratch space,
// we wind up using a T that has interior mutability and is thus itself not
// Sync. So what we *really* want is for our Pool<T> to by Sync even when T is
// not Sync (but is at least Send).
//
// The only non-sync aspect of a Pool is its 'owner_val' field, which is used
// to implement faster access to a pool value in the common case of a pool
// being accessed in the same thread in which it was created. The 'stack' field
// is also shared, but a Mutex<T> where T: Send is already Sync. So we only
// need to worry about 'owner_val'.
//
// The key is to guarantee that 'owner_val' can only ever be accessed from one
// thread. In our implementation below, we guarantee this by only returning the
// 'owner_val' when the ID of the current thread matches the ID of the thread
// that created the Pool. Since this can only ever be one thread, it follows
// that only one thread can access 'owner_val' at any point in time. Thus, it
// is safe to declare that Pool<T> is Sync when T is Send.
//
// NOTE: It would also be possible to make the owning thread be the *first*
// thread that tries to get a value out of a Pool. However, the current
// implementation is a little simpler and it's not clear if making the first
// thread (rather than the creating thread) is meaningfully better.
//
// If there is a way to achieve our performance goals using safe code, then
// I would very much welcome a patch. As it stands, the implementation below
// tries to balance safety with performance. The case where a Regex is used
// from multiple threads simultaneously will suffer a bit since getting a cache
// will require unlocking a mutex.
unsafe impl<T: Send> Sync for Pool<T> {}

impl<T: ::std::fmt::Debug> ::std::fmt::Debug for Pool<T> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        f.debug_struct("Pool")
            .field("stack", &self.stack)
            .field("owner", &self.owner)
            .field("owner_val", &self.owner_val)
            .finish()
    }
}

/// A guard that is returned when a caller requests a value from the pool.
///
/// The purpose of the guard is to use RAII to automatically put the value back
/// in the pool once it's dropped.
#[derive(Debug)]
pub struct PoolGuard<'a, T: Send> {
    /// The pool that this guard is attached to.
    pool: &'a Pool<T>,
    /// This is None when the guard represents the special "owned" value. In
    /// which case, the value is retrieved from 'pool.owner_val'.
    value: Option<Box<T>>,
}

impl<T: Send> Pool<T> {
    /// Create a new pool. The given closure is used to create values in the
    /// pool when necessary.
    pub fn new(create: CreateFn<T>) -> Pool<T> {
        let owner = AtomicUsize::new(0);
        let owner_val = create();
        Pool { stack: Mutex::new(vec![]), create, owner, owner_val }
    }

    /// Get a value from the pool. The caller is guaranteed to have exclusive
    /// access to the given value.
    ///
    /// Note that there is no guarantee provided about which value in the
    /// pool is returned. That is, calling get, dropping the guard (causing
    /// the value to go back into the pool) and then calling get again is NOT
    /// guaranteed to return the same value received in the first get call.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub fn get(&self) -> PoolGuard<'_, T> {
        // Our fast path checks if the caller is the thread that "owns" this
        // pool. Or stated differently, whether it is the first thread that
        // tried to extract a value from the pool. If it is, then we can return
        // a T to the caller without going through a mutex.
        //
        // SAFETY: We must guarantee that only one thread gets access to this
        // value. Since a thread is uniquely identified by the THREAD_ID thread
        // local, it follows that is the caller's thread ID is equal to the
        // owner, then only one thread may receive this value.
        let caller = THREAD_ID.with(|id| *id);
        let owner = self.owner.load(Ordering::Relaxed);
        if caller == owner {
            return self.guard_owned();
        }
        self.get_slow(caller, owner)
    }

    /// This is the "slow" version that goes through a mutex to pop an
    /// allocated value off a stack to return to the caller. (Or, if the stack
    /// is empty, a new value is created.)
    ///
    /// If the pool has no owner, then this will set the owner.
    #[cold]
    fn get_slow(&self, caller: usize, owner: usize) -> PoolGuard<'_, T> {
        use std::sync::atomic::Ordering::Relaxed;

        if owner == 0 {
            // The sentinel 0 value means this pool is not yet owned. We
            // try to atomically set the owner. If we do, then this thread
            // becomes the owner and we can return a guard that represents
            // the special T for the owner.
            let res = self.owner.compare_exchange(0, caller, Relaxed, Relaxed);
            if res.is_ok() {
                return self.guard_owned();
            }
        }
        let mut stack = self.stack.lock().unwrap();
        let value = match stack.pop() {
            None => Box::new((self.create)()),
            Some(value) => value,
        };
        self.guard_stack(value)
    }

    /// Puts a value back into the pool. Callers don't need to call this. Once
    /// the guard that's returned by 'get' is dropped, it is put back into the
    /// pool automatically.
    fn put(&self, value: Box<T>) {
        let mut stack = self.stack.lock().unwrap();
        stack.push(value);
    }

    /// Create a guard that represents the special owned T.
    fn guard_owned(&self) -> PoolGuard<'_, T> {
        PoolGuard { pool: self, value: None }
    }

    /// Create a guard that contains a value from the pool's stack.
    fn guard_stack(&self, value: Box<T>) -> PoolGuard<'_, T> {
        PoolGuard { pool: self, value: Some(value) }
    }
}

impl<'a, T: Send> PoolGuard<'a, T> {
    /// Return the underlying value.
    pub fn value(&self) -> &T {
        match self.value {
            None => &self.pool.owner_val,
            Some(ref v) => &**v,
        }
    }
}

impl<'a, T: Send> Drop for PoolGuard<'a, T> {
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            self.pool.put(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{RefUnwindSafe, UnwindSafe};

    use super::*;

    #[test]
    fn oibits() {
        use crate::exec::ProgramCache;

        fn has_oibits<T: Send + Sync + UnwindSafe + RefUnwindSafe>() {}
        has_oibits::<Pool<ProgramCache>>();
    }

    // Tests that Pool implements the "single owner" optimization. That is, the
    // thread that first accesses the pool gets its own copy, while all other
    // threads get distinct copies.
    #[test]
    fn thread_owner_optimization() {
        use std::cell::RefCell;
        use std::sync::Arc;

        let pool: Arc<Pool<RefCell<Vec<char>>>> =
            Arc::new(Pool::new(Box::new(|| RefCell::new(vec!['a']))));
        pool.get().value().borrow_mut().push('x');

        let pool1 = pool.clone();
        let t1 = std::thread::spawn(move || {
            let guard = pool1.get();
            let v = guard.value();
            v.borrow_mut().push('y');
        });

        let pool2 = pool.clone();
        let t2 = std::thread::spawn(move || {
            let guard = pool2.get();
            let v = guard.value();
            v.borrow_mut().push('z');
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // If we didn't implement the single owner optimization, then one of
        // the threads above is likely to have mutated the [a, x] vec that
        // we stuffed in the pool before spawning the threads. But since
        // neither thread was first to access the pool, and because of the
        // optimization, we should be guaranteed that neither thread mutates
        // the special owned pool value.
        //
        // (Technically this is an implementation detail and not a contract of
        // Pool's API.)
        assert_eq!(vec!['a', 'x'], *pool.get().value().borrow());
    }
}
