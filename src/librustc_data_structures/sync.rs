// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This mdoule defines types which are thread safe if cfg!(parallel_queries) is true.
//!
//! `Lrc` is an alias of either Rc or Arc.
//!
//! `Lock` is a mutex.
//! It internally uses `parking_lot::Mutex` if cfg!(parallel_queries) is true,
//! `RefCell` otherwise.
//!
//! `RwLock` is a read-write lock.
//! It internally uses `parking_lot::RwLock` if cfg!(parallel_queries) is true,
//! `RefCell` otherwise.
//!
//! `LockCell` is a thread safe version of `Cell`, with `set` and `get` operations.
//! It can never deadlock. It uses `Cell` when
//! cfg!(parallel_queries) is false, otherwise it is a `Lock`.
//!
//! `MTLock` is a mutex which disappears if cfg!(parallel_queries) is false.
//!
//! `rustc_global!` gives us a way to declare variables which are intended to be
//! global for the current rustc session. This currently maps to thread-locals,
//! since rustdoc uses the rustc libraries in multiple threads.
//! These globals should eventually be moved into the `Session` structure.
//!
//! `rustc_erase_owner!` erases a OwningRef owner into Erased or Erased + Send + Sync
//! depending on the value of cfg!(parallel_queries).

use std::cmp::Ordering;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::fmt;
use owning_ref::{Erased, OwningRef};

cfg_if! {
    if #[cfg(not(parallel_queries))] {
        pub auto trait Send {}
        pub auto trait Sync {}

        impl<T: ?Sized> Send for T {}
        impl<T: ?Sized> Sync for T {}

        #[macro_export]
        macro_rules! rustc_erase_owner {
            ($v:expr) => {
                $v.erase_owner()
            }
        }

        pub type MetadataRef = OwningRef<Box<Erased>, [u8]>;

        pub use std::rc::Rc as Lrc;
        pub use std::cell::Ref as ReadGuard;
        pub use std::cell::RefMut as WriteGuard;
        pub use std::cell::RefMut as LockGuard;

        pub use std::cell::RefCell as RwLock;
        use std::cell::RefCell as InnerLock;

        use std::cell::Cell;

        #[derive(Debug)]
        pub struct MTLock<T>(T);

        impl<T> MTLock<T> {
            #[inline(always)]
            pub fn new(inner: T) -> Self {
                MTLock(inner)
            }

            #[inline(always)]
            pub fn into_inner(self) -> T {
                self.0
            }

            #[inline(always)]
            pub fn get_mut(&mut self) -> &mut T {
                &mut self.0
            }

            #[inline(always)]
            pub fn lock(&self) -> &T {
                &self.0
            }

            #[inline(always)]
            pub fn borrow(&self) -> &T {
                &self.0
            }

            #[inline(always)]
            pub fn borrow_mut(&self) -> &T {
                &self.0
            }
        }

        // FIXME: Probably a bad idea (in the threaded case)
        impl<T: Clone> Clone for MTLock<T> {
            #[inline]
            fn clone(&self) -> Self {
                MTLock(self.0.clone())
            }
        }

        pub struct LockCell<T>(Cell<T>);

        impl<T> LockCell<T> {
            #[inline(always)]
            pub fn new(inner: T) -> Self {
                LockCell(Cell::new(inner))
            }

            #[inline(always)]
            pub fn into_inner(self) -> T {
                self.0.into_inner()
            }

            #[inline(always)]
            pub fn set(&self, new_inner: T) {
                self.0.set(new_inner);
            }

            #[inline(always)]
            pub fn get(&self) -> T where T: Copy {
                self.0.get()
            }

            #[inline(always)]
            pub fn set_mut(&mut self, new_inner: T) {
                self.0.set(new_inner);
            }

            #[inline(always)]
            pub fn get_mut(&mut self) -> T where T: Copy {
                self.0.get()
            }
        }

        impl<T> LockCell<Option<T>> {
            #[inline(always)]
            pub fn take(&self) -> Option<T> {
                unsafe { (*self.0.as_ptr()).take() }
            }
        }
    } else {
        pub use std::marker::Send as Send;
        pub use std::marker::Sync as Sync;

        pub use parking_lot::RwLockReadGuard as ReadGuard;
        pub use parking_lot::RwLockWriteGuard as WriteGuard;

        pub use parking_lot::MutexGuard as LockGuard;

        use parking_lot;

        pub use std::sync::Arc as Lrc;

        pub use self::Lock as MTLock;

        use parking_lot::Mutex as InnerLock;

        pub type MetadataRef = OwningRef<Box<Erased + Send + Sync>, [u8]>;

        /// This makes locks panic if they are already held.
        /// It is only useful when you are running in a single thread
        const ERROR_CHECKING: bool = false;

        #[macro_export]
        macro_rules! rustc_erase_owner {
            ($v:expr) => {{
                let v = $v;
                ::rustc_data_structures::sync::assert_send_sync_val(&v);
                v.erase_send_sync_owner()
            }}
        }

        pub struct LockCell<T>(Lock<T>);

        impl<T> LockCell<T> {
            #[inline(always)]
            pub fn new(inner: T) -> Self {
                LockCell(Lock::new(inner))
            }

            #[inline(always)]
            pub fn into_inner(self) -> T {
                self.0.into_inner()
            }

            #[inline(always)]
            pub fn set(&self, new_inner: T) {
                *self.0.lock() = new_inner;
            }

            #[inline(always)]
            pub fn get(&self) -> T where T: Copy {
                *self.0.lock()
            }

            #[inline(always)]
            pub fn set_mut(&mut self, new_inner: T) {
                *self.0.get_mut() = new_inner;
            }

            #[inline(always)]
            pub fn get_mut(&mut self) -> T where T: Copy {
                *self.0.get_mut()
            }
        }

        impl<T> LockCell<Option<T>> {
            #[inline(always)]
            pub fn take(&self) -> Option<T> {
                self.0.lock().take()
            }
        }

        #[derive(Debug)]
        pub struct RwLock<T>(parking_lot::RwLock<T>);

        impl<T> RwLock<T> {
            #[inline(always)]
            pub fn new(inner: T) -> Self {
                RwLock(parking_lot::RwLock::new(inner))
            }

            #[inline(always)]
            pub fn borrow(&self) -> ReadGuard<T> {
                if ERROR_CHECKING {
                    self.0.try_read().expect("lock was already held")
                } else {
                    self.0.read()
                }
            }

            #[inline(always)]
            pub fn borrow_mut(&self) -> WriteGuard<T> {
                if ERROR_CHECKING {
                    self.0.try_write().expect("lock was already held")
                } else {
                    self.0.write()
                }
            }
        }

        // FIXME: Probably a bad idea
        impl<T: Clone> Clone for RwLock<T> {
            #[inline]
            fn clone(&self) -> Self {
                RwLock::new(self.borrow().clone())
            }
        }
    }
}

pub fn assert_sync<T: ?Sized + Sync>() {}
pub fn assert_send_sync_val<T: ?Sized + Sync + Send>(_t: &T) {}

#[macro_export]
#[allow_internal_unstable]
macro_rules! rustc_global {
    // empty (base case for the recursion)
    () => {};

    // process multiple declarations
    ($(#[$attr:meta])* $vis:vis static $name:ident: $t:ty = $init:expr; $($rest:tt)*) => (
        thread_local!($(#[$attr])* $vis static $name: $t = $init);
        rustc_global!($($rest)*);
    );

    // handle a single declaration
    ($(#[$attr:meta])* $vis:vis static $name:ident: $t:ty = $init:expr) => (
        thread_local!($(#[$attr])* $vis static $name: $t = $init);
    );
}

#[macro_export]
macro_rules! rustc_access_global {
    ($name:path, $callback:expr) => {
        $name.with($callback)
    }
}

impl<T: Copy + Debug> Debug for LockCell<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("LockCell")
            .field("value", &self.get())
            .finish()
    }
}

impl<T:Default> Default for LockCell<T> {
    /// Creates a `LockCell<T>`, with the `Default` value for T.
    #[inline]
    fn default() -> LockCell<T> {
        LockCell::new(Default::default())
    }
}

impl<T:PartialEq + Copy> PartialEq for LockCell<T> {
    #[inline]
    fn eq(&self, other: &LockCell<T>) -> bool {
        self.get() == other.get()
    }
}

impl<T:Eq + Copy> Eq for LockCell<T> {}

impl<T:PartialOrd + Copy> PartialOrd for LockCell<T> {
    #[inline]
    fn partial_cmp(&self, other: &LockCell<T>) -> Option<Ordering> {
        self.get().partial_cmp(&other.get())
    }

    #[inline]
    fn lt(&self, other: &LockCell<T>) -> bool {
        self.get() < other.get()
    }

    #[inline]
    fn le(&self, other: &LockCell<T>) -> bool {
        self.get() <= other.get()
    }

    #[inline]
    fn gt(&self, other: &LockCell<T>) -> bool {
        self.get() > other.get()
    }

    #[inline]
    fn ge(&self, other: &LockCell<T>) -> bool {
        self.get() >= other.get()
    }
}

impl<T:Ord + Copy> Ord for LockCell<T> {
    #[inline]
    fn cmp(&self, other: &LockCell<T>) -> Ordering {
        self.get().cmp(&other.get())
    }
}

#[derive(Debug)]
pub struct Lock<T>(InnerLock<T>);

impl<T> Lock<T> {
    #[inline(always)]
    pub fn new(inner: T) -> Self {
        Lock(InnerLock::new(inner))
    }

    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.0.into_inner()
    }

    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut()
    }

    #[cfg(parallel_queries)]
    #[inline(always)]
    pub fn lock(&self) -> LockGuard<T> {
        if ERROR_CHECKING {
            self.0.try_lock().expect("lock was already held")
        } else {
            self.0.lock()
        }
    }

    #[cfg(not(parallel_queries))]
    #[inline(always)]
    pub fn lock(&self) -> LockGuard<T> {
        self.0.borrow_mut()
    }

    #[inline(always)]
    pub fn borrow(&self) -> LockGuard<T> {
        self.lock()
    }

    #[inline(always)]
    pub fn borrow_mut(&self) -> LockGuard<T> {
        self.lock()
    }
}

// FIXME: Probably a bad idea
impl<T: Clone> Clone for Lock<T> {
    #[inline]
    fn clone(&self) -> Self {
        Lock::new(self.borrow().clone())
    }
}
