// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Task local data management

Allows storing arbitrary types inside task-local-data (TLD), to be accessed
anywhere within a task, keyed by a global pointer parameterized over the type of
the TLD slot. Useful for dynamic variables, singletons, and interfacing with
foreign code with bad callback interfaces.

To declare a new key for storing local data of a particular type, use the
`local_data_key!` macro. This macro will expand to a `static` item appropriately
named and annotated. This name is then passed to the functions in this module to
modify/read the slot specified by the key.

```rust
local_data_key!(key_int: int)
local_data_key!(key_vector: Vec<int>)

key_int.replace(Some(3));
assert_eq!(*key_int.get().unwrap(), 3);

key_vector.replace(Some(vec![4]));
assert_eq!(*key_vector.get().unwrap(), vec![4]);
```

*/

// Casting 'Arcane Sight' reveals an overwhelming aura of Transmutation
// magic.

use core::prelude::*;

use alloc::heap;
use collections::treemap::TreeMap;
use collections::MutableMap;
use core::cmp;
use core::kinds::marker;
use core::mem;
use core::ptr;
use core::fmt;
use core::cell::UnsafeCell;

use local::Local;
use task::{Task, LocalStorage};

/**
 * Indexes a task-local data slot. This pointer is used for comparison to
 * differentiate keys from one another. The actual type `T` is not used anywhere
 * as a member of this type, except that it is parameterized with it to define
 * the type of each key's value.
 *
 * The value of each Key is of the singleton enum KeyValue. These also have the
 * same name as `Key` and their purpose is to take up space in the programs data
 * sections to ensure that each value of the `Key` type points to a unique
 * location.
 */
pub type Key<T> = &'static KeyValue<T>;

#[allow(missing_doc)]
pub enum KeyValue<T> { Key }

// The task-local-map stores all TLD information for the currently running
// task. It is stored as an owned pointer into the runtime, and it's only
// allocated when TLD is used for the first time.
//
// TLD values are boxed up, with a loan count stored in the box. The box is
// necessary given how TLD maps are constructed, but theoretically in the
// future this could be rewritten to statically construct TLD offsets at
// compile-time to get O(1) lookup. At that time, the box can be removed.
//
// A very common usage pattern for TLD is to use replace(None) to extract a
// value from TLD, work with it, and then store it (or a derived/new value)
// back with replace(v). We take special care to reuse the allocation in this
// case for performance reasons.
//
// However, that does mean that if a value is replaced with None, the
// allocation will stay alive and the entry will stay in the TLD map until the
// task deallocates. This makes the assumption that every key inserted into a
// given task's TLD is going to be present for a majority of the rest of the
// task's lifetime, but that's a fairly safe assumption, and there's very
// little downside as long as it holds true for most keys.
//
// The Map type must be public in order to allow rustrt to see it.
//
// We'd like to use HashMap here, but it uses TLD in its construction (it uses
// the task-local rng). We could try to provide our own source of randomness,
// except it also lives in libstd (which is a client of us) so we can't even
// reference it. Instead, use TreeMap, which provides reasonable performance.
#[doc(hidden)]
pub type Map = TreeMap<uint, TLDValue>;
#[unsafe_no_drop_flag]
struct TLDValue {
    // box_ptr is a pointer to TLDValueBox<T>. It can never be null.
    box_ptr: *mut (),
    // drop_fn is the function that knows how to drop the box_ptr.
    drop_fn: unsafe fn(p: *mut ())
}

struct TLDValueBox<T> {
    // value is only initialized when refcount >= 1.
    value: T,
    // refcount of 0 means uninitialized value, 1 means initialized, 2+ means
    // borrowed.
    // NB: we use UnsafeCell instead of Cell because Ref should be allowed to
    // be Sync. The only mutation occurs when a Ref is created or destroyed,
    // so there's no issue with &Ref being thread-safe.
    refcount: UnsafeCell<uint>
}

// Gets the map from the runtime. Lazily initialises if not done so already.
unsafe fn get_local_map<'a>() -> Option<&'a mut Map> {
    if !Local::exists(None::<Task>) { return None }

    let task: *mut Task = Local::unsafe_borrow();
    match &mut (*task).storage {
        // If the at_exit function is already set, then we just need to take
        // a loan out on the TLD map stored inside
        &LocalStorage(Some(ref mut map_ptr)) => {
            return Some(map_ptr);
        }
        // If this is the first time we've accessed TLD, perform similar
        // actions to the oldsched way of doing things.
        &LocalStorage(ref mut slot) => {
            *slot = Some(TreeMap::new());
            match *slot {
                Some(ref mut map_ptr) => { return Some(map_ptr) }
                None => fail!("unreachable code"),
            }
        }
    }
}

/// A RAII immutable reference to a task-local value.
///
/// The task-local data can be accessed through this value, and when this
/// structure is dropped it will return the borrow on the data.
#[cfg(not(stage0))]
pub struct Ref<T:'static> {
    // FIXME #12808: strange names to try to avoid interfering with
    // field accesses of the contained type via Deref
    _inner: &'static TLDValueBox<T>,
    _marker: marker::NoSend
}

/// stage0 only
#[cfg(stage0)]
pub struct Ref<T> {
    // FIXME #12808: strange names to try to avoid interfering with
    // field accesses of the contained type via Deref
    _inner: &'static TLDValueBox<T>,
    _marker: marker::NoSend
}

fn key_to_key_value<T: 'static>(key: Key<T>) -> uint {
    key as *const _ as uint
}

impl<T: 'static> KeyValue<T> {
    /// Replaces a value in task local data.
    ///
    /// If this key is already present in TLD, then the previous value is
    /// replaced with the provided data, and then returned.
    ///
    /// # Failure
    ///
    /// This function will fail if the key is present in TLD and currently on
    /// loan with the `get` method.
    ///
    /// It will also fail if there is no local task (because the current thread
    /// is not owned by the runtime).
    ///
    /// # Example
    ///
    /// ```
    /// local_data_key!(foo: int)
    ///
    /// assert_eq!(foo.replace(Some(10)), None);
    /// assert_eq!(foo.replace(Some(4)), Some(10));
    /// assert_eq!(foo.replace(None), Some(4));
    /// ```
    pub fn replace(&'static self, data: Option<T>) -> Option<T> {
        let map = match unsafe { get_local_map() } {
            Some(map) => map,
            None => fail!("must have a local task to insert into TLD"),
        };
        let keyval = key_to_key_value(self);

        // The following match takes a mutable borrow on the map. In order to insert
        // our data if the key isn't present, we need to let the match end first.
        let data = match (map.find_mut(&keyval), data) {
            (None, Some(data)) => {
                // The key doesn't exist and we need to insert it. To make borrowck
                // happy, return it up a scope and insert it there.
                data
            }
            (None, None) => {
                // The key doesn't exist and we're trying to replace it with nothing.
                // Do nothing.
                return None
            }
            (Some(slot), data) => {
                // We have a slot with a box.
                let value_box = slot.box_ptr as *mut TLDValueBox<T>;
                let refcount = unsafe { *(*value_box).refcount.get() };
                return match (refcount, data) {
                    (0, None) => {
                        // The current value is uninitialized and we have no new value.
                        // Do nothing.
                        None
                    }
                    (0, Some(new_value)) => {
                        // The current value is uninitialized and we're storing a new value.
                        unsafe {
                            ptr::write(&mut (*value_box).value, new_value);
                            *(*value_box).refcount.get() = 1;
                            None
                        }
                    }
                    (1, None) => {
                        // We have an initialized value and we're removing it.
                        unsafe {
                            let ret = ptr::read(&(*value_box).value);
                            *(*value_box).refcount.get() = 0;
                            Some(ret)
                        }
                    }
                    (1, Some(new_value)) => {
                        // We have an initialized value and we're replacing it.
                        let value_ref = unsafe { &mut (*value_box).value };
                        let ret = mem::replace(value_ref, new_value);
                        // Refcount is already 1, leave it as that.
                        Some(ret)
                    }
                    _ => {
                        // Refcount is 2+, which means we have a live borrow.
                        fail!("TLD value cannot be replaced because it is already borrowed");
                    }
                }
            }
        };
        // If we've reached this point, we need to insert into the map.
        map.insert(keyval, TLDValue::new(data));
        None
    }

    /// Borrows a value from TLD.
    ///
    /// If `None` is returned, then this key is not present in TLD. If `Some`
    /// is returned, then the returned data is a smart pointer representing a
    /// new loan on this TLD key. While on loan, this key cannot be altered via
    /// the `replace` method.
    ///
    /// # Example
    ///
    /// ```
    /// local_data_key!(key: int)
    ///
    /// assert!(key.get().is_none());
    ///
    /// key.replace(Some(3));
    /// assert_eq!(*key.get().unwrap(), 3);
    /// ```
    pub fn get(&'static self) -> Option<Ref<T>> {
        use collections::Map;

        let map = match unsafe { get_local_map() } {
            Some(map) => map,
            None => return None,
        };
        let keyval = key_to_key_value(self);

        match map.find(&keyval) {
            Some(slot) => {
                let value_box = slot.box_ptr as *mut TLDValueBox<T>;
                if unsafe { *(*value_box).refcount.get() } >= 1 {
                    unsafe {
                        *(*value_box).refcount.get() += 1;
                        Some(Ref {
                            _inner: &*value_box,
                            _marker: marker::NoSend
                        })
                    }
                } else {
                    None
                }
            }
            None => None
        }
    }

    // it's not clear if this is the right design for a public API, or if
    // there's even a need for this as a public API, but our benchmarks need
    // this to ensure consistent behavior on each run.
    #[cfg(test)]
    fn clear(&'static self) {
        let map = match unsafe { get_local_map() } {
            Some(map) => map,
            None => return
        };
        let keyval = key_to_key_value(self);
        self.replace(None); // ensure we have no outstanding borrows
        map.remove(&keyval);
    }
}

impl<T: 'static> Deref<T> for Ref<T> {
    #[inline(always)]
    fn deref<'a>(&'a self) -> &'a T {
        &self._inner.value
    }
}

impl<T: 'static + fmt::Show> fmt::Show for Ref<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<T: cmp::PartialEq + 'static> cmp::PartialEq for Ref<T> {
    fn eq(&self, other: &Ref<T>) -> bool {
        (**self).eq(&**other)
    }
    fn ne(&self, other: &Ref<T>) -> bool {
        (**self).ne(&**other)
    }
}

impl<T: cmp::Eq + 'static> cmp::Eq for Ref<T> {}

impl<T: cmp::PartialOrd + 'static> cmp::PartialOrd for Ref<T> {
    fn partial_cmp(&self, other: &Ref<T>) -> Option<cmp::Ordering> {
        (**self).partial_cmp(&**other)
    }
    fn lt(&self, other: &Ref<T>) -> bool { (**self).lt(&**other) }
    fn le(&self, other: &Ref<T>) -> bool { (**self).le(&**other) }
    fn gt(&self, other: &Ref<T>) -> bool { (**self).gt(&**other) }
    fn ge(&self, other: &Ref<T>) -> bool { (**self).ge(&**other) }
}

impl<T: cmp::Ord + 'static> cmp::Ord for Ref<T> {
    fn cmp(&self, other: &Ref<T>) -> cmp::Ordering {
        (**self).cmp(&**other)
    }
}

#[unsafe_destructor]
impl<T: 'static> Drop for Ref<T> {
    fn drop(&mut self) {
        unsafe {
            *self._inner.refcount.get() -= 1;
        }
    }
}

impl TLDValue {
    fn new<T>(value: T) -> TLDValue {
        let box_ptr = unsafe {
            let allocation = heap::allocate(mem::size_of::<TLDValueBox<T>>(),
                                            mem::min_align_of::<TLDValueBox<T>>());
            let value_box = allocation as *mut TLDValueBox<T>;
            ptr::write(value_box, TLDValueBox {
                value: value,
                refcount: UnsafeCell::new(1)
            });
            value_box as *mut ()
        };
        // Destruction of TLDValue needs to know how to properly deallocate the TLDValueBox,
        // so we need our own custom destructor function.
        unsafe fn d<T>(p: *mut ()) {
            let value_box = p as *mut TLDValueBox<T>;
            debug_assert!(*(*value_box).refcount.get() < 2, "TLDValue destructed while borrowed");
            // use a RAII type here to ensure we always deallocate even if we fail while
            // running the destructor for the value.
            struct Guard<T> {
                p: *mut TLDValueBox<T>
            }
            #[unsafe_destructor]
            impl<T> Drop for Guard<T> {
                fn drop(&mut self) {
                    let size = mem::size_of::<TLDValueBox<T>>();
                    let align = mem::align_of::<TLDValueBox<T>>();
                    unsafe { heap::deallocate(self.p as *mut u8, size, align); }
                }
            }
            let _guard = Guard::<T> { p: value_box };
            if *(*value_box).refcount.get() != 0 {
                // the contained value is valid; drop it
                ptr::read(&(*value_box).value);
            }
            // the box will be deallocated by the guard
        }
        TLDValue {
            box_ptr: box_ptr,
            drop_fn: d::<T>
        }
    }
}


impl Drop for TLDValue {
    fn drop(&mut self) {
        // box_ptr should always be non-null. Check it anyway just to be thorough
        if !self.box_ptr.is_null() {
            unsafe { (self.drop_fn)(self.box_ptr) }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use std::prelude::*;
    use std::gc::{Gc, GC};
    use super::*;
    use std::task;

    #[test]
    fn test_tls_multitask() {
        static my_key: Key<String> = &Key;
        my_key.replace(Some("parent data".to_string()));
        task::spawn(proc() {
            // TLD shouldn't carry over.
            assert!(my_key.get().is_none());
            my_key.replace(Some("child data".to_string()));
            assert!(my_key.get().get_ref().as_slice() == "child data");
            // should be cleaned up for us
        });

        // Must work multiple times
        assert!(my_key.get().unwrap().as_slice() == "parent data");
        assert!(my_key.get().unwrap().as_slice() == "parent data");
        assert!(my_key.get().unwrap().as_slice() == "parent data");
    }

    #[test]
    fn test_tls_overwrite() {
        static my_key: Key<String> = &Key;
        my_key.replace(Some("first data".to_string()));
        my_key.replace(Some("next data".to_string())); // Shouldn't leak.
        assert!(my_key.get().unwrap().as_slice() == "next data");
    }

    #[test]
    fn test_tls_pop() {
        static my_key: Key<String> = &Key;
        my_key.replace(Some("weasel".to_string()));
        assert!(my_key.replace(None).unwrap() == "weasel".to_string());
        // Pop must remove the data from the map.
        assert!(my_key.replace(None).is_none());
    }

    #[test]
    fn test_tls_crust_automorestack_memorial_bug() {
        // This might result in a stack-canary clobber if the runtime fails to
        // set sp_limit to 0 when calling the cleanup extern - it might
        // automatically jump over to the rust stack, which causes next_c_sp
        // to get recorded as something within a rust stack segment. Then a
        // subsequent upcall (esp. for logging, think vsnprintf) would run on
        // a stack smaller than 1 MB.
        static my_key: Key<String> = &Key;
        task::spawn(proc() {
            my_key.replace(Some("hax".to_string()));
        });
    }

    #[test]
    fn test_tls_multiple_types() {
        static str_key: Key<String> = &Key;
        static box_key: Key<Gc<()>> = &Key;
        static int_key: Key<int> = &Key;
        task::spawn(proc() {
            str_key.replace(Some("string data".to_string()));
            box_key.replace(Some(box(GC) ()));
            int_key.replace(Some(42));
        });
    }

    #[test]
    fn test_tls_overwrite_multiple_types() {
        static str_key: Key<String> = &Key;
        static box_key: Key<Gc<()>> = &Key;
        static int_key: Key<int> = &Key;
        task::spawn(proc() {
            str_key.replace(Some("string data".to_string()));
            str_key.replace(Some("string data 2".to_string()));
            box_key.replace(Some(box(GC) ()));
            box_key.replace(Some(box(GC) ()));
            int_key.replace(Some(42));
            // This could cause a segfault if overwriting-destruction is done
            // with the crazy polymorphic transmute rather than the provided
            // finaliser.
            int_key.replace(Some(31337));
        });
    }

    #[test]
    #[should_fail]
    fn test_tls_cleanup_on_failure() {
        static str_key: Key<String> = &Key;
        static box_key: Key<Gc<()>> = &Key;
        static int_key: Key<int> = &Key;
        str_key.replace(Some("parent data".to_string()));
        box_key.replace(Some(box(GC) ()));
        task::spawn(proc() {
            str_key.replace(Some("string data".to_string()));
            box_key.replace(Some(box(GC) ()));
            int_key.replace(Some(42));
            fail!();
        });
        // Not quite nondeterministic.
        int_key.replace(Some(31337));
        fail!();
    }

    #[test]
    fn test_cleanup_drops_values() {
        let (tx, rx) = channel::<()>();
        struct Dropper {
            tx: Sender<()>
        };
        impl Drop for Dropper {
            fn drop(&mut self) {
                self.tx.send(());
            }
        }
        static key: Key<Dropper> = &Key;
        let _ = task::try(proc() {
            key.replace(Some(Dropper{ tx: tx }));
        });
        // At this point the task has been cleaned up and the TLD dropped.
        // If the channel doesn't have a value now, then the Sender was leaked.
        assert_eq!(rx.try_recv(), Ok(()));
    }

    #[test]
    fn test_static_pointer() {
        static key: Key<&'static int> = &Key;
        static VALUE: int = 0;
        key.replace(Some(&VALUE));
    }

    #[test]
    fn test_owned() {
        static key: Key<Box<int>> = &Key;
        key.replace(Some(box 1));

        {
            let k1 = key.get().unwrap();
            let k2 = key.get().unwrap();
            let k3 = key.get().unwrap();
            assert_eq!(**k1, 1);
            assert_eq!(**k2, 1);
            assert_eq!(**k3, 1);
        }
        key.replace(Some(box 2));
        assert_eq!(**key.get().unwrap(), 2);
    }

    #[test]
    fn test_same_key_type() {
        static key1: Key<int> = &Key;
        static key2: Key<int> = &Key;
        static key3: Key<int> = &Key;
        static key4: Key<int> = &Key;
        static key5: Key<int> = &Key;
        key1.replace(Some(1));
        key2.replace(Some(2));
        key3.replace(Some(3));
        key4.replace(Some(4));
        key5.replace(Some(5));

        assert_eq!(*key1.get().unwrap(), 1);
        assert_eq!(*key2.get().unwrap(), 2);
        assert_eq!(*key3.get().unwrap(), 3);
        assert_eq!(*key4.get().unwrap(), 4);
        assert_eq!(*key5.get().unwrap(), 5);
    }

    #[test]
    #[should_fail]
    fn test_nested_get_set1() {
        static key: Key<int> = &Key;
        assert_eq!(key.replace(Some(4)), None);

        let _k = key.get();
        key.replace(Some(4));
    }

    // ClearKey is a RAII class that ensures the keys are cleared from the map.
    // This is so repeated runs of a benchmark don't bloat the map with extra
    // keys and distort the measurements.
    // It's not used on the tests because the tests run in separate tasks.
    struct ClearKey<T>(Key<T>);
    #[unsafe_destructor]
    impl<T: 'static> Drop for ClearKey<T> {
        fn drop(&mut self) {
            let ClearKey(ref key) = *self;
            key.clear();
        }
    }

    #[bench]
    fn bench_replace_none(b: &mut test::Bencher) {
        static key: Key<uint> = &Key;
        let _clear = ClearKey(key);
        key.replace(None);
        b.iter(|| {
            key.replace(None)
        });
    }

    #[bench]
    fn bench_replace_some(b: &mut test::Bencher) {
        static key: Key<uint> = &Key;
        let _clear = ClearKey(key);
        key.replace(Some(1u));
        b.iter(|| {
            key.replace(Some(2))
        });
    }

    #[bench]
    fn bench_replace_none_some(b: &mut test::Bencher) {
        static key: Key<uint> = &Key;
        let _clear = ClearKey(key);
        key.replace(Some(0u));
        b.iter(|| {
            let old = key.replace(None).unwrap();
            let new = old + 1;
            key.replace(Some(new))
        });
    }

    #[bench]
    fn bench_100_keys_replace_last(b: &mut test::Bencher) {
        static keys: [KeyValue<uint>, ..100] = [Key, ..100];
        let _clear = keys.iter().map(ClearKey).collect::<Vec<ClearKey<uint>>>();
        for (i, key) in keys.iter().enumerate() {
            key.replace(Some(i));
        }
        b.iter(|| {
            let key: Key<uint> = &keys[99];
            key.replace(Some(42))
        });
    }

    #[bench]
    fn bench_1000_keys_replace_last(b: &mut test::Bencher) {
        static keys: [KeyValue<uint>, ..1000] = [Key, ..1000];
        let _clear = keys.iter().map(ClearKey).collect::<Vec<ClearKey<uint>>>();
        for (i, key) in keys.iter().enumerate() {
            key.replace(Some(i));
        }
        b.iter(|| {
            let key: Key<uint> = &keys[999];
            key.replace(Some(42))
        });
        for key in keys.iter() { key.clear(); }
    }

    #[bench]
    fn bench_get(b: &mut test::Bencher) {
        static key: Key<uint> = &Key;
        let _clear = ClearKey(key);
        key.replace(Some(42));
        b.iter(|| {
            key.get()
        });
    }

    #[bench]
    fn bench_100_keys_get_last(b: &mut test::Bencher) {
        static keys: [KeyValue<uint>, ..100] = [Key, ..100];
        let _clear = keys.iter().map(ClearKey).collect::<Vec<ClearKey<uint>>>();
        for (i, key) in keys.iter().enumerate() {
            key.replace(Some(i));
        }
        b.iter(|| {
            let key: Key<uint> = &keys[99];
            key.get()
        });
    }

    #[bench]
    fn bench_1000_keys_get_last(b: &mut test::Bencher) {
        static keys: [KeyValue<uint>, ..1000] = [Key, ..1000];
        let _clear = keys.iter().map(ClearKey).collect::<Vec<ClearKey<uint>>>();
        for (i, key) in keys.iter().enumerate() {
            key.replace(Some(i));
        }
        b.iter(|| {
            let key: Key<uint> = &keys[999];
            key.get()
        });
    }
}
