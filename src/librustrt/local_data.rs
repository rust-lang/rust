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

Allows storing arbitrary types inside task-local-storage (TLS), to be accessed
anywhere within a task, keyed by a global pointer parameterized over the type of
the TLS slot.  Useful for dynamic variables, singletons, and interfacing with
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

use alloc::owned::Box;
use collections::vec::Vec;
use core::kinds::marker;
use core::mem;
use core::raw;

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

#[doc(hidden)]
trait LocalData {}
impl<T: 'static> LocalData for T {}

// The task-local-map stores all TLS information for the currently running task.
// It is stored as an owned pointer into the runtime, and it's only allocated
// when TLS is used for the first time. This map must be very carefully
// constructed because it has many mutable loans unsoundly handed out on it to
// the various invocations of TLS requests.
//
// One of the most important operations is loaning a value via `get` to a
// caller. In doing so, the slot that the TLS entry is occupying cannot be
// invalidated because upon returning its loan state must be updated. Currently
// the TLS map is a vector, but this is possibly dangerous because the vector
// can be reallocated/moved when new values are pushed onto it.
//
// This problem currently isn't solved in a very elegant way. Inside the `get`
// function, it internally "invalidates" all references after the loan is
// finished and looks up into the vector again. In theory this will prevent
// pointers from being moved under our feet so long as LLVM doesn't go too crazy
// with the optimizations.
//
// n.b. If TLS is used heavily in future, this could be made more efficient with
//      a proper map.
#[doc(hidden)]
pub type Map = Vec<Option<(*const u8, TLSValue, uint)>>;
type TLSValue = Box<LocalData + Send>;

// Gets the map from the runtime. Lazily initialises if not done so already.
unsafe fn get_local_map() -> Option<&mut Map> {
    if !Local::exists(None::<Task>) { return None }

    let task: *mut Task = Local::unsafe_borrow();
    match &mut (*task).storage {
        // If the at_exit function is already set, then we just need to take
        // a loan out on the TLS map stored inside
        &LocalStorage(Some(ref mut map_ptr)) => {
            return Some(map_ptr);
        }
        // If this is the first time we've accessed TLS, perform similar
        // actions to the oldsched way of doing things.
        &LocalStorage(ref mut slot) => {
            *slot = Some(Vec::new());
            match *slot {
                Some(ref mut map_ptr) => { return Some(map_ptr) }
                None => fail!("unreachable code"),
            }
        }
    }
}

fn key_to_key_value<T: 'static>(key: Key<T>) -> *const u8 {
    key as *const KeyValue<T> as *const u8
}

/// An RAII immutable reference to a task-local value.
///
/// The task-local data can be accessed through this value, and when this
/// structure is dropped it will return the borrow on the data.
pub struct Ref<T> {
    // FIXME #12808: strange names to try to avoid interfering with
    // field accesses of the contained type via Deref
    _ptr: &'static T,
    _key: Key<T>,
    _index: uint,
    _nosend: marker::NoSend,
}

impl<T: 'static> KeyValue<T> {
    /// Replaces a value in task local storage.
    ///
    /// If this key is already present in TLS, then the previous value is
    /// replaced with the provided data, and then returned.
    ///
    /// # Failure
    ///
    /// This function will fail if this key is present in TLS and currently on
    /// loan with the `get` method.
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

        // When the task-local map is destroyed, all the data needs to be
        // cleaned up. For this reason we can't do some clever tricks to store
        // '~T' as a '*c_void' or something like that. To solve the problem, we
        // cast everything to a trait (LocalData) which is then stored inside
        // the map.  Upon destruction of the map, all the objects will be
        // destroyed and the traits have enough information about them to
        // destroy themselves.
        //
        // Additionally, the type of the local data map must ascribe to Send, so
        // we do the transmute here to add the Send bound back on. This doesn't
        // actually matter because TLS will always own the data (until its moved
        // out) and we're not actually sending it to other schedulers or
        // anything.
        let newval = data.map(|d| {
            let d = box d as Box<LocalData>;
            let d: Box<LocalData + Send> = unsafe { mem::transmute(d) };
            (keyval, d, 0)
        });

        let pos = match self.find(map) {
            Some((i, _, &0)) => Some(i),
            Some((_, _, _)) => fail!("TLS value cannot be replaced because it \
                                      is already borrowed"),
            None => map.iter().position(|entry| entry.is_none()),
        };

        match pos {
            Some(i) => {
                mem::replace(map.get_mut(i), newval).map(|(_, data, _)| {
                    // Move `data` into transmute to get out the memory that it
                    // owns, we must free it manually later.
                    let t: raw::TraitObject = unsafe { mem::transmute(data) };
                    let alloc: Box<T> = unsafe { mem::transmute(t.data) };

                    // Now that we own `alloc`, we can just move out of it as we
                    // would with any other data.
                    *alloc
                })
            }
            None => {
                map.push(newval);
                None
            }
        }
    }

    /// Borrows a value from TLS.
    ///
    /// If `None` is returned, then this key is not present in TLS. If `Some` is
    /// returned, then the returned data is a smart pointer representing a new
    /// loan on this TLS key. While on loan, this key cannot be altered via the
    /// `replace` method.
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
        let map = match unsafe { get_local_map() } {
            Some(map) => map,
            None => return None,
        };

        self.find(map).map(|(pos, data, loan)| {
            *loan += 1;

            // data was created with `~T as ~LocalData`, so we extract
            // pointer part of the trait, (as ~T), and then use
            // compiler coercions to achieve a '&' pointer.
            let ptr = unsafe {
                let data = data as *const Box<LocalData + Send>
                                as *const raw::TraitObject;
                &mut *((*data).data as *mut T)
            };
            Ref { _ptr: ptr, _index: pos, _nosend: marker::NoSend, _key: self }
        })
    }

    fn find<'a>(&'static self,
                map: &'a mut Map) -> Option<(uint, &'a TLSValue, &'a mut uint)>{
        let key_value = key_to_key_value(self);
        map.mut_iter().enumerate().filter_map(|(i, entry)| {
            match *entry {
                Some((k, ref data, ref mut loan)) if k == key_value => {
                    Some((i, data, loan))
                }
                _ => None
            }
        }).next()
    }
}

impl<T: 'static> Deref<T> for Ref<T> {
    fn deref<'a>(&'a self) -> &'a T { self._ptr }
}

#[unsafe_destructor]
impl<T: 'static> Drop for Ref<T> {
    fn drop(&mut self) {
        let map = unsafe { get_local_map().unwrap() };

        let (_, _, ref mut loan) = *map.get_mut(self._index).get_mut_ref();
        *loan -= 1;
    }
}

#[cfg(test)]
mod tests {
    use std::prelude::*;
    use std::gc::{Gc, GC};
    use super::*;
    use std::task;

    #[test]
    fn test_tls_multitask() {
        static my_key: Key<String> = &Key;
        my_key.replace(Some("parent data".to_string()));
        task::spawn(proc() {
            // TLS shouldn't carry over.
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
        key.replace(Some(4));

        let _k = key.get();
        key.replace(Some(4));
    }
}
