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
use std::local_data;

local_data_key!(key_int: int)
local_data_key!(key_vector: ~[int])

local_data::set(key_int, 3);
local_data::get(key_int, |opt| assert_eq!(opt.map(|x| *x), Some(3)));

local_data::set(key_vector, ~[4]);
local_data::get(key_vector, |opt| assert_eq!(*opt.unwrap(), ~[4]));
```

*/

// Casting 'Arcane Sight' reveals an overwhelming aura of Transmutation
// magic.

use cast;
use iter::{Iterator};
use kinds::Send;
use mem::replace;
use option::{None, Option, Some};
use owned::Box;
use rt::task::{Task, LocalStorage};
use slice::{ImmutableVector, MutableVector};
use vec::Vec;

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

#[allow(missing_doc)]
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
pub type Map = Vec<Option<(*u8, TLSValue, LoanState)>>;
type TLSValue = Box<LocalData:Send>;

// Gets the map from the runtime. Lazily initialises if not done so already.
unsafe fn get_local_map() -> &mut Map {
    use rt::local::Local;

    let task: *mut Task = Local::unsafe_borrow();
    match &mut (*task).storage {
        // If the at_exit function is already set, then we just need to take
        // a loan out on the TLS map stored inside
        &LocalStorage(Some(ref mut map_ptr)) => {
            return map_ptr;
        }
        // If this is the first time we've accessed TLS, perform similar
        // actions to the oldsched way of doing things.
        &LocalStorage(ref mut slot) => {
            *slot = Some(vec!());
            match *slot {
                Some(ref mut map_ptr) => { return map_ptr }
                None => abort()
            }
        }
    }
}

#[deriving(Eq)]
enum LoanState {
    NoLoan, ImmLoan, MutLoan
}

impl LoanState {
    fn describe(&self) -> &'static str {
        match *self {
            NoLoan => "no loan",
            ImmLoan => "immutable",
            MutLoan => "mutable"
        }
    }
}

fn key_to_key_value<T: 'static>(key: Key<T>) -> *u8 {
    unsafe { cast::transmute(key) }
}

/// Removes a task-local value from task-local storage. This will return
/// Some(value) if the key was present in TLS, otherwise it will return None.
///
/// A runtime assertion will be triggered it removal of TLS value is attempted
/// while the value is still loaned out via `get` or `get_mut`.
pub fn pop<T: 'static>(key: Key<T>) -> Option<T> {
    let map = unsafe { get_local_map() };
    let key_value = key_to_key_value(key);

    for entry in map.mut_iter() {
        match *entry {
            Some((k, _, loan)) if k == key_value => {
                if loan != NoLoan {
                    fail!("TLS value cannot be removed because it is currently \
                          borrowed as {}", loan.describe());
                }
                // Move the data out of the `entry` slot via prelude::replace.
                // This is guaranteed to succeed because we already matched
                // on `Some` above.
                let data = match replace(entry, None) {
                    Some((_, data, _)) => data,
                    None => abort()
                };

                // Move `data` into transmute to get out the memory that it
                // owns, we must free it manually later.
                let (_vtable, alloc): (uint, Box<T>) = unsafe {
                    cast::transmute(data)
                };

                // Now that we own `alloc`, we can just move out of it as we
                // would with any other data.
                return Some(*alloc);
            }
            _ => {}
        }
    }
    return None;
}

/// Retrieves a value from TLS. The closure provided is yielded `Some` of a
/// reference to the value located in TLS if one exists, or `None` if the key
/// provided is not present in TLS currently.
///
/// It is considered a runtime error to attempt to get a value which is already
/// on loan via the `get_mut` method provided.
pub fn get<T: 'static, U>(key: Key<T>, f: |Option<&T>| -> U) -> U {
    get_with(key, ImmLoan, f)
}

/// Retrieves a mutable value from TLS. The closure provided is yielded `Some`
/// of a reference to the mutable value located in TLS if one exists, or `None`
/// if the key provided is not present in TLS currently.
///
/// It is considered a runtime error to attempt to get a value which is already
/// on loan via this or the `get` methods.
pub fn get_mut<T: 'static, U>(key: Key<T>, f: |Option<&mut T>| -> U) -> U {
    get_with(key, MutLoan, |x| {
        match x {
            None => f(None),
            // We're violating a lot of compiler guarantees with this
            // invocation of `transmute`, but we're doing runtime checks to
            // ensure that it's always valid (only one at a time).
            //
            // there is no need to be upset!
            Some(x) => { f(Some(unsafe { cast::transmute::<&_, &mut _>(x) })) }
        }
    })
}

fn get_with<T:'static,
            U>(
            key: Key<T>,
            state: LoanState,
            f: |Option<&T>| -> U)
            -> U {
    // This function must be extremely careful. Because TLS can store owned
    // values, and we must have some form of `get` function other than `pop`,
    // this function has to give a `&` reference back to the caller.
    //
    // One option is to return the reference, but this cannot be sound because
    // the actual lifetime of the object is not known. The slot in TLS could not
    // be modified until the object goes out of scope, but the TLS code cannot
    // know when this happens.
    //
    // For this reason, the reference is yielded to a specified closure. This
    // way the TLS code knows exactly what the lifetime of the yielded pointer
    // is, allowing callers to acquire references to owned data. This is also
    // sound so long as measures are taken to ensure that while a TLS slot is
    // loaned out to a caller, it's not modified recursively.
    let map = unsafe { get_local_map() };
    let key_value = key_to_key_value(key);

    let pos = map.iter().position(|entry| {
        match *entry {
            Some((k, _, _)) if k == key_value => true, _ => false
        }
    });
    match pos {
        None => { return f(None); }
        Some(i) => {
            let ret;
            let mut return_loan = false;
            match *map.get_mut(i) {
                Some((_, ref data, ref mut loan)) => {
                    match (state, *loan) {
                        (_, NoLoan) => {
                            *loan = state;
                            return_loan = true;
                        }
                        (ImmLoan, ImmLoan) => {}
                        (want, cur) => {
                            fail!("TLS slot cannot be borrowed as {} because \
                                    it is already borrowed as {}",
                                  want.describe(), cur.describe());
                        }
                    }
                    // data was created with `box T as Box<LocalData>`, so we
                    // extract pointer part of the trait, (as Box<T>), and
                    // then use compiler coercions to achieve a '&' pointer.
                    unsafe {
                        match *cast::transmute::<&TLSValue,
                                                 &(uint, Box<T>)>(data){
                            (_vtable, ref alloc) => {
                                let value: &T = *alloc;
                                ret = f(Some(value));
                            }
                        }
                    }
                }
                _ => abort()
            }

            // n.b. 'data' and 'loans' are both invalid pointers at the point
            // 'f' returned because `f` could have appended more TLS items which
            // in turn relocated the vector. Hence we do another lookup here to
            // fixup the loans.
            if return_loan {
                match *map.get_mut(i) {
                    Some((_, _, ref mut loan)) => { *loan = NoLoan; }
                    None => abort()
                }
            }
            return ret;
        }
    }
}

fn abort() -> ! {
    use intrinsics;
    unsafe { intrinsics::abort() }
}

/// Inserts a value into task local storage. If the key is already present in
/// TLS, then the previous value is removed and replaced with the provided data.
///
/// It is considered a runtime error to attempt to set a key which is currently
/// on loan via the `get` or `get_mut` methods.
pub fn set<T: 'static>(key: Key<T>, data: T) {
    let map = unsafe { get_local_map() };
    let keyval = key_to_key_value(key);

    // When the task-local map is destroyed, all the data needs to be cleaned
    // up. For this reason we can't do some clever tricks to store 'Box<T>' as
    // a '*c_void' or something like that. To solve the problem, we cast
    // everything to a trait (LocalData) which is then stored inside the map.
    // Upon destruction of the map, all the objects will be destroyed and the
    // traits have enough information about them to destroy themselves.
    let data = box data as Box<LocalData:>;

    fn insertion_position(map: &mut Map,
                          key: *u8) -> Option<uint> {
        // First see if the map contains this key already
        let curspot = map.iter().position(|entry| {
            match *entry {
                Some((ekey, _, loan)) if key == ekey => {
                    if loan != NoLoan {
                        fail!("TLS value cannot be overwritten because it is
                               already borrowed as {}", loan.describe())
                    }
                    true
                }
                _ => false,
            }
        });
        // If it doesn't contain the key, just find a slot that's None
        match curspot {
            Some(i) => Some(i),
            None => map.iter().position(|entry| entry.is_none())
        }
    }

    // The type of the local data map must ascribe to Send, so we do the
    // transmute here to add the Send bound back on. This doesn't actually
    // matter because TLS will always own the data (until its moved out) and
    // we're not actually sending it to other schedulers or anything.
    let data: Box<LocalData:Send> = unsafe { cast::transmute(data) };
    match insertion_position(map, keyval) {
        Some(i) => { *map.get_mut(i) = Some((keyval, data, NoLoan)); }
        None => { map.push(Some((keyval, data, NoLoan))); }
    }
}

/// Modifies a task-local value by temporarily removing it from task-local
/// storage and then re-inserting if `Some` is returned from the closure.
///
/// This function will have the same runtime errors as generated from `pop` and
/// `set` (the key must not currently be on loan
pub fn modify<T: 'static>(key: Key<T>, f: |Option<T>| -> Option<T>) {
    match f(pop(key)) {
        Some(next) => { set(key, next); }
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::*;
    use owned::Box;
    use task;

    #[test]
    fn test_tls_multitask() {
        static my_key: Key<~str> = &Key;
        set(my_key, "parent data".to_owned());
        task::spawn(proc() {
            // TLS shouldn't carry over.
            assert!(get(my_key, |k| k.map(|k| (*k).clone())).is_none());
            set(my_key, "child data".to_owned());
            assert!(get(my_key, |k| k.map(|k| (*k).clone())).unwrap() ==
                    "child data".to_owned());
            // should be cleaned up for us
        });
        // Must work multiple times
        assert!(get(my_key, |k| k.map(|k| (*k).clone())).unwrap() == "parent data".to_owned());
        assert!(get(my_key, |k| k.map(|k| (*k).clone())).unwrap() == "parent data".to_owned());
        assert!(get(my_key, |k| k.map(|k| (*k).clone())).unwrap() == "parent data".to_owned());
    }

    #[test]
    fn test_tls_overwrite() {
        static my_key: Key<~str> = &Key;
        set(my_key, "first data".to_owned());
        set(my_key, "next data".to_owned()); // Shouldn't leak.
        assert!(get(my_key, |k| k.map(|k| (*k).clone())).unwrap() == "next data".to_owned());
    }

    #[test]
    fn test_tls_pop() {
        static my_key: Key<~str> = &Key;
        set(my_key, "weasel".to_owned());
        assert!(pop(my_key).unwrap() == "weasel".to_owned());
        // Pop must remove the data from the map.
        assert!(pop(my_key).is_none());
    }

    #[test]
    fn test_tls_modify() {
        static my_key: Key<~str> = &Key;
        modify(my_key, |data| {
            match data {
                Some(ref val) => fail!("unwelcome value: {}", *val),
                None           => Some("first data".to_owned())
            }
        });
        modify(my_key, |data| {
            match data.as_ref().map(|s| s.as_slice()) {
                Some("first data") => Some("next data".to_owned()),
                Some(ref val)       => fail!("wrong value: {}", *val),
                None                 => fail!("missing value")
            }
        });
        assert!(pop(my_key).unwrap() == "next data".to_owned());
    }

    #[test]
    fn test_tls_crust_automorestack_memorial_bug() {
        // This might result in a stack-canary clobber if the runtime fails to
        // set sp_limit to 0 when calling the cleanup extern - it might
        // automatically jump over to the rust stack, which causes next_c_sp
        // to get recorded as something within a rust stack segment. Then a
        // subsequent upcall (esp. for logging, think vsnprintf) would run on
        // a stack smaller than 1 MB.
        static my_key: Key<~str> = &Key;
        task::spawn(proc() {
            set(my_key, "hax".to_owned());
        });
    }

    #[test]
    fn test_tls_multiple_types() {
        static str_key: Key<~str> = &Key;
        static box_key: Key<@()> = &Key;
        static int_key: Key<int> = &Key;
        task::spawn(proc() {
            set(str_key, "string data".to_owned());
            set(box_key, @());
            set(int_key, 42);
        });
    }

    #[test]
    #[allow(dead_code)]
    fn test_tls_overwrite_multiple_types() {
        static str_key: Key<~str> = &Key;
        static box_key: Key<@()> = &Key;
        static int_key: Key<int> = &Key;
        task::spawn(proc() {
            set(str_key, "string data".to_owned());
            set(str_key, "string data 2".to_owned());
            set(box_key, @());
            set(box_key, @());
            set(int_key, 42);
            // This could cause a segfault if overwriting-destruction is done
            // with the crazy polymorphic transmute rather than the provided
            // finaliser.
            set(int_key, 31337);
        });
    }

    #[test]
    #[should_fail]
    fn test_tls_cleanup_on_failure() {
        static str_key: Key<~str> = &Key;
        static box_key: Key<@()> = &Key;
        static int_key: Key<int> = &Key;
        set(str_key, "parent data".to_owned());
        set(box_key, @());
        task::spawn(proc() {
            // spawn_linked
            set(str_key, "string data".to_owned());
            set(box_key, @());
            set(int_key, 42);
            fail!();
        });
        // Not quite nondeterministic.
        set(int_key, 31337);
        fail!();
    }

    #[test]
    fn test_static_pointer() {
        static key: Key<&'static int> = &Key;
        static VALUE: int = 0;
        let v: &'static int = &VALUE;
        set(key, v);
    }

    #[test]
    fn test_owned() {
        static key: Key<Box<int>> = &Key;
        set(key, box 1);

        get(key, |v| {
            get(key, |v| {
                get(key, |v| {
                    assert_eq!(**v.unwrap(), 1);
                });
                assert_eq!(**v.unwrap(), 1);
            });
            assert_eq!(**v.unwrap(), 1);
        });
        set(key, box 2);
        get(key, |v| {
            assert_eq!(**v.unwrap(), 2);
        })
    }

    #[test]
    fn test_get_mut() {
        static key: Key<int> = &Key;
        set(key, 1);

        get_mut(key, |v| {
            *v.unwrap() = 2;
        });

        get(key, |v| {
            assert_eq!(*v.unwrap(), 2);
        })
    }

    #[test]
    fn test_same_key_type() {
        static key1: Key<int> = &Key;
        static key2: Key<int> = &Key;
        static key3: Key<int> = &Key;
        static key4: Key<int> = &Key;
        static key5: Key<int> = &Key;
        set(key1, 1);
        set(key2, 2);
        set(key3, 3);
        set(key4, 4);
        set(key5, 5);

        get(key1, |x| assert_eq!(*x.unwrap(), 1));
        get(key2, |x| assert_eq!(*x.unwrap(), 2));
        get(key3, |x| assert_eq!(*x.unwrap(), 3));
        get(key4, |x| assert_eq!(*x.unwrap(), 4));
        get(key5, |x| assert_eq!(*x.unwrap(), 5));
    }

    #[test]
    #[should_fail]
    fn test_nested_get_set1() {
        static key: Key<int> = &Key;
        set(key, 4);
        get(key, |_| {
            set(key, 4);
        })
    }

    #[test]
    #[should_fail]
    fn test_nested_get_mut2() {
        static key: Key<int> = &Key;
        set(key, 4);
        get(key, |_| {
            get_mut(key, |_| {})
        })
    }

    #[test]
    #[should_fail]
    fn test_nested_get_mut3() {
        static key: Key<int> = &Key;
        set(key, 4);
        get_mut(key, |_| {
            get(key, |_| {})
        })
    }

    #[test]
    #[should_fail]
    fn test_nested_get_mut4() {
        static key: Key<int> = &Key;
        set(key, 4);
        get_mut(key, |_| {
            get_mut(key, |_| {})
        })
    }
}
