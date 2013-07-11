// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(missing_doc)];

use cast;
use libc;
use local_data;
use prelude::*;
use ptr;
use sys;
use task::rt;
use util;

use super::rt::rust_task;
use rt::task::{Task, LocalStorage};

pub enum Handle {
    OldHandle(*rust_task),
    NewHandle(*mut LocalStorage)
}

impl Handle {
    pub fn new() -> Handle {
        use rt::{context, OldTaskContext};
        use rt::local::Local;
        unsafe {
            match context() {
                OldTaskContext => {
                    OldHandle(rt::rust_get_task())
                }
                _ => {
                    let task = Local::unsafe_borrow::<Task>();
                    NewHandle(&mut (*task).storage)
                }
            }
        }
    }
}

trait LocalData {}
impl<T: 'static> LocalData for T {}

// The task-local-map actually stores all TLS information. Right now it's a list
// of triples of (key, value, loans). The key is a code pointer (right now at
// least), the value is a trait so destruction can work, and the loans value
// is a count of the number of times the value is currently on loan via
// `local_data_get`.
//
// TLS is designed to be able to store owned data, so `local_data_get` must
// return a borrowed pointer to this data. In order to have a proper lifetime, a
// borrowed pointer is instead yielded to a closure specified to the `get`
// function. As a result, it would be unsound to perform `local_data_set` on the
// same key inside of a `local_data_get`, so we ensure at runtime that this does
// not happen.
//
// n.b. Has to be a pointer at outermost layer; the foreign call returns void *.
//
// n.b. If TLS is used heavily in future, this could be made more efficient with
// a proper map.
type TaskLocalMap = ~[Option<(*libc::c_void, TLSValue, uint)>];
type TLSValue = ~LocalData:;

fn cleanup_task_local_map(map_ptr: *libc::c_void) {
    unsafe {
        assert!(!map_ptr.is_null());
        // Get and keep the single reference that was created at the
        // beginning.
        let _map: TaskLocalMap = cast::transmute(map_ptr);
        // All local_data will be destroyed along with the map.
    }
}

// Gets the map from the runtime. Lazily initialises if not done so already.
unsafe fn get_local_map(handle: Handle) -> &mut TaskLocalMap {

    unsafe fn oldsched_map(task: *rust_task) -> &mut TaskLocalMap {
        extern fn cleanup_extern_cb(map_ptr: *libc::c_void) {
            cleanup_task_local_map(map_ptr);
        }

        // Relies on the runtime initialising the pointer to null.
        // Note: the map is an owned pointer and is "owned" by TLS. It is moved
        // into the tls slot for this task, and then mutable loans are taken
        // from this slot to modify the map.
        let map_ptr = rt::rust_get_task_local_data(task);
        if (*map_ptr).is_null() {
            // First time TLS is used, create a new map and set up the necessary
            // TLS information for its safe destruction
            let map: TaskLocalMap = ~[];
            *map_ptr = cast::transmute(map);
            rt::rust_task_local_data_atexit(task, cleanup_extern_cb);
        }
        return cast::transmute(map_ptr);
    }

    unsafe fn newsched_map(local: *mut LocalStorage) -> &mut TaskLocalMap {
        // This is based on the same idea as the oldsched code above.
        match &mut *local {
            // If the at_exit function is already set, then we just need to take
            // a loan out on the TLS map stored inside
            &LocalStorage(ref mut map_ptr, Some(_)) => {
                assert!(map_ptr.is_not_null());
                return cast::transmute(map_ptr);
            }
            // If this is the first time we've accessed TLS, perform similar
            // actions to the oldsched way of doing things.
            &LocalStorage(ref mut map_ptr, ref mut at_exit) => {
                assert!(map_ptr.is_null());
                assert!(at_exit.is_none());
                let map: TaskLocalMap = ~[];
                *map_ptr = cast::transmute(map);
                *at_exit = Some(cleanup_task_local_map);
                return cast::transmute(map_ptr);
            }
        }
    }

    match handle {
        OldHandle(task) => oldsched_map(task),
        NewHandle(local_storage) => newsched_map(local_storage)
    }
}

unsafe fn key_to_key_value<T: 'static>(key: local_data::Key<T>) -> *libc::c_void {
    let pair: sys::Closure = cast::transmute(key);
    return pair.code as *libc::c_void;
}

pub unsafe fn local_pop<T: 'static>(handle: Handle,
                                    key: local_data::Key<T>) -> Option<T> {
    let map = get_local_map(handle);
    let key_value = key_to_key_value(key);

    for map.mut_iter().advance |entry| {
        match *entry {
            Some((k, _, loans)) if k == key_value => {
                if loans != 0 {
                    fail!("TLS value has been loaned via get already");
                }
                // Move the data out of the `entry` slot via util::replace. This
                // is guaranteed to succeed because we already matched on `Some`
                // above.
                let data = match util::replace(entry, None) {
                    Some((_, data, _)) => data,
                    None => libc::abort(),
                };

                // Move `data` into transmute to get out the memory that it
                // owns, we must free it manually later.
                let (_vtable, box): (uint, ~~T) = cast::transmute(data);

                // Read the box's value (using the compiler's built-in
                // auto-deref functionality to obtain a pointer to the base)
                let ret = ptr::read_ptr(cast::transmute::<&T, *mut T>(*box));

                // Finally free the allocated memory. we don't want this to
                // actually touch the memory inside because it's all duplicated
                // now, so the box is transmuted to a 0-sized type. We also use
                // a type which references `T` because currently the layout
                // could depend on whether T contains managed pointers or not.
                let _: ~~[T, ..0] = cast::transmute(box);

                // Everything is now deallocated, and we own the value that was
                // located inside TLS, so we now return it.
                return Some(ret);
            }
            _ => {}
        }
    }
    return None;
}

pub unsafe fn local_get<T: 'static, U>(handle: Handle,
                                       key: local_data::Key<T>,
                                       f: &fn(Option<&T>) -> U) -> U {
    // This does in theory take multiple mutable loans on the tls map, but the
    // references returned are never removed because the map is only increasing
    // in size (it never shrinks).
    let map = get_local_map(handle);
    let key_value = key_to_key_value(key);
    for map.mut_iter().advance |entry| {
        match *entry {
            Some((k, ref data, ref mut loans)) if k == key_value => {
                let ret;
                *loans = *loans + 1;
                // data was created with `~~T as ~LocalData`, so we extract
                // pointer part of the trait, (as ~~T), and then use compiler
                // coercions to achieve a '&' pointer
                match *cast::transmute::<&TLSValue, &(uint, ~~T)>(data) {
                    (_vtable, ref box) => {
                        let value: &T = **box;
                        ret = f(Some(value));
                    }
                }
                *loans = *loans - 1;
                return ret;
            }
            _ => {}
        }
    }
    return f(None);
}

pub unsafe fn local_set<T: 'static>(handle: Handle,
                                    key: local_data::Key<T>,
                                    data: T) {
    let map = get_local_map(handle);
    let keyval = key_to_key_value(key);

    // When the task-local map is destroyed, all the data needs to be cleaned
    // up. For this reason we can't do some clever tricks to store '~T' as a
    // '*c_void' or something like that. To solve the problem, we cast
    // everything to a trait (LocalData) which is then stored inside the map.
    // Upon destruction of the map, all the objects will be destroyed and the
    // traits have enough information about them to destroy themselves.
    //
    // FIXME(#7673): This should be "~data as ~LocalData" (without the colon at
    //               the end, and only one sigil)
    let data = ~~data as ~LocalData:;

    fn insertion_position(map: &mut TaskLocalMap,
                          key: *libc::c_void) -> Option<uint> {
        // First see if the map contains this key already
        let curspot = map.iter().position(|entry| {
            match *entry {
                Some((ekey, _, loans)) if key == ekey => {
                    if loans != 0 {
                        fail!("TLS value has been loaned via get already");
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

    match insertion_position(map, keyval) {
        Some(i) => { map[i] = Some((keyval, data, 0)); }
        None => { map.push(Some((keyval, data, 0))); }
    }
}
