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
use local_data::LocalDataKey;
use prelude::*;
use sys;
use task::rt;

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

// The task-local-map actuall stores all TLS information. Right now it's a list
// of key-value pairs. Each value is an actual Rust type so that when the map is
// destroyed all of the contents are destroyed. Each of the keys are actually
// addresses which don't need to be destroyed.
//
// n.b. Has to be a pointer at outermost layer; the foreign call returns void *.
//
// n.b. If TLS is used heavily in future, this could be made more efficient with
// a proper map.
type TaskLocalMap = ~[Option<(*libc::c_void, @LocalData)>];

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

unsafe fn key_to_key_value<T: 'static>(key: LocalDataKey<T>) -> *libc::c_void {
    let pair: sys::Closure = cast::transmute(key);
    return pair.code as *libc::c_void;
}

// If returning Some(..), returns with @T with the map's reference. Careful!
unsafe fn local_data_lookup<T: 'static>(map: &TaskLocalMap,
                                        key: LocalDataKey<T>)
                                            -> Option<(uint, @T)>
{
    use managed::raw::BoxRepr;

    let key_value = key_to_key_value(key);
    for map.iter().enumerate().advance |(i, entry)| {
        match *entry {
            Some((k, ref data)) if k == key_value => {
                // We now have the correct 'data' as type @LocalData which we
                // need to somehow transmute this back to @T. This was
                // originally stored into the map as:
                //
                //    let data = @T;
                //    let element = @data as @LocalData;
                //    insert(key, element);
                //
                // This means that the element stored is a 2-word pair (because
                // it's a trait). The second element is the vtable (we don't
                // need it), and the first element is actually '@@T'. Not only
                // is this @@T, but it's a pointer to the base of the @@T (box
                // and all), so we have to traverse this to find the actual
                // pointer that we want.
                let (_vtable, box) =
                    *cast::transmute::<&@LocalData, &(uint, *BoxRepr)>(data);
                let ptr: &@T = cast::transmute(&(*box).data);
                return Some((i, *ptr));
            }
            _ => {}
        }
    }
    return None;
}

pub unsafe fn local_pop<T: 'static>(handle: Handle,
                                    key: LocalDataKey<T>) -> Option<@T> {
    let map = get_local_map(handle);
    match local_data_lookup(map, key) {
        Some((index, data)) => {
            map[index] = None;
            Some(data)
        }
        None => None
    }
}

pub unsafe fn local_get<T: 'static>(handle: Handle,
                                    key: LocalDataKey<T>) -> Option<@T> {
    match local_data_lookup(get_local_map(handle), key) {
        Some((_, data)) => Some(data),
        None => None
    }
}

pub unsafe fn local_set<T: 'static>(handle: Handle,
                                    key: LocalDataKey<T>,
                                    data: @T) {
    let map = get_local_map(handle);
    let keyval = key_to_key_value(key);

    // When the task-local map is destroyed, all the data needs to be cleaned
    // up. For this reason we can't do some clever tricks to store '@T' as a
    // '*c_void' or something like that. To solve the problem, we cast
    // everything to a trait (LocalData) which is then stored inside the map.
    // Upon destruction of the map, all the objects will be destroyed and the
    // traits have enough information about them to destroy themselves.
    let entry = Some((keyval, @data as @LocalData));

    match local_data_lookup(map, key) {
        Some((index, _)) => { map[index] = entry; }
        None => {
            // Find an empty slot. If not, grow the vector.
            match map.iter().position(|x| x.is_none()) {
                Some(empty_index) => { map[empty_index] = entry; }
                None => { map.push(entry); }
            }
        }
    }
}
