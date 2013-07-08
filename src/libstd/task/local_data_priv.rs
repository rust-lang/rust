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
use cmp::Eq;
use libc;
use prelude::*;
use task::rt;
use local_data::LocalDataKey;

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

pub trait LocalData { }
impl<T: 'static> LocalData for @T { }

impl Eq for @LocalData {
    fn eq(&self, other: &@LocalData) -> bool {
        unsafe {
            let ptr_a: &(uint, uint) = cast::transmute(self);
            let ptr_b: &(uint, uint) = cast::transmute(other);
            return ptr_a == ptr_b;
        }
    }
    fn ne(&self, other: &@LocalData) -> bool { !(*self).eq(other) }
}

// If TLS is used heavily in future, this could be made more efficient with a
// proper map.
type TaskLocalElement = (*libc::c_void, *libc::c_void, @LocalData);
// Has to be a pointer at outermost layer; the foreign call returns void *.
type TaskLocalMap = @mut ~[Option<TaskLocalElement>];

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
unsafe fn get_local_map(handle: Handle) -> TaskLocalMap {
    match handle {
        OldHandle(task) => get_task_local_map(task),
        NewHandle(local_storage) => get_newsched_local_map(local_storage)
    }
}

unsafe fn get_task_local_map(task: *rust_task) -> TaskLocalMap {

    extern fn cleanup_task_local_map_extern_cb(map_ptr: *libc::c_void) {
        cleanup_task_local_map(map_ptr);
    }

    // Relies on the runtime initialising the pointer to null.
    // Note: The map's box lives in TLS invisibly referenced once. Each time
    // we retrieve it for get/set, we make another reference, which get/set
    // drop when they finish. No "re-storing after modifying" is needed.
    let map_ptr = rt::rust_get_task_local_data(task);
    if map_ptr.is_null() {
        let map: TaskLocalMap = @mut ~[];
        // NB: This bumps the ref count before converting to an unsafe pointer,
        // keeping the map alive until TLS is destroyed
        rt::rust_set_task_local_data(task, cast::transmute(map));
        rt::rust_task_local_data_atexit(task, cleanup_task_local_map_extern_cb);
        map
    } else {
        let map = cast::transmute(map_ptr);
        let nonmut = cast::transmute::<TaskLocalMap,
                                       @~[Option<TaskLocalElement>]>(map);
        cast::bump_box_refcount(nonmut);
        map
    }
}

unsafe fn get_newsched_local_map(local: *mut LocalStorage) -> TaskLocalMap {
    match &mut *local {
        &LocalStorage(map_ptr, Some(_)) => {
            assert!(map_ptr.is_not_null());
            let map = cast::transmute(map_ptr);
            let nonmut = cast::transmute::<TaskLocalMap,
            @~[Option<TaskLocalElement>]>(map);
            cast::bump_box_refcount(nonmut);
            return map;
        }
        &LocalStorage(ref mut map_ptr, ref mut at_exit) => {
            assert!((*map_ptr).is_null());
            let map: TaskLocalMap = @mut ~[];
            *map_ptr = cast::transmute(map);
            let at_exit_fn: ~fn(*libc::c_void) = |p|cleanup_task_local_map(p);
            *at_exit = Some(at_exit_fn);
            return map;
        }
    }
}

unsafe fn key_to_key_value<T: 'static>(key: LocalDataKey<T>) -> *libc::c_void {
    // Keys are closures, which are (fnptr,envptr) pairs. Use fnptr.
    // Use reinterpret_cast -- transmute would leak (forget) the closure.
    let pair: (*libc::c_void, *libc::c_void) = cast::transmute_copy(&key);
    pair.first()
}

// If returning Some(..), returns with @T with the map's reference. Careful!
unsafe fn local_data_lookup<T: 'static>(
    map: TaskLocalMap, key: LocalDataKey<T>)
    -> Option<(uint, *libc::c_void)> {

    let key_value = key_to_key_value(key);
    let map_pos = (*map).iter().position_(|entry|
        match *entry {
            Some((k,_,_)) => k == key_value,
            None => false
        }
    );
    do map_pos.map |index| {
        // .get() is guaranteed because of "None { false }" above.
        let (_, data_ptr, _) = (*map)[*index].get();
        (*index, data_ptr)
    }
}

unsafe fn local_get_helper<T: 'static>(
    handle: Handle, key: LocalDataKey<T>,
    do_pop: bool) -> Option<@T> {

    let map = get_local_map(handle);
    // Interpreturn our findings from the map
    do local_data_lookup(map, key).map |result| {
        // A reference count magically appears on 'data' out of thin air. It
        // was referenced in the local_data box, though, not here, so before
        // overwriting the local_data_box we need to give an extra reference.
        // We must also give an extra reference when not removing.
        let (index, data_ptr) = *result;
        let data: @T = cast::transmute(data_ptr);
        cast::bump_box_refcount(data);
        if do_pop {
            map[index] = None;
        }
        data
    }
}


pub unsafe fn local_pop<T: 'static>(
    handle: Handle,
    key: LocalDataKey<T>) -> Option<@T> {

    local_get_helper(handle, key, true)
}

pub unsafe fn local_get<T: 'static>(
    handle: Handle,
    key: LocalDataKey<T>) -> Option<@T> {

    local_get_helper(handle, key, false)
}

pub unsafe fn local_set<T: 'static>(
    handle: Handle, key: LocalDataKey<T>, data: @T) {

    let map = get_local_map(handle);
    // Store key+data as *voids. Data is invisibly referenced once; key isn't.
    let keyval = key_to_key_value(key);
    // We keep the data in two forms: one as an unsafe pointer, so we can get
    // it back by casting; another in an existential box, so the reference we
    // own on it can be dropped when the box is destroyed. The unsafe pointer
    // does not have a reference associated with it, so it may become invalid
    // when the box is destroyed.
    let data_ptr = *cast::transmute::<&@T, &*libc::c_void>(&data);
    let data_box = @data as @LocalData;
    // Construct new entry to store in the map.
    let new_entry = Some((keyval, data_ptr, data_box));
    // Find a place to put it.
    match local_data_lookup(map, key) {
        Some((index, _old_data_ptr)) => {
            // Key already had a value set, _old_data_ptr, whose reference
            // will get dropped when the local_data box is overwritten.
            map[index] = new_entry;
        }
        None => {
            // Find an empty slot. If not, grow the vector.
            match (*map).iter().position_(|x| x.is_none()) {
                Some(empty_index) => { map[empty_index] = new_entry; }
                None => { map.push(new_entry); }
            }
        }
    }
}

pub unsafe fn local_modify<T: 'static>(
    handle: Handle, key: LocalDataKey<T>,
    modify_fn: &fn(Option<@T>) -> Option<@T>) {

    // Could be more efficient by doing the lookup work, but this is easy.
    let newdata = modify_fn(local_pop(handle, key));
    if newdata.is_some() {
        local_set(handle, key, newdata.unwrap());
    }
}
