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
use managed::raw::BoxRepr;
use prelude::*;
use ptr;
use sys;
use task::rt;
use unstable::intrinsics;
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

// The task-local-map actuall stores all TLS information. Right now it's a list
// of triples of (key, value, loans). The key is a code pointer (right now at
// least), the value is a trait so destruction can work, and the loans value
// is a count of the number of times the value is currently on loan via
// `local_data_get`.
//
// TLS is designed to be able to store owned data, so `local_data_get` must
// return a borrowed pointer to this data. In order to have a proper lifetime, a
// borrowed pointer is insted yielded to a closure specified to the `get`
// function. As a result, it would be unsound to perform `local_data_set` on the
// same key inside of a `local_data_get`, so we ensure at runtime that this does
// not happen.
//
// n.b. Has to be a pointer at outermost layer; the foreign call returns void *.
//
// n.b. If TLS is used heavily in future, this could be made more efficient with
// a proper map.
type TaskLocalMap = ~[Option<(*libc::c_void, TLSValue, uint)>];
type TLSValue = @LocalData;

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

unsafe fn transmute_back<'a, T>(data: &'a TLSValue) -> (*BoxRepr, &'a T) {
    // Currently, a TLSValue is an '@Trait' instance which means that its actual
    // representation is a pair of (vtable, box). Also, because of issue #7673
    // the box actually points to another box which has the data. Hence, to get
    // a pointer to the actual value that we're interested in, we decode the
    // trait pointer and pass through one layer of boxes to get to the actual
    // data we're interested in.
    //
    // The reference count of the containing @Trait box is already taken care of
    // because the TLSValue is owned by the containing TLS map which means that
    // the reference count is at least one. Extra protections are then added at
    // runtime to ensure that once a loan on a value in TLS has been given out,
    // the value isn't modified by another user.
    let (_vt, box) = *cast::transmute::<&TLSValue, &(uint, *BoxRepr)>(data);

    return (box, cast::transmute(&(*box).data));
}

pub unsafe fn local_pop<T: 'static>(handle: Handle,
                                    key: LocalDataKey<T>) -> Option<T> {
    // If you've never seen horrendously unsafe code written in rust before,
    // just feel free to look a bit farther...
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

                // First, via some various cheats/hacks, we extract the value
                // contained within the TLS box. This leaves a big chunk of
                // memory which needs to be deallocated now.
                let (chunk, inside) = transmute_back(&data);
                let inside = cast::transmute_mut(inside);
                let ret = ptr::read_ptr(inside);

                // Forget the trait box because we're about to manually
                // deallocate the other box. And for my next trick (kids don't
                // try this at home), transmute the chunk of @ memory from the
                // @-trait box to a pointer to a zero-sized '@' block which will
                // then cause it to get properly deallocated, but it won't touch
                // any of the uninitialized memory beyond the end.
                cast::forget(data);
                let chunk: *mut BoxRepr = cast::transmute(chunk);
                (*chunk).header.type_desc =
                    cast::transmute(intrinsics::get_tydesc::<()>());
                let _: @() = cast::transmute(chunk);

                return Some(ret);
            }
            _ => {}
        }
    }
    return None;
}

pub unsafe fn local_get<T: 'static, U>(handle: Handle,
                                       key: LocalDataKey<T>,
                                       f: &fn(Option<&T>) -> U) -> U {
    // This does in theory take multiple mutable loans on the tls map, but the
    // references returned are never removed because the map is only increasing
    // in size (it never shrinks).
    let map = get_local_map(handle);
    let key_value = key_to_key_value(key);
    for map.mut_iter().advance |entry| {
        match *entry {
            Some((k, ref data, ref mut loans)) if k == key_value => {
                *loans = *loans + 1;
                let (_, val) = transmute_back(data);
                let ret = f(Some(val));
                *loans = *loans - 1;
                return ret;
            }
            _ => {}
        }
    }
    return f(None);
}

// FIXME(#7673): This shouldn't require '@', it should use '~'
pub unsafe fn local_set<T: 'static>(handle: Handle,
                                    key: LocalDataKey<@T>,
                                    data: @T) {
    let map = get_local_map(handle);
    let keyval = key_to_key_value(key);

    // When the task-local map is destroyed, all the data needs to be cleaned
    // up. For this reason we can't do some clever tricks to store '@T' as a
    // '*c_void' or something like that. To solve the problem, we cast
    // everything to a trait (LocalData) which is then stored inside the map.
    // Upon destruction of the map, all the objects will be destroyed and the
    // traits have enough information about them to destroy themselves.
    let data = @data as @LocalData;

    // First, try to insert it if we already have it.
    for map.mut_iter().advance |entry| {
        match *entry {
            Some((key, ref mut value, loans)) if key == keyval => {
                if loans != 0 {
                    fail!("TLS value has been loaned via get already");
                }
                util::replace(value, data);
                return;
            }
            _ => {}
        }
    }
    // Next, search for an open spot
    for map.mut_iter().advance |entry| {
        match *entry {
            Some(*) => {}
            None => {
                *entry = Some((keyval, data, 0));
                return;
            }
        }
    }
    // Finally push it on the end of the list
    map.push(Some((keyval, data, 0)));
}
