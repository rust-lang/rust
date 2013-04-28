// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * A task-based interface to the uv loop
 *
 * The I/O task runs in its own single-threaded scheduler.  By using the
 * `interact` function you can execute code in a uv callback.
 */

use ll = uv_ll;

use core::libc::c_void;
use core::libc;
use core::comm::{stream, Port, Chan, SharedChan};
use core::ptr::addr_of;

/// Used to abstract-away direct interaction with a libuv loop.
pub struct IoTask {
    async_handle: *ll::uv_async_t,
    op_chan: SharedChan<IoTaskMsg>
}

impl Clone for IoTask {
    fn clone(&self) -> IoTask {
        IoTask{
            async_handle: self.async_handle,
            op_chan: self.op_chan.clone()
        }
    }
}

pub fn spawn_iotask(task: task::TaskBuilder) -> IoTask {

    let (iotask_port, iotask_chan) = stream();

    do task.sched_mode(task::SingleThreaded).spawn {
        debug!("entering libuv task");
        run_loop(&iotask_chan);
        debug!("libuv task exiting");
    };

    iotask_port.recv()
}


/**
 * Provide a callback to be processed by `iotask`
 *
 * The primary way to do operations again a running `iotask` that
 * doesn't involve creating a uv handle via `safe_handle`
 *
 * # Warning
 *
 * This function is the only safe way to interact with _any_ `iotask`.
 * Using functions in the `uv::ll` module outside of the `cb` passed into
 * this function is _very dangerous_.
 *
 * # Arguments
 *
 * * iotask - a uv I/O task that you want to do operations against
 * * cb - a function callback to be processed on the running loop's
 * thread. The only parameter passed in is an opaque pointer representing the
 * running `uv_loop_t*`. In the context of this callback, it is safe to use
 * this pointer to do various uv_* API calls contained within the `uv::ll`
 * module. It is not safe to send the `loop_ptr` param to this callback out
 * via ports/chans.
 */
pub fn interact(iotask: &IoTask, cb: ~fn(*c_void)) {
    send_msg(iotask, Interaction(cb));
}

/**
 * Shut down the I/O task
 *
 * Is used to signal to the loop that it should close the internally-held
 * async handle and do a sanity check to make sure that all other handles are
 * closed, causing a failure otherwise.
 */
pub fn exit(iotask: &IoTask) {
    send_msg(iotask, TeardownLoop);
}


// INTERNAL API

enum IoTaskMsg {
    Interaction(~fn(*libc::c_void)),
    TeardownLoop
}

/// Run the loop and begin handling messages
fn run_loop(iotask_ch: &Chan<IoTask>) {

    unsafe {
        debug!("creating loop");
        let loop_ptr = ll::loop_new();

        // set up the special async handle we'll use to allow multi-task
        // communication with this loop
        let async = ll::async_t();
        let async_handle = addr_of(&async);

        // associate the async handle with the loop
        ll::async_init(loop_ptr, async_handle, wake_up_cb);

        let (msg_po, msg_ch) = stream::<IoTaskMsg>();

        // initialize our loop data and store it in the loop
        let data: IoTaskLoopData = IoTaskLoopData {
            async_handle: async_handle,
            msg_po: msg_po
        };
        ll::set_data_for_uv_handle(async_handle, addr_of(&data));

        // Send out a handle through which folks can talk to us
        // while we dwell in the I/O loop
        let iotask = IoTask{
            async_handle: async_handle,
            op_chan: SharedChan::new(msg_ch)
        };
        iotask_ch.send(iotask);

        debug!("about to run uv loop");
        // enter the loop... this blocks until the loop is done..
        ll::run(loop_ptr);
        debug!("uv loop ended");
        ll::loop_delete(loop_ptr);
    }
}

// data that lives for the lifetime of the high-evel oo
struct IoTaskLoopData {
    async_handle: *ll::uv_async_t,
    msg_po: Port<IoTaskMsg>,
}

fn send_msg(iotask: &IoTask,
            msg: IoTaskMsg) {
    iotask.op_chan.send(msg);
    unsafe {
        ll::async_send(iotask.async_handle);
    }
}

/// Dispatch all pending messages
extern fn wake_up_cb(async_handle: *ll::uv_async_t,
                    status: int) {

    debug!("wake_up_cb extern.. handle: %? status: %?",
                     async_handle, status);

    unsafe {
        let loop_ptr = ll::get_loop_for_uv_handle(async_handle);
        let data =
            ll::get_data_for_uv_handle(async_handle) as *IoTaskLoopData;
        let msg_po = &(*data).msg_po;

        while msg_po.peek() {
            match msg_po.recv() {
                Interaction(ref cb) => (*cb)(loop_ptr),
                TeardownLoop => begin_teardown(data)
            }
        }
    }
}

fn begin_teardown(data: *IoTaskLoopData) {
    unsafe {
        debug!("iotask begin_teardown() called, close async_handle");
        let async_handle = (*data).async_handle;
        ll::close(async_handle as *c_void, tear_down_close_cb);
    }
}
extern fn tear_down_walk_cb(handle: *libc::c_void, arg: *libc::c_void) {
    debug!("IN TEARDOWN WALK CB");
    // pretty much, if we still have an active handle and it is *not*
    // the async handle that facilities global loop communication, we
    // want to barf out and fail
    assert!(handle == arg);
}

extern fn tear_down_close_cb(handle: *ll::uv_async_t) {
    unsafe {
        let loop_ptr = ll::get_loop_for_uv_handle(handle);
        debug!("in tear_down_close_cb");
        ll::walk(loop_ptr, tear_down_walk_cb, handle as *libc::c_void);
    }
}

#[cfg(test)]
extern fn async_close_cb(handle: *ll::uv_async_t) {
    unsafe {
        debug!("async_close_cb handle %?", handle);
        let exit_ch = &(*(ll::get_data_for_uv_handle(handle)
                        as *AhData)).exit_ch;
        let exit_ch = exit_ch.clone();
        exit_ch.send(());
    }
}

#[cfg(test)]
extern fn async_handle_cb(handle: *ll::uv_async_t, status: libc::c_int) {
    unsafe {
        debug!("async_handle_cb handle %? status %?",handle,status);
        ll::close(handle, async_close_cb);
    }
}

#[cfg(test)]
struct AhData {
    iotask: IoTask,
    exit_ch: SharedChan<()>
}

#[cfg(test)]
fn impl_uv_iotask_async(iotask: &IoTask) {
    let async_handle = ll::async_t();
    let ah_ptr = ptr::addr_of(&async_handle);
    let (exit_po, exit_ch) = stream::<()>();
    let ah_data = AhData {
        iotask: iotask.clone(),
        exit_ch: SharedChan::new(exit_ch)
    };
    let ah_data_ptr: *AhData = ptr::to_unsafe_ptr(&ah_data);
    debug!("about to interact");
    do interact(iotask) |loop_ptr| {
        unsafe {
            debug!("interacting");
            ll::async_init(loop_ptr, ah_ptr, async_handle_cb);
            ll::set_data_for_uv_handle(
                ah_ptr, ah_data_ptr as *libc::c_void);
            ll::async_send(ah_ptr);
        }
    };
    debug!("waiting for async close");
    exit_po.recv();
}

// this fn documents the bear minimum neccesary to roll your own
// high_level_loop
#[cfg(test)]
fn spawn_test_loop(exit_ch: ~Chan<()>) -> IoTask {
    let (iotask_port, iotask_ch) = stream::<IoTask>();
    do task::spawn_sched(task::ManualThreads(1u)) {
        debug!("about to run a test loop");
        run_loop(&iotask_ch);
        exit_ch.send(());
    };
    return iotask_port.recv();
}

#[cfg(test)]
extern fn lifetime_handle_close(handle: *libc::c_void) {
    debug!("lifetime_handle_close ptr %?", handle);
}

#[cfg(test)]
extern fn lifetime_async_callback(handle: *libc::c_void,
                                 status: libc::c_int) {
    debug!("lifetime_handle_close ptr %? status %?",
                    handle, status);
}

#[test]
fn test_uv_iotask_async() {
    let (exit_po, exit_ch) = stream::<()>();
    let iotask = &spawn_test_loop(~exit_ch);

    debug!("spawned iotask");

    // using this handle to manage the lifetime of the
    // high_level_loop, as it will exit the first time one of
    // the impl_uv_hl_async() is cleaned up with no one ref'd
    // handles on the loop (Which can happen under
    // race-condition type situations.. this ensures that the
    // loop lives until, at least, all of the
    // impl_uv_hl_async() runs have been called, at least.
    let (work_exit_po, work_exit_ch) = stream::<()>();
    let work_exit_ch = SharedChan::new(work_exit_ch);
    for iter::repeat(7u) {
        let iotask_clone = iotask.clone();
        let work_exit_ch_clone = work_exit_ch.clone();
        do task::spawn_sched(task::ManualThreads(1u)) {
            debug!("async");
            impl_uv_iotask_async(&iotask_clone);
            debug!("done async");
            work_exit_ch_clone.send(());
        };
    };
    for iter::repeat(7u) {
        debug!("waiting");
        work_exit_po.recv();
    };
    debug!(~"sending teardown_loop msg..");
    exit(iotask);
    exit_po.recv();
    debug!(~"after recv on exit_po.. exiting..");
}
