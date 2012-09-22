/*!
 * A task-based interface to the uv loop
 *
 * The I/O task runs in its own single-threaded scheduler.  By using the
 * `interact` function you can execute code in a uv callback.
 */

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

export IoTask;
export spawn_iotask;
export interact;
export exit;

use libc::c_void;
use ptr::addr_of;
use comm = core::comm;
use comm::{Port, Chan, listen};
use task::TaskBuilder;
use ll = uv_ll;

/// Used to abstract-away direct interaction with a libuv loop.
enum IoTask {
    IoTask_({
        async_handle: *ll::uv_async_t,
        op_chan: Chan<IoTaskMsg>
    })
}

fn spawn_iotask(+task: task::TaskBuilder) -> IoTask {

    do listen |iotask_ch| {

        do task.sched_mode(task::SingleThreaded).spawn {
            debug!("entering libuv task");
            run_loop(iotask_ch);
            debug!("libuv task exiting");
        };

        iotask_ch.recv()
    }
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
unsafe fn interact(iotask: IoTask,
                   +cb: fn~(*c_void)) {
    send_msg(iotask, Interaction(move cb));
}

/**
 * Shut down the I/O task
 *
 * Is used to signal to the loop that it should close the internally-held
 * async handle and do a sanity check to make sure that all other handles are
 * closed, causing a failure otherwise.
 */
fn exit(iotask: IoTask) unsafe {
    send_msg(iotask, TeardownLoop);
}


// INTERNAL API

enum IoTaskMsg {
    Interaction (fn~(*libc::c_void)),
    TeardownLoop
}

/// Run the loop and begin handling messages
fn run_loop(iotask_ch: Chan<IoTask>) unsafe {

    let loop_ptr = ll::loop_new();

    // set up the special async handle we'll use to allow multi-task
    // communication with this loop
    let async = ll::async_t();
    let async_handle = addr_of(async);

    // associate the async handle with the loop
    ll::async_init(loop_ptr, async_handle, wake_up_cb);

    // initialize our loop data and store it in the loop
    let data: IoTaskLoopData = {
        async_handle: async_handle,
        msg_po: Port()
    };
    ll::set_data_for_uv_handle(async_handle, addr_of(data));

    // Send out a handle through which folks can talk to us
    // while we dwell in the I/O loop
    let iotask = IoTask_({
        async_handle: async_handle,
        op_chan: data.msg_po.chan()
    });
    iotask_ch.send(iotask);

    log(debug, ~"about to run uv loop");
    // enter the loop... this blocks until the loop is done..
    ll::run(loop_ptr);
    log(debug, ~"uv loop ended");
    ll::loop_delete(loop_ptr);
}

// data that lives for the lifetime of the high-evel oo
type IoTaskLoopData = {
    async_handle: *ll::uv_async_t,
    msg_po: Port<IoTaskMsg>
};

fn send_msg(iotask: IoTask,
            +msg: IoTaskMsg) unsafe {
    iotask.op_chan.send(move msg);
    ll::async_send(iotask.async_handle);
}

/// Dispatch all pending messages
extern fn wake_up_cb(async_handle: *ll::uv_async_t,
                    status: int) unsafe {

    log(debug, fmt!("wake_up_cb extern.. handle: %? status: %?",
                     async_handle, status));

    let loop_ptr = ll::get_loop_for_uv_handle(async_handle);
    let data = ll::get_data_for_uv_handle(async_handle) as *IoTaskLoopData;
    let msg_po = (*data).msg_po;

    while msg_po.peek() {
        match msg_po.recv() {
          Interaction(cb) => cb(loop_ptr),
          TeardownLoop => begin_teardown(data)
        }
    }
}

fn begin_teardown(data: *IoTaskLoopData) unsafe {
    log(debug, ~"iotask begin_teardown() called, close async_handle");
    let async_handle = (*data).async_handle;
    ll::close(async_handle as *c_void, tear_down_close_cb);
}

extern fn tear_down_close_cb(handle: *ll::uv_async_t) unsafe {
    let loop_ptr = ll::get_loop_for_uv_handle(handle);
    let loop_refs = ll::loop_refcount(loop_ptr);
    log(debug, fmt!("tear_down_close_cb called, closing handle at %? refs %?",
                    handle, loop_refs));
    assert loop_refs == 1i32;
}

#[cfg(test)]
mod test {
    #[legacy_exports];
    extern fn async_close_cb(handle: *ll::uv_async_t) unsafe {
        log(debug, fmt!("async_close_cb handle %?", handle));
        let exit_ch = (*(ll::get_data_for_uv_handle(handle)
                        as *AhData)).exit_ch;
        core::comm::send(exit_ch, ());
    }
    extern fn async_handle_cb(handle: *ll::uv_async_t, status: libc::c_int)
        unsafe {
        log(debug, fmt!("async_handle_cb handle %? status %?",handle,status));
        ll::close(handle, async_close_cb);
    }
    type AhData = {
        iotask: IoTask,
        exit_ch: comm::Chan<()>
    };
    fn impl_uv_iotask_async(iotask: IoTask) unsafe {
        let async_handle = ll::async_t();
        let ah_ptr = ptr::addr_of(async_handle);
        let exit_po = core::comm::Port::<()>();
        let exit_ch = core::comm::Chan(exit_po);
        let ah_data = {
            iotask: iotask,
            exit_ch: exit_ch
        };
        let ah_data_ptr = ptr::addr_of(ah_data);
        do interact(iotask) |loop_ptr| unsafe {
            ll::async_init(loop_ptr, ah_ptr, async_handle_cb);
            ll::set_data_for_uv_handle(ah_ptr, ah_data_ptr as *libc::c_void);
            ll::async_send(ah_ptr);
        };
        core::comm::recv(exit_po);
    }

    // this fn documents the bear minimum neccesary to roll your own
    // high_level_loop
    unsafe fn spawn_test_loop(exit_ch: comm::Chan<()>) -> IoTask {
        let iotask_port = comm::Port::<IoTask>();
        let iotask_ch = comm::Chan(iotask_port);
        do task::spawn_sched(task::ManualThreads(1u)) {
            run_loop(iotask_ch);
            exit_ch.send(());
        };
        return core::comm::recv(iotask_port);
    }

    extern fn lifetime_handle_close(handle: *libc::c_void) unsafe {
        log(debug, fmt!("lifetime_handle_close ptr %?", handle));
    }

    extern fn lifetime_async_callback(handle: *libc::c_void,
                                     status: libc::c_int) {
        log(debug, fmt!("lifetime_handle_close ptr %? status %?",
                        handle, status));
    }

    #[test]
    fn test_uv_iotask_async() unsafe {
        let exit_po = core::comm::Port::<()>();
        let exit_ch = core::comm::Chan(exit_po);
        let iotask = spawn_test_loop(exit_ch);

        // using this handle to manage the lifetime of the high_level_loop,
        // as it will exit the first time one of the impl_uv_hl_async() is
        // cleaned up with no one ref'd handles on the loop (Which can happen
        // under race-condition type situations.. this ensures that the loop
        // lives until, at least, all of the impl_uv_hl_async() runs have been
        // called, at least.
        let work_exit_po = core::comm::Port::<()>();
        let work_exit_ch = core::comm::Chan(work_exit_po);
        for iter::repeat(7u) {
            do task::spawn_sched(task::ManualThreads(1u)) {
                impl_uv_iotask_async(iotask);
                core::comm::send(work_exit_ch, ());
            };
        };
        for iter::repeat(7u) {
            core::comm::recv(work_exit_po);
        };
        log(debug, ~"sending teardown_loop msg..");
        exit(iotask);
        core::comm::recv(exit_po);
        log(debug, ~"after recv on exit_po.. exiting..");
    }
}
