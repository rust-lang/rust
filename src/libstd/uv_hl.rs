#[doc = "
High-level bindings to work with the libuv library.

This module is geared towards library developers who want to
provide a high-level, abstracted interface to some set of
libuv functionality.
"];

import ll = uv_ll;

export high_level_loop;
export interact, prepare_loop;

#[doc = "
Used to abstract-away direct interaction with a libuv loop.

# Fields

* async_handle - a pointer to a uv_async_t struct used to 'poke'
the C uv loop to process any pending callbacks

* op_chan - a channel used to send function callbacks to be processed
by the C uv loop
"]
type high_level_loop = {
    async_handle: *ll::uv_async_t,
    op_chan: comm::chan<fn~(*libc::c_void)>
};

#[doc = "
Pass in a callback to be processed on the running libuv loop's thread

# Fields

* a_loop - a high_level_loop record that represents a channel of
communication with an active libuv loop running on a thread
somwhere in the current process

* cb - a function callback to be processed on the running loop's
thread. The only parameter is an opaque pointer to the running
uv_loop_t. You can use this pointer to initiate or continue any
operations against the loop
"]
unsafe fn interact(a_loop: high_level_loop,
                      -cb: fn~(*libc::c_void)) {
    comm::send(a_loop.op_chan, cb);
    ll::async_send(a_loop.async_handle);
}

#[doc = "
Prepares a clean, inactive uv_loop_t* to be used with any of the
functions in the `uv::hl` module.

Library developers can use this function to prepare a given
`uv_loop_t*`, whose lifecycle they manage, to be used, ran
and controlled with the tools in this module.

After this is ran against a loop, a library developer can run
the loop in its own thread and then use the returned
`high_level_loop` to interact with it.

# Fields

* loop_ptr - a pointer to a newly created `uv_loop_t*` with no
handles registered (this will interfere with the internal lifecycle
management this module provides). Ideally, this should be called
immediately after using `uv::ll::loop_new()`

# Returns

A `high_level_loop` record that can be used to interact with the
loop (after you use `uv::ll::run()` on the `uv_loop_t*`, of course
"]
unsafe fn prepare_loop(loop_ptr: *libc::c_void)
    -> high_level_loop {
    // will probably need to stake out a data record
    // here, as well, to keep whatever state we want to
    // use with the loop

    // move this into a malloc
    let async = ll::async_t();
    let async_ptr = ptr::addr_of(async);
    let op_port = comm::port::<fn~(*libc::c_void)>();
    let async_result = ll::async_init(loop_ptr,
                                      async_ptr,
                                      interact_poke);
    if (async_result != 0i32) {
        fail ll::get_last_err_info(loop_ptr);
    }
    // need to store the port and async_ptr in the top-level
    // of the provided loop ..
    ret { async_handle: async_ptr,
         op_chan: comm::chan::<fn~(*libc::c_void)>(op_port)
        };
}

// this will be invoked by a called to uv::hl::interact(), so
// we'll drain the port of pending callbacks, processing each
crust fn interact_poke(async_handle: *libc::c_void) {
    // nothing here, yet.
    log(debug, #fmt("interact_poke crust.. handle: %?",
                     async_handle));
}