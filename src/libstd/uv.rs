#[doc = "
Rust mappings to libuv (http://github.com/joyent/libuv)

This crate provides a low-level mapping to libuv, a library for
running an asynchronous event loop, with extensive IO operations.

This crate is seeing heavy work, currently, and the final API layout
should not be inferred from its current form.

The base module contains a set of safe functions for creating
an event loop that runs within a single task, but allows operations
against it from other tasks, but funneling it through a uv_async
request which reads from a port that users write to.

The 'll' module contains low-level, bare-metal mappings to the libuv
C-api. All functions within this module are marked unsafe and should
be used, primarily, for composing rust-idiomatic abstractions. In
lieu of satisfactory documention for the 'll' module, itself, libuv's
uv.h should be consulted.
"];

import map::hashmap;
export loop_new, loop_delete, run, close, run_in_bg;
export async_init, async_send;
export timer_init, timer_start, timer_stop;
export uv_ip4_addr, uv_ip6_addr;
export direct;

// these are processed solely in the
// process_operation() crust fn below
enum uv_operation {
    op_async_init([u8]),
    op_close(uv_handle, *libc::c_void),
    op_timer_init([u8]),
    op_timer_start([u8], *libc::c_void, u32, u32),
    op_timer_stop([u8], *libc::c_void, fn~(uv_handle)),
    op_teardown(*libc::c_void)
}

enum uv_handle {
    uv_async([u8], uv_loop),
    uv_timer([u8], uv_loop),
}

enum uv_msg {
    // requests from library users
    msg_run(comm::chan<bool>),
    msg_run_in_bg(),
    msg_async_init(fn~(uv_handle), fn~(uv_handle)),
    msg_async_send([u8]),
    msg_close(uv_handle, fn~()),
    msg_timer_init(fn~(uv_handle)),
    msg_timer_start([u8], u32, u32, fn~(uv_handle)),
    msg_timer_stop([u8], fn~(uv_handle)),

    // dispatches from libuv
    uv_async_init([u8], *libc::c_void),
    uv_async_send([u8]),
    uv_close([u8]),
    uv_timer_init([u8], *libc::c_void),
    uv_timer_call([u8]),
    uv_timer_stop([u8], fn~(uv_handle)),
    uv_end(),
    uv_teardown_check()
}

type uv_loop_data = {
    operation_port: comm::port<uv_operation>,
    rust_loop_chan: comm::chan<uv_msg>
};

enum uv_loop {
    uv_loop_new(comm::chan<uv_msg>, *libc::c_void)
}

// libuv struct mappings
type uv_ip4_addr = {
    ip: [u8],
    port: int
};
type uv_ip6_addr = uv_ip4_addr;

enum uv_handle_type {
    UNKNOWN_HANDLE = 0,
    UV_TCP,
    UV_UDP,
    UV_NAMED_PIPE,
    UV_TTY,
    UV_FILE,
    UV_TIMER,
    UV_PREPARE,
    UV_CHECK,
    UV_IDLE,
    UV_ASYNC,
    UV_ARES_TASK,
    UV_ARES_EVENT,
    UV_PROCESS,
    UV_FS_EVENT
}

type handle_type = libc::c_uint;

type uv_handle_fields = {
   loop_handle: *libc::c_void,
   type_: handle_type,
   close_cb: *u8,
   mut data: *libc::c_void,
};

// unix size: 8
type uv_err_t = {
    code: libc::c_int,
    sys_errno_: libc::c_int
};

// don't create one of these directly. instead,
// count on it appearing in libuv callbacks or embedded
// in other types as a pointer to be used in other
// operations (so mostly treat it as opaque, once you
// have it in this form..)
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "win32")]
type uv_stream_t = {
    fields: uv_handle_fields
};

// unix size: 272
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
type uv_tcp_t = {
    fields: uv_handle_fields,
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8, a10: *u8, a11: *u8,
    a12: *u8, a13: *u8, a14: *u8, a15: *u8,
    a16: *u8, a17: *u8, a18: *u8, a19: *u8,
    a20: *u8, a21: *u8, a22: *u8, a23: *u8,
    a24: *u8, a25: *u8, a26: *u8, a27: *u8,
    a28: *u8, a29: *u8
};
// win32 size: 240 (120)
#[cfg(target_os = "win32")]
type uv_tcp_t = {
    fields: uv_handle_fields,
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8, a10: *u8, a11: *u8,
    a12: *u8, a13: *u8, a14: *u8, a15: *u8,
    a16: *u8, a17: *u8, a18: *u8, a19: *u8,
    a20: *u8, a21: *u8, a22: *u8, a23: *u8,
    a24: *u8, a25: *u8
};
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
fn gen_stub_uv_tcp_t() -> uv_tcp_t {
    ret { fields: { loop_handle: ptr::null(), type_: 0u32,
                    close_cb: ptr::null(),
                    mut data: ptr::null() },
        a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8, a03: 0 as *u8,
        a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8, a07: 0 as *u8,
        a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8, a11: 0 as *u8,
        a12: 0 as *u8, a13: 0 as *u8, a14: 0 as *u8, a15: 0 as *u8,
        a16: 0 as *u8, a17: 0 as *u8, a18: 0 as *u8, a19: 0 as *u8,
        a20: 0 as *u8, a21: 0 as *u8, a22: 0 as *u8, a23: 0 as *u8,
        a24: 0 as *u8, a25: 0 as *u8, a26: 0 as *u8, a27: 0 as *u8,
        a28: 0 as *u8, a29: 0 as *u8
    };
}
#[cfg(target_os = "win32")]
fn gen_stub_uv_tcp_t() -> uv_tcp_t {
    ret { fields: { loop_handle: ptr::null(), type_: 0u32,
                    close_cb: ptr::null(),
                    mut data: ptr::null() },
        a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8, a03: 0 as *u8,
        a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8, a07: 0 as *u8,
        a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8, a11: 0 as *u8,
        a12: 0 as *u8, a13: 0 as *u8, a14: 0 as *u8, a15: 0 as *u8,
        a16: 0 as *u8, a17: 0 as *u8, a18: 0 as *u8, a19: 0 as *u8,
        a20: 0 as *u8, a21: 0 as *u8, a22: 0 as *u8, a23: 0 as *u8,
        a24: 0 as *u8, a25: 0 as *u8
    };
}

// unix size: 48
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
type uv_connect_t = {
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8
};
// win32 size: 88 (44)
#[cfg(target_os = "win32")]
type uv_connect_t = {
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8, a10: *u8
};
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
fn gen_stub_uv_connect_t() -> uv_connect_t {
    ret {
        a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8, a03: 0 as *u8,
        a04: 0 as *u8, a05: 0 as *u8
    };
}
#[cfg(target_os = "win32")]
fn gen_stub_uv_connect_t() -> uv_connect_t {
    ret {
        a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8, a03: 0 as *u8,
        a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8, a07: 0 as *u8,
        a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8
    };
}

// unix size: 16
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "win32")]
type uv_buf_t = {
    base: *u8,
    len: libc::size_t
};
// no gen stub method.. should create
// it via uv::direct::buf_init()

// unix size: 144
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
type uv_write_t = {
    fields: uv_handle_fields,
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8, a10: *u8, a11: *u8,
    a12: *u8, a13: *u8
};
// win32 size: 136 (68)
#[cfg(target_os = "win32")]
type uv_write_t = {
    fields: uv_handle_fields,
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8, a10: *u8, a11: *u8,
    a12: *u8
};
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
fn gen_stub_uv_write_t() -> uv_write_t {
    ret { fields: { loop_handle: ptr::null(), type_: 0u32,
                    close_cb: ptr::null(),
                    mut data: ptr::null() },
        a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8, a03: 0 as *u8,
        a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8, a07: 0 as *u8,
        a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8, a11: 0 as *u8,
        a12: 0 as *u8, a13: 0 as *u8
    };
}
#[cfg(target_os = "win32")]
fn gen_stub_uv_write_t() -> uv_write_t {
    ret { fields: { loop_handle: ptr::null(), type_: 0u32,
                    close_cb: ptr::null(),
                    mut data: ptr::null() },
        a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8, a03: 0 as *u8,
        a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8, a07: 0 as *u8,
        a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8, a11: 0 as *u8,
        a12: 0 as *u8
    };
}
// unix size: 120
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "win32")]
type uv_async_t = {
    fields: uv_handle_fields,
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8, a10: *u8
};
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "win32")]
fn gen_stub_uv_async_t() -> uv_async_t {
    ret { fields: { loop_handle: ptr::null(), type_: 0u32,
                    close_cb: ptr::null(),
                    mut data: ptr::null() },
        a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8, a03: 0 as *u8,
        a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8, a07: 0 as *u8,
        a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8
    };
}

// unix size: 16
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "win32")]
type sockaddr_in = {
    mut sin_family: u16,
    mut sin_port: u16,
    mut sin_addr: u32, // in_addr: this is an opaque, per-platform struct
    mut sin_zero: (u8, u8, u8, u8, u8, u8, u8, u8)
};

// unix size: 28 .. make due w/ 32
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "win32")]
type sockaddr_in6 = {
    a0: *u8, a1: *u8,
    a2: *u8, a3: *u8
};

#[nolink]
native mod rustrt {
    fn rust_uv_loop_new() -> *libc::c_void;
    fn rust_uv_loop_delete(lp: *libc::c_void);
    fn rust_uv_loop_set_data(
        lp: *libc::c_void,
        data: *uv_loop_data);
    fn rust_uv_bind_op_cb(lp: *libc::c_void, cb: *u8)
        -> *libc::c_void;
    fn rust_uv_stop_op_cb(handle: *libc::c_void);
    fn rust_uv_run(loop_handle: *libc::c_void);
    fn rust_uv_close(handle: *libc::c_void, cb: *u8);
    fn rust_uv_hilvl_close(handle: *libc::c_void, cb: *u8);
    fn rust_uv_hilvl_close_async(handle: *libc::c_void);
    fn rust_uv_hilvl_close_timer(handle: *libc::c_void);
    fn rust_uv_async_send(handle: *uv_async_t);
    fn rust_uv_async_init(loop_handle: *libc::c_void,
                          async_handle: *uv_async_t,
                          cb: *u8) -> libc::c_int;
    fn rust_uv_hilvl_async_init(
        loop_handle: *libc::c_void,
        cb: *u8,
        id: *u8) -> *libc::c_void;
    fn rust_uv_timer_init(
        loop_handle: *libc::c_void,
        cb: *u8,
        id: *u8) -> *libc::c_void;
    fn rust_uv_timer_start(
        timer_handle: *libc::c_void,
        timeout: libc::c_uint,
        repeat: libc::c_uint);
    fn rust_uv_timer_stop(handle: *libc::c_void);
    fn rust_uv_free(ptr: *libc::c_void);
    fn rust_uv_tcp_init(
        loop_handle: *libc::c_void,
        handle_ptr: *uv_tcp_t) -> libc::c_int;
    fn rust_uv_buf_init(base: *u8, len: libc::size_t)
        -> uv_buf_t;
    fn rust_uv_last_error(loop_handle: *libc::c_void) -> uv_err_t;
    // FIXME ref #2064
    fn rust_uv_strerror(err: *uv_err_t) -> *libc::c_char;
    // FIXME ref #2064
    fn rust_uv_err_name(err: *uv_err_t) -> *libc::c_char;
    fn rust_uv_ip4_test_verify_port_val(++addr: sockaddr_in,
                                        expected: libc::c_uint)
        -> bool;
    fn rust_uv_ip4_addr(ip: *u8, port: libc::c_int)
        -> sockaddr_in;
    // FIXME ref #2064
    fn rust_uv_tcp_connect(connect_ptr: *uv_connect_t,
                           tcp_handle_ptr: *uv_tcp_t,
                           ++after_cb: *u8,
                           ++addr: *sockaddr_in) -> libc::c_int;
    // FIXME ref 2064
    fn rust_uv_tcp_bind(tcp_server: *uv_tcp_t,
                        ++addr: *sockaddr_in) -> libc::c_int;
    fn rust_uv_listen(stream: *libc::c_void, backlog: libc::c_int,
                      cb: *u8) -> libc::c_int;
    fn rust_uv_accept(server: *libc::c_void, client: *libc::c_void)
        -> libc::c_int;
    fn rust_uv_write(req: *libc::c_void, stream: *libc::c_void,
             ++buf_in: *uv_buf_t, buf_cnt: libc::c_int,
             cb: *u8) -> libc::c_int;
    fn rust_uv_read_start(stream: *libc::c_void, on_alloc: *u8,
                          on_read: *u8) -> libc::c_int;
    fn rust_uv_read_stop(stream: *libc::c_void) -> libc::c_int;
    fn rust_uv_malloc_buf_base_of(sug_size: libc::size_t) -> *u8;
    fn rust_uv_free_base_of_buf(++buf: uv_buf_t);

    // sizeof testing helpers
    fn rust_uv_helper_uv_tcp_t_size() -> libc::c_uint;
    fn rust_uv_helper_uv_connect_t_size() -> libc::c_uint;
    fn rust_uv_helper_uv_buf_t_size() -> libc::c_uint;
    fn rust_uv_helper_uv_write_t_size() -> libc::c_uint;
    fn rust_uv_helper_uv_err_t_size() -> libc::c_uint;
    fn rust_uv_helper_sockaddr_in_size() -> libc::c_uint;
    fn rust_uv_helper_uv_async_t_size() -> libc::c_uint;

    // data accessors for rust-mapped uv structs
    fn rust_uv_get_stream_handle_from_connect_req(
        connect_req: *uv_connect_t)
        -> *uv_stream_t;
    fn rust_uv_get_stream_handle_from_write_req(
        write_req: *uv_write_t)
        -> *uv_stream_t;
    fn rust_uv_get_loop_for_uv_handle(handle: *libc::c_void)
        -> *libc::c_void;
    fn rust_uv_get_data_for_uv_handle(handle: *libc::c_void)
        -> *libc::c_void;
    fn rust_uv_set_data_for_uv_handle(handle: *libc::c_void,
                                      data: *libc::c_void);
    fn rust_uv_get_data_for_req(req: *libc::c_void) -> *libc::c_void;
    fn rust_uv_set_data_for_req(req: *libc::c_void,
                                data: *libc::c_void);
    fn rust_uv_get_base_from_buf(++buf: uv_buf_t) -> *u8;
    fn rust_uv_get_len_from_buf(++buf: uv_buf_t) -> libc::size_t;
}

// this module is structured around functions that directly
// expose libuv functionality and data structures. for use
// in higher level mappings
mod direct {
    unsafe fn loop_new() -> *libc::c_void {
        ret rustrt::rust_uv_loop_new();
    }

    unsafe fn loop_delete(loop_handle: *libc::c_void) {
        rustrt::rust_uv_loop_delete(loop_handle);
    }

    unsafe fn run(loop_handle: *libc::c_void) {
        rustrt::rust_uv_run(loop_handle);
    }

    unsafe fn close(handle: *libc::c_void, cb: *u8) {
        rustrt::rust_uv_close(handle, cb);
    }

    unsafe fn tcp_init(loop_handle: *libc::c_void, handle: *uv_tcp_t)
        -> libc::c_int {
        ret rustrt::rust_uv_tcp_init(loop_handle, handle);
    }
    // FIXME ref #2064
    unsafe fn tcp_connect(connect_ptr: *uv_connect_t,
                          tcp_handle_ptr: *uv_tcp_t,
                          addr_ptr: *sockaddr_in,
                          ++after_connect_cb: *u8)
    -> libc::c_int {
        let address = *addr_ptr;
        io::println(#fmt("b4 native tcp_connect--addr port: %u cb: %u",
                         address.sin_port as uint, after_connect_cb as uint));
        ret rustrt::rust_uv_tcp_connect(connect_ptr, tcp_handle_ptr,
                                    after_connect_cb, addr_ptr);
    }
    // FIXME ref #2064
    unsafe fn tcp_bind(tcp_server_ptr: *uv_tcp_t,
                       addr_ptr: *sockaddr_in) -> libc::c_int {
        ret rustrt::rust_uv_tcp_bind(tcp_server_ptr,
                                     addr_ptr);
    }

    unsafe fn listen(stream: *libc::c_void, backlog: libc::c_int,
                     cb: *u8) -> libc::c_int {
        ret rustrt::rust_uv_listen(stream, backlog, cb);
    }

    unsafe fn accept(server: *libc::c_void, client: *libc::c_void)
        -> libc::c_int {
        ret rustrt::rust_uv_accept(server, client);
    }

    unsafe fn write(req: *libc::c_void, stream: *libc::c_void,
             buf_in: *[uv_buf_t], cb: *u8) -> libc::c_int {
        let buf_ptr = vec::unsafe::to_ptr(*buf_in);
        let buf_cnt = vec::len(*buf_in) as i32;
        ret rustrt::rust_uv_write(req, stream, buf_ptr, buf_cnt, cb);
    }
    unsafe fn read_start(stream: *uv_stream_t, on_alloc: *u8,
                         on_read: *u8) -> libc::c_int {
        ret rustrt::rust_uv_read_start(stream as *libc::c_void,
                                       on_alloc, on_read);
    }

    unsafe fn read_stop(stream: *uv_stream_t) -> libc::c_int {
        ret rustrt::rust_uv_read_stop(stream as *libc::c_void);
    }

    unsafe fn last_error(loop_handle: *libc::c_void) -> uv_err_t {
        ret rustrt::rust_uv_last_error(loop_handle);
    }

    unsafe fn strerror(err: *uv_err_t) -> *libc::c_char {
        ret rustrt::rust_uv_strerror(err);
    }
    unsafe fn err_name(err: *uv_err_t) -> *libc::c_char {
        ret rustrt::rust_uv_err_name(err);
    }

    unsafe fn async_init(loop_handle: *libc::c_void,
                         async_handle: *uv_async_t,
                         cb: *u8) -> libc::c_int {
        ret rustrt::rust_uv_async_init(loop_handle,
                                       async_handle,
                                       cb);
    }

    unsafe fn async_send(async_handle: *uv_async_t) {
        ret rustrt::rust_uv_async_send(async_handle);
    }

    // libuv struct initializers
    unsafe fn tcp_t() -> uv_tcp_t {
        ret gen_stub_uv_tcp_t();
    }
    unsafe fn connect_t() -> uv_connect_t {
        ret gen_stub_uv_connect_t();
    }
    unsafe fn write_t() -> uv_write_t {
        ret gen_stub_uv_write_t();
    }
    unsafe fn async_t() -> uv_async_t {
        ret gen_stub_uv_async_t();
    }
    unsafe fn get_loop_for_uv_handle(handle: *libc::c_void)
        -> *libc::c_void {
        ret rustrt::rust_uv_get_loop_for_uv_handle(handle);
    }
    unsafe fn get_stream_handle_from_connect_req(connect: *uv_connect_t)
        -> *uv_stream_t {
        ret rustrt::rust_uv_get_stream_handle_from_connect_req(
            connect);
    }
    unsafe fn get_stream_handle_from_write_req(
        write_req: *uv_write_t)
        -> *uv_stream_t {
        ret rustrt::rust_uv_get_stream_handle_from_write_req(
            write_req);
    }

    unsafe fn get_data_for_uv_handle(handle: *libc::c_void) -> *libc::c_void {
        ret rustrt::rust_uv_get_data_for_uv_handle(handle);
    }
    unsafe fn set_data_for_uv_handle(handle: *libc::c_void,
                        data: *libc::c_void) {
        rustrt::rust_uv_set_data_for_uv_handle(handle, data);
    }
    unsafe fn get_data_for_req(req: *libc::c_void) -> *libc::c_void {
        ret rustrt::rust_uv_get_data_for_req(req);
    }
    unsafe fn set_data_for_req(req: *libc::c_void,
                        data: *libc::c_void) {
        rustrt::rust_uv_set_data_for_req(req, data);
    }
    unsafe fn get_base_from_buf(buf: uv_buf_t) -> *u8 {
        ret rustrt::rust_uv_get_base_from_buf(buf);
    }
    unsafe fn get_len_from_buf(buf: uv_buf_t) -> libc::size_t {
        ret rustrt::rust_uv_get_len_from_buf(buf);
    }
    unsafe fn buf_init(input: *u8, len: uint) -> uv_buf_t {
        ret rustrt::rust_uv_buf_init(input, len);
    }
    unsafe fn ip4_addr(ip: str, port: int)
    -> sockaddr_in {
        let mut addr_vec = str::bytes(ip);
        addr_vec += [0u8]; // add null terminator
        let addr_vec_ptr = vec::unsafe::to_ptr(addr_vec);
        let ip_back = str::from_bytes(addr_vec);
        io::println(#fmt("vec val: '%s' length: %u",
                         ip_back, vec::len(addr_vec)));
        ret rustrt::rust_uv_ip4_addr(addr_vec_ptr,
                                     port as libc::c_int);
    }
    unsafe fn malloc_buf_base_of(suggested_size: libc::size_t)
        -> *u8 {
        ret rustrt::rust_uv_malloc_buf_base_of(suggested_size);
    }
    unsafe fn free_base_of_buf(buf: uv_buf_t) {
        rustrt::rust_uv_free_base_of_buf(buf);
    }

    unsafe fn print_last_err_info(uv_loop: *libc::c_void) {
        let err = direct::last_error(uv_loop);
        let err_ptr = ptr::addr_of(err);
        let err_name = str::unsafe::from_c_str(direct::err_name(err_ptr));
        let err_msg = str::unsafe::from_c_str(direct::strerror(err_ptr));
        io::println(#fmt("LIBUV ERROR: name: %s msg: %s",
                        err_name, err_msg));
    }
}

// public functions
fn loop_new() -> uv_loop unsafe {
    let ret_recv_port: comm::port<uv_loop> =
        comm::port();
    let ret_recv_chan: comm::chan<uv_loop> =
        comm::chan(ret_recv_port);

    task::spawn_sched(task::manual_threads(1u)) {||
        // our beloved uv_loop_t ptr
        let loop_handle = rustrt::
            rust_uv_loop_new();

        // this port/chan pair are used to send messages to
        // libuv. libuv processes any pending messages on the
        // port (via crust) after receiving an async "wakeup"
        // on a special uv_async_t handle created below
        let operation_port = comm::port::<uv_operation>();
        let operation_chan = comm::chan::<uv_operation>(
            operation_port);

        // this port/chan pair as used in the while() loop
        // below. It takes dispatches, originating from libuv
        // callbacks, to invoke handles registered by the
        // user
        let rust_loop_port = comm::port::<uv_msg>();
        let rust_loop_chan =
            comm::chan::<uv_msg>(rust_loop_port);
        // let the task-spawner return
        let user_uv_loop = uv_loop_new(rust_loop_chan, loop_handle);
        comm::send(ret_recv_chan, copy(user_uv_loop));

        // create our "special" async handle that will
        // allow all operations against libuv to be
        // "buffered" in the operation_port, for processing
        // from the thread that libuv runs on
        let loop_data: uv_loop_data = {
            operation_port: operation_port,
            rust_loop_chan: rust_loop_chan
        };
        rustrt::rust_uv_loop_set_data(
            loop_handle,
            ptr::addr_of(loop_data)); // pass an opaque C-ptr
                                      // to libuv, this will be
                                      // in the process_operation
                                      // crust fn
        let op_handle = rustrt::rust_uv_bind_op_cb(
            loop_handle,
            process_operation);

        // all state goes here
        let handles: map::hashmap<[u8], *libc::c_void> =
            map::bytes_hash();
        let id_to_handle: map::hashmap<[u8], uv_handle> =
            map::bytes_hash();
        let after_cbs: map::hashmap<[u8], fn~(uv_handle)> =
            map::bytes_hash();
        let close_callbacks: map::hashmap<[u8], fn~()> =
            map::bytes_hash();
        let async_cbs: map::hashmap<[u8], fn~(uv_handle)> =
            map::bytes_hash();
        let timer_cbs: map::hashmap<[u8], fn~(uv_handle)> =
            map::bytes_hash();

        // the main loop that this task blocks on.
        // should have the same lifetime as the C libuv
        // event loop.
        let mut keep_going = true;
        while keep_going {
            alt comm::recv(rust_loop_port) {
              msg_run(end_chan) {
                // start the libuv event loop
                // we'll also do a uv_async_send with
                // the operation handle to have the
                // loop process any pending operations
                // once its up and running
                task::spawn_sched(task::manual_threads(1u)) {||
                    // make sure we didn't start the loop
                    // without the user registering handles
                    comm::send(rust_loop_chan, uv_teardown_check);
                    // this call blocks
                    rustrt::rust_uv_run(loop_handle);
                    // when we're done, msg the
                    // end chan
                    comm::send(end_chan, true);
                    comm::send(rust_loop_chan, uv_end);
                };
              }

              msg_run_in_bg {
                task::spawn_sched(task::manual_threads(1u)) {||
                    // see note above
                    comm::send(rust_loop_chan, uv_teardown_check);
                    // this call blocks
                    rustrt::rust_uv_run(loop_handle);
                };
              }

              msg_close(handle, cb) {
                let id = get_id_from_handle(handle);
                close_callbacks.insert(id, cb);
                let handle_ptr = handles.get(id);
                let op = op_close(handle, handle_ptr);

                pass_to_libuv(op_handle, operation_chan, op);
              }
              uv_close(id) {
                handles.remove(id);
                let handle = id_to_handle.get(id);
                id_to_handle.remove(id);
                alt handle {
                  uv_async(id, _) {
                    async_cbs.remove(id);
                  }
                  uv_timer(id, _) {
                    timer_cbs.remove(id);
                  }
                  _ {
                    fail "unknown form of uv_handle encountered "
                        + "in uv_close handler";
                  }
                }
                let cb = close_callbacks.get(id);
                close_callbacks.remove(id);
                task::spawn {||
                    cb();
                };
                // ask the rust loop to check and see if there
                // are no more user-registered handles
                comm::send(rust_loop_chan, uv_teardown_check);
              }

              msg_async_init(callback, after_cb) {
                // create a new async handle
                // with the id as the handle's
                // data and save the callback for
                // invocation on msg_async_send
                let id = gen_handle_id();
                handles.insert(id, ptr::null());
                async_cbs.insert(id, callback);
                after_cbs.insert(id, after_cb);
                let op = op_async_init(id);
                pass_to_libuv(op_handle, operation_chan, op);
              }
              uv_async_init(id, async_handle) {
                // libuv created a handle, which is
                // passed back to us. save it and
                // then invoke the supplied callback
                // for after completion
                handles.insert(id, async_handle);
                let after_cb = after_cbs.get(id);
                after_cbs.remove(id);
                let async = uv_async(id, user_uv_loop);
                id_to_handle.insert(id, copy(async));
                task::spawn {||
                    after_cb(async);
                };
              }

              msg_async_send(id) {
                let async_handle = handles.get(id);
                do_send(async_handle);
              }
              uv_async_send(id) {
                let async_cb = async_cbs.get(id);
                task::spawn {||
                    let the_loop = user_uv_loop;
                    async_cb(uv_async(id, the_loop));
                };
              }

              msg_timer_init(after_cb) {
                let id = gen_handle_id();
                handles.insert(id, ptr::null());
                after_cbs.insert(id, after_cb);
                let op = op_timer_init(id);
                pass_to_libuv(op_handle, operation_chan, op);
              }
              uv_timer_init(id, handle) {
                handles.insert(id, handle);
                let after_cb = after_cbs.get(id);
                after_cbs.remove(id);
                let new_timer = uv_timer(id, user_uv_loop);
                id_to_handle.insert(id, copy(new_timer));
                task::spawn {||
                    after_cb(new_timer);
                };
              }

              uv_timer_call(id) {
                let cb = timer_cbs.get(id);
                let the_timer = id_to_handle.get(id);
                task::spawn {||
                    cb(the_timer);
                };
              }

              msg_timer_start(id, timeout, repeat, timer_call_cb) {
                timer_cbs.insert(id, timer_call_cb);
                let handle = handles.get(id);
                let op = op_timer_start(id, handle, timeout,
                                        repeat);
                pass_to_libuv(op_handle, operation_chan, op);
              }

              msg_timer_stop(id, after_cb) {
                let handle = handles.get(id);
                let op = op_timer_stop(id, handle, after_cb);
                pass_to_libuv(op_handle, operation_chan, op);
              }
              uv_timer_stop(id, after_cb) {
                let the_timer = id_to_handle.get(id);
                after_cb(the_timer);
              }

              uv_teardown_check() {
                // here we're checking if there are no user-registered
                // handles (and the loop is running), if this is the
                // case, then we need to unregister the op_handle via
                // a uv_close() call, thus allowing libuv to return
                // on its own.
                if (handles.size() == 0u) {
                    let op = op_teardown(op_handle);
                    pass_to_libuv(op_handle, operation_chan, op);
                }
              }

              uv_end() {
                keep_going = false;
              }

              _ { fail "unknown form of uv_msg received"; }
            }
        }
    };
    ret comm::recv(ret_recv_port);
}

fn loop_delete(lp: uv_loop) {
    let loop_ptr = get_loop_ptr_from_uv_loop(lp);
    rustrt::rust_uv_loop_delete(loop_ptr);
}

fn run(lp: uv_loop) {
    let end_port = comm::port::<bool>();
    let end_chan = comm::chan::<bool>(end_port);
    let loop_chan = get_loop_chan_from_uv_loop(lp);
    comm::send(loop_chan, msg_run(end_chan));
    comm::recv(end_port);
}

fn run_in_bg(lp: uv_loop) {
    let loop_chan = get_loop_chan_from_uv_loop(lp);
    comm::send(loop_chan, msg_run_in_bg);
}

fn async_init (
    lp: uv_loop,
    async_cb: fn~(uv_handle),
    after_cb: fn~(uv_handle)) {
    let msg = msg_async_init(async_cb, after_cb);
    let loop_chan = get_loop_chan_from_uv_loop(lp);
    comm::send(loop_chan, msg);
}

fn async_send(async: uv_handle) {
    alt async {
      uv_async(id, lp) {
        let loop_chan = get_loop_chan_from_uv_loop(lp);
        comm::send(loop_chan, msg_async_send(id));
      }
      _ {
        fail "attempting to call async_send() with a" +
            " uv_async uv_handle";
      }
    }
}

fn close(h: uv_handle, cb: fn~()) {
    let loop_chan = get_loop_chan_from_handle(h);
    comm::send(loop_chan, msg_close(h, cb));
}

fn timer_init(lp: uv_loop, after_cb: fn~(uv_handle)) {
    let msg = msg_timer_init(after_cb);
    let loop_chan = get_loop_chan_from_uv_loop(lp);
    comm::send(loop_chan, msg);
}

fn timer_start(the_timer: uv_handle, timeout: u32, repeat:u32,
               timer_cb: fn~(uv_handle)) {
    alt the_timer {
      uv_timer(id, lp) {
        let msg = msg_timer_start(id, timeout, repeat, timer_cb);
        let loop_chan = get_loop_chan_from_uv_loop(lp);
        comm::send(loop_chan, msg);
      }
      _ {
        fail "can only pass a uv_timer form of uv_handle to "+
             " uv::timer_start()";
      }
    }
}

fn timer_stop(the_timer: uv_handle, after_cb: fn~(uv_handle)) {
    alt the_timer {
      uv_timer(id, lp) {
        let loop_chan = get_loop_chan_from_uv_loop(lp);
        let msg = msg_timer_stop(id, after_cb);
        comm::send(loop_chan, msg);
      }
      _ {
        fail "only uv_timer form is allowed in calls to "+
             " uv::timer_stop()";
      }
    }
}

// internal functions
fn pass_to_libuv(
        op_handle: *libc::c_void,
        operation_chan: comm::chan<uv_operation>,
        op: uv_operation) unsafe {
    comm::send(operation_chan, copy(op));
    do_send(op_handle);
}
fn do_send(h: *libc::c_void) {
    rustrt::rust_uv_async_send(h as *uv_async_t);
}
fn gen_handle_id() -> [u8] {
    ret rand::rng().gen_bytes(16u);
}
fn get_handle_id_from(buf: *u8) -> [u8] unsafe {
    ret vec::unsafe::from_buf(buf, 16u);
}

fn get_loop_chan_from_data(data: *uv_loop_data)
        -> comm::chan<uv_msg> unsafe {
    ret (*data).rust_loop_chan;
}

fn get_loop_chan_from_handle(handle: uv_handle)
    -> comm::chan<uv_msg> {
    alt handle {
      uv_async(id,lp) | uv_timer(id,lp) {
        let loop_chan = get_loop_chan_from_uv_loop(lp);
        ret loop_chan;
      }
      _ {
        fail "unknown form of uv_handle for get_loop_chan_from "
             + " handle";
      }
    }
}

fn get_loop_ptr_from_uv_loop(lp: uv_loop) -> *libc::c_void {
    alt lp {
      uv_loop_new(loop_chan, loop_ptr) {
        ret loop_ptr;
      }
    }
}
fn get_loop_chan_from_uv_loop(lp: uv_loop) -> comm::chan<uv_msg> {
    alt lp {
      uv_loop_new(loop_chan, loop_ptr) {
        ret loop_chan;
      }
    }
}

fn get_id_from_handle(handle: uv_handle) -> [u8] {
    alt handle {
      uv_async(id,lp) | uv_timer(id,lp) {
        ret id;
      }
      _ {
        fail "unknown form of uv_handle for get_id_from handle";
      }
    }
}

// crust
crust fn process_operation(
        lp: *libc::c_void,
        data: *uv_loop_data) unsafe {
    let op_port = (*data).operation_port;
    let loop_chan = get_loop_chan_from_data(data);
    let mut op_pending = comm::peek(op_port);
    while(op_pending) {
        alt comm::recv(op_port) {
          op_async_init(id) {
            let id_ptr = vec::unsafe::to_ptr(id);
            let async_handle = rustrt::rust_uv_hilvl_async_init(
                lp,
                process_async_send,
                id_ptr);
            comm::send(loop_chan, uv_async_init(
                id,
                async_handle));
          }
          op_close(handle, handle_ptr) {
            handle_op_close(handle, handle_ptr);
          }
          op_timer_init(id) {
            let id_ptr = vec::unsafe::to_ptr(id);
            let timer_handle = rustrt::rust_uv_timer_init(
                lp,
                process_timer_call,
                id_ptr);
            comm::send(loop_chan, uv_timer_init(
                id,
                timer_handle));
          }
          op_timer_start(id, handle, timeout, repeat) {
            rustrt::rust_uv_timer_start(handle, timeout,
                                              repeat);
          }
          op_timer_stop(id, handle, after_cb) {
            rustrt::rust_uv_timer_stop(handle);
            comm::send(loop_chan, uv_timer_stop(id, after_cb));
          }
          op_teardown(op_handle) {
            // this is the last msg that'll be processed by
            // this fn, in the current lifetime of the handle's
            // uv_loop_t
            rustrt::rust_uv_stop_op_cb(op_handle);
          }
          _ { fail "unknown form of uv_operation received"; }
        }
        op_pending = comm::peek(op_port);
    }
}

fn handle_op_close(handle: uv_handle, handle_ptr: *libc::c_void) {
    // it's just like im doing C
    alt handle {
      uv_async(id, lp) {
        let cb = process_close_async;
        rustrt::rust_uv_hilvl_close(
            handle_ptr, cb);
      }
      uv_timer(id, lp) {
        let cb = process_close_timer;
        rustrt::rust_uv_hilvl_close(
            handle_ptr, cb);
      }
      _ {
        fail "unknown form of uv_handle encountered " +
            "in process_operation/op_close";
      }
    }
}

crust fn process_async_send(id_buf: *u8, data: *uv_loop_data)
    unsafe {
    let handle_id = get_handle_id_from(id_buf);
    let loop_chan = get_loop_chan_from_data(data);
    comm::send(loop_chan, uv_async_send(handle_id));
}

crust fn process_timer_call(id_buf: *u8, data: *uv_loop_data)
    unsafe {
    let handle_id = get_handle_id_from(id_buf);
    let loop_chan = get_loop_chan_from_data(data);
    comm::send(loop_chan, uv_timer_call(handle_id));
}

fn process_close_common(id: [u8], data: *uv_loop_data)
    unsafe {
    // notify the rust loop that their handle is closed, then
    // the caller will invoke a per-handle-type c++ func to
    // free allocated memory
    let loop_chan = get_loop_chan_from_data(data);
    comm::send(loop_chan, uv_close(id));
}

crust fn process_close_async(
    id_buf: *u8,
    handle_ptr: *libc::c_void,
    data: *uv_loop_data)
    unsafe {
    let id = get_handle_id_from(id_buf);
    rustrt::rust_uv_hilvl_close_async(handle_ptr);
    // at this point, the handle and its data has been
    // released. notify the rust loop to remove the
    // handle and its data and call the user-supplied
    // close cb
    process_close_common(id, data);
}

crust fn process_close_timer(
    id_buf: *u8,
    handle_ptr: *libc::c_void,
    data: *uv_loop_data)
    unsafe {
    let id = get_handle_id_from(id_buf);
    rustrt::rust_uv_hilvl_close_timer(handle_ptr);
    process_close_common(id, data);
}


#[test]
fn test_uv_new_loop_no_handles() {
    let test_loop = uv::loop_new();
    uv::run(test_loop); // this should return immediately
                    // since there aren't any handles..
    uv::loop_delete(test_loop);
}

#[test]
#[ignore(cfg(target_os = "freebsd"))]
fn test_uv_simple_async() {
    let test_loop = uv::loop_new();
    let exit_port = comm::port::<bool>();
    let exit_chan = comm::chan::<bool>(exit_port);
    uv::async_init(test_loop, {|new_async|
        uv::close(new_async) {||
            comm::send(exit_chan, true);
        };
    }, {|new_async|
        uv::async_send(new_async);
    });
    uv::run(test_loop);
    let result = comm::recv(exit_port);
    assert result;
    uv::loop_delete(test_loop);
}

#[test]
#[ignore(cfg(target_os = "freebsd"))]
fn test_uv_timer() {
    let test_loop = uv::loop_new();
    let exit_port = comm::port::<bool>();
    let exit_chan = comm::chan::<bool>(exit_port);
    uv::timer_init(test_loop) {|new_timer|
        uv::timer_start(new_timer, 1u32, 0u32) {|started_timer|
            uv::timer_stop(started_timer) {|stopped_timer|
                uv::close(stopped_timer) {||
                    comm::send(exit_chan, true);
                };
            };
        };
    };
    uv::run(test_loop);
    assert comm::recv(exit_port);
    uv::loop_delete(test_loop);
}

enum tcp_read_data {
    tcp_read_eof,
    tcp_read_more([u8]),
    tcp_read_error
}

type request_wrapper = {
    write_req: *uv_write_t,
    req_buf: *[uv_buf_t],
    read_chan: *comm::chan<str>
};

crust fn after_close_cb(handle: *libc::c_void) {
    io::println("after uv_close!");
}

crust fn on_alloc_cb(handle: *libc::c_void,
                     suggested_size: libc::size_t) -> uv_buf_t
    unsafe {
    io::println("on_alloc_cb!");
    let char_ptr = direct::malloc_buf_base_of(suggested_size);
    ret direct::buf_init(char_ptr, suggested_size);
}

crust fn on_read_cb(stream: *uv_stream_t, nread: libc::ssize_t,
                    ++buf: uv_buf_t) unsafe {
    io::println(#fmt("CLIENT entering on_read_cb nred: %d", nread));
    if (nread > 0) {
        // we have data
        io::println(#fmt("CLIENT read: data! nread: %d", nread));
        direct::read_stop(stream);
        let client_data = direct::
            get_data_for_uv_handle(stream as *libc::c_void)
              as *request_wrapper;
        let buf_base = direct::get_base_from_buf(buf);
        let buf_len = direct::get_len_from_buf(buf);
        let bytes = vec::unsafe::from_buf(buf_base, buf_len);
        let read_chan = *((*client_data).read_chan);
        let msg_from_server = str::from_bytes(bytes);
        comm::send(read_chan, msg_from_server);
        direct::close(stream as *libc::c_void, after_close_cb)
    }
    else if (nread == -1) {
        // err .. possibly EOF
        io::println("read: eof!");
    }
    else {
        // nread == 0 .. do nothing, just free buf as below
        io::println("read: do nothing!");
    }
    // when we're done
    direct::free_base_of_buf(buf);
    io::println("CLIENT exiting on_read_cb");
}

crust fn on_write_complete_cb(write_req: *uv_write_t,
                              status: libc::c_int) unsafe {
    io::println(#fmt("CLIENT beginning on_write_complete_cb status: %d",
                     status as int));
    let stream = direct::get_stream_handle_from_write_req(write_req);
    io::println(#fmt("CLIENT on_write_complete_cb: tcp:%d write_handle:%d",
        stream as int, write_req as int));
    let result = direct::read_start(stream, on_alloc_cb, on_read_cb);
    io::println(#fmt("CLIENT ending on_write_complete_cb .. status: %d",
                     result as int));
}

crust fn on_connect_cb(connect_req_ptr: *uv_connect_t,
                             status: libc::c_int) unsafe {
    io::println(#fmt("beginning on_connect_cb .. status: %d",
                     status as int));
    let stream =
        direct::get_stream_handle_from_connect_req(connect_req_ptr);
    if (status == 0i32) {
        io::println("on_connect_cb: in status=0 if..");
        let client_data = direct::get_data_for_req(
            connect_req_ptr as *libc::c_void)
            as *request_wrapper;
        let write_handle = (*client_data).write_req as *libc::c_void;
        io::println(#fmt("on_connect_cb: tcp stream: %d write_handle addr %d",
                        stream as int, write_handle as int));
        let write_result = direct::write(write_handle,
                          stream as *libc::c_void,
                          (*client_data).req_buf,
                          on_write_complete_cb);
        io::println(#fmt("on_connect_cb: direct::write() status: %d",
                         write_result as int));
    }
    else {
        let test_loop = direct::get_loop_for_uv_handle(stream as *libc::c_void);
        direct::print_last_err_info(test_loop);
        assert false;
    }
    io::println("finishing on_connect_cb");
}

fn impl_uv_tcp_request(ip: str, port: int, req_str: str,
                      client_chan: *comm::chan<str>) unsafe {
    let test_loop = direct::loop_new();
    let tcp_handle = direct::tcp_t();
    let tcp_handle_ptr = ptr::addr_of(tcp_handle);
    let connect_handle = direct::connect_t();
    let connect_req_ptr = ptr::addr_of(connect_handle);

    // this is the persistent payload of data that we
    // need to pass around to get this example to work.
    // In C, this would be a malloc'd or stack-allocated
    // struct that we'd cast to a void* and store as the
    // data field in our uv_connect_t struct
    let req_str_bytes = str::bytes(req_str);
    let req_msg_ptr: *u8 = vec::unsafe::to_ptr(req_str_bytes);
    let req_msg = [
        direct::buf_init(req_msg_ptr, vec::len(req_str_bytes))
    ];
    // this is the enclosing record, we'll pass a ptr to
    // this to C..
    let write_handle = direct::write_t();
    let write_handle_ptr = ptr::addr_of(write_handle);
    io::println(#fmt("tcp req: tcp stream: %d write_handle: %d",
                     tcp_handle_ptr as int,
                     write_handle_ptr as int));
    let client_data = { writer_handle: write_handle_ptr,
                req_buf: ptr::addr_of(req_msg),
                read_chan: client_chan };

    let tcp_init_result = direct::tcp_init(
        test_loop as *libc::c_void, tcp_handle_ptr);
    if (tcp_init_result == 0i32) {
        io::println("sucessful tcp_init_result");

        io::println("building addr...");
        let addr = direct::ip4_addr(ip, port);
        // FIXME ref #2064
        let addr_ptr = ptr::addr_of(addr);
        io::println(#fmt("after build addr in rust. port: %u",
                         addr.sin_port as uint));

        // this should set up the connection request..
        io::println(#fmt("before calling tcp_connect .. connect cb ptr: %u ",
                        on_connect_cb as uint));
        let tcp_connect_result = direct::tcp_connect(
            connect_req_ptr, tcp_handle_ptr,
            addr_ptr, on_connect_cb);
        if (tcp_connect_result == 0i32) {
            // not set the data on the connect_req
            // until its initialized
            direct::set_data_for_req(
                connect_req_ptr as *libc::c_void,
                ptr::addr_of(client_data) as *libc::c_void);
            direct::set_data_for_uv_handle(
                tcp_handle_ptr as *libc::c_void,
                ptr::addr_of(client_data) as *libc::c_void);
            io::println("before run tcp req loop");
            direct::run(test_loop);
            io::println("after run tcp req loop");
        }
        else {
           io::println("direct::tcp_connect() failure");
           assert false;
        }
    }
    else {
        io::println("direct::tcp_init() failure");
        assert false;
    }
    direct::loop_delete(test_loop);

}

crust fn server_after_close_cb(handle: *libc::c_void) unsafe {
    io::println("SERVER server stream closed, should exit loop...");
}

crust fn client_stream_after_close_cb(handle: *libc::c_void)
    unsafe {
    io::println("SERVER: closed client stream, now closing server stream");
    let client_data = direct::get_data_for_uv_handle(
        handle) as
        *tcp_server_data;
    direct::close((*client_data).server as *libc::c_void,
                  server_after_close_cb);
}

crust fn after_server_resp_write(req: *uv_write_t) unsafe {
    let client_stream_ptr =
        direct::get_stream_handle_from_write_req(req);
    let client_data = direct::get_data_for_uv_handle(
        client_stream_ptr as *libc::c_void) as
        *tcp_server_data;
    io::println("SERVER: resp sent... closing client stream");
    direct::close(client_stream_ptr as *libc::c_void,
                  client_stream_after_close_cb)
}

crust fn on_server_read_cb(client_stream_ptr: *uv_stream_t,
                           nread: libc::ssize_t,
                           ++buf: uv_buf_t) unsafe {
    if (nread > 0) {
        // we have data
        io::println(#fmt("SERVER read: data! nread: %d", nread));

        // pull out the contents of the write from the client
        let buf_base = direct::get_base_from_buf(buf);
        let buf_len = direct::get_len_from_buf(buf);
        let bytes = vec::unsafe::from_buf(buf_base, buf_len);
        let request_str = str::from_bytes(bytes);

        let client_data = direct::get_data_for_uv_handle(
            client_stream_ptr as *libc::c_void) as *tcp_server_data;

        let server_kill_msg = (*client_data).server_kill_msg;
        let write_req = (*client_data).server_write_req;
        if (str::contains(request_str, server_kill_msg)) {
            io::println("SERVER: client request contains server_kill_msg!");
            io::println("SERVER: sending response to client");
            direct::read_stop(client_stream_ptr);
            let server_chan = *((*client_data).server_chan);
            comm::send(server_chan, request_str);
            let write_result = direct::write(
                write_req as *libc::c_void,
                client_stream_ptr as *libc::c_void,
                (*client_data).server_resp_buf,
                after_server_resp_write);
            io::println(#fmt("SERVER: resp write result: %d",
                        write_result as int));
            if (write_result != 0i32) {
                io::println("non-zero result for server resp direct::write()");
                direct::print_last_err_info(
                    direct::get_loop_for_uv_handle(client_stream_ptr
                        as *libc::c_void));
                assert false;
            }
        }
        else {
            io::println("SERVER: client req DOESNT contain server_kill_msg!");
        }
    }
    else if (nread == -1) {
        // err .. possibly EOF
        io::println("read: eof!");
    }
    else {
        // nread == 0 .. do nothing, just free buf as below
        io::println("read: do nothing!");
    }
    // when we're done
    direct::free_base_of_buf(buf);
    io::println("SERVER exiting on_read_cb");
}

crust fn server_connection_cb(server_stream_ptr: *uv_stream_t,
                              status: libc::c_int) unsafe {
    io::println("client connecting!");
    let test_loop = direct::get_loop_for_uv_handle(
                           server_stream_ptr as *libc::c_void);
    let server_data = direct::get_data_for_uv_handle(
        server_stream_ptr as *libc::c_void) as *tcp_server_data;
    let client_stream_ptr = (*server_data).client;
    let client_init_result = direct::tcp_init(test_loop,
                                              client_stream_ptr);
    direct::set_data_for_uv_handle(
        client_stream_ptr as *libc::c_void,
        server_data as *libc::c_void);
    if (client_init_result == 0i32) {
        io::println("successfully initialized client stream");
        let accept_result = direct::accept(server_stream_ptr as
                                             *libc::c_void,
                                           client_stream_ptr as
                                             *libc::c_void);
        if (accept_result == 0i32) {
            // start reading
            let read_result = direct::read_start(client_stream_ptr
                                                   as *uv_stream_t,
                                                 on_alloc_cb,
                                                 on_server_read_cb);
            if (read_result == 0i32) {
                io::println("successful server read start");
            }
            else {
                io::println(#fmt("server_connection_cb: bad read:%d",
                                read_result as int));
                assert false;
            }
        }
        else {
            io::println(#fmt("server_connection_cb: bad accept: %d",
                        accept_result as int));
            assert false;
        }
    }
    else {
        io::println(#fmt("server_connection_cb: bad client init: %d",
                    client_init_result as int));
        assert false;
    }
}

type tcp_server_data = {
    client: *uv_tcp_t,
    server: *uv_tcp_t,
    server_kill_msg: str,
    server_resp_buf: *[uv_buf_t],
    server_chan: *comm::chan<str>,
    server_write_req: *uv_write_t
};

type async_handle_data = {
    continue_chan: *comm::chan<bool>
};

crust fn async_close_cb(handle: *libc::c_void) {
    io::println("SERVER: closing async cb...");
}

crust fn continue_async_cb(async_handle: *uv_async_t,
                           status: libc::c_int) unsafe {
    // once we're in the body of this callback,
    // the tcp server's loop is set up, so we
    // can continue on to let the tcp client
    // do its thang
    let data = direct::get_data_for_uv_handle(
        async_handle as *libc::c_void) as *async_handle_data;
    let continue_chan = *((*data).continue_chan);
    comm::send(continue_chan, true);
    direct::close(async_handle as *libc::c_void, async_close_cb);
}

fn impl_uv_tcp_server(server_ip: str,
                      server_port: int,
                      kill_server_msg: str,
                      server_resp_msg: str,
                      server_chan: *comm::chan<str>,
                      continue_chan: *comm::chan<bool>) unsafe {
    let test_loop = direct::loop_new();
    let tcp_server = direct::tcp_t();
    let tcp_server_ptr = ptr::addr_of(tcp_server);

    let tcp_client = direct::tcp_t();
    let tcp_client_ptr = ptr::addr_of(tcp_client);

    let server_write_req = direct::write_t();
    let server_write_req_ptr = ptr::addr_of(server_write_req);

    let resp_str_bytes = str::bytes(server_resp_msg);
    let resp_msg_ptr: *u8 = vec::unsafe::to_ptr(resp_str_bytes);
    let resp_msg = [
        direct::buf_init(resp_msg_ptr, vec::len(resp_str_bytes))
    ];

    let continue_async_handle = direct::async_t();
    let continue_async_handle_ptr =
        ptr::addr_of(continue_async_handle);
    let async_data =
        { continue_chan: continue_chan };
    let async_data_ptr = ptr::addr_of(async_data);

    let server_data: tcp_server_data = {
        client: tcp_client_ptr,
        server: tcp_server_ptr,
        server_kill_msg: kill_server_msg,
        server_resp_buf: ptr::addr_of(resp_msg),
        server_chan: server_chan,
        server_write_req: server_write_req_ptr
    };
    let server_data_ptr = ptr::addr_of(server_data);
    direct::set_data_for_uv_handle(tcp_server_ptr as *libc::c_void,
                                   server_data_ptr as *libc::c_void);

    // uv_tcp_init()
    let tcp_init_result = direct::tcp_init(
        test_loop as *libc::c_void, tcp_server_ptr);
    if (tcp_init_result == 0i32) {
        let server_addr = direct::ip4_addr(server_ip, server_port);
        // FIXME ref #2064
        let server_addr_ptr = ptr::addr_of(server_addr);

        // uv_tcp_bind()
        let bind_result = direct::tcp_bind(tcp_server_ptr,
                                           server_addr_ptr);
        if (bind_result == 0i32) {
            io::println("successful uv_tcp_bind, listening");

            // uv_listen()
            let listen_result = direct::listen(tcp_server_ptr as
                                                 *libc::c_void,
                                               128i32,
                                               server_connection_cb);
            if (listen_result == 0i32) {
                // let the test know it can set up the tcp server,
                // now.. this may still present a race, not sure..
                let async_result = direct::async_init(test_loop,
                                   continue_async_handle_ptr,
                                   continue_async_cb);
                if (async_result == 0i32) {
                    direct::set_data_for_uv_handle(
                        continue_async_handle_ptr as *libc::c_void,
                        async_data_ptr as *libc::c_void);
                    direct::async_send(continue_async_handle_ptr);
                    // uv_run()
                    direct::run(test_loop);
                    io::println("server uv::run() has returned");
                }
                else {
                    io::println(#fmt("uv_async_init failure: %d",
                            async_result as int));
                    assert false;
                }
            }
            else {
                io::println(#fmt("non-zero result on uv_listen: %d",
                            listen_result as int));
                assert false;
            }
        }
        else {
            io::println(#fmt("non-zero result on uv_tcp_bind: %d",
                        bind_result as int));
            assert false;
        }
    }
    else {
        io::println(#fmt("non-zero result on uv_tcp_init: %d",
                    tcp_init_result as int));
        assert false;
    }
    direct::loop_delete(test_loop);
}

#[test]
#[ignore(cfg(target_os = "freebsd"))]
fn test_uv_tcp_server_and_request() unsafe {
    let bind_ip = "0.0.0.0";
    let request_ip = "127.0.0.1";
    let port = 8888;
    let kill_server_msg = "does a dog have buddha nature?";
    let server_resp_msg = "mu!";
    let client_port = comm::port::<str>();
    let client_chan = comm::chan::<str>(client_port);
    let server_port = comm::port::<str>();
    let server_chan = comm::chan::<str>(server_port);

    let continue_port = comm::port::<bool>();
    let continue_chan = comm::chan::<bool>(continue_port);
    let continue_chan_ptr = ptr::addr_of(continue_chan);

    task::spawn_sched(task::manual_threads(1u)) {||
        impl_uv_tcp_server(bind_ip, port,
                           kill_server_msg,
                           server_resp_msg,
                           ptr::addr_of(server_chan),
                           continue_chan_ptr);
    };

    // block until the server up is.. possibly a race?
    io::println("before receiving on server continue_port");
    comm::recv(continue_port);
    io::println("received on continue port, set up tcp client");

    task::spawn_sched(task::manual_threads(1u)) {||
        impl_uv_tcp_request(request_ip, port,
                           kill_server_msg,
                           ptr::addr_of(client_chan));
    };

    let msg_from_client = comm::recv(server_port);
    let msg_from_server = comm::recv(client_port);

    assert str::contains(msg_from_client, kill_server_msg);
    assert str::contains(msg_from_server, server_resp_msg);
}

// struct size tests
#[test]
#[ignore(cfg(target_os = "freebsd"))]
fn test_uv_struct_size_uv_tcp_t() {
    let native_handle_size = rustrt::rust_uv_helper_uv_tcp_t_size();
    let rust_handle_size = sys::size_of::<uv_tcp_t>();
    let output = #fmt("uv_tcp_t -- native: %u rust: %u",
                      native_handle_size as uint, rust_handle_size);
    io::println(output);
    assert native_handle_size as uint == rust_handle_size;
}
#[test]
#[ignore(cfg(target_os = "freebsd"))]
fn test_uv_struct_size_uv_connect_t() {
    let native_handle_size =
        rustrt::rust_uv_helper_uv_connect_t_size();
    let rust_handle_size = sys::size_of::<uv_connect_t>();
    let output = #fmt("uv_connect_t -- native: %u rust: %u",
                      native_handle_size as uint, rust_handle_size);
    io::println(output);
    assert native_handle_size as uint == rust_handle_size;
}
#[test]
#[ignore(cfg(target_os = "freebsd"))]
fn test_uv_struct_size_uv_buf_t() {
    let native_handle_size =
        rustrt::rust_uv_helper_uv_buf_t_size();
    let rust_handle_size = sys::size_of::<uv_buf_t>();
    let output = #fmt("uv_buf_t -- native: %u rust: %u",
                      native_handle_size as uint, rust_handle_size);
    io::println(output);
    assert native_handle_size as uint == rust_handle_size;
}
#[test]
#[ignore(cfg(target_os = "freebsd"))]
fn test_uv_struct_size_uv_write_t() {
    let native_handle_size =
        rustrt::rust_uv_helper_uv_write_t_size();
    let rust_handle_size = sys::size_of::<uv_write_t>();
    let output = #fmt("uv_write_t -- native: %u rust: %u",
                      native_handle_size as uint, rust_handle_size);
    io::println(output);
    assert native_handle_size as uint == rust_handle_size;
}

#[test]
#[ignore(cfg(target_os = "freebsd"))]
fn test_uv_struct_size_sockaddr_in() {
    let native_handle_size =
        rustrt::rust_uv_helper_sockaddr_in_size();
    let rust_handle_size = sys::size_of::<sockaddr_in>();
    let output = #fmt("sockaddr_in -- native: %u rust: %u",
                      native_handle_size as uint, rust_handle_size);
    io::println(output);
    assert native_handle_size as uint == rust_handle_size;
}

#[test]
#[ignore(cfg(target_os = "freebsd"))]
fn test_uv_struct_size_uv_async_t() {
    let native_handle_size =
        rustrt::rust_uv_helper_uv_async_t_size();
    let rust_handle_size = sys::size_of::<uv_async_t>();
    let output = #fmt("uv_async_t -- native: %u rust: %u",
                      native_handle_size as uint, rust_handle_size);
    io::println(output);
    assert native_handle_size as uint == rust_handle_size;
}

fn impl_uv_byval_test() unsafe {
    let addr = direct::ip4_addr("173.194.33.111", 80);
    io::println(#fmt("after build addr in rust. port: %u",
                     addr.sin_port as uint));
    assert rustrt::rust_uv_ip4_test_verify_port_val(addr,
                                   addr.sin_port as libc::c_uint);
    io::println(#fmt("after build addr in rust. port: %u",
                     addr.sin_port as uint));
}
#[test]
#[ignore(cfg(target_os = "freebsd"))]
fn test_uv_ip4_byval_passing_test() {
    impl_uv_byval_test();
}
