#[doc = "
Low-level bindings to the libuv library.

This module contains a set of direct, 'bare-metal' wrappers around
the libuv C-API.

Also contained herein are a set of rust records that map, in
approximate memory-size, to the libuv data structures. The record
implementations are adjusted, per-platform, to match their respective
representations.

There are also a collection of helper functions to ease interacting
with the low-level API (such as a function to return the latest
libuv error as a rust-formatted string).

As new functionality, existant in uv.h, is added to the rust stdlib,
the mappings should be added in this module.

This module's implementation will hopefully be, eventually, replaced
with per-platform, generated source files from rust-bindgen.
"];

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
// unix size: 120
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
type uv_async_t = {
    fields: uv_handle_fields,
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8, a10: *u8
};
// win32 size 132 (68)
#[cfg(target_os = "win32")]
type uv_async_t = {
    fields: uv_handle_fields,
    a00: *u8, a01: *u8, a02: *u8, a03: *u8,
    a04: *u8, a05: *u8, a06: *u8, a07: *u8,
    a08: *u8, a09: *u8, a10: *u8, a11: *u8,
    a12: *u8
};

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

mod uv_ll_struct_stubgen {
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn gen_stub_uv_tcp_t() -> uv_tcp_t {
        ret { fields: { loop_handle: ptr::null(), type_: 0u32,
                        close_cb: ptr::null(),
                        mut data: ptr::null() },
            a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
            a03: 0 as *u8,
            a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
            a07: 0 as *u8,
            a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8,
            a11: 0 as *u8,
            a12: 0 as *u8, a13: 0 as *u8, a14: 0 as *u8,
            a15: 0 as *u8,
            a16: 0 as *u8, a17: 0 as *u8, a18: 0 as *u8,
            a19: 0 as *u8,
            a20: 0 as *u8, a21: 0 as *u8, a22: 0 as *u8,
            a23: 0 as *u8,
            a24: 0 as *u8, a25: 0 as *u8, a26: 0 as *u8,
            a27: 0 as *u8,
            a28: 0 as *u8, a29: 0 as *u8
        };
    }
    #[cfg(target_os = "win32")]
    fn gen_stub_uv_tcp_t() -> uv_tcp_t {
        ret { fields: { loop_handle: ptr::null(), type_: 0u32,
                        close_cb: ptr::null(),
                        mut data: ptr::null() },
            a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
            a03: 0 as *u8,
            a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
            a07: 0 as *u8,
            a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8,
            a11: 0 as *u8,
            a12: 0 as *u8, a13: 0 as *u8, a14: 0 as *u8,
            a15: 0 as *u8,
            a16: 0 as *u8, a17: 0 as *u8, a18: 0 as *u8,
            a19: 0 as *u8,
            a20: 0 as *u8, a21: 0 as *u8, a22: 0 as *u8,
            a23: 0 as *u8,
            a24: 0 as *u8, a25: 0 as *u8
        };
    }
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn gen_stub_uv_connect_t() -> uv_connect_t {
        ret {
            a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
            a03: 0 as *u8,
            a04: 0 as *u8, a05: 0 as *u8
        };
    }
    #[cfg(target_os = "win32")]
    fn gen_stub_uv_connect_t() -> uv_connect_t {
        ret {
            a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
            a03: 0 as *u8,
            a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
            a07: 0 as *u8,
            a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8
        };
    }
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn gen_stub_uv_async_t() -> uv_async_t {
        ret { fields: { loop_handle: ptr::null(), type_: 0u32,
                        close_cb: ptr::null(),
                        mut data: ptr::null() },
            a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
            a03: 0 as *u8,
            a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
            a07: 0 as *u8,
            a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8
        };
    }
    #[cfg(target_os = "win32")]
    fn gen_stub_uv_async_t() -> uv_async_t {
        ret { fields: { loop_handle: ptr::null(), type_: 0u32,
                        close_cb: ptr::null(),
                        mut data: ptr::null() },
            a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
            a03: 0 as *u8,
            a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
            a07: 0 as *u8,
            a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8,
            a11: 0 as *u8,
            a12: 0 as *u8
        };
    }
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn gen_stub_uv_write_t() -> uv_write_t {
        ret { fields: { loop_handle: ptr::null(), type_: 0u32,
                        close_cb: ptr::null(),
                        mut data: ptr::null() },
            a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
            a03: 0 as *u8,
            a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
            a07: 0 as *u8,
            a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8,
            a11: 0 as *u8,
            a12: 0 as *u8, a13: 0 as *u8
        };
    }
    #[cfg(target_os = "win32")]
    fn gen_stub_uv_write_t() -> uv_write_t {
        ret { fields: { loop_handle: ptr::null(), type_: 0u32,
                        close_cb: ptr::null(),
                        mut data: ptr::null() },
            a00: 0 as *u8, a01: 0 as *u8, a02: 0 as *u8,
            a03: 0 as *u8,
            a04: 0 as *u8, a05: 0 as *u8, a06: 0 as *u8,
            a07: 0 as *u8,
            a08: 0 as *u8, a09: 0 as *u8, a10: 0 as *u8,
            a11: 0 as *u8,
            a12: 0 as *u8
        };
    }
}

#[nolink]
native mod rustrt {
    fn rust_uv_loop_new() -> *libc::c_void;
    fn rust_uv_loop_delete(lp: *libc::c_void);
    fn rust_uv_run(loop_handle: *libc::c_void);
    fn rust_uv_close(handle: *libc::c_void, cb: *u8);
    fn rust_uv_async_send(handle: *uv_async_t);
    fn rust_uv_async_init(loop_handle: *libc::c_void,
                          async_handle: *uv_async_t,
                          cb: *u8) -> libc::c_int;
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
    ret uv_ll_struct_stubgen::gen_stub_uv_tcp_t();
}
unsafe fn connect_t() -> uv_connect_t {
    ret uv_ll_struct_stubgen::gen_stub_uv_connect_t();
}
unsafe fn write_t() -> uv_write_t {
    ret uv_ll_struct_stubgen::gen_stub_uv_write_t();
}
unsafe fn async_t() -> uv_async_t {
    ret uv_ll_struct_stubgen::gen_stub_uv_async_t();
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

unsafe fn get_last_err_info(uv_loop: *libc::c_void) -> str {
    let err = last_error(uv_loop);
    let err_ptr = ptr::addr_of(err);
    let err_name = str::unsafe::from_c_str(err_name(err_ptr));
    let err_msg = str::unsafe::from_c_str(strerror(err_ptr));
    ret #fmt("LIBUV ERROR: name: %s msg: %s",
                    err_name, err_msg);
}