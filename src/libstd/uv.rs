/*
This is intended to be a low-level binding to libuv that very closely mimics
the C libuv API. Does very little right now pending scheduler improvements.
*/

export sanity_check;
export loop_t, idle_t;
export loop_new, loop_delete, default_loop, run, unref;
export idle_init, idle_start;
export idle_new;

import core::ctypes;

#[link_name = "rustrt"]
native mod uv {
    fn rust_uv_loop_new() -> *loop_t;
    fn rust_uv_loop_delete(loop: *loop_t);
    fn rust_uv_default_loop() -> *loop_t;
    fn rust_uv_run(loop: *loop_t) -> ctypes::c_int;
    fn rust_uv_unref(loop: *loop_t);
    fn rust_uv_idle_init(loop: *loop_t, idle: *idle_t) -> ctypes::c_int;
    fn rust_uv_idle_start(idle: *idle_t, cb: idle_cb) -> ctypes::c_int;
}

#[link_name = "rustrt"]
native mod helpers {
    fn rust_uv_size_of_idle_t() -> ctypes::size_t;
}

type opaque_cb = *ctypes::void;

type handle_type = ctypes::enum;

type close_cb = opaque_cb;
type idle_cb = opaque_cb;

type handle_private_fields = {
    a00: ctypes::c_int,
    a01: ctypes::c_int,
    a02: ctypes::c_int,
    a03: ctypes::c_int,
    a04: ctypes::c_int,
    a05: ctypes::c_int,
    a06: int,
    a07: int,
    a08: int,
    a09: int,
    a10: int,
    a11: int,
    a12: int
};

type handle_fields = {
    loop: *loop_t,
    type_: handle_type,
    close_cb: close_cb,
    data: *ctypes::void,
    private: handle_private_fields
};

type handle_t = {
    fields: handle_fields
};

type loop_t = int;




type idle_t = {
    fields: handle_fields
    /* private: idle_private_fields */
};

fn idle_init(loop: *loop_t, idle: *idle_t) -> ctypes::c_int {
    uv::rust_uv_idle_init(loop, idle)
}

fn idle_start(idle: *idle_t, cb: idle_cb) -> ctypes::c_int {
    uv::rust_uv_idle_start(idle, cb)
}




fn default_loop() -> *loop_t {
    uv::rust_uv_default_loop()
}

fn loop_new() -> *loop_t {
    uv::rust_uv_loop_new()
}

fn loop_delete(loop: *loop_t) {
    uv::rust_uv_loop_delete(loop)
}

fn run(loop: *loop_t) -> ctypes::c_int {
    uv::rust_uv_run(loop)
}

fn unref(loop: *loop_t) {
    uv::rust_uv_unref(loop)
}


fn sanity_check() {
    fn check_size(t: str, uv: ctypes::size_t, rust: ctypes::size_t) {
        #debug("size of %s: uv: %u, rust: %u", t, uv, rust);
        assert uv <= rust;
    }
    check_size("idle_t",
               helpers::rust_uv_size_of_idle_t(),
               sys::size_of::<idle_t>());
}

fn handle_fields_new() -> handle_fields {
    {
        loop: ptr::null(),
        type_: 0u32,
        close_cb: ptr::null(),
        data: ptr::null(),
        private: {
            a00: 0i32,
            a01: 0i32,
            a02: 0i32,
            a03: 0i32,
            a04: 0i32,
            a05: 0i32,
            a06: 0,
            a07: 0,
            a08: 0,
            a09: 0,
            a10: 0,
            a11: 0,
            a12: 0
        }
    }
}

fn idle_new() -> idle_t {
    {
        fields: handle_fields_new()
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_sanity_check() {
        sanity_check();
    }

    // From test-ref.c
    mod test_ref {

        #[test]
        fn ref() {
            let loop = loop_new();
            run(loop);
            loop_delete(loop);
        }

        #[test]
        fn idle_ref() {
            let loop = loop_new();
            let h = idle_new();
            idle_init(loop, ptr::addr_of(h));
            idle_start(ptr::addr_of(h), ptr::null());
            unref(loop);
            run(loop);
            loop_delete(loop);
        }

        #[test]
        fn async_ref() {
            /*
            let loop = loop_new();
            let h = async_new();
            async_init(loop, ptr::addr_of(h), ptr::null());
            unref(loop);
            run(loop);
            loop_delete(loop);
            */
        }
    }
}
