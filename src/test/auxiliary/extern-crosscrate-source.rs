#[link(name = "externcallback",
       vers = "0.1")];

#[crate_type = "lib"];

extern mod rustrt {
    fn rust_dbg_call(cb: *u8,
                     data: libc::uintptr_t) -> libc::uintptr_t;
}

fn fact(n: uint) -> uint {
    debug!("n = %?", n);
    rustrt::rust_dbg_call(cb, n)
}

extern fn cb(data: libc::uintptr_t) -> libc::uintptr_t {
    if data == 1u {
        data
    } else {
        fact(data - 1u) * data
    }
}
