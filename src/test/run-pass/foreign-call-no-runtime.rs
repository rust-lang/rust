use std::cast;
use std::libc;
use std::unstable::run_in_bare_thread;

#[link(name = "rustrt")]
extern {
    fn rust_dbg_call(cb: extern "C" fn(libc::uintptr_t),
                     data: libc::uintptr_t) -> libc::uintptr_t;
}

pub fn main() {
    unsafe {
        run_in_bare_thread(proc() {
            let i = &100;
            rust_dbg_call(callback, cast::transmute(i));
        });
    }
}

extern fn callback(data: libc::uintptr_t) {
    unsafe {
        let data: *int = cast::transmute(data);
        assert_eq!(*data, 100);
    }
}
