#[cfg(target_os = "linux")];
#[cfg(target_os = "macos")];

import std::uv;
import std::ptr;

#[test]
fn sanity_check() {
    uv::sanity_check();
}

// From test-ref.c
mod test_ref {

    #[test]
    fn ref() {
        let loop = uv::loop_new();
        uv::run(loop);
        uv::loop_delete(loop);
    }

    #[test]
    fn idle_ref() {
        let loop = uv::loop_new();
        let h = uv::idle_new();
        uv::idle_init(loop, ptr::addr_of(h));
        uv::idle_start(ptr::addr_of(h), ptr::null());
        uv::unref(loop);
        uv::run(loop);
        uv::loop_delete(loop);
    }

    #[test]
    fn async_ref() {
        /*
        let loop = uv::loop_new();
        let h = uv::async_new();
        uv::async_init(loop, ptr::addr_of(h), ptr::null());
        uv::unref(loop);
        uv::run(loop);
        uv::loop_delete(loop);
        */
    }
}