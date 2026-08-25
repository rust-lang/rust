//@ aux-build: tls-rlib.rs
//@ aux-build: tls-export.rs
//@ run-pass

#![feature(cfg_target_thread_local)]

#[cfg(target_thread_local)]
extern crate tls_export;

fn main() {
    #[cfg(target_thread_local)]
    {
        // Check that we get the real address of the `FOO` TLS in the dylib
        assert_eq!(&tls_export::FOO as *const bool as usize, tls_export::foo_addr());

        // Check that we get the real address of the `BAR` TLS in the rlib linked into the dylib
        assert_eq!(&tls_export::BAR as *const bool as usize, tls_export::bar_addr());
    }
}
