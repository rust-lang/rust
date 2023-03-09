// aux-build: tls-export.rs
// run-pass

#![feature(cfg_target_thread_local)]

#[cfg(target_thread_local)]
extern crate tls_export;

fn main() {
    // Check that we get the real address of the TLS in the dylib
    #[cfg(target_thread_local)]
    assert_eq!(&tls_export::FOO as *const bool as usize, tls_export::foo_addr());
}
