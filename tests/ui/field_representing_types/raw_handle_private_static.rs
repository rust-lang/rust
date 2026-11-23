//@ run-pass
//@ aux-build:raw_handle_private_static_aux.rs
//@ edition: 2024

extern crate raw_handle_private_static_aux;

fn main() {
    assert_eq!(raw_handle_private_static_aux::read(), 7);
}
