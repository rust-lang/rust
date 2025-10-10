//@ edition: 2024
//@ check-pass
//@ compile-flags: --cfg r#struct --cfg r#enum --cfg r#async --cfg r#impl --cfg r#trait
//@ compile-flags: --check-cfg 'cfg(r#struct)' --check-cfg 'cfg(r#enum)' --check-cfg 'cfg(r#async)' --check-cfg 'cfg(r#impl)' --check-cfg 'cfg(r#trait)'

fn main() {}
