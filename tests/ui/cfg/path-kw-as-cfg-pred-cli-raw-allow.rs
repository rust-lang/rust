//@ edition: 2024
//@ check-pass
//@ compile-flags: --cfg r#struct --cfg r#priv
//@ compile-flags: --check-cfg 'cfg(r#struct)' --check-cfg 'cfg(r#priv)'

fn main() {}
