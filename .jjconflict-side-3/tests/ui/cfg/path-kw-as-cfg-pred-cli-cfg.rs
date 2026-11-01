//@ edition: 2024

//@ revisions: cfg_crate cfg_super cfg_self_lower cfg_self_upper
//@ revisions: cfg_raw_crate cfg_raw_super cfg_raw_self_lower cfg_raw_self_upper
//@ revisions: cfg_struct cfg_priv cfg_underscore cfg_raw_underscore

//@ [cfg_crate]compile-flags: --cfg crate
//@ [cfg_super]compile-flags: --cfg super
//@ [cfg_self_lower]compile-flags: --cfg self
//@ [cfg_self_upper]compile-flags: --cfg Self

//@ [cfg_raw_crate]compile-flags: --cfg r#crate
//@ [cfg_raw_super]compile-flags: --cfg r#super
//@ [cfg_raw_self_lower]compile-flags: --cfg r#self
//@ [cfg_raw_self_upper]compile-flags: --cfg r#Self

//@ [cfg_struct]compile-flags: --cfg struct
//@ [cfg_priv]compile-flags: --cfg priv
//@ [cfg_underscore]compile-flags: --cfg _
//@ [cfg_raw_underscore]compile-flags: --cfg r#_

fn main() {}

//~? ERROR invalid `--cfg` argument
