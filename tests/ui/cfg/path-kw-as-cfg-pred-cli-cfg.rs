//@ edition: 2024
//@ check-fail

//@ revisions: cfg_crate cfg_super cfg_self_lower cfg_self_upper
//@ revisions: cfg_struct cfg_enum cfg_async cfg_impl cfg_trait
//@ revisions: cfg_raw_crate cfg_raw_super cfg_raw_self_lower cfg_raw_self_upper

//@ [cfg_crate]compile-flags: --cfg crate
//@ [cfg_super]compile-flags: --cfg super
//@ [cfg_self_lower]compile-flags: --cfg self
//@ [cfg_self_upper]compile-flags: --cfg Self

//@ [cfg_struct]compile-flags: --cfg struct
//@ [cfg_enum]compile-flags: --cfg enum
//@ [cfg_async]compile-flags: --cfg async
//@ [cfg_impl]compile-flags: --cfg impl
//@ [cfg_trait]compile-flags: --cfg trait

//@ [cfg_raw_crate]compile-flags: --cfg r#crate
//@ [cfg_raw_super]compile-flags: --cfg r#super
//@ [cfg_raw_self_lower]compile-flags: --cfg r#self
//@ [cfg_raw_self_upper]compile-flags: --cfg r#Self

fn main() {}

//~? ERROR invalid `--cfg` argument
