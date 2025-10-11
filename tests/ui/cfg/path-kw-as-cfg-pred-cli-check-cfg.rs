//@ edition: 2024
//@ check-fail

//@ revisions: check_cfg_crate check_cfg_super check_cfg_self_lower check_cfg_self_upper
//@ revisions: check_cfg_struct check_cfg_enum check_cfg_async check_cfg_impl check_cfg_trait
//@ revisions: check_cfg_raw_crate check_cfg_raw_super check_cfg_raw_self_lower
//@ revisions: check_cfg_raw_self_upper

//@ [check_cfg_crate]compile-flags: --check-cfg 'cfg(crate)'
//@ [check_cfg_super]compile-flags: --check-cfg 'cfg(super)'
//@ [check_cfg_self_lower]compile-flags: --check-cfg 'cfg(self)'
//@ [check_cfg_self_upper]compile-flags: --check-cfg 'cfg(Self)'

//@ [check_cfg_struct]compile-flags: --check-cfg 'cfg(struct)'
//@ [check_cfg_enum]compile-flags: --check-cfg 'cfg(enum)'
//@ [check_cfg_async]compile-flags: --check-cfg 'cfg(async)'
//@ [check_cfg_impl]compile-flags: --check-cfg 'cfg(impl)'
//@ [check_cfg_trait]compile-flags: --check-cfg 'cfg(trait)'

//@ [check_cfg_raw_crate]compile-flags: --check-cfg 'cfg(r#crate)'
//@ [check_cfg_raw_super]compile-flags: --check-cfg 'cfg(r#super)'
//@ [check_cfg_raw_self_lower]compile-flags: --check-cfg 'cfg(r#self)'
//@ [check_cfg_raw_self_upper]compile-flags: --check-cfg 'cfg(r#Self)'

fn main() {}

//~? ERROR invalid `--check-cfg` argument
