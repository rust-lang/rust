//@ edition: 2021

//@ revisions: check_cfg_crate check_cfg_super check_cfg_self_lower check_cfg_self_upper
//@ revisions: check_cfg_raw_crate check_cfg_raw_super check_cfg_raw_self_lower
//@ revisions: check_cfg_raw_self_upper
//@ revisions: check_cfg_struct check_cfg_priv check_cfg_underscore check_cfg_raw_underscore

//@ [check_cfg_crate]compile-flags: --check-cfg 'cfg(crate)'
//@ [check_cfg_super]compile-flags: --check-cfg 'cfg(super)'
//@ [check_cfg_self_lower]compile-flags: --check-cfg 'cfg(self)'
//@ [check_cfg_self_upper]compile-flags: --check-cfg 'cfg(Self)'

//@ [check_cfg_raw_crate]compile-flags: --check-cfg 'cfg(r#crate)'
//@ [check_cfg_raw_super]compile-flags: --check-cfg 'cfg(r#super)'
//@ [check_cfg_raw_self_lower]compile-flags: --check-cfg 'cfg(r#self)'
//@ [check_cfg_raw_self_upper]compile-flags: --check-cfg 'cfg(r#Self)'

//@ [check_cfg_struct]compile-flags: --check-cfg 'cfg(struct)'
//@ [check_cfg_priv]compile-flags: --check-cfg 'cfg(priv)'
//@ [check_cfg_underscore]compile-flags: --check-cfg 'cfg(_)'
//@ [check_cfg_raw_underscore]compile-flags: --check-cfg 'cfg(r#_)'

fn main() {}

//~? ERROR invalid `--check-cfg` argument
