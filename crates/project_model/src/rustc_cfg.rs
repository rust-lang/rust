//! Runs `rustc --print cfg` to get built-in cfg flags.

use std::process::Command;

use crate::{cfg_flag::CfgFlag, utf8_stdout};

pub(crate) fn get(target: Option<&str>) -> Vec<CfgFlag> {
    let _p = profile::span("rustc_cfg::get");
    let mut res = Vec::with_capacity(6 * 2 + 1);

    // Some nightly-only cfgs, which are required for stdlib
    res.push(CfgFlag::Atom("target_thread_local".into()));
    for &ty in ["8", "16", "32", "64", "cas", "ptr"].iter() {
        for &key in ["target_has_atomic", "target_has_atomic_load_store"].iter() {
            res.push(CfgFlag::KeyValue { key: key.to_string(), value: ty.into() });
        }
    }

    let rustc_cfgs = {
        let mut cmd = Command::new(toolchain::rustc());
        cmd.args(&["--print", "cfg", "-O"]);
        if let Some(target) = target {
            cmd.args(&["--target", target]);
        }
        utf8_stdout(cmd)
    };

    match rustc_cfgs {
        Ok(rustc_cfgs) => res.extend(rustc_cfgs.lines().map(|it| it.parse().unwrap())),
        Err(e) => log::error!("failed to get rustc cfgs: {:#}", e),
    }

    res
}
