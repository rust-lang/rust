//! RA Proc Macro Server
//!
//! This library is able to call compiled Rust custom derive dynamic libraries on arbitrary code.
//! The general idea here is based on <https://github.com/fedochet/rust-proc-macro-expander>.
//!
//! But we adapt it to better fit RA needs:
//!
//! * We use `tt` for proc-macro `TokenStream` server, it is easier to manipulate and interact with
//!   RA than `proc-macro2` token stream.
//! * By **copying** the whole rustc `lib_proc_macro` code, we are able to build this with `stable`
//!   rustc rather than `unstable`. (Although in general ABI compatibility is still an issue)…

#![cfg(feature = "sysroot-abi")]
#![feature(proc_macro_internals, proc_macro_diagnostic, proc_macro_span)]
#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]
#![allow(unreachable_pub)]

extern crate proc_macro;

mod dylib;
mod server;
mod proc_macros;

use std::{
    collections::{hash_map::Entry, HashMap},
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    thread,
    time::SystemTime,
};

use proc_macro_api::{
    msg::{self, CURRENT_API_VERSION},
    ProcMacroKind,
};

use ::tt::token_id as tt;

// see `build.rs`
include!(concat!(env!("OUT_DIR"), "/rustc_version.rs"));

#[derive(Default)]
pub struct ProcMacroSrv {
    expanders: HashMap<(PathBuf, SystemTime), dylib::Expander>,
}

const EXPANDER_STACK_SIZE: usize = 8 * 1024 * 1024;

impl ProcMacroSrv {
    pub fn expand(&mut self, task: msg::ExpandMacro) -> Result<msg::FlatTree, msg::PanicMessage> {
        let expander = self.expander(task.lib.as_ref()).map_err(|err| {
            debug_assert!(false, "should list macros before asking to expand");
            msg::PanicMessage(format!("failed to load macro: {err}"))
        })?;

        let prev_env = EnvSnapshot::new();
        for (k, v) in &task.env {
            env::set_var(k, v);
        }
        let prev_working_dir = match task.current_dir {
            Some(dir) => {
                let prev_working_dir = std::env::current_dir().ok();
                if let Err(err) = std::env::set_current_dir(&dir) {
                    eprintln!("Failed to set the current working dir to {dir}. Error: {err:?}")
                }
                prev_working_dir
            }
            None => None,
        };

        let macro_body = task.macro_body.to_subtree(CURRENT_API_VERSION);
        let attributes = task.attributes.map(|it| it.to_subtree(CURRENT_API_VERSION));
        let result = thread::scope(|s| {
            let thread = thread::Builder::new()
                .stack_size(EXPANDER_STACK_SIZE)
                .name(task.macro_name.clone())
                .spawn_scoped(s, || {
                    expander
                        .expand(&task.macro_name, &macro_body, attributes.as_ref())
                        .map(|it| msg::FlatTree::new(&it, CURRENT_API_VERSION))
                });
            let res = match thread {
                Ok(handle) => handle.join(),
                Err(e) => std::panic::resume_unwind(Box::new(e)),
            };

            match res {
                Ok(res) => res,
                Err(e) => std::panic::resume_unwind(e),
            }
        });

        prev_env.rollback();

        if let Some(dir) = prev_working_dir {
            if let Err(err) = std::env::set_current_dir(&dir) {
                eprintln!(
                    "Failed to set the current working dir to {}. Error: {:?}",
                    dir.display(),
                    err
                )
            }
        }

        result.map_err(msg::PanicMessage)
    }

    pub fn list_macros(
        &mut self,
        dylib_path: &Path,
    ) -> Result<Vec<(String, ProcMacroKind)>, String> {
        let expander = self.expander(dylib_path)?;
        Ok(expander.list_macros())
    }

    fn expander(&mut self, path: &Path) -> Result<&dylib::Expander, String> {
        let time = fs::metadata(path)
            .and_then(|it| it.modified())
            .map_err(|err| format!("Failed to get file metadata for {}: {err}", path.display()))?;

        Ok(match self.expanders.entry((path.to_path_buf(), time)) {
            Entry::Vacant(v) => {
                v.insert(dylib::Expander::new(path).map_err(|err| {
                    format!("Cannot create expander for {}: {err}", path.display())
                })?)
            }
            Entry::Occupied(e) => e.into_mut(),
        })
    }
}

pub struct PanicMessage {
    message: Option<String>,
}

impl PanicMessage {
    pub fn as_str(&self) -> Option<String> {
        self.message.clone()
    }
}

struct EnvSnapshot {
    vars: HashMap<OsString, OsString>,
}

impl EnvSnapshot {
    fn new() -> EnvSnapshot {
        EnvSnapshot { vars: env::vars_os().collect() }
    }

    fn rollback(self) {}
}

impl Drop for EnvSnapshot {
    fn drop(&mut self) {
        for (name, value) in env::vars_os() {
            let old_value = self.vars.remove(&name);
            if old_value != Some(value) {
                match old_value {
                    None => env::remove_var(name),
                    Some(old_value) => env::set_var(name, old_value),
                }
            }
        }
        for (name, old_value) in self.vars.drain() {
            env::set_var(name, old_value)
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub fn proc_macro_test_dylib_path() -> std::path::PathBuf {
    proc_macro_test::PROC_MACRO_TEST_LOCATION.into()
}
