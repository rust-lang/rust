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

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]
#![cfg_attr(
    feature = "sysroot-abi",
    feature(proc_macro_internals, proc_macro_diagnostic, proc_macro_span)
)]
#![allow(unreachable_pub)]

mod dylib;
mod abis;

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
    msg::{ExpandMacro, FlatTree, PanicMessage},
    ProcMacroKind,
};

#[derive(Default)]
pub(crate) struct ProcMacroSrv {
    expanders: HashMap<(PathBuf, SystemTime), dylib::Expander>,
}

const EXPANDER_STACK_SIZE: usize = 8 * 1024 * 1024;

impl ProcMacroSrv {
    pub fn expand(&mut self, task: ExpandMacro) -> Result<FlatTree, PanicMessage> {
        let expander = self.expander(task.lib.as_ref()).map_err(|err| {
            debug_assert!(false, "should list macros before asking to expand");
            PanicMessage(format!("failed to load macro: {}", err))
        })?;

        let prev_env = EnvSnapshot::new();
        for (k, v) in &task.env {
            env::set_var(k, v);
        }
        let prev_working_dir = match task.current_dir {
            Some(dir) => {
                let prev_working_dir = std::env::current_dir().ok();
                if let Err(err) = std::env::set_current_dir(&dir) {
                    eprintln!("Failed to set the current working dir to {}. Error: {:?}", dir, err)
                }
                prev_working_dir
            }
            None => None,
        };

        let macro_body = task.macro_body.to_subtree();
        let attributes = task.attributes.map(|it| it.to_subtree());
        let result = thread::scope(|s| {
            let thread = thread::Builder::new()
                .stack_size(EXPANDER_STACK_SIZE)
                .name(task.macro_name.clone())
                .spawn_scoped(s, || {
                    expander
                        .expand(&task.macro_name, &macro_body, attributes.as_ref())
                        .map(|it| FlatTree::new(&it))
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

        result.map_err(PanicMessage)
    }

    pub(crate) fn list_macros(
        &mut self,
        dylib_path: &Path,
    ) -> Result<Vec<(String, ProcMacroKind)>, String> {
        let expander = self.expander(dylib_path)?;
        Ok(expander.list_macros())
    }

    fn expander(&mut self, path: &Path) -> Result<&dylib::Expander, String> {
        let time = fs::metadata(path).and_then(|it| it.modified()).map_err(|err| {
            format!("Failed to get file metadata for {}: {:?}", path.display(), err)
        })?;

        Ok(match self.expanders.entry((path.to_path_buf(), time)) {
            Entry::Vacant(v) => v.insert(dylib::Expander::new(path).map_err(|err| {
                format!("Cannot create expander for {}: {:?}", path.display(), err)
            })?),
            Entry::Occupied(e) => e.into_mut(),
        })
    }
}

struct EnvSnapshot {
    vars: HashMap<OsString, OsString>,
}

impl EnvSnapshot {
    fn new() -> EnvSnapshot {
        EnvSnapshot { vars: env::vars_os().collect() }
    }

    fn rollback(self) {
        let mut old_vars = self.vars;
        for (name, value) in env::vars_os() {
            let old_value = old_vars.remove(&name);
            if old_value != Some(value) {
                match old_value {
                    None => env::remove_var(name),
                    Some(old_value) => env::set_var(name, old_value),
                }
            }
        }
        for (name, old_value) in old_vars {
            env::set_var(name, old_value)
        }
    }
}

pub mod cli;

#[cfg(test)]
mod tests;
