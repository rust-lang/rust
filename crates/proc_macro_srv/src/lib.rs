//! RA Proc Macro Server
//!
//! This library is able to call compiled Rust custom derive dynamic libraries on arbitrary code.
//! The general idea here is based on https://github.com/fedochet/rust-proc-macro-expander.
//!
//! But we adapt it to better fit RA needs:
//!
//! * We use `tt` for proc-macro `TokenStream` server, it is easier to manipulate and interact with
//!   RA than `proc-macro2` token stream.
//! * By **copying** the whole rustc `lib_proc_macro` code, we are able to build this with `stable`
//!   rustc rather than `unstable`. (Although in general ABI compatibility is still an issue)…
#![allow(unreachable_pub)]

#[allow(dead_code)]
#[doc(hidden)]
mod proc_macro;

#[doc(hidden)]
mod rustc_server;

mod dylib;

use proc_macro::bridge::client::TokenStream;
use proc_macro_api::{ExpansionResult, ExpansionTask, ListMacrosResult, ListMacrosTask};
use std::{
    collections::{hash_map::Entry, HashMap},
    env, fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

#[derive(Default)]
pub(crate) struct ProcMacroSrv {
    expanders: HashMap<(PathBuf, SystemTime), dylib::Expander>,
}

impl ProcMacroSrv {
    pub fn expand(&mut self, task: &ExpansionTask) -> Result<ExpansionResult, String> {
        let expander = self.expander(&task.lib)?;

        let mut prev_env = HashMap::new();
        for (k, v) in &task.env {
            prev_env.insert(k.as_str(), env::var_os(k));
            env::set_var(k, v);
        }

        let result = expander.expand(&task.macro_name, &task.macro_body, task.attributes.as_ref());

        for (k, _) in &task.env {
            match &prev_env[k.as_str()] {
                Some(v) => env::set_var(k, v),
                None => env::remove_var(k),
            }
        }

        match result {
            Ok(expansion) => Ok(ExpansionResult { expansion }),
            Err(msg) => {
                let msg = msg.as_str().unwrap_or("<unknown error>");
                Err(format!("proc-macro panicked: {}", msg))
            }
        }
    }

    pub fn list_macros(&mut self, task: &ListMacrosTask) -> Result<ListMacrosResult, String> {
        let expander = self.expander(&task.lib)?;
        Ok(ListMacrosResult { macros: expander.list_macros() })
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

pub mod cli;

#[cfg(test)]
mod tests;
