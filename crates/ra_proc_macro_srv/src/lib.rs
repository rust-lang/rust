//! RA Proc Macro Server
//!
//! This library is able to call compiled Rust custom derive dynamic libraries on arbitrary code.
//! The general idea here is based on https://github.com/fedochet/rust-proc-macro-expander.
//!
//! But we adapt it to better fit RA needs:
//!
//! * We use `ra_tt` for proc-macro `TokenStream` server, it is easier to manipulate and interact with
//!   RA than `proc-macro2` token stream.
//! * By **copying** the whole rustc `lib_proc_macro` code, we are able to build this with `stable`
//!   rustc rather than `unstable`. (Although in gerenal ABI compatibility is still an issue)

#[allow(dead_code)]
#[doc(hidden)]
mod proc_macro;

#[doc(hidden)]
mod rustc_server;

mod dylib;

use proc_macro::bridge::client::TokenStream;
use ra_proc_macro::{ExpansionResult, ExpansionTask, ListMacrosResult, ListMacrosTask};
use std::path::Path;

pub(crate) fn expand_task(task: &ExpansionTask) -> Result<ExpansionResult, String> {
    let expander = create_expander(&task.lib);

    match expander.expand(&task.macro_name, &task.macro_body, task.attributes.as_ref()) {
        Ok(expansion) => Ok(ExpansionResult { expansion }),
        Err(msg) => {
            Err(format!("Cannot perform expansion for {}: error {:?}", &task.macro_name, msg))
        }
    }
}

pub(crate) fn list_macros(task: &ListMacrosTask) -> ListMacrosResult {
    let expander = create_expander(&task.lib);

    ListMacrosResult { macros: expander.list_macros() }
}

fn create_expander(lib: &Path) -> dylib::Expander {
    dylib::Expander::new(lib)
        .unwrap_or_else(|err| panic!("Cannot create expander for {}: {:?}", lib.display(), err))
}

pub mod cli;

#[cfg(test)]
mod tests;
