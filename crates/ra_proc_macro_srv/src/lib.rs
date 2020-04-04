//! RA Proc Macro Server
//!
//! This library is able to call compiled Rust custom derive dynamic libraries on arbitrary code.
//! The general idea here is based on https://github.com/fedochet/rust-proc-macro-expander.
//!
//! But we change some several design for fitting RA needs:
//!
//! * We use `ra_tt` for proc-macro `TokenStream` server, it is easy to manipute and interact with
//!   RA then proc-macro2 token stream.
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

pub fn expand_task(task: &ExpansionTask) -> Result<ExpansionResult, String> {
    let expander = dylib::Expander::new(&task.lib)
        .expect(&format!("Cannot expand with provided libraries: ${:?}", &task.lib));

    match expander.expand(&task.macro_name, &task.macro_body, task.attributes.as_ref()) {
        Ok(expansion) => Ok(ExpansionResult { expansion }),
        Err(msg) => {
            let reason = format!(
                "Cannot perform expansion for {}: error {:?}!",
                &task.macro_name,
                msg.as_str()
            );
            Err(reason)
        }
    }
}

pub fn list_macros(task: &ListMacrosTask) -> Result<ListMacrosResult, String> {
    let expander = dylib::Expander::new(&task.lib)
        .expect(&format!("Cannot expand with provided libraries: ${:?}", &task.lib));

    match expander.list_macros() {
        Ok(macros) => Ok(ListMacrosResult { macros }),
        Err(msg) => {
            let reason =
                format!("Cannot perform expansion for {:?}: error {:?}!", &task.lib, msg.as_str());
            Err(reason)
        }
    }
}
