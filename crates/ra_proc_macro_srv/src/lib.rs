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

use ra_proc_macro::{ExpansionResult, ExpansionTask, ListMacrosResult, ListMacrosTask};

pub fn expand_task(_task: &ExpansionTask) -> Result<ExpansionResult, String> {
    unimplemented!()
}

pub fn list_macros(_task: &ListMacrosTask) -> Result<ListMacrosResult, String> {
    unimplemented!()
}
