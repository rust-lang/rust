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

#![cfg(any(feature = "sysroot-abi", rust_analyzer))]
#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]
#![feature(proc_macro_internals, proc_macro_diagnostic, proc_macro_span)]
#![allow(unreachable_pub, internal_features, clippy::disallowed_types, clippy::print_stderr)]

extern crate proc_macro;
#[cfg(feature = "in-rust-tree")]
extern crate rustc_driver as _;

#[cfg(not(feature = "in-rust-tree"))]
extern crate ra_ap_rustc_lexer as rustc_lexer;
#[cfg(feature = "in-rust-tree")]
extern crate rustc_lexer;

mod dylib;
mod proc_macros;
mod server_impl;

use std::{
    collections::{hash_map::Entry, HashMap},
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    thread,
};

use paths::{Utf8Path, Utf8PathBuf};
use proc_macro_api::{
    msg::{
        self, deserialize_span_data_index_map, serialize_span_data_index_map, ExpnGlobals,
        SpanMode, TokenId, CURRENT_API_VERSION,
    },
    ProcMacroKind,
};
use span::Span;

use crate::server_impl::TokenStream;

pub const RUSTC_VERSION_STRING: &str = env!("RUSTC_VERSION");

pub struct ProcMacroSrv<'env> {
    expanders: HashMap<Utf8PathBuf, dylib::Expander>,
    span_mode: SpanMode,
    env: &'env EnvSnapshot,
}

impl<'env> ProcMacroSrv<'env> {
    pub fn new(env: &'env EnvSnapshot) -> Self {
        Self { expanders: Default::default(), span_mode: Default::default(), env }
    }
}

const EXPANDER_STACK_SIZE: usize = 8 * 1024 * 1024;

impl ProcMacroSrv<'_> {
    pub fn set_span_mode(&mut self, span_mode: SpanMode) {
        self.span_mode = span_mode;
    }

    pub fn span_mode(&self) -> SpanMode {
        self.span_mode
    }

    pub fn expand(
        &mut self,
        msg::ExpandMacro { lib, env, current_dir, data }: msg::ExpandMacro,
    ) -> Result<(msg::FlatTree, Vec<u32>), msg::PanicMessage> {
        let span_mode = self.span_mode;
        let snapped_env = self.env;
        let expander = self
            .expander(lib.as_ref())
            .map_err(|err| msg::PanicMessage(format!("failed to load macro: {err}")))?;

        let prev_env = EnvChange::apply(snapped_env, env, current_dir.as_ref().map(<_>::as_ref));

        let result = match span_mode {
            SpanMode::Id => expand_id(data, expander).map(|it| (it, vec![])),
            SpanMode::RustAnalyzer => expand_ra_span(data, expander),
        };

        prev_env.rollback();

        result.map_err(msg::PanicMessage)
    }

    pub fn list_macros(
        &mut self,
        dylib_path: &Utf8Path,
    ) -> Result<Vec<(String, ProcMacroKind)>, String> {
        let expander = self.expander(dylib_path)?;
        Ok(expander.list_macros())
    }

    fn expander(&mut self, path: &Utf8Path) -> Result<&dylib::Expander, String> {
        let expander = || {
            dylib::Expander::new(path)
                .map_err(|err| format!("Cannot create expander for {path}: {err}",))
        };

        Ok(match self.expanders.entry(path.to_path_buf()) {
            Entry::Vacant(v) => v.insert(expander()?),
            Entry::Occupied(mut e) => {
                let time = fs::metadata(path).and_then(|it| it.modified()).ok();
                if Some(e.get().modified_time()) != time {
                    e.insert(expander()?);
                }
                e.into_mut()
            }
        })
    }
}

trait ProcMacroSrvSpan: Copy {
    type Server: proc_macro::bridge::server::Server<TokenStream = TokenStream<Self>>;
    fn make_server(call_site: Self, def_site: Self, mixed_site: Self) -> Self::Server;
}

impl ProcMacroSrvSpan for TokenId {
    type Server = server_impl::token_id::TokenIdServer;

    fn make_server(call_site: Self, def_site: Self, mixed_site: Self) -> Self::Server {
        Self::Server { call_site, def_site, mixed_site }
    }
}
impl ProcMacroSrvSpan for Span {
    type Server = server_impl::rust_analyzer_span::RaSpanServer;
    fn make_server(call_site: Self, def_site: Self, mixed_site: Self) -> Self::Server {
        Self::Server {
            call_site,
            def_site,
            mixed_site,
            tracked_env_vars: Default::default(),
            tracked_paths: Default::default(),
        }
    }
}

fn expand_id(
    msg::ExpandMacroData {
        macro_body,
        macro_name,
        attributes,
        has_global_spans: ExpnGlobals { serialize: _, def_site, call_site, mixed_site },
        span_data_table: _,
    }: msg::ExpandMacroData,
    expander: &dylib::Expander,
) -> Result<msg::FlatTree, String> {
    let def_site = TokenId(def_site as u32);
    let call_site = TokenId(call_site as u32);
    let mixed_site = TokenId(mixed_site as u32);

    let macro_body = macro_body.to_subtree_unresolved(CURRENT_API_VERSION);
    let attributes = attributes.map(|it| it.to_subtree_unresolved(CURRENT_API_VERSION));
    let result = thread::scope(|s| {
        let thread = thread::Builder::new()
            .stack_size(EXPANDER_STACK_SIZE)
            .name(macro_name.clone())
            .spawn_scoped(s, || {
                expander
                    .expand(&macro_name, macro_body, attributes, def_site, call_site, mixed_site)
                    .map(|it| msg::FlatTree::new_raw(&it, CURRENT_API_VERSION))
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
    result
}

fn expand_ra_span(
    msg::ExpandMacroData {
        macro_body,
        macro_name,
        attributes,
        has_global_spans: ExpnGlobals { serialize: _, def_site, call_site, mixed_site },
        span_data_table,
    }: msg::ExpandMacroData,
    expander: &dylib::Expander,
) -> Result<(msg::FlatTree, Vec<u32>), String> {
    let mut span_data_table = deserialize_span_data_index_map(&span_data_table);

    let def_site = span_data_table[def_site];
    let call_site = span_data_table[call_site];
    let mixed_site = span_data_table[mixed_site];

    let macro_body = macro_body.to_subtree_resolved(CURRENT_API_VERSION, &span_data_table);
    let attributes =
        attributes.map(|it| it.to_subtree_resolved(CURRENT_API_VERSION, &span_data_table));
    // Note, we spawn a new thread here so that thread locals allocation don't accumulate (this
    // includes the proc-macro symbol interner)
    let result = thread::scope(|s| {
        let thread = thread::Builder::new()
            .stack_size(EXPANDER_STACK_SIZE)
            .name(macro_name.clone())
            .spawn_scoped(s, || {
                expander
                    .expand(&macro_name, macro_body, attributes, def_site, call_site, mixed_site)
                    .map(|it| {
                        (
                            msg::FlatTree::new(&it, CURRENT_API_VERSION, &mut span_data_table),
                            serialize_span_data_index_map(&span_data_table),
                        )
                    })
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
    result
}

pub struct PanicMessage {
    message: Option<String>,
}

impl PanicMessage {
    pub fn into_string(self) -> Option<String> {
        self.message
    }
}

pub struct EnvSnapshot {
    vars: HashMap<OsString, OsString>,
}

impl Default for EnvSnapshot {
    fn default() -> EnvSnapshot {
        EnvSnapshot { vars: env::vars_os().collect() }
    }
}

struct EnvChange<'snap> {
    changed_vars: Vec<String>,
    prev_working_dir: Option<PathBuf>,
    snap: &'snap EnvSnapshot,
}

impl<'snap> EnvChange<'snap> {
    fn apply(
        snap: &'snap EnvSnapshot,
        new_vars: Vec<(String, String)>,
        current_dir: Option<&Path>,
    ) -> EnvChange<'snap> {
        let prev_working_dir = match current_dir {
            Some(dir) => {
                let prev_working_dir = std::env::current_dir().ok();
                if let Err(err) = std::env::set_current_dir(dir) {
                    eprintln!(
                        "Failed to set the current working dir to {}. Error: {err:?}",
                        dir.display()
                    )
                }
                prev_working_dir
            }
            None => None,
        };
        EnvChange {
            snap,
            changed_vars: new_vars
                .into_iter()
                .map(|(k, v)| {
                    env::set_var(&k, v);
                    k
                })
                .collect(),
            prev_working_dir,
        }
    }

    fn rollback(self) {}
}

impl Drop for EnvChange<'_> {
    fn drop(&mut self) {
        for name in self.changed_vars.drain(..) {
            match self.snap.vars.get::<std::ffi::OsStr>(name.as_ref()) {
                Some(prev_val) => env::set_var(name, prev_val),
                None => env::remove_var(name),
            }
        }

        if let Some(dir) = &self.prev_working_dir {
            if let Err(err) = std::env::set_current_dir(dir) {
                eprintln!(
                    "Failed to set the current working dir to {}. Error: {:?}",
                    dir.display(),
                    err
                )
            }
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub fn proc_macro_test_dylib_path() -> paths::Utf8PathBuf {
    proc_macro_test::PROC_MACRO_TEST_LOCATION.into()
}
