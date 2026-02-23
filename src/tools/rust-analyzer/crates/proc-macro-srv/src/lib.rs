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
#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]
#![feature(proc_macro_internals, proc_macro_diagnostic, proc_macro_span)]
#![allow(
    unreachable_pub,
    internal_features,
    clippy::disallowed_types,
    clippy::print_stderr,
    unused_crate_dependencies
)]
#![deny(deprecated_safe, clippy::undocumented_unsafe_blocks)]

#[cfg(not(feature = "in-rust-tree"))]
extern crate proc_macro as rustc_proc_macro;
#[cfg(feature = "in-rust-tree")]
extern crate rustc_driver as _;
#[cfg(feature = "in-rust-tree")]
extern crate rustc_proc_macro;

#[cfg(not(feature = "in-rust-tree"))]
extern crate ra_ap_rustc_lexer as rustc_lexer;
#[cfg(feature = "in-rust-tree")]
extern crate rustc_lexer;

mod bridge;
mod dylib;
mod server_impl;
mod token_stream;

use std::{
    collections::{HashMap, hash_map::Entry},
    env,
    ffi::OsString,
    fs,
    ops::Range,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, PoisonError},
    thread,
};

use paths::{Utf8Path, Utf8PathBuf};
use span::Span;
use temp_dir::TempDir;

pub use crate::server_impl::token_id::SpanId;

pub use rustc_proc_macro::Delimiter;
pub use span;

pub use crate::bridge::*;
pub use crate::server_impl::literal_from_str;
pub use crate::token_stream::{TokenStream, TokenStreamIter, literal_to_string};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ProcMacroKind {
    CustomDerive,
    Attr,
    Bang,
}

pub const RUSTC_VERSION_STRING: &str = env!("RUSTC_VERSION");

pub struct ProcMacroSrv<'env> {
    expanders: Mutex<HashMap<Utf8PathBuf, Arc<dylib::Expander>>>,
    env: &'env EnvSnapshot,
    temp_dir: TempDir,
}

impl<'env> ProcMacroSrv<'env> {
    pub fn new(env: &'env EnvSnapshot) -> Self {
        Self {
            expanders: Default::default(),
            env,
            temp_dir: TempDir::with_prefix("proc-macro-srv").unwrap(),
        }
    }

    pub fn join_spans(&self, first: Span, second: Span) -> Option<Span> {
        first.join(second, |_, _| {
            // FIXME: Once we can talk back to the client, implement a "long join" request for anchors
            // that differ in [AstId]s as joining those spans requires resolving the AstIds.
            None
        })
    }
}

#[derive(Debug)]
pub enum ProcMacroClientError {
    Cancelled { reason: String },
    Io(std::io::Error),
    Protocol(String),
    Eof,
}

#[derive(Debug)]
pub enum ProcMacroPanicMarker {
    Cancelled { reason: String },
    Internal { reason: String },
}

pub type ProcMacroClientHandle<'a> = &'a mut (dyn ProcMacroClientInterface + Sync + Send);

pub trait ProcMacroClientInterface {
    fn file(&mut self, file_id: span::FileId) -> String;
    fn source_text(&mut self, span: Span) -> Option<String>;
    fn local_file(&mut self, file_id: span::FileId) -> Option<String>;
    /// Line and column are 1-based.
    fn line_column(&mut self, span: Span) -> Option<(u32, u32)>;

    fn byte_range(&mut self, span: Span) -> Range<usize>;
}

const EXPANDER_STACK_SIZE: usize = 8 * 1024 * 1024;

pub enum ExpandError {
    Panic(PanicMessage),
    Cancelled { reason: Option<String> },
    Internal { reason: Option<String> },
}

impl ExpandError {
    pub fn into_string(self) -> Option<String> {
        match self {
            ExpandError::Panic(panic_message) => panic_message.into_string(),
            ExpandError::Cancelled { reason } => reason,
            ExpandError::Internal { reason } => reason,
        }
    }
}

impl ProcMacroSrv<'_> {
    pub fn expand<S: ProcMacroSrvSpan>(
        &self,
        lib: impl AsRef<Utf8Path>,
        env: &[(String, String)],
        current_dir: Option<impl AsRef<Path>>,
        macro_name: &str,
        macro_body: token_stream::TokenStream<S>,
        attribute: Option<token_stream::TokenStream<S>>,
        def_site: S,
        call_site: S,
        mixed_site: S,
        callback: Option<ProcMacroClientHandle<'_>>,
    ) -> Result<token_stream::TokenStream<S>, ExpandError> {
        let snapped_env = self.env;
        let expander = self.expander(lib.as_ref()).map_err(|err| ExpandError::Internal {
            reason: Some(format!("failed to load macro: {err}")),
        })?;

        let prev_env = EnvChange::apply(snapped_env, env, current_dir.as_ref().map(<_>::as_ref));

        // Note, we spawn a new thread here so that thread locals allocation don't accumulate (this
        // includes the proc-macro symbol interner)
        let result = thread::scope(|s| {
            let thread = thread::Builder::new()
                .stack_size(EXPANDER_STACK_SIZE)
                .name(macro_name.to_owned())
                .spawn_scoped(s, move || {
                    expander.expand(
                        macro_name, macro_body, attribute, def_site, call_site, mixed_site,
                        callback,
                    )
                });
            match thread.unwrap().join() {
                Ok(res) => res.map_err(ExpandError::Panic),

                Err(payload) => {
                    if let Some(marker) = payload.downcast_ref::<ProcMacroPanicMarker>() {
                        return match marker {
                            ProcMacroPanicMarker::Cancelled { reason } => {
                                Err(ExpandError::Cancelled { reason: Some(reason.clone()) })
                            }
                            ProcMacroPanicMarker::Internal { reason } => {
                                Err(ExpandError::Internal { reason: Some(reason.clone()) })
                            }
                        };
                    }

                    std::panic::resume_unwind(payload)
                }
            }
        });
        prev_env.rollback();

        result
    }

    pub fn list_macros(
        &self,
        dylib_path: &Utf8Path,
    ) -> Result<Vec<(String, ProcMacroKind)>, String> {
        let expander = self.expander(dylib_path)?;
        Ok(expander.list_macros().map(|(k, v)| (k.to_owned(), v)).collect())
    }

    fn expander(&self, path: &Utf8Path) -> Result<Arc<dylib::Expander>, String> {
        let expander = || {
            let expander = dylib::Expander::new(&self.temp_dir, path)
                .map_err(|err| format!("Cannot create expander for {path}: {err}",));
            expander.map(Arc::new)
        };

        Ok(
            match self
                .expanders
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .entry(path.to_path_buf())
            {
                Entry::Vacant(v) => v.insert(expander()?).clone(),
                Entry::Occupied(mut e) => {
                    let time = fs::metadata(path).and_then(|it| it.modified()).ok();
                    if Some(e.get().modified_time()) != time {
                        e.insert(expander()?);
                    }
                    e.get().clone()
                }
            },
        )
    }
}

pub trait ProcMacroSrvSpan: Copy + Send + Sync {
    type Server<'a>: rustc_proc_macro::bridge::server::Server<
            TokenStream = crate::token_stream::TokenStream<Self>,
        >;
    fn make_server<'a>(
        call_site: Self,
        def_site: Self,
        mixed_site: Self,
        callback: Option<ProcMacroClientHandle<'a>>,
    ) -> Self::Server<'a>;
}

impl ProcMacroSrvSpan for SpanId {
    type Server<'a> = server_impl::token_id::SpanIdServer<'a>;

    fn make_server<'a>(
        call_site: Self,
        def_site: Self,
        mixed_site: Self,
        callback: Option<ProcMacroClientHandle<'a>>,
    ) -> Self::Server<'a> {
        Self::Server {
            call_site,
            def_site,
            mixed_site,
            callback,
            tracked_env_vars: Default::default(),
            tracked_paths: Default::default(),
        }
    }
}

impl ProcMacroSrvSpan for Span {
    type Server<'a> = server_impl::rust_analyzer_span::RaSpanServer<'a>;
    fn make_server<'a>(
        call_site: Self,
        def_site: Self,
        mixed_site: Self,
        callback: Option<ProcMacroClientHandle<'a>>,
    ) -> Self::Server<'a> {
        Self::Server {
            call_site,
            def_site,
            mixed_site,
            callback,
            tracked_env_vars: Default::default(),
            tracked_paths: Default::default(),
        }
    }
}

#[derive(Debug, Clone)]
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

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct EnvChange<'snap> {
    changed_vars: Vec<&'snap str>,
    prev_working_dir: Option<PathBuf>,
    snap: &'snap EnvSnapshot,
    _guard: std::sync::MutexGuard<'snap, ()>,
}

impl<'snap> EnvChange<'snap> {
    fn apply(
        snap: &'snap EnvSnapshot,
        new_vars: &'snap [(String, String)],
        current_dir: Option<&Path>,
    ) -> EnvChange<'snap> {
        let guard = ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
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
                .iter()
                .map(|(k, v)| {
                    // SAFETY: We have acquired the environment lock
                    unsafe { env::set_var(k, v) };
                    &**k
                })
                .collect(),
            prev_working_dir,
            _guard: guard,
        }
    }

    fn rollback(self) {}
}

impl Drop for EnvChange<'_> {
    fn drop(&mut self) {
        for name in self.changed_vars.drain(..) {
            // SAFETY: We have acquired the environment lock
            unsafe {
                match self.snap.vars.get::<std::ffi::OsStr>(name.as_ref()) {
                    Some(prev_val) => env::set_var(name, prev_val),
                    None => env::remove_var(name),
                }
            }
        }

        if let Some(dir) = &self.prev_working_dir
            && let Err(err) = std::env::set_current_dir(dir)
        {
            eprintln!(
                "Failed to set the current working dir to {}. Error: {:?}",
                dir.display(),
                err
            )
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub fn proc_macro_test_dylib_path() -> paths::Utf8PathBuf {
    proc_macro_test::PROC_MACRO_TEST_LOCATION.into()
}
