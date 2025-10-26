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
#![cfg_attr(not(feature = "sysroot-abi"), allow(unused_crate_dependencies))]
#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]
#![feature(proc_macro_internals, proc_macro_diagnostic, proc_macro_span)]
#![allow(unreachable_pub, internal_features, clippy::disallowed_types, clippy::print_stderr)]
#![deny(deprecated_safe, clippy::undocumented_unsafe_blocks)]

extern crate proc_macro;
#[cfg(feature = "in-rust-tree")]
extern crate rustc_driver as _;

#[cfg(not(feature = "in-rust-tree"))]
extern crate ra_ap_rustc_lexer as rustc_lexer;
#[cfg(feature = "in-rust-tree")]
extern crate rustc_lexer;

mod dylib;
mod server_impl;

use std::{
    collections::{HashMap, hash_map::Entry},
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, PoisonError},
    thread,
};

use paths::{Utf8Path, Utf8PathBuf};
use span::Span;
use temp_dir::TempDir;

use crate::server_impl::TokenStream;

pub use crate::server_impl::token_id::SpanId;

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
}

const EXPANDER_STACK_SIZE: usize = 8 * 1024 * 1024;

impl ProcMacroSrv<'_> {
    pub fn expand<S: ProcMacroSrvSpan>(
        &self,
        lib: impl AsRef<Utf8Path>,
        env: &[(String, String)],
        current_dir: Option<impl AsRef<Path>>,
        macro_name: &str,
        macro_body: tt::TopSubtree<S>,
        attribute: Option<tt::TopSubtree<S>>,
        def_site: S,
        call_site: S,
        mixed_site: S,
    ) -> Result<Vec<tt::TokenTree<S>>, PanicMessage> {
        let snapped_env = self.env;
        let expander = self.expander(lib.as_ref()).map_err(|err| PanicMessage {
            message: Some(format!("failed to load macro: {err}")),
        })?;

        let prev_env = EnvChange::apply(snapped_env, env, current_dir.as_ref().map(<_>::as_ref));

        // Note, we spawn a new thread here so that thread locals allocation don't accumulate (this
        // includes the proc-macro symbol interner)
        let result = thread::scope(|s| {
            let thread = thread::Builder::new()
                .stack_size(EXPANDER_STACK_SIZE)
                .name(macro_name.to_owned())
                .spawn_scoped(s, move || {
                    expander
                        .expand(
                            macro_name,
                            server_impl::TopSubtree(macro_body.0.into_vec()),
                            attribute.map(|it| server_impl::TopSubtree(it.0.into_vec())),
                            def_site,
                            call_site,
                            mixed_site,
                        )
                        .map(|tt| tt.0)
                });
            match thread.unwrap().join() {
                Ok(res) => res,
                Err(e) => std::panic::resume_unwind(e),
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

pub trait ProcMacroSrvSpan: Copy + Send {
    type Server: proc_macro::bridge::server::Server<TokenStream = TokenStream<Self>>;
    fn make_server(call_site: Self, def_site: Self, mixed_site: Self) -> Self::Server;
}

impl ProcMacroSrvSpan for SpanId {
    type Server = server_impl::token_id::SpanIdServer;

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
