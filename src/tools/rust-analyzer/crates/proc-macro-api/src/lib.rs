//! Client-side Proc-Macro crate
//!
//! We separate proc-macro expanding logic to an extern program to allow
//! different implementations (e.g. wasm or dylib loading). And this crate
//! is used to provide basic infrastructure for communication between two
//! processes: Client (RA itself), Server (the external program)

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

pub mod msg;
mod process;
mod version;

use paths::AbsPathBuf;
use std::{
    ffi::OsStr,
    fmt, io,
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};

use ::tt::token_id as tt;

use crate::{
    msg::{ExpandMacro, FlatTree, PanicMessage},
    process::ProcMacroProcessSrv,
};

pub use version::{read_dylib_info, read_version, RustCInfo};

#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub enum ProcMacroKind {
    CustomDerive,
    FuncLike,
    Attr,
}

/// A handle to an external process which load dylibs with macros (.so or .dll)
/// and runs actual macro expansion functions.
#[derive(Debug)]
pub struct ProcMacroServer {
    /// Currently, the proc macro process expands all procedural macros sequentially.
    ///
    /// That means that concurrent salsa requests may block each other when expanding proc macros,
    /// which is unfortunate, but simple and good enough for the time being.
    ///
    /// Therefore, we just wrap the `ProcMacroProcessSrv` in a mutex here.
    process: Arc<Mutex<ProcMacroProcessSrv>>,
}

pub struct MacroDylib {
    path: AbsPathBuf,
}

impl MacroDylib {
    // FIXME: this is buggy due to TOCTOU, we should check the version in the
    // macro process instead.
    pub fn new(path: AbsPathBuf) -> io::Result<MacroDylib> {
        let _p = profile::span("MacroDylib::new");

        let info = version::read_dylib_info(&path)?;
        if info.version.0 < 1 || info.version.1 < 47 {
            let msg = format!("proc-macro {} built by {info:#?} is not supported by rust-analyzer, please update your Rust version.", path.display());
            return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
        }

        Ok(MacroDylib { path })
    }
}

/// A handle to a specific macro (a `#[proc_macro]` annotated function).
///
/// It exists within a context of a specific [`ProcMacroProcess`] -- currently
/// we share a single expander process for all macros.
#[derive(Debug, Clone)]
pub struct ProcMacro {
    process: Arc<Mutex<ProcMacroProcessSrv>>,
    dylib_path: AbsPathBuf,
    name: String,
    kind: ProcMacroKind,
}

impl Eq for ProcMacro {}
impl PartialEq for ProcMacro {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.kind == other.kind
            && self.dylib_path == other.dylib_path
            && Arc::ptr_eq(&self.process, &other.process)
    }
}

pub struct ServerError {
    pub message: String,
    pub io: Option<io::Error>,
}

impl fmt::Display for ServerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)?;
        if let Some(io) = &self.io {
            f.write_str(": ")?;
            io.fmt(f)?;
        }
        Ok(())
    }
}

pub struct MacroPanic {
    pub message: String,
}

impl ProcMacroServer {
    /// Spawns an external process as the proc macro server and returns a client connected to it.
    pub fn spawn(
        process_path: AbsPathBuf,
        args: impl IntoIterator<Item = impl AsRef<OsStr>> + Clone,
    ) -> io::Result<ProcMacroServer> {
        let process = ProcMacroProcessSrv::run(process_path, args)?;
        Ok(ProcMacroServer { process: Arc::new(Mutex::new(process)) })
    }

    pub fn load_dylib(&self, dylib: MacroDylib) -> Result<Vec<ProcMacro>, ServerError> {
        let _p = profile::span("ProcMacroClient::load_dylib");
        let macros =
            self.process.lock().unwrap_or_else(|e| e.into_inner()).find_proc_macros(&dylib.path)?;

        match macros {
            Ok(macros) => Ok(macros
                .into_iter()
                .map(|(name, kind)| ProcMacro {
                    process: self.process.clone(),
                    name,
                    kind,
                    dylib_path: dylib.path.clone(),
                })
                .collect()),
            Err(message) => Err(ServerError { message, io: None }),
        }
    }
}

impl ProcMacro {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kind(&self) -> ProcMacroKind {
        self.kind
    }

    pub fn expand(
        &self,
        subtree: &tt::Subtree,
        attr: Option<&tt::Subtree>,
        env: Vec<(String, String)>,
    ) -> Result<Result<tt::Subtree, PanicMessage>, ServerError> {
        let current_dir = env
            .iter()
            .find(|(name, _)| name == "CARGO_MANIFEST_DIR")
            .map(|(_, value)| value.clone());

        let task = ExpandMacro {
            macro_body: FlatTree::new(subtree),
            macro_name: self.name.to_string(),
            attributes: attr.map(FlatTree::new),
            lib: self.dylib_path.to_path_buf().into(),
            env,
            current_dir,
        };

        let request = msg::Request::ExpandMacro(task);
        let response = self.process.lock().unwrap_or_else(|e| e.into_inner()).send_task(request)?;
        match response {
            msg::Response::ExpandMacro(it) => Ok(it.map(FlatTree::to_subtree)),
            msg::Response::ListMacros(..) | msg::Response::ApiVersionCheck(..) => {
                Err(ServerError { message: "unexpected response".to_string(), io: None })
            }
        }
    }
}
