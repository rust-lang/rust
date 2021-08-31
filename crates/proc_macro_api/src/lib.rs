//! Client-side Proc-Macro crate
//!
//! We separate proc-macro expanding logic to an extern program to allow
//! different implementations (e.g. wasm or dylib loading). And this crate
//! is used to provide basic infrastructure for communication between two
//! processes: Client (RA itself), Server (the external program)

pub mod msg;
mod process;
mod rpc;
mod version;

use paths::{AbsPath, AbsPathBuf};
use std::{
    ffi::OsStr,
    io,
    sync::{Arc, Mutex},
};

use tt::{SmolStr, Subtree};

use crate::process::ProcMacroProcessSrv;

pub use rpc::{
    flat::FlatTree, ExpansionResult, ExpansionTask, ListMacrosResult, ListMacrosTask, ProcMacroKind,
};
pub use version::{read_dylib_info, RustCInfo};

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

/// A handle to a specific macro (a `#[proc_macro]` annotated function).
///
/// It exists withing a context of a specific [`ProcMacroProcess`] -- currently
/// we share a single expander process for all macros.
#[derive(Debug, Clone)]
pub struct ProcMacro {
    process: Arc<Mutex<ProcMacroProcessSrv>>,
    dylib_path: AbsPathBuf,
    name: SmolStr,
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

impl ProcMacroServer {
    /// Spawns an external process as the proc macro server and returns a client connected to it.
    pub fn spawn(
        process_path: AbsPathBuf,
        args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    ) -> io::Result<ProcMacroServer> {
        let process = ProcMacroProcessSrv::run(process_path, args)?;
        Ok(ProcMacroServer { process: Arc::new(Mutex::new(process)) })
    }

    pub fn load_dylib(&self, dylib_path: &AbsPath) -> Vec<ProcMacro> {
        let _p = profile::span("ProcMacroClient::by_dylib_path");
        match version::read_dylib_info(dylib_path) {
            Ok(info) => {
                if info.version.0 < 1 || info.version.1 < 47 {
                    eprintln!("proc-macro {} built by {:#?} is not supported by Rust Analyzer, please update your rust version.", dylib_path.display(), info);
                }
            }
            Err(err) => {
                eprintln!(
                    "proc-macro {} failed to find the given version. Reason: {}",
                    dylib_path.display(),
                    err
                );
            }
        }

        let macros = match self
            .process
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .find_proc_macros(dylib_path)
        {
            Err(err) => {
                eprintln!("Failed to find proc macros. Error: {:#?}", err);
                return vec![];
            }
            Ok(macros) => macros,
        };

        macros
            .into_iter()
            .map(|(name, kind)| ProcMacro {
                process: self.process.clone(),
                name: name.into(),
                kind,
                dylib_path: dylib_path.to_path_buf(),
            })
            .collect()
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
        subtree: &Subtree,
        attr: Option<&Subtree>,
        env: Vec<(String, String)>,
    ) -> Result<Subtree, tt::ExpansionError> {
        let task = ExpansionTask {
            macro_body: FlatTree::new(subtree),
            macro_name: self.name.to_string(),
            attributes: attr.map(FlatTree::new),
            lib: self.dylib_path.to_path_buf().into(),
            env,
        };

        let result: ExpansionResult = self
            .process
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .send_task(msg::Request::ExpansionMacro(task))?;
        Ok(result.expansion.to_subtree())
    }
}
