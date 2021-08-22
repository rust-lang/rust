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

pub use rpc::{ExpansionResult, ExpansionTask, ListMacrosResult, ListMacrosTask, ProcMacroKind};
pub use version::{read_dylib_info, RustCInfo};

#[derive(Debug, Clone)]
pub struct ProcMacroProcessExpander {
    process: Arc<Mutex<ProcMacroProcessSrv>>,
    dylib_path: AbsPathBuf,
    name: SmolStr,
    kind: ProcMacroKind,
}

impl Eq for ProcMacroProcessExpander {}
impl PartialEq for ProcMacroProcessExpander {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.kind == other.kind
            && self.dylib_path == other.dylib_path
            && Arc::ptr_eq(&self.process, &other.process)
    }
}

impl ProcMacroProcessExpander {
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
            macro_body: subtree.clone(),
            macro_name: self.name.to_string(),
            attributes: attr.cloned(),
            lib: self.dylib_path.to_path_buf(),
            env,
        };

        let result: ExpansionResult = self
            .process
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .send_task(msg::Request::ExpansionMacro(task))?;
        Ok(result.expansion)
    }
}

#[derive(Debug)]
pub struct ProcMacroClient {
    /// Currently, the proc macro process expands all procedural macros sequentially.
    ///
    /// That means that concurrent salsa requests may block each other when expanding proc macros,
    /// which is unfortunate, but simple and good enough for the time being.
    ///
    /// Therefore, we just wrap the `ProcMacroProcessSrv` in a mutex here.
    process: Arc<Mutex<ProcMacroProcessSrv>>,
}

impl ProcMacroClient {
    /// Spawns an external process as the proc macro server and returns a client connected to it.
    pub fn extern_process(
        process_path: AbsPathBuf,
        args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    ) -> io::Result<ProcMacroClient> {
        let process = ProcMacroProcessSrv::run(process_path, args)?;
        Ok(ProcMacroClient { process: Arc::new(Mutex::new(process)) })
    }

    pub fn by_dylib_path(&self, dylib_path: &AbsPath) -> Vec<ProcMacroProcessExpander> {
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
            .map(|(name, kind)| ProcMacroProcessExpander {
                process: self.process.clone(),
                name: name.into(),
                kind,
                dylib_path: dylib_path.to_path_buf(),
            })
            .collect()
    }
}
