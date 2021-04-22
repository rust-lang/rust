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

use base_db::{Env, ProcMacro};
use std::{
    ffi::OsStr,
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

use tt::{SmolStr, Subtree};

use crate::process::{ProcMacroProcessSrv, ProcMacroProcessThread};

pub use rpc::{ExpansionResult, ExpansionTask, ListMacrosResult, ListMacrosTask, ProcMacroKind};
pub use version::{read_dylib_info, RustCInfo};

#[derive(Debug, Clone)]
struct ProcMacroProcessExpander {
    process: Arc<ProcMacroProcessSrv>,
    dylib_path: PathBuf,
    name: SmolStr,
}

impl Eq for ProcMacroProcessExpander {}
impl PartialEq for ProcMacroProcessExpander {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.dylib_path == other.dylib_path
            && Arc::ptr_eq(&self.process, &other.process)
    }
}

impl base_db::ProcMacroExpander for ProcMacroProcessExpander {
    fn expand(
        &self,
        subtree: &Subtree,
        attr: Option<&Subtree>,
        env: &Env,
    ) -> Result<Subtree, tt::ExpansionError> {
        let task = ExpansionTask {
            macro_body: subtree.clone(),
            macro_name: self.name.to_string(),
            attributes: attr.cloned(),
            lib: self.dylib_path.to_path_buf(),
            env: env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
        };

        let result: ExpansionResult = self.process.send_task(msg::Request::ExpansionMacro(task))?;
        Ok(result.expansion)
    }
}

#[derive(Debug)]
pub struct ProcMacroClient {
    process: Arc<ProcMacroProcessSrv>,
    thread: ProcMacroProcessThread,
}

impl ProcMacroClient {
    pub fn extern_process(
        process_path: PathBuf,
        args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    ) -> io::Result<ProcMacroClient> {
        let (thread, process) = ProcMacroProcessSrv::run(process_path, args)?;
        Ok(ProcMacroClient { process: Arc::new(process), thread })
    }

    pub fn by_dylib_path(&self, dylib_path: &Path) -> Vec<ProcMacro> {
        let _p = profile::span("ProcMacroClient::by_dylib_path");
        match version::read_dylib_info(dylib_path) {
            Ok(info) => {
                if info.version.0 < 1 || info.version.1 < 47 {
                    eprintln!("proc-macro {} built by {:#?} is not supported by Rust Analyzer, please update your rust version.", dylib_path.to_string_lossy(), info);
                }
            }
            Err(err) => {
                eprintln!(
                    "proc-macro {} failed to find the given version. Reason: {}",
                    dylib_path.to_string_lossy(),
                    err
                );
            }
        }

        let macros = match self.process.find_proc_macros(dylib_path) {
            Err(err) => {
                eprintln!("Failed to find proc macros. Error: {:#?}", err);
                return vec![];
            }
            Ok(macros) => macros,
        };

        macros
            .into_iter()
            .map(|(name, kind)| {
                let name = SmolStr::new(&name);
                let kind = match kind {
                    ProcMacroKind::CustomDerive => base_db::ProcMacroKind::CustomDerive,
                    ProcMacroKind::FuncLike => base_db::ProcMacroKind::FuncLike,
                    ProcMacroKind::Attr => base_db::ProcMacroKind::Attr,
                };
                let expander = Arc::new(ProcMacroProcessExpander {
                    process: self.process.clone(),
                    name: name.clone(),
                    dylib_path: dylib_path.into(),
                });

                ProcMacro { name, kind, expander }
            })
            .collect()
    }
}
