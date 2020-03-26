//! Client-side Proc-Macro crate
//!
//! We separate proc-macro expanding logic to an extern program to allow
//! different implementations (e.g. wasm or dylib loading). And this crate
//! is used to provide basic infrastructure  for communication between two
//! processes: Client (RA itself), Server (the external program)

use ra_tt::{SmolStr, Subtree};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcMacroProcessExpander {
    process: Arc<ProcMacroProcessSrv>,
    name: SmolStr,
}

impl ra_tt::TokenExpander for ProcMacroProcessExpander {
    fn expand(
        &self,
        _subtree: &Subtree,
        _attr: Option<&Subtree>,
    ) -> Result<Subtree, ra_tt::ExpansionError> {
        // FIXME: do nothing for now
        Ok(Subtree::default())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcMacroProcessSrv {
    path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcMacroClient {
    Process { process: Arc<ProcMacroProcessSrv> },
    Dummy,
}

impl ProcMacroClient {
    pub fn extern_process(process_path: &Path) -> ProcMacroClient {
        let process = ProcMacroProcessSrv { path: process_path.into() };
        ProcMacroClient::Process { process: Arc::new(process) }
    }

    pub fn dummy() -> ProcMacroClient {
        ProcMacroClient::Dummy
    }

    pub fn by_dylib_path(
        &self,
        _dylib_path: &Path,
    ) -> Vec<(SmolStr, Arc<dyn ra_tt::TokenExpander>)> {
        // FIXME: return empty for now
        vec![]
    }
}
