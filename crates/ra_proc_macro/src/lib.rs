//! Client-side Proc-Macro crate
//!
//! We separate proc-macro expanding logic to an extern program to allow
//! different implementations (e.g. wasm or dylib loading). And this crate
//! is used to provide basic infrastructure  for communication between two
//! processes: Client (RA itself), Server (the external program)

use ra_mbe::ExpandError;
use ra_tt::Subtree;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

trait ProcMacroExpander: std::fmt::Debug + Send + Sync + std::panic::RefUnwindSafe {
    fn custom_derive(&self, subtree: &Subtree, derive_name: &str) -> Result<Subtree, ExpandError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcMacroProcessExpander {
    process: Arc<ProcMacroProcessSrv>,
}

impl ProcMacroExpander for ProcMacroProcessExpander {
    fn custom_derive(
        &self,
        _subtree: &Subtree,
        _derive_name: &str,
    ) -> Result<Subtree, ExpandError> {
        // FIXME: do nothing for now
        Ok(Subtree::default())
    }
}

#[derive(Debug, Clone)]
pub struct ProcMacro {
    expander: Arc<dyn ProcMacroExpander>,
    name: String,
}

impl Eq for ProcMacro {}
impl PartialEq for ProcMacro {
    fn eq(&self, other: &ProcMacro) -> bool {
        self.name == other.name && Arc::ptr_eq(&self.expander, &other.expander)
    }
}

impl ProcMacro {
    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn custom_derive(&self, subtree: &Subtree) -> Result<Subtree, ExpandError> {
        self.expander.custom_derive(subtree, &self.name)
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

    pub fn by_dylib_path(&self, _dylib_path: &Path) -> Vec<ProcMacro> {
        // FIXME: return empty for now
        vec![]
    }
}
