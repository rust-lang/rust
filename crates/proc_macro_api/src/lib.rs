//! Client-side Proc-Macro crate
//!
//! We separate proc-macro expanding logic to an extern program to allow
//! different implementations (e.g. wasm or dylib loading). And this crate
//! is used to provide basic infrastructure for communication between two
//! processes: Client (RA itself), Server (the external program)

mod rpc;
mod process;
pub mod msg;

use std::{
    ffi::OsStr,
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

use tt::{SmolStr, Subtree};

use crate::process::{ProcMacroProcessSrv, ProcMacroProcessThread};

pub use rpc::{ExpansionResult, ExpansionTask, ListMacrosResult, ListMacrosTask, ProcMacroKind};

#[derive(Debug, Clone)]
pub struct ProcMacroProcessExpander {
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

impl tt::TokenExpander for ProcMacroProcessExpander {
    fn expand(
        &self,
        subtree: &Subtree,
        attr: Option<&Subtree>,
    ) -> Result<Subtree, tt::ExpansionError> {
        let task = ExpansionTask {
            macro_body: subtree.clone(),
            macro_name: self.name.to_string(),
            attributes: attr.cloned(),
            lib: self.dylib_path.to_path_buf(),
        };

        let result: ExpansionResult = self.process.send_task(msg::Request::ExpansionMacro(task))?;
        Ok(result.expansion)
    }
}

#[derive(Debug)]
enum ProcMacroClientKind {
    Process { process: Arc<ProcMacroProcessSrv>, thread: ProcMacroProcessThread },
    Dummy,
}

#[derive(Debug)]
pub struct ProcMacroClient {
    kind: ProcMacroClientKind,
}

impl ProcMacroClient {
    pub fn extern_process(
        process_path: PathBuf,
        args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    ) -> io::Result<ProcMacroClient> {
        let (thread, process) = ProcMacroProcessSrv::run(process_path, args)?;
        Ok(ProcMacroClient {
            kind: ProcMacroClientKind::Process { process: Arc::new(process), thread },
        })
    }

    pub fn dummy() -> ProcMacroClient {
        ProcMacroClient { kind: ProcMacroClientKind::Dummy }
    }

    pub fn by_dylib_path(&self, dylib_path: &Path) -> Vec<(SmolStr, Arc<dyn tt::TokenExpander>)> {
        match &self.kind {
            ProcMacroClientKind::Dummy => vec![],
            ProcMacroClientKind::Process { process, .. } => {
                let macros = match process.find_proc_macros(dylib_path) {
                    Err(err) => {
                        eprintln!("Failed to find proc macros. Error: {:#?}", err);
                        return vec![];
                    }
                    Ok(macros) => macros,
                };

                macros
                    .into_iter()
                    .filter_map(|(name, kind)| {
                        match kind {
                            ProcMacroKind::CustomDerive | ProcMacroKind::FuncLike => {
                                let name = SmolStr::new(&name);
                                let expander: Arc<dyn tt::TokenExpander> =
                                    Arc::new(ProcMacroProcessExpander {
                                        process: process.clone(),
                                        name: name.clone(),
                                        dylib_path: dylib_path.into(),
                                    });
                                Some((name, expander))
                            }
                            // FIXME: Attribute macro are currently unsupported.
                            ProcMacroKind::Attr => None,
                        }
                    })
                    .collect()
            }
        }
    }
}
