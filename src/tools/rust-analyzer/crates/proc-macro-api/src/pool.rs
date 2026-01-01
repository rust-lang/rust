use std::sync::Arc;

use crate::{
    MacroDylib, ProcMacro, ServerError, bidirectional_protocol::SubCallback,
    process::ProcMacroServerProcess,
};

#[derive(Debug)]
pub(crate) struct ProcMacroServerPool {
    workers: Vec<Arc<ProcMacroServerProcess>>,
}

impl ProcMacroServerPool {
    pub(crate) fn new(workers: Vec<Arc<ProcMacroServerProcess>>) -> Self {
        Self { workers }
    }
}

impl ProcMacroServerPool {
    pub(crate) fn exited(&self) -> Option<&ServerError> {
        for worker in &self.workers {
            if let Some(e) = worker.exited() {
                return Some(e);
            }
        }
        None
    }

    pub(crate) fn load_dylib(
        &self,
        dylib: &MacroDylib,
        _callback: Option<SubCallback<'_>>,
    ) -> Result<Vec<ProcMacro>, ServerError> {
        let _p = tracing::info_span!("ProcMacroServer::load_dylib").entered();
        let mut all_macros = Vec::new();

        for worker in &self.workers {
            let dylib_path = Arc::new(dylib.path.clone());
            let dylib_last_modified = std::fs::metadata(dylib_path.as_path())
                .ok()
                .and_then(|metadata| metadata.modified().ok());
            let macros = worker.load_dylib(&dylib.path, None)?;

            for (name, kind) in macros {
                all_macros.push(ProcMacro {
                    process: worker.clone(),
                    name: name.into(),
                    kind,
                    dylib_path: Arc::new(dylib.path.clone()),
                    dylib_last_modified,
                });
            }
        }

        Ok(all_macros)
    }
}

pub(crate) fn default_pool_size() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(4)
}
