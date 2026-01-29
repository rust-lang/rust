//! A pool of proc-macro server processes
use std::sync::Arc;

use crate::{
    MacroDylib, ProcMacro, ServerError, bidirectional_protocol::SubCallback,
    process::ProcMacroServerProcess,
};

#[derive(Debug, Clone)]
pub(crate) struct ProcMacroServerPool {
    workers: Arc<[ProcMacroServerProcess]>,
    version: u32,
}

impl ProcMacroServerPool {
    pub(crate) fn new(workers: Vec<ProcMacroServerProcess>) -> Self {
        let version = workers[0].version();
        Self { workers: workers.into(), version }
    }
}

impl ProcMacroServerPool {
    pub(crate) fn exited(&self) -> Option<&ServerError> {
        for worker in &*self.workers {
            worker.exited()?;
        }
        self.workers[0].exited()
    }

    pub(crate) fn pick_process(&self) -> Result<&ProcMacroServerProcess, ServerError> {
        let mut best: Option<&ProcMacroServerProcess> = None;
        let mut best_load = u32::MAX;

        for w in self.workers.iter().filter(|w| w.exited().is_none()) {
            let load = w.number_of_active_req();

            if load == 0 {
                return Ok(w);
            }

            if load < best_load {
                best = Some(w);
                best_load = load;
            }
        }

        best.ok_or_else(|| ServerError {
            message: "all proc-macro server workers have exited".into(),
            io: None,
        })
    }

    pub(crate) fn load_dylib(
        &self,
        dylib: &MacroDylib,
        callback: Option<SubCallback<'_>>,
    ) -> Result<Vec<ProcMacro>, ServerError> {
        let _span = tracing::info_span!("ProcMacroServer::load_dylib").entered();

        let dylib_path = Arc::new(dylib.path.clone());
        let dylib_last_modified =
            std::fs::metadata(dylib_path.as_path()).ok().and_then(|m| m.modified().ok());

        let (first, rest) = self.workers.split_first().expect("worker pool must not be empty");

        let macros = first
            .find_proc_macros(&dylib.path, callback)?
            .map_err(|e| ServerError { message: e, io: None })?;

        for worker in rest {
            worker
                .find_proc_macros(&dylib.path, callback)?
                .map_err(|e| ServerError { message: e, io: None })?;
        }

        Ok(macros
            .into_iter()
            .map(|(name, kind)| ProcMacro {
                pool: self.clone(),
                name: name.into(),
                kind,
                dylib_path: dylib_path.clone(),
                dylib_last_modified,
            })
            .collect())
    }

    pub(crate) fn version(&self) -> u32 {
        self.version
    }
}
