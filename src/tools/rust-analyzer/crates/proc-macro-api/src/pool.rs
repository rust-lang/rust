use std::sync::Arc;

use tt::Span;

use crate::{
    MacroDylib, ProcMacro, ServerError, bidirectional_protocol::SubCallback,
    process::ProcMacroServerProcess,
};

#[derive(Debug, Clone)]
pub(crate) struct ProcMacroServerPool {
    workers: Arc<[ProcMacroServerProcess]>,
}

impl ProcMacroServerPool {
    pub(crate) fn new(workers: Vec<ProcMacroServerProcess>) -> Self {
        Self { workers: workers.into() }
    }
}

impl ProcMacroServerPool {
    pub(crate) fn exited(&self) -> Option<&ServerError> {
        for worker in &*self.workers {
            worker.exited()?;
        }
        self.workers[0].exited()
    }

    fn pick_process(&self) -> &ProcMacroServerProcess {
        for workers in &*self.workers {
            if workers.can_use() {
                return workers;
            }
        }
        &self.workers[0]
    }

    pub(crate) fn load_dylib(
        &self,
        dylib: &MacroDylib,
        _callback: Option<SubCallback<'_>>,
    ) -> Result<Vec<ProcMacro>, ServerError> {
        let _p = tracing::info_span!("ProcMacroServer::load_dylib").entered();
        let mut all_macros = Vec::new();

        for worker in &*self.workers {
            let dylib_path = Arc::new(dylib.path.clone());
            let dylib_last_modified = std::fs::metadata(dylib_path.as_path())
                .ok()
                .and_then(|metadata| metadata.modified().ok());
            let macros = worker.find_proc_macros(&dylib.path, None)?.unwrap();

            for (name, kind) in macros {
                all_macros.push(ProcMacro {
                    process: self.clone(),
                    name: name.into(),
                    kind,
                    dylib_path: Arc::new(dylib.path.clone()),
                    dylib_last_modified,
                });
            }
        }

        Ok(all_macros)
    }

    pub(crate) fn expand(
        &self,
        proc_macro: &ProcMacro,
        subtree: tt::SubtreeView<'_>,
        attr: Option<tt::SubtreeView<'_>>,
        env: Vec<(String, String)>,
        def_site: Span,
        call_site: Span,
        mixed_site: Span,
        current_dir: String,
        callback: Option<SubCallback<'_>>,
    ) -> Result<Result<tt::TopSubtree, String>, ServerError> {
        let process = self.pick_process();

        let (mut subtree, mut attr) = (subtree, attr);
        let (mut subtree_changed, mut attr_changed);
        if proc_macro.needs_fixup_change(process) {
            subtree_changed = tt::TopSubtree::from_subtree(subtree);
            proc_macro.change_fixup_to_match_old_server(&mut subtree_changed);
            subtree = subtree_changed.view();

            if let Some(attr) = &mut attr {
                attr_changed = tt::TopSubtree::from_subtree(*attr);
                proc_macro.change_fixup_to_match_old_server(&mut attr_changed);
                *attr = attr_changed.view();
            }
        }

        process.expand(
            proc_macro,
            subtree,
            attr,
            env,
            def_site,
            call_site,
            mixed_site,
            current_dir,
            callback,
        )
    }
}

pub(crate) fn default_pool_size() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(4)
}
