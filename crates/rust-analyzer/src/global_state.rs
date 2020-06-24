//! The context or environment in which the language server functions. In our
//! server implementation this is know as the `WorldState`.
//!
//! Each tick provides an immutable snapshot of the state as `WorldSnapshot`.

use std::{convert::TryFrom, sync::Arc};

use crossbeam_channel::{unbounded, Receiver};
use lsp_types::Url;
use parking_lot::RwLock;
use ra_db::{CrateId, SourceRoot, VfsPath};
use ra_flycheck::{Flycheck, FlycheckConfig};
use ra_ide::{Analysis, AnalysisChange, AnalysisHost, CrateGraph, FileId};
use ra_project_model::{CargoWorkspace, ProcMacroClient, ProjectWorkspace, Target};
use stdx::format_to;
use vfs::{file_set::FileSetConfig, loader::Handle, AbsPath, AbsPathBuf};

use crate::{
    config::{Config, FilesWatcher},
    diagnostics::{CheckFixes, DiagnosticCollection},
    from_proto,
    line_endings::LineEndings,
    main_loop::ReqQueue,
    request_metrics::{LatestRequests, RequestMetrics},
    to_proto::url_from_abs_path,
    Result,
};
use rustc_hash::{FxHashMap, FxHashSet};

fn create_flycheck(workspaces: &[ProjectWorkspace], config: &FlycheckConfig) -> Option<Flycheck> {
    // FIXME: Figure out the multi-workspace situation
    workspaces.iter().find_map(|w| match w {
        ProjectWorkspace::Cargo { cargo, .. } => {
            let cargo_project_root = cargo.workspace_root().to_path_buf();
            Some(Flycheck::new(config.clone(), cargo_project_root.into()))
        }
        ProjectWorkspace::Json { .. } => {
            log::warn!("Cargo check watching only supported for cargo workspaces, disabling");
            None
        }
    })
}

#[derive(Eq, PartialEq)]
pub(crate) enum Status {
    Loading,
    Ready,
}

impl Default for Status {
    fn default() -> Self {
        Status::Loading
    }
}

/// `GlobalState` is the primary mutable state of the language server
///
/// The most interesting components are `vfs`, which stores a consistent
/// snapshot of the file systems, and `analysis_host`, which stores our
/// incremental salsa database.
pub(crate) struct GlobalState {
    pub(crate) config: Config,
    pub(crate) workspaces: Arc<Vec<ProjectWorkspace>>,
    pub(crate) analysis_host: AnalysisHost,
    pub(crate) loader: Box<dyn vfs::loader::Handle>,
    pub(crate) task_receiver: Receiver<vfs::loader::Message>,
    pub(crate) flycheck: Option<Flycheck>,
    pub(crate) diagnostics: DiagnosticCollection,
    pub(crate) mem_docs: FxHashSet<VfsPath>,
    pub(crate) vfs: Arc<RwLock<(vfs::Vfs, FxHashMap<FileId, LineEndings>)>>,
    pub(crate) status: Status,
    pub(crate) req_queue: ReqQueue,
    pub(crate) latest_requests: Arc<RwLock<LatestRequests>>,
    source_root_config: SourceRootConfig,
    _proc_macro_client: ProcMacroClient,
}

/// An immutable snapshot of the world's state at a point in time.
pub(crate) struct GlobalStateSnapshot {
    pub(crate) config: Config,
    pub(crate) workspaces: Arc<Vec<ProjectWorkspace>>,
    pub(crate) analysis: Analysis,
    pub(crate) check_fixes: CheckFixes,
    pub(crate) latest_requests: Arc<RwLock<LatestRequests>>,
    vfs: Arc<RwLock<(vfs::Vfs, FxHashMap<FileId, LineEndings>)>>,
}

impl GlobalState {
    pub(crate) fn new(
        workspaces: Vec<ProjectWorkspace>,
        lru_capacity: Option<usize>,
        config: Config,
        req_queue: ReqQueue,
    ) -> GlobalState {
        let mut change = AnalysisChange::new();

        let project_folders = ProjectFolders::new(&workspaces);

        let (task_sender, task_receiver) = unbounded::<vfs::loader::Message>();
        let mut vfs = vfs::Vfs::default();

        let proc_macro_client = match &config.proc_macro_srv {
            None => ProcMacroClient::dummy(),
            Some((path, args)) => match ProcMacroClient::extern_process(path.into(), args) {
                Ok(it) => it,
                Err(err) => {
                    log::error!(
                        "Failed to run ra_proc_macro_srv from path {}, error: {:?}",
                        path.display(),
                        err
                    );
                    ProcMacroClient::dummy()
                }
            },
        };

        let mut loader = {
            let loader = vfs_notify::LoaderHandle::spawn(Box::new(move |msg| {
                task_sender.send(msg).unwrap()
            }));
            Box::new(loader)
        };
        let watch = match config.files.watcher {
            FilesWatcher::Client => vec![],
            FilesWatcher::Notify => project_folders.watch,
        };
        loader.set_config(vfs::loader::Config { load: project_folders.load, watch });

        // Create crate graph from all the workspaces
        let mut crate_graph = CrateGraph::default();
        let mut load = |path: &AbsPath| {
            let contents = loader.load_sync(path);
            let path = vfs::VfsPath::from(path.to_path_buf());
            vfs.set_file_contents(path.clone(), contents);
            vfs.file_id(&path)
        };
        for ws in workspaces.iter() {
            crate_graph.extend(ws.to_crate_graph(
                config.cargo.target.as_deref(),
                &proc_macro_client,
                &mut load,
            ));
        }
        change.set_crate_graph(crate_graph);

        let flycheck = config.check.as_ref().and_then(|c| create_flycheck(&workspaces, c));

        let mut analysis_host = AnalysisHost::new(lru_capacity);
        analysis_host.apply_change(change);
        let mut res = GlobalState {
            config,
            workspaces: Arc::new(workspaces),
            analysis_host,
            loader,
            task_receiver,
            flycheck,
            diagnostics: Default::default(),
            mem_docs: FxHashSet::default(),
            vfs: Arc::new(RwLock::new((vfs, FxHashMap::default()))),
            status: Status::default(),
            req_queue,
            latest_requests: Default::default(),
            source_root_config: project_folders.source_root_config,
            _proc_macro_client: proc_macro_client,
        };
        res.process_changes();
        res
    }

    pub fn update_configuration(&mut self, config: Config) {
        self.analysis_host.update_lru_capacity(config.lru_capacity);
        if config.check != self.config.check {
            self.flycheck =
                config.check.as_ref().and_then(|it| create_flycheck(&self.workspaces, it));
        }

        self.config = config;
    }

    pub fn process_changes(&mut self) -> bool {
        let change = {
            let mut change = AnalysisChange::new();
            let (vfs, line_endings_map) = &mut *self.vfs.write();
            let changed_files = vfs.take_changes();
            if changed_files.is_empty() {
                return false;
            }

            let fs_op = changed_files.iter().any(|it| it.is_created_or_deleted());
            if fs_op {
                let roots = self.source_root_config.partition(&vfs);
                change.set_roots(roots)
            }

            for file in changed_files {
                let text = if file.exists() {
                    let bytes = vfs.file_contents(file.file_id).to_vec();
                    match String::from_utf8(bytes).ok() {
                        Some(text) => {
                            let (text, line_endings) = LineEndings::normalize(text);
                            line_endings_map.insert(file.file_id, line_endings);
                            Some(Arc::new(text))
                        }
                        None => None,
                    }
                } else {
                    None
                };
                change.change_file(file.file_id, text);
            }
            change
        };

        self.analysis_host.apply_change(change);
        true
    }

    pub fn snapshot(&self) -> GlobalStateSnapshot {
        GlobalStateSnapshot {
            config: self.config.clone(),
            workspaces: Arc::clone(&self.workspaces),
            analysis: self.analysis_host.analysis(),
            vfs: Arc::clone(&self.vfs),
            latest_requests: Arc::clone(&self.latest_requests),
            check_fixes: Arc::clone(&self.diagnostics.check_fixes),
        }
    }

    pub fn maybe_collect_garbage(&mut self) {
        self.analysis_host.maybe_collect_garbage()
    }

    pub fn collect_garbage(&mut self) {
        self.analysis_host.collect_garbage()
    }

    pub(crate) fn complete_request(&mut self, request: RequestMetrics) {
        self.latest_requests.write().record(request)
    }
}

impl GlobalStateSnapshot {
    pub(crate) fn analysis(&self) -> &Analysis {
        &self.analysis
    }

    pub(crate) fn url_to_file_id(&self, url: &Url) -> Result<FileId> {
        let path = from_proto::abs_path(url)?;
        let path = path.into();
        let res =
            self.vfs.read().0.file_id(&path).ok_or_else(|| format!("file not found: {}", path))?;
        Ok(res)
    }

    pub(crate) fn file_id_to_url(&self, id: FileId) -> Url {
        file_id_to_url(&self.vfs.read().0, id)
    }

    pub(crate) fn file_line_endings(&self, id: FileId) -> LineEndings {
        self.vfs.read().1[&id]
    }

    pub(crate) fn anchored_path(&self, file_id: FileId, path: &str) -> Url {
        let mut base = self.vfs.read().0.file_path(file_id);
        base.pop();
        let path = base.join(path);
        let path = path.as_path().unwrap();
        url_from_abs_path(&path)
    }

    pub(crate) fn cargo_target_for_crate_root(
        &self,
        crate_id: CrateId,
    ) -> Option<(&CargoWorkspace, Target)> {
        let file_id = self.analysis().crate_root(crate_id).ok()?;
        let path = self.vfs.read().0.file_path(file_id);
        let path = path.as_path()?;
        self.workspaces.iter().find_map(|ws| match ws {
            ProjectWorkspace::Cargo { cargo, .. } => {
                cargo.target_by_root(&path).map(|it| (cargo, it))
            }
            ProjectWorkspace::Json { .. } => None,
        })
    }

    pub(crate) fn status(&self) -> String {
        let mut buf = String::new();
        if self.workspaces.is_empty() {
            buf.push_str("no workspaces\n")
        } else {
            buf.push_str("workspaces:\n");
            for w in self.workspaces.iter() {
                format_to!(buf, "{} packages loaded\n", w.n_packages());
            }
        }
        buf.push_str("\nanalysis:\n");
        buf.push_str(
            &self
                .analysis
                .status()
                .unwrap_or_else(|_| "Analysis retrieval was cancelled".to_owned()),
        );
        buf
    }
}

pub(crate) fn file_id_to_url(vfs: &vfs::Vfs, id: FileId) -> Url {
    let path = vfs.file_path(id);
    let path = path.as_path().unwrap();
    url_from_abs_path(&path)
}

#[derive(Default)]
pub(crate) struct ProjectFolders {
    pub(crate) load: Vec<vfs::loader::Entry>,
    pub(crate) watch: Vec<usize>,
    pub(crate) source_root_config: SourceRootConfig,
}

impl ProjectFolders {
    pub(crate) fn new(workspaces: &[ProjectWorkspace]) -> ProjectFolders {
        let mut res = ProjectFolders::default();
        let mut fsc = FileSetConfig::builder();
        let mut local_filesets = vec![];

        for root in workspaces.iter().flat_map(|it| it.to_roots()) {
            let path = root.path().to_owned();

            let mut file_set_roots: Vec<VfsPath> = vec![];

            let entry = if root.is_member() {
                vfs::loader::Entry::local_cargo_package(path.to_path_buf())
            } else {
                vfs::loader::Entry::cargo_package_dependency(path.to_path_buf())
            };
            res.load.push(entry);
            if root.is_member() {
                res.watch.push(res.load.len() - 1);
            }

            if let Some(out_dir) = root.out_dir() {
                let out_dir = AbsPathBuf::try_from(out_dir.to_path_buf()).unwrap();
                res.load.push(vfs::loader::Entry::rs_files_recursively(out_dir.clone()));
                if root.is_member() {
                    res.watch.push(res.load.len() - 1);
                }
                file_set_roots.push(out_dir.into());
            }
            file_set_roots.push(path.to_path_buf().into());

            if root.is_member() {
                local_filesets.push(fsc.len());
            }
            fsc.add_file_set(file_set_roots)
        }

        let fsc = fsc.build();
        res.source_root_config = SourceRootConfig { fsc, local_filesets };

        res
    }
}

#[derive(Default, Debug)]
pub(crate) struct SourceRootConfig {
    pub(crate) fsc: FileSetConfig,
    pub(crate) local_filesets: Vec<usize>,
}

impl SourceRootConfig {
    pub(crate) fn partition(&self, vfs: &vfs::Vfs) -> Vec<SourceRoot> {
        self.fsc
            .partition(vfs)
            .into_iter()
            .enumerate()
            .map(|(idx, file_set)| {
                let is_local = self.local_filesets.contains(&idx);
                if is_local {
                    SourceRoot::new_local(file_set)
                } else {
                    SourceRoot::new_library(file_set)
                }
            })
            .collect()
    }
}
