//! Project loading & configuration updates.
//!
//! This is quite tricky. The main problem is time and changes -- there's no
//! fixed "project" rust-analyzer is working with, "current project" is itself
//! mutable state. For example, when the user edits `Cargo.toml` by adding a new
//! dependency, project model changes. What's more, switching project model is
//! not instantaneous -- it takes time to run `cargo metadata` and (for proc
//! macros) `cargo check`.
//!
//! The main guiding principle here is, as elsewhere in rust-analyzer,
//! robustness. We try not to assume that the project model exists or is
//! correct. Instead, we try to provide a best-effort service. Even if the
//! project is currently loading and we don't have a full project model, we
//! still want to respond to various  requests.
use std::{collections::hash_map::Entry, mem, sync::Arc};

use flycheck::{FlycheckConfig, FlycheckHandle};
use hir::db::DefDatabase;
use ide::Change;
use ide_db::{
    base_db::{
        CrateGraph, Env, ProcMacro, ProcMacroExpander, ProcMacroExpansionError, ProcMacroKind,
        ProcMacroLoadResult, SourceRoot, VfsPath,
    },
    FxHashMap,
};
use itertools::Itertools;
use proc_macro_api::{MacroDylib, ProcMacroServer};
use project_model::{PackageRoot, ProjectWorkspace, WorkspaceBuildScripts};
use syntax::SmolStr;
use vfs::{file_set::FileSetConfig, AbsPath, AbsPathBuf, ChangeKind};

use crate::{
    config::{Config, FilesWatcher, LinkedProject},
    global_state::GlobalState,
    lsp_ext,
    main_loop::Task,
    op_queue::Cause,
};

use ::tt::token_id as tt;

#[derive(Debug)]
pub(crate) enum ProjectWorkspaceProgress {
    Begin,
    Report(String),
    End(Vec<anyhow::Result<ProjectWorkspace>>),
}

#[derive(Debug)]
pub(crate) enum BuildDataProgress {
    Begin,
    Report(String),
    End((Arc<Vec<ProjectWorkspace>>, Vec<anyhow::Result<WorkspaceBuildScripts>>)),
}

impl GlobalState {
    pub(crate) fn is_quiescent(&self) -> bool {
        !(self.last_reported_status.is_none()
            || self.fetch_workspaces_queue.op_in_progress()
            || self.fetch_build_data_queue.op_in_progress()
            || self.vfs_progress_config_version < self.vfs_config_version
            || self.vfs_progress_n_done < self.vfs_progress_n_total)
    }

    pub(crate) fn update_configuration(&mut self, config: Config) {
        let _p = profile::span("GlobalState::update_configuration");
        let old_config = mem::replace(&mut self.config, Arc::new(config));
        if self.config.lru_capacity() != old_config.lru_capacity() {
            self.analysis_host.update_lru_capacity(self.config.lru_capacity());
        }
        if self.config.linked_projects() != old_config.linked_projects() {
            self.fetch_workspaces_queue.request_op("linked projects changed".to_string())
        } else if self.config.flycheck() != old_config.flycheck() {
            self.reload_flycheck();
        }

        if self.analysis_host.raw_database().enable_proc_attr_macros()
            != self.config.expand_proc_attr_macros()
        {
            self.analysis_host
                .raw_database_mut()
                .set_enable_proc_attr_macros(self.config.expand_proc_attr_macros());
        }
    }

    pub(crate) fn current_status(&self) -> lsp_ext::ServerStatusParams {
        let mut status = lsp_ext::ServerStatusParams {
            health: lsp_ext::Health::Ok,
            quiescent: self.is_quiescent(),
            message: None,
        };

        if self.proc_macro_changed {
            status.health = lsp_ext::Health::Warning;
            status.message =
                Some("Reload required due to source changes of a procedural macro.".into())
        }
        if let Err(_) = self.fetch_build_data_error() {
            status.health = lsp_ext::Health::Warning;
            status.message =
                Some("Failed to run build scripts of some packages, check the logs.".to_string());
        }
        if !self.config.cargo_autoreload()
            && self.is_quiescent()
            && self.fetch_workspaces_queue.op_requested()
        {
            status.health = lsp_ext::Health::Warning;
            status.message = Some("Workspace reload required".to_string())
        }

        if let Err(_) = self.fetch_workspace_error() {
            status.health = lsp_ext::Health::Error;
            status.message = Some("Failed to load workspaces".to_string())
        }

        if self.config.linked_projects().is_empty()
            && self.config.detached_files().is_empty()
            && self.config.notifications().cargo_toml_not_found
        {
            status.health = lsp_ext::Health::Warning;
            status.message = Some("Failed to discover workspace".to_string())
        }

        status
    }

    pub(crate) fn fetch_workspaces(&mut self, cause: Cause) {
        tracing::info!(%cause, "will fetch workspaces");

        self.task_pool.handle.spawn_with_sender({
            let linked_projects = self.config.linked_projects();
            let detached_files = self.config.detached_files().to_vec();
            let cargo_config = self.config.cargo();

            move |sender| {
                let progress = {
                    let sender = sender.clone();
                    move |msg| {
                        sender
                            .send(Task::FetchWorkspace(ProjectWorkspaceProgress::Report(msg)))
                            .unwrap()
                    }
                };

                sender.send(Task::FetchWorkspace(ProjectWorkspaceProgress::Begin)).unwrap();

                let mut workspaces = linked_projects
                    .iter()
                    .map(|project| match project {
                        LinkedProject::ProjectManifest(manifest) => {
                            project_model::ProjectWorkspace::load(
                                manifest.clone(),
                                &cargo_config,
                                &progress,
                            )
                        }
                        LinkedProject::InlineJsonProject(it) => {
                            Ok(project_model::ProjectWorkspace::load_inline(
                                it.clone(),
                                cargo_config.target.as_deref(),
                                &cargo_config.extra_env,
                            ))
                        }
                    })
                    .collect::<Vec<_>>();

                if !detached_files.is_empty() {
                    workspaces.push(project_model::ProjectWorkspace::load_detached_files(
                        detached_files,
                        &cargo_config,
                    ));
                }

                tracing::info!("did fetch workspaces {:?}", workspaces);
                sender
                    .send(Task::FetchWorkspace(ProjectWorkspaceProgress::End(workspaces)))
                    .unwrap();
            }
        });
    }

    pub(crate) fn fetch_build_data(&mut self, cause: Cause) {
        tracing::info!(%cause, "will fetch build data");
        let workspaces = Arc::clone(&self.workspaces);
        let config = self.config.cargo();
        self.task_pool.handle.spawn_with_sender(move |sender| {
            sender.send(Task::FetchBuildData(BuildDataProgress::Begin)).unwrap();

            let progress = {
                let sender = sender.clone();
                move |msg| {
                    sender.send(Task::FetchBuildData(BuildDataProgress::Report(msg))).unwrap()
                }
            };
            let res = ProjectWorkspace::run_all_build_scripts(&workspaces, &config, &progress);

            sender.send(Task::FetchBuildData(BuildDataProgress::End((workspaces, res)))).unwrap();
        });
    }

    pub(crate) fn switch_workspaces(&mut self, cause: Cause) {
        let _p = profile::span("GlobalState::switch_workspaces");
        tracing::info!(%cause, "will switch workspaces");

        if let Err(_) = self.fetch_workspace_error() {
            if !self.workspaces.is_empty() {
                // It only makes sense to switch to a partially broken workspace
                // if we don't have any workspace at all yet.
                return;
            }
        }

        let Some(workspaces) = self.fetch_workspaces_queue.last_op_result() else { return; };
        let workspaces =
            workspaces.iter().filter_map(|res| res.as_ref().ok().cloned()).collect::<Vec<_>>();

        let same_workspaces = workspaces.len() == self.workspaces.len()
            && workspaces
                .iter()
                .zip(self.workspaces.iter())
                .all(|(l, r)| l.eq_ignore_build_data(r));

        if same_workspaces {
            let (workspaces, build_scripts) = self.fetch_build_data_queue.last_op_result();
            if Arc::ptr_eq(workspaces, &self.workspaces) {
                tracing::debug!("set build scripts to workspaces");

                let workspaces = workspaces
                    .iter()
                    .cloned()
                    .zip(build_scripts)
                    .map(|(mut ws, bs)| {
                        ws.set_build_scripts(bs.as_ref().ok().cloned().unwrap_or_default());
                        ws
                    })
                    .collect::<Vec<_>>();

                // Workspaces are the same, but we've updated build data.
                self.workspaces = Arc::new(workspaces);
            } else {
                tracing::info!("build scripts do not match the version of the active workspace");
                // Current build scripts do not match the version of the active
                // workspace, so there's nothing for us to update.
                return;
            }
        } else {
            tracing::debug!("abandon build scripts for workspaces");

            // Here, we completely changed the workspace (Cargo.toml edit), so
            // we don't care about build-script results, they are stale.
            // FIXME: can we abort the build scripts here?
            self.workspaces = Arc::new(workspaces);
        }

        if let FilesWatcher::Client = self.config.files().watcher {
            let registration_options = lsp_types::DidChangeWatchedFilesRegistrationOptions {
                watchers: self
                    .workspaces
                    .iter()
                    .flat_map(|ws| ws.to_roots())
                    .filter(|it| it.is_local)
                    .flat_map(|root| {
                        root.include.into_iter().flat_map(|it| {
                            [
                                format!("{}/**/*.rs", it.display()),
                                format!("{}/**/Cargo.toml", it.display()),
                                format!("{}/**/Cargo.lock", it.display()),
                            ]
                        })
                    })
                    .map(|glob_pattern| lsp_types::FileSystemWatcher {
                        glob_pattern: lsp_types::GlobPattern::String(glob_pattern),
                        kind: None,
                    })
                    .collect(),
            };
            let registration = lsp_types::Registration {
                id: "workspace/didChangeWatchedFiles".to_string(),
                method: "workspace/didChangeWatchedFiles".to_string(),
                register_options: Some(serde_json::to_value(registration_options).unwrap()),
            };
            self.send_request::<lsp_types::request::RegisterCapability>(
                lsp_types::RegistrationParams { registrations: vec![registration] },
                |_, _| (),
            );
        }

        let mut change = Change::new();

        let files_config = self.config.files();
        let project_folders = ProjectFolders::new(&self.workspaces, &files_config.exclude);

        if self.proc_macro_clients.is_empty() {
            if let Some((path, path_manually_set)) = self.config.proc_macro_srv() {
                tracing::info!("Spawning proc-macro servers");
                self.proc_macro_clients = self
                    .workspaces
                    .iter()
                    .map(|ws| {
                        let (path, args): (_, &[_]) = if path_manually_set {
                            tracing::debug!(
                                "Pro-macro server path explicitly set: {}",
                                path.display()
                            );
                            (path.clone(), &[])
                        } else {
                            match ws.find_sysroot_proc_macro_srv() {
                                Some(server_path) => (server_path, &[]),
                                None => (path.clone(), &["proc-macro"]),
                            }
                        };

                        tracing::info!(?args, "Using proc-macro server at {}", path.display(),);
                        ProcMacroServer::spawn(path.clone(), args).map_err(|err| {
                            let error = format!(
                                "Failed to run proc-macro server from path {}, error: {:?}",
                                path.display(),
                                err
                            );
                            tracing::error!(error);
                            error
                        })
                    })
                    .collect()
            };
        }

        let watch = match files_config.watcher {
            FilesWatcher::Client => vec![],
            FilesWatcher::Server => project_folders.watch,
        };
        self.vfs_config_version += 1;
        self.loader.handle.set_config(vfs::loader::Config {
            load: project_folders.load,
            watch,
            version: self.vfs_config_version,
        });

        // Create crate graph from all the workspaces
        let crate_graph = {
            let dummy_replacements = self.config.dummy_replacements();

            let vfs = &mut self.vfs.write().0;
            let loader = &mut self.loader;
            let mem_docs = &self.mem_docs;
            let mut load = move |path: &AbsPath| {
                let _p = profile::span("switch_workspaces::load");
                let vfs_path = vfs::VfsPath::from(path.to_path_buf());
                if !mem_docs.contains(&vfs_path) {
                    let contents = loader.handle.load_sync(path);
                    vfs.set_file_contents(vfs_path.clone(), contents);
                }
                let res = vfs.file_id(&vfs_path);
                if res.is_none() {
                    tracing::warn!("failed to load {}", path.display())
                }
                res
            };

            let mut crate_graph = CrateGraph::default();
            for (idx, ws) in self.workspaces.iter().enumerate() {
                let proc_macro_client = match self.proc_macro_clients.get(idx) {
                    Some(res) => res.as_ref().map_err(|e| &**e),
                    None => Err("Proc macros are disabled"),
                };
                let mut load_proc_macro = move |crate_name: &str, path: &AbsPath| {
                    load_proc_macro(
                        proc_macro_client,
                        path,
                        dummy_replacements.get(crate_name).map(|v| &**v).unwrap_or_default(),
                    )
                };
                crate_graph.extend(ws.to_crate_graph(
                    &mut load_proc_macro,
                    &mut load,
                    &self.config.cargo().extra_env,
                ));
            }
            crate_graph
        };
        change.set_crate_graph(crate_graph);

        self.source_root_config = project_folders.source_root_config;

        self.analysis_host.apply_change(change);
        self.process_changes();
        self.reload_flycheck();
        tracing::info!("did switch workspaces");
    }

    pub(super) fn fetch_workspace_error(&self) -> Result<(), String> {
        let mut buf = String::new();

        let Some(last_op_result) = self.fetch_workspaces_queue.last_op_result() else { return Ok(()) };
        if last_op_result.is_empty() {
            stdx::format_to!(buf, "rust-analyzer failed to discover workspace");
        } else {
            for ws in last_op_result {
                if let Err(err) = ws {
                    stdx::format_to!(buf, "rust-analyzer failed to load workspace: {:#}\n", err);
                }
            }
        }

        if buf.is_empty() {
            return Ok(());
        }

        Err(buf)
    }

    pub(super) fn fetch_build_data_error(&self) -> Result<(), String> {
        let mut buf = String::new();

        for ws in &self.fetch_build_data_queue.last_op_result().1 {
            match ws {
                Ok(data) => match data.error() {
                    Some(stderr) => stdx::format_to!(buf, "{:#}\n", stderr),
                    _ => (),
                },
                // io errors
                Err(err) => stdx::format_to!(buf, "{:#}\n", err),
            }
        }

        if buf.is_empty() {
            Ok(())
        } else {
            Err(buf)
        }
    }

    fn reload_flycheck(&mut self) {
        let _p = profile::span("GlobalState::reload_flycheck");
        let config = self.config.flycheck();
        let sender = self.flycheck_sender.clone();
        let invocation_strategy = match config {
            FlycheckConfig::CargoCommand { .. } => flycheck::InvocationStrategy::PerWorkspace,
            FlycheckConfig::CustomCommand { invocation_strategy, .. } => invocation_strategy,
        };

        self.flycheck = match invocation_strategy {
            flycheck::InvocationStrategy::Once => vec![FlycheckHandle::spawn(
                0,
                Box::new(move |msg| sender.send(msg).unwrap()),
                config,
                self.config.root_path().clone(),
            )],
            flycheck::InvocationStrategy::PerWorkspace => {
                self.workspaces
                    .iter()
                    .enumerate()
                    .filter_map(|(id, w)| match w {
                        ProjectWorkspace::Cargo { cargo, .. } => Some((id, cargo.workspace_root())),
                        ProjectWorkspace::Json { project, .. } => {
                            // Enable flychecks for json projects if a custom flycheck command was supplied
                            // in the workspace configuration.
                            match config {
                                FlycheckConfig::CustomCommand { .. } => Some((id, project.path())),
                                _ => None,
                            }
                        }
                        ProjectWorkspace::DetachedFiles { .. } => None,
                    })
                    .map(|(id, root)| {
                        let sender = sender.clone();
                        FlycheckHandle::spawn(
                            id,
                            Box::new(move |msg| sender.send(msg).unwrap()),
                            config.clone(),
                            root.to_path_buf(),
                        )
                    })
                    .collect()
            }
        }
        .into();
    }
}

#[derive(Default)]
pub(crate) struct ProjectFolders {
    pub(crate) load: Vec<vfs::loader::Entry>,
    pub(crate) watch: Vec<usize>,
    pub(crate) source_root_config: SourceRootConfig,
}

impl ProjectFolders {
    pub(crate) fn new(
        workspaces: &[ProjectWorkspace],
        global_excludes: &[AbsPathBuf],
    ) -> ProjectFolders {
        let mut res = ProjectFolders::default();
        let mut fsc = FileSetConfig::builder();
        let mut local_filesets = vec![];

        // Dedup source roots
        // Depending on the project setup, we can have duplicated source roots, or for example in
        // the case of the rustc workspace, we can end up with two source roots that are almost the
        // same but not quite, like:
        // PackageRoot { is_local: false, include: [AbsPathBuf(".../rust/src/tools/miri/cargo-miri")], exclude: [] }
        // PackageRoot {
        //     is_local: true,
        //     include: [AbsPathBuf(".../rust/src/tools/miri/cargo-miri"), AbsPathBuf(".../rust/build/x86_64-pc-windows-msvc/stage0-tools/x86_64-pc-windows-msvc/release/build/cargo-miri-85801cd3d2d1dae4/out")],
        //     exclude: [AbsPathBuf(".../rust/src/tools/miri/cargo-miri/.git"), AbsPathBuf(".../rust/src/tools/miri/cargo-miri/target")]
        // }
        //
        // The first one comes from the explicit rustc workspace which points to the rustc workspace itself
        // The second comes from the rustc workspace that we load as the actual project workspace
        // These `is_local` differing in this kind of way gives us problems, especially when trying to filter diagnostics as we don't report diagnostics for external libraries.
        // So we need to deduplicate these, usually it would be enough to deduplicate by `include`, but as the rustc example shows here that doesn't work,
        // so we need to also coalesce the includes if they overlap.

        let mut roots: Vec<_> = workspaces
            .iter()
            .flat_map(|ws| ws.to_roots())
            .update(|root| root.include.sort())
            .sorted_by(|a, b| a.include.cmp(&b.include))
            .collect();

        // map that tracks indices of overlapping roots
        let mut overlap_map = FxHashMap::<_, Vec<_>>::default();
        let mut done = false;

        while !mem::replace(&mut done, true) {
            // maps include paths to indices of the corresponding root
            let mut include_to_idx = FxHashMap::default();
            // Find and note down the indices of overlapping roots
            for (idx, root) in roots.iter().enumerate().filter(|(_, it)| !it.include.is_empty()) {
                for include in &root.include {
                    match include_to_idx.entry(include) {
                        Entry::Occupied(e) => {
                            overlap_map.entry(*e.get()).or_default().push(idx);
                        }
                        Entry::Vacant(e) => {
                            e.insert(idx);
                        }
                    }
                }
            }
            for (k, v) in overlap_map.drain() {
                done = false;
                for v in v {
                    let r = mem::replace(
                        &mut roots[v],
                        PackageRoot { is_local: false, include: vec![], exclude: vec![] },
                    );
                    roots[k].is_local |= r.is_local;
                    roots[k].include.extend(r.include);
                    roots[k].exclude.extend(r.exclude);
                }
                roots[k].include.sort();
                roots[k].exclude.sort();
                roots[k].include.dedup();
                roots[k].exclude.dedup();
            }
        }

        for root in roots.into_iter().filter(|it| !it.include.is_empty()) {
            let file_set_roots: Vec<VfsPath> =
                root.include.iter().cloned().map(VfsPath::from).collect();

            let entry = {
                let mut dirs = vfs::loader::Directories::default();
                dirs.extensions.push("rs".into());
                dirs.include.extend(root.include);
                dirs.exclude.extend(root.exclude);
                for excl in global_excludes {
                    if dirs
                        .include
                        .iter()
                        .any(|incl| incl.starts_with(excl) || excl.starts_with(incl))
                    {
                        dirs.exclude.push(excl.clone());
                    }
                }

                vfs::loader::Entry::Directories(dirs)
            };

            if root.is_local {
                res.watch.push(res.load.len());
            }
            res.load.push(entry);

            if root.is_local {
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
        let _p = profile::span("SourceRootConfig::partition");
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

/// Load the proc-macros for the given lib path, replacing all expanders whose names are in `dummy_replace`
/// with an identity dummy expander.
pub(crate) fn load_proc_macro(
    server: Result<&ProcMacroServer, &str>,
    path: &AbsPath,
    dummy_replace: &[Box<str>],
) -> ProcMacroLoadResult {
    let server = server.map_err(ToOwned::to_owned)?;
    let res: Result<Vec<_>, String> = (|| {
        let dylib = MacroDylib::new(path.to_path_buf())
            .map_err(|io| format!("Proc-macro dylib loading failed: {io}"))?;
        let vec = server.load_dylib(dylib).map_err(|e| format!("{e}"))?;
        if vec.is_empty() {
            return Err("proc macro library returned no proc macros".to_string());
        }
        Ok(vec
            .into_iter()
            .map(|expander| expander_to_proc_macro(expander, dummy_replace))
            .collect())
    })();
    return match res {
        Ok(proc_macros) => {
            tracing::info!(
                "Loaded proc-macros for {}: {:?}",
                path.display(),
                proc_macros.iter().map(|it| it.name.clone()).collect::<Vec<_>>()
            );
            Ok(proc_macros)
        }
        Err(e) => {
            tracing::warn!("proc-macro loading for {} failed: {e}", path.display());
            Err(e)
        }
    };

    fn expander_to_proc_macro(
        expander: proc_macro_api::ProcMacro,
        dummy_replace: &[Box<str>],
    ) -> ProcMacro {
        let name = SmolStr::from(expander.name());
        let kind = match expander.kind() {
            proc_macro_api::ProcMacroKind::CustomDerive => ProcMacroKind::CustomDerive,
            proc_macro_api::ProcMacroKind::FuncLike => ProcMacroKind::FuncLike,
            proc_macro_api::ProcMacroKind::Attr => ProcMacroKind::Attr,
        };
        let expander: Arc<dyn ProcMacroExpander> =
            if dummy_replace.iter().any(|replace| &**replace == name) {
                match kind {
                    ProcMacroKind::Attr => Arc::new(IdentityExpander),
                    _ => Arc::new(EmptyExpander),
                }
            } else {
                Arc::new(Expander(expander))
            };
        ProcMacro { name, kind, expander }
    }

    #[derive(Debug)]
    struct Expander(proc_macro_api::ProcMacro);

    impl ProcMacroExpander for Expander {
        fn expand(
            &self,
            subtree: &tt::Subtree,
            attrs: Option<&tt::Subtree>,
            env: &Env,
        ) -> Result<tt::Subtree, ProcMacroExpansionError> {
            let env = env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            match self.0.expand(subtree, attrs, env) {
                Ok(Ok(subtree)) => Ok(subtree),
                Ok(Err(err)) => Err(ProcMacroExpansionError::Panic(err.0)),
                Err(err) => Err(ProcMacroExpansionError::System(err.to_string())),
            }
        }
    }

    /// Dummy identity expander, used for attribute proc-macros that are deliberately ignored by the user.
    #[derive(Debug)]
    struct IdentityExpander;

    impl ProcMacroExpander for IdentityExpander {
        fn expand(
            &self,
            subtree: &tt::Subtree,
            _: Option<&tt::Subtree>,
            _: &Env,
        ) -> Result<tt::Subtree, ProcMacroExpansionError> {
            Ok(subtree.clone())
        }
    }

    /// Empty expander, used for proc-macros that are deliberately ignored by the user.
    #[derive(Debug)]
    struct EmptyExpander;

    impl ProcMacroExpander for EmptyExpander {
        fn expand(
            &self,
            _: &tt::Subtree,
            _: Option<&tt::Subtree>,
            _: &Env,
        ) -> Result<tt::Subtree, ProcMacroExpansionError> {
            Ok(tt::Subtree::empty())
        }
    }
}

pub(crate) fn should_refresh_for_change(path: &AbsPath, change_kind: ChangeKind) -> bool {
    const IMPLICIT_TARGET_FILES: &[&str] = &["build.rs", "src/main.rs", "src/lib.rs"];
    const IMPLICIT_TARGET_DIRS: &[&str] = &["src/bin", "examples", "tests", "benches"];

    let file_name = match path.file_name().unwrap_or_default().to_str() {
        Some(it) => it,
        None => return false,
    };

    if let "Cargo.toml" | "Cargo.lock" = file_name {
        return true;
    }
    if change_kind == ChangeKind::Modify {
        return false;
    }

    // .cargo/config{.toml}
    if path.extension().unwrap_or_default() != "rs" {
        let is_cargo_config = matches!(file_name, "config.toml" | "config")
            && path.parent().map(|parent| parent.as_ref().ends_with(".cargo")).unwrap_or(false);
        return is_cargo_config;
    }

    if IMPLICIT_TARGET_FILES.iter().any(|it| path.as_ref().ends_with(it)) {
        return true;
    }
    let parent = match path.parent() {
        Some(it) => it,
        None => return false,
    };
    if IMPLICIT_TARGET_DIRS.iter().any(|it| parent.as_ref().ends_with(it)) {
        return true;
    }
    if file_name == "main.rs" {
        let grand_parent = match parent.parent() {
            Some(it) => it,
            None => return false,
        };
        if IMPLICIT_TARGET_DIRS.iter().any(|it| grand_parent.as_ref().ends_with(it)) {
            return true;
        }
    }
    false
}
