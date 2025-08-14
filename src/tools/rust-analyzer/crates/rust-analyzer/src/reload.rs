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
// FIXME: This is a mess that needs some untangling work
use std::{iter, mem};

use hir::{ChangeWithProcMacros, ProcMacrosBuilder, db::DefDatabase};
use ide_db::{
    FxHashMap,
    base_db::{CrateGraphBuilder, ProcMacroLoadingError, ProcMacroPaths, salsa::Durability},
};
use itertools::Itertools;
use load_cargo::{ProjectFolders, load_proc_macro};
use lsp_types::FileSystemWatcher;
use proc_macro_api::ProcMacroClient;
use project_model::{ManifestPath, ProjectWorkspace, ProjectWorkspaceKind, WorkspaceBuildScripts};
use stdx::{format_to, thread::ThreadIntent};
use triomphe::Arc;
use vfs::{AbsPath, AbsPathBuf, ChangeKind};

use crate::{
    config::{Config, FilesWatcher, LinkedProject},
    flycheck::{FlycheckConfig, FlycheckHandle},
    global_state::{
        FetchBuildDataResponse, FetchWorkspaceRequest, FetchWorkspaceResponse, GlobalState,
    },
    lsp_ext,
    main_loop::{DiscoverProjectParam, Task},
    op_queue::Cause,
};
use tracing::{debug, info};

#[derive(Debug)]
pub(crate) enum ProjectWorkspaceProgress {
    Begin,
    Report(String),
    End(Vec<anyhow::Result<ProjectWorkspace>>, bool),
}

#[derive(Debug)]
pub(crate) enum BuildDataProgress {
    Begin,
    Report(String),
    End((Arc<Vec<ProjectWorkspace>>, Vec<anyhow::Result<WorkspaceBuildScripts>>)),
}

#[derive(Debug)]
pub(crate) enum ProcMacroProgress {
    Begin,
    Report(String),
    End(ChangeWithProcMacros),
}

impl GlobalState {
    /// Is the server quiescent?
    ///
    /// This indicates that we've fully loaded the projects and
    /// are ready to do semantic work.
    pub(crate) fn is_quiescent(&self) -> bool {
        self.vfs_done
            && self.fetch_ws_receiver.is_none()
            && !self.fetch_workspaces_queue.op_in_progress()
            && !self.fetch_build_data_queue.op_in_progress()
            && !self.fetch_proc_macros_queue.op_in_progress()
            && !self.discover_workspace_queue.op_in_progress()
            && self.vfs_progress_config_version >= self.vfs_config_version
    }

    /// Is the server ready to respond to analysis dependent LSP requests?
    ///
    /// Unlike `is_quiescent`, this returns false when we're indexing
    /// the project, because we're holding the salsa lock and cannot
    /// respond to LSP requests that depend on salsa data.
    fn is_fully_ready(&self) -> bool {
        self.is_quiescent() && !self.prime_caches_queue.op_in_progress()
    }

    pub(crate) fn update_configuration(&mut self, config: Config) {
        let _p = tracing::info_span!("GlobalState::update_configuration").entered();
        let old_config = mem::replace(&mut self.config, Arc::new(config));
        if self.config.lru_parse_query_capacity() != old_config.lru_parse_query_capacity() {
            self.analysis_host.update_lru_capacity(self.config.lru_parse_query_capacity());
        }
        if self.config.lru_query_capacities_config() != old_config.lru_query_capacities_config() {
            self.analysis_host.update_lru_capacities(
                &self.config.lru_query_capacities_config().cloned().unwrap_or_default(),
            );
        }

        if self.config.linked_or_discovered_projects() != old_config.linked_or_discovered_projects()
        {
            let req = FetchWorkspaceRequest { path: None, force_crate_graph_reload: false };
            self.fetch_workspaces_queue.request_op("discovered projects changed".to_owned(), req)
        } else if self.config.flycheck(None) != old_config.flycheck(None) {
            self.reload_flycheck();
        }

        if self.analysis_host.raw_database().expand_proc_attr_macros()
            != self.config.expand_proc_attr_macros()
        {
            self.analysis_host.raw_database_mut().set_expand_proc_attr_macros_with_durability(
                self.config.expand_proc_attr_macros(),
                Durability::HIGH,
            );
        }

        if self.config.cargo(None) != old_config.cargo(None) {
            let req = FetchWorkspaceRequest { path: None, force_crate_graph_reload: false };
            self.fetch_workspaces_queue.request_op("cargo config changed".to_owned(), req)
        }

        if self.config.cfg_set_test(None) != old_config.cfg_set_test(None) {
            let req = FetchWorkspaceRequest { path: None, force_crate_graph_reload: false };
            self.fetch_workspaces_queue.request_op("cfg_set_test config changed".to_owned(), req)
        }
    }

    pub(crate) fn current_status(&self) -> lsp_ext::ServerStatusParams {
        let mut status = lsp_ext::ServerStatusParams {
            health: lsp_ext::Health::Ok,
            quiescent: self.is_fully_ready(),
            message: None,
        };
        let mut message = String::new();

        if !self.config.cargo_autoreload_config(None)
            && self.is_quiescent()
            && self.fetch_workspaces_queue.op_requested()
            && self.config.discover_workspace_config().is_none()
        {
            status.health |= lsp_ext::Health::Warning;
            message.push_str("Auto-reloading is disabled and the workspace has changed, a manual workspace reload is required.\n\n");
        }

        if self.build_deps_changed {
            status.health |= lsp_ext::Health::Warning;
            message.push_str(
                "Proc-macros and/or build scripts have changed and need to be rebuilt.\n\n",
            );
        }
        if self.fetch_build_data_error().is_err() {
            status.health |= lsp_ext::Health::Warning;
            message.push_str("Failed to run build scripts of some packages.\n\n");
            message.push_str("Please refer to the logs for more details on the errors.");
        }
        if let Some(err) = &self.config_errors {
            status.health |= lsp_ext::Health::Warning;
            format_to!(message, "{err}\n");
        }
        if let Some(err) = &self.last_flycheck_error {
            status.health |= lsp_ext::Health::Warning;
            message.push_str(err);
            message.push('\n');
        }

        if self.config.linked_or_discovered_projects().is_empty()
            && self.config.detached_files().is_empty()
        {
            status.health |= lsp_ext::Health::Warning;
            message.push_str("Failed to discover workspace.\n");
            message.push_str("Consider adding the `Cargo.toml` of the workspace to the [`linkedProjects`](https://rust-analyzer.github.io/book/configuration.html#linkedProjects) setting.\n\n");
        }
        if self.fetch_workspace_error().is_err() {
            status.health |= lsp_ext::Health::Error;
            message.push_str("Failed to load workspaces.");

            if self.config.has_linked_projects() {
                message.push_str(
                    "`rust-analyzer.linkedProjects` have been specified, which may be incorrect. Specified project paths:\n",
                );
                message
                    .push_str(&format!("    {}", self.config.linked_manifests().format("\n    ")));
                if self.config.has_linked_project_jsons() {
                    message.push_str("\nAdditionally, one or more project jsons are specified")
                }
            }
            message.push_str("\n\n");
        }

        if !self.workspaces.is_empty() {
            self.check_workspaces_msrv().for_each(|e| {
                status.health |= lsp_ext::Health::Warning;
                format_to!(message, "{e}");
            });

            let proc_macro_clients = self.proc_macro_clients.iter().chain(iter::repeat(&None));

            for (ws, proc_macro_client) in self.workspaces.iter().zip(proc_macro_clients) {
                if let ProjectWorkspaceKind::Cargo { error: Some(error), .. }
                | ProjectWorkspaceKind::DetachedFile {
                    cargo: Some((_, _, Some(error))), ..
                } = &ws.kind
                {
                    status.health |= lsp_ext::Health::Warning;
                    format_to!(
                        message,
                        "Failed to read Cargo metadata with dependencies for `{}`: {:#}\n\n",
                        ws.manifest_or_root(),
                        error
                    );
                }
                if let Some(err) = ws.sysroot.error() {
                    status.health |= lsp_ext::Health::Warning;
                    format_to!(
                        message,
                        "Workspace `{}` has sysroot errors: ",
                        ws.manifest_or_root()
                    );
                    message.push_str(err);
                    message.push_str("\n\n");
                }
                if let ProjectWorkspaceKind::Cargo { rustc: Err(Some(err)), .. } = &ws.kind {
                    status.health |= lsp_ext::Health::Warning;
                    format_to!(
                        message,
                        "Failed loading rustc_private crates for workspace `{}`: ",
                        ws.manifest_or_root()
                    );
                    message.push_str(err);
                    message.push_str("\n\n");
                };
                match proc_macro_client {
                    Some(Err(err)) => {
                        status.health |= lsp_ext::Health::Warning;
                        format_to!(
                            message,
                            "Failed spawning proc-macro server for workspace `{}`: {err}",
                            ws.manifest_or_root()
                        );
                        message.push_str("\n\n");
                    }
                    Some(Ok(client)) => {
                        if let Some(err) = client.exited() {
                            status.health |= lsp_ext::Health::Warning;
                            format_to!(
                                message,
                                "proc-macro server for workspace `{}` exited: {err}",
                                ws.manifest_or_root()
                            );
                            message.push_str("\n\n");
                        }
                    }
                    // sysroot was explicitly not set so we didn't discover a server
                    None => {}
                }
            }
        }

        if !message.is_empty() {
            status.message = Some(message.trim_end().to_owned());
        }

        status
    }

    pub(crate) fn fetch_workspaces(
        &mut self,
        cause: Cause,
        path: Option<AbsPathBuf>,
        force_crate_graph_reload: bool,
    ) {
        info!(%cause, "will fetch workspaces");

        self.task_pool.handle.spawn_with_sender(ThreadIntent::Worker, {
            let linked_projects = self.config.linked_or_discovered_projects();
            let detached_files: Vec<_> = self
                .config
                .detached_files()
                .iter()
                .cloned()
                .map(ManifestPath::try_from)
                .filter_map(Result::ok)
                .collect();
            let cargo_config = self.config.cargo(None);
            let discover_command = self.config.discover_workspace_config().cloned();
            let is_quiescent = !(self.discover_workspace_queue.op_in_progress()
                || self.vfs_progress_config_version < self.vfs_config_version
                || !self.vfs_done);

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

                if let (Some(_command), Some(path)) = (&discover_command, &path) {
                    let build = linked_projects.iter().find_map(|project| match project {
                        LinkedProject::InlineProjectJson(it) => it.crate_by_buildfile(path),
                        _ => None,
                    });

                    if let Some(build) = build
                        && is_quiescent
                    {
                        let path = AbsPathBuf::try_from(build.build_file)
                            .expect("Unable to convert to an AbsPath");
                        let arg = DiscoverProjectParam::Buildfile(path);
                        sender.send(Task::DiscoverLinkedProjects(arg)).unwrap();
                    }
                }

                let mut workspaces = linked_projects
                    .iter()
                    .map(|project| match project {
                        LinkedProject::ProjectManifest(manifest) => {
                            debug!(path = %manifest, "loading project from manifest");

                            project_model::ProjectWorkspace::load(
                                manifest.clone(),
                                &cargo_config,
                                &progress,
                            )
                        }
                        LinkedProject::InlineProjectJson(it) => {
                            let workspace = project_model::ProjectWorkspace::load_inline(
                                it.clone(),
                                &cargo_config,
                                &progress,
                            );
                            Ok(workspace)
                        }
                    })
                    .collect::<Vec<_>>();

                let mut i = 0;
                while i < workspaces.len() {
                    if let Ok(w) = &workspaces[i] {
                        let dupes: Vec<_> = workspaces[i + 1..]
                            .iter()
                            .positions(|it| it.as_ref().is_ok_and(|ws| ws.eq_ignore_build_data(w)))
                            .collect();
                        dupes.into_iter().rev().for_each(|d| {
                            _ = workspaces.remove(d + i + 1);
                        });
                    }
                    i += 1;
                }

                if !detached_files.is_empty() {
                    workspaces.extend(project_model::ProjectWorkspace::load_detached_files(
                        detached_files,
                        &cargo_config,
                    ));
                }

                info!(?workspaces, "did fetch workspaces");
                sender
                    .send(Task::FetchWorkspace(ProjectWorkspaceProgress::End(
                        workspaces,
                        force_crate_graph_reload,
                    )))
                    .unwrap();
            }
        });
    }

    pub(crate) fn fetch_build_data(&mut self, cause: Cause) {
        info!(%cause, "will fetch build data");
        let workspaces = Arc::clone(&self.workspaces);
        let config = self.config.cargo(None);
        let root_path = self.config.root_path().clone();

        self.task_pool.handle.spawn_with_sender(ThreadIntent::Worker, move |sender| {
            sender.send(Task::FetchBuildData(BuildDataProgress::Begin)).unwrap();

            let progress = {
                let sender = sender.clone();
                move |msg| {
                    sender.send(Task::FetchBuildData(BuildDataProgress::Report(msg))).unwrap()
                }
            };
            let res = ProjectWorkspace::run_all_build_scripts(
                &workspaces,
                &config,
                &progress,
                &root_path,
            );

            sender.send(Task::FetchBuildData(BuildDataProgress::End((workspaces, res)))).unwrap();
        });
    }

    pub(crate) fn fetch_proc_macros(
        &mut self,
        cause: Cause,
        mut change: ChangeWithProcMacros,
        paths: Vec<ProcMacroPaths>,
    ) {
        info!(%cause, "will load proc macros");
        let ignored_proc_macros = self.config.ignored_proc_macros(None).clone();
        let proc_macro_clients = self.proc_macro_clients.clone();

        self.task_pool.handle.spawn_with_sender(ThreadIntent::Worker, move |sender| {
            sender.send(Task::LoadProcMacros(ProcMacroProgress::Begin)).unwrap();

            let ignored_proc_macros = &ignored_proc_macros;
            let progress = {
                let sender = sender.clone();
                &move |msg| {
                    sender.send(Task::LoadProcMacros(ProcMacroProgress::Report(msg))).unwrap()
                }
            };

            let mut builder = ProcMacrosBuilder::default();
            let proc_macro_clients = proc_macro_clients.iter().chain(iter::repeat(&None));
            for (client, paths) in proc_macro_clients.zip(paths) {
                for (crate_id, res) in paths.iter() {
                    let expansion_res = match client {
                        Some(Ok(client)) => match res {
                            Ok((crate_name, path)) => {
                                progress(format!("loading proc-macros: {path}"));
                                let ignored_proc_macros = ignored_proc_macros
                                    .iter()
                                    .find_map(|(name, macros)| {
                                        eq_ignore_underscore(name, crate_name).then_some(&**macros)
                                    })
                                    .unwrap_or_default();

                                load_proc_macro(client, path, ignored_proc_macros)
                            }
                            Err(e) => Err(e.clone()),
                        },
                        Some(Err(e)) => Err(ProcMacroLoadingError::ProcMacroSrvError(
                            e.to_string().into_boxed_str(),
                        )),
                        None => Err(ProcMacroLoadingError::ProcMacroSrvError(
                            "proc-macro-srv is not running".into(),
                        )),
                    };
                    builder.insert(*crate_id, expansion_res)
                }
            }

            change.set_proc_macros(builder);
            sender.send(Task::LoadProcMacros(ProcMacroProgress::End(change))).unwrap();
        });
    }

    pub(crate) fn switch_workspaces(&mut self, cause: Cause) {
        let _p = tracing::info_span!("GlobalState::switch_workspaces").entered();
        tracing::info!(%cause, "will switch workspaces");

        let Some(FetchWorkspaceResponse { workspaces, force_crate_graph_reload }) =
            self.fetch_workspaces_queue.last_op_result()
        else {
            return;
        };
        let switching_from_empty_workspace = self.workspaces.is_empty();

        info!(%cause, ?force_crate_graph_reload, %switching_from_empty_workspace);
        if self.fetch_workspace_error().is_err() && !switching_from_empty_workspace {
            if *force_crate_graph_reload {
                self.recreate_crate_graph(cause, false);
            }
            // It only makes sense to switch to a partially broken workspace
            // if we don't have any workspace at all yet.
            return;
        }

        let workspaces =
            workspaces.iter().filter_map(|res| res.as_ref().ok().cloned()).collect::<Vec<_>>();

        let same_workspaces = workspaces.len() == self.workspaces.len()
            && workspaces
                .iter()
                .zip(self.workspaces.iter())
                .all(|(l, r)| l.eq_ignore_build_data(r));

        if same_workspaces {
            if switching_from_empty_workspace {
                // Switching from empty to empty is a no-op
                return;
            }
            if let Some(FetchBuildDataResponse { workspaces, build_scripts }) =
                self.fetch_build_data_queue.last_op_result()
            {
                if Arc::ptr_eq(workspaces, &self.workspaces) {
                    info!("set build scripts to workspaces");

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
                    info!("same workspace, but new build data");
                    self.workspaces = Arc::new(workspaces);
                } else {
                    info!("build scripts do not match the version of the active workspace");
                    if *force_crate_graph_reload {
                        self.recreate_crate_graph(cause, switching_from_empty_workspace);
                    }

                    // Current build scripts do not match the version of the active
                    // workspace, so there's nothing for us to update.
                    return;
                }
            } else {
                if *force_crate_graph_reload {
                    self.recreate_crate_graph(cause, switching_from_empty_workspace);
                }

                // No build scripts but unchanged workspaces, nothing to do here
                return;
            }
        } else {
            info!("abandon build scripts for workspaces");

            // Here, we completely changed the workspace (Cargo.toml edit), so
            // we don't care about build-script results, they are stale.
            // FIXME: can we abort the build scripts here if they are already running?
            self.workspaces = Arc::new(workspaces);
            self.check_workspaces_msrv().for_each(|message| {
                self.send_notification::<lsp_types::notification::ShowMessage>(
                    lsp_types::ShowMessageParams { typ: lsp_types::MessageType::WARNING, message },
                );
            });

            if self.config.run_build_scripts(None) {
                self.build_deps_changed = false;
                self.fetch_build_data_queue.request_op("workspace updated".to_owned(), ());

                if !switching_from_empty_workspace {
                    // `switch_workspaces()` will be called again when build scripts already run, which should
                    // take a short time. If we update the workspace now we will invalidate proc macros and cfgs,
                    // and then when build scripts complete we will invalidate them again.
                    return;
                }
            }
        }

        if let FilesWatcher::Client = self.config.files().watcher {
            let filter = self
                .workspaces
                .iter()
                .flat_map(|ws| ws.to_roots())
                .filter(|it| it.is_local)
                .map(|it| it.include);

            let mut watchers: Vec<FileSystemWatcher> =
                if self.config.did_change_watched_files_relative_pattern_support() {
                    // When relative patterns are supported by the client, prefer using them
                    filter
                        .flat_map(|include| {
                            include.into_iter().flat_map(|base| {
                                [
                                    (base.clone(), "**/*.rs"),
                                    (base.clone(), "**/Cargo.{lock,toml}"),
                                    (base, "**/rust-analyzer.toml"),
                                ]
                            })
                        })
                        .map(|(base, pat)| lsp_types::FileSystemWatcher {
                            glob_pattern: lsp_types::GlobPattern::Relative(
                                lsp_types::RelativePattern {
                                    base_uri: lsp_types::OneOf::Right(
                                        lsp_types::Url::from_file_path(base).unwrap(),
                                    ),
                                    pattern: pat.to_owned(),
                                },
                            ),
                            kind: None,
                        })
                        .collect()
                } else {
                    // When they're not, integrate the base to make them into absolute patterns
                    filter
                        .flat_map(|include| {
                            include.into_iter().flat_map(|base| {
                                [
                                    format!("{base}/**/*.rs"),
                                    format!("{base}/**/Cargo.{{toml,lock}}"),
                                    format!("{base}/**/rust-analyzer.toml"),
                                ]
                            })
                        })
                        .map(|glob_pattern| lsp_types::FileSystemWatcher {
                            glob_pattern: lsp_types::GlobPattern::String(glob_pattern),
                            kind: None,
                        })
                        .collect()
                };

            // Also explicitly watch any build files configured in JSON project files.
            for ws in self.workspaces.iter() {
                if let ProjectWorkspaceKind::Json(project_json) = &ws.kind {
                    for (_, krate) in project_json.crates() {
                        let Some(build) = &krate.build else {
                            continue;
                        };
                        watchers.push(lsp_types::FileSystemWatcher {
                            glob_pattern: lsp_types::GlobPattern::String(
                                build.build_file.to_string(),
                            ),
                            kind: None,
                        });
                    }
                }
            }

            watchers.extend(
                iter::once(Config::user_config_dir_path().as_deref())
                    .chain(self.workspaces.iter().map(|ws| ws.manifest().map(ManifestPath::as_ref)))
                    .flatten()
                    .map(|glob_pattern| lsp_types::FileSystemWatcher {
                        glob_pattern: lsp_types::GlobPattern::String(glob_pattern.to_string()),
                        kind: None,
                    }),
            );

            let registration_options =
                lsp_types::DidChangeWatchedFilesRegistrationOptions { watchers };
            let registration = lsp_types::Registration {
                id: "workspace/didChangeWatchedFiles".to_owned(),
                method: "workspace/didChangeWatchedFiles".to_owned(),
                register_options: Some(serde_json::to_value(registration_options).unwrap()),
            };
            self.send_request::<lsp_types::request::RegisterCapability>(
                lsp_types::RegistrationParams { registrations: vec![registration] },
                |_, _| (),
            );
        }

        let files_config = self.config.files();
        let project_folders = ProjectFolders::new(
            &self.workspaces,
            &files_config.exclude,
            Config::user_config_dir_path().as_deref(),
        );

        if (self.proc_macro_clients.len() < self.workspaces.len() || !same_workspaces)
            && self.config.expand_proc_macros()
        {
            info!("Spawning proc-macro servers");

            self.proc_macro_clients = Arc::from_iter(self.workspaces.iter().map(|ws| {
                let path = match self.config.proc_macro_srv() {
                    Some(path) => path,
                    None => match ws.find_sysroot_proc_macro_srv()? {
                        Ok(path) => path,
                        Err(e) => return Some(Err(e)),
                    },
                };

                let env: FxHashMap<_, _> = match &ws.kind {
                    ProjectWorkspaceKind::Cargo { cargo, .. }
                    | ProjectWorkspaceKind::DetachedFile { cargo: Some((cargo, ..)), .. } => cargo
                        .env()
                        .into_iter()
                        .map(|(k, v)| (k.clone(), Some(v.clone())))
                        .chain(
                            self.config.extra_env(None).iter().map(|(k, v)| (k.clone(), v.clone())),
                        )
                        .chain(
                            ws.sysroot
                                .root()
                                .filter(|_| {
                                    !self.config.extra_env(None).contains_key("RUSTUP_TOOLCHAIN")
                                        && std::env::var_os("RUSTUP_TOOLCHAIN").is_none()
                                })
                                .map(|it| ("RUSTUP_TOOLCHAIN".to_owned(), Some(it.to_string()))),
                        )
                        .collect(),

                    _ => Default::default(),
                };
                info!("Using proc-macro server at {path}");

                Some(ProcMacroClient::spawn(&path, &env).map_err(|err| {
                    tracing::error!(
                        "Failed to run proc-macro server from path {path}, error: {err:?}",
                    );
                    anyhow::format_err!(
                        "Failed to run proc-macro server from path {path}, error: {err:?}",
                    )
                }))
            }))
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
        self.source_root_config = project_folders.source_root_config;
        self.local_roots_parent_map = Arc::new(self.source_root_config.source_root_parent_map());

        info!(?cause, "recreating the crate graph");
        self.recreate_crate_graph(cause, switching_from_empty_workspace);

        info!("did switch workspaces");
    }

    fn recreate_crate_graph(&mut self, cause: String, initial_build: bool) {
        info!(?cause, "Building Crate Graph");
        self.report_progress(
            "Building CrateGraph",
            crate::lsp::utils::Progress::Begin,
            None,
            None,
            None,
        );

        // crate graph construction relies on these paths, record them so when one of them gets
        // deleted or created we trigger a reconstruction of the crate graph
        self.crate_graph_file_dependencies.clear();
        self.detached_files = self
            .workspaces
            .iter()
            .filter_map(|ws| match &ws.kind {
                ProjectWorkspaceKind::DetachedFile { file, .. } => Some(file.clone()),
                _ => None,
            })
            .collect();

        let (crate_graph, proc_macro_paths) = {
            // Create crate graph from all the workspaces
            let vfs = &self.vfs.read().0;
            let load = |path: &AbsPath| {
                let vfs_path = vfs::VfsPath::from(path.to_path_buf());
                self.crate_graph_file_dependencies.insert(vfs_path.clone());
                vfs.file_id(&vfs_path).and_then(|(file_id, excluded)| {
                    (excluded == vfs::FileExcluded::No).then_some(file_id)
                })
            };

            ws_to_crate_graph(&self.workspaces, self.config.extra_env(None), load)
        };
        let mut change = ChangeWithProcMacros::default();
        if initial_build || !self.config.expand_proc_macros() {
            if self.config.expand_proc_macros() {
                change.set_proc_macros(
                    crate_graph
                        .iter()
                        .map(|id| (id, Err(ProcMacroLoadingError::NotYetBuilt)))
                        .collect(),
                );
            } else {
                change.set_proc_macros(
                    crate_graph
                        .iter()
                        .map(|id| (id, Err(ProcMacroLoadingError::Disabled)))
                        .collect(),
                );
            }

            change.set_crate_graph(crate_graph);
            self.analysis_host.apply_change(change);

            self.finish_loading_crate_graph();
        } else {
            change.set_crate_graph(crate_graph);
            self.fetch_proc_macros_queue.request_op(cause, (change, proc_macro_paths));
        }

        self.report_progress(
            "Building CrateGraph",
            crate::lsp::utils::Progress::End,
            None,
            None,
            None,
        );
    }

    pub(crate) fn finish_loading_crate_graph(&mut self) {
        self.process_changes();
        self.reload_flycheck();
    }

    pub(super) fn fetch_workspace_error(&self) -> Result<(), String> {
        let mut buf = String::new();

        let Some(FetchWorkspaceResponse { workspaces, .. }) =
            self.fetch_workspaces_queue.last_op_result()
        else {
            return Ok(());
        };

        if workspaces.is_empty() && self.config.discover_workspace_config().is_none() {
            stdx::format_to!(buf, "rust-analyzer failed to fetch workspace");
        } else {
            for ws in workspaces {
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

        let Some(FetchBuildDataResponse { build_scripts, .. }) =
            &self.fetch_build_data_queue.last_op_result()
        else {
            return Ok(());
        };

        for script in build_scripts {
            match script {
                Ok(data) => {
                    if let Some(stderr) = data.error() {
                        stdx::format_to!(buf, "{:#}\n", stderr)
                    }
                }
                // io errors
                Err(err) => stdx::format_to!(buf, "{:#}\n", err),
            }
        }

        if buf.is_empty() { Ok(()) } else { Err(buf) }
    }

    fn reload_flycheck(&mut self) {
        let _p = tracing::info_span!("GlobalState::reload_flycheck").entered();
        let config = self.config.flycheck(None);
        let sender = self.flycheck_sender.clone();
        let invocation_strategy = match config {
            FlycheckConfig::CargoCommand { .. } => {
                crate::flycheck::InvocationStrategy::PerWorkspace
            }
            FlycheckConfig::CustomCommand { ref invocation_strategy, .. } => {
                invocation_strategy.clone()
            }
        };

        self.flycheck = match invocation_strategy {
            crate::flycheck::InvocationStrategy::Once => {
                vec![FlycheckHandle::spawn(
                    0,
                    sender,
                    config,
                    None,
                    self.config.root_path().clone(),
                    None,
                )]
            }
            crate::flycheck::InvocationStrategy::PerWorkspace => {
                self.workspaces
                    .iter()
                    .enumerate()
                    .filter_map(|(id, ws)| {
                        Some((
                            id,
                            match &ws.kind {
                                ProjectWorkspaceKind::Cargo { cargo, .. }
                                | ProjectWorkspaceKind::DetachedFile {
                                    cargo: Some((cargo, _, _)),
                                    ..
                                } => (cargo.workspace_root(), Some(cargo.manifest_path())),
                                ProjectWorkspaceKind::Json(project) => {
                                    // Enable flychecks for json projects if a custom flycheck command was supplied
                                    // in the workspace configuration.
                                    match config {
                                        FlycheckConfig::CustomCommand { .. } => {
                                            (project.path(), None)
                                        }
                                        _ => return None,
                                    }
                                }
                                ProjectWorkspaceKind::DetachedFile { .. } => return None,
                            },
                            ws.sysroot.root().map(ToOwned::to_owned),
                        ))
                    })
                    .map(|(id, (root, manifest_path), sysroot_root)| {
                        FlycheckHandle::spawn(
                            id,
                            sender.clone(),
                            config.clone(),
                            sysroot_root,
                            root.to_path_buf(),
                            manifest_path.map(|it| it.to_path_buf()),
                        )
                    })
                    .collect()
            }
        }
        .into();
    }
}

// FIXME: Move this into load-cargo?
pub fn ws_to_crate_graph(
    workspaces: &[ProjectWorkspace],
    extra_env: &FxHashMap<String, Option<String>>,
    mut load: impl FnMut(&AbsPath) -> Option<vfs::FileId>,
) -> (CrateGraphBuilder, Vec<ProcMacroPaths>) {
    let mut crate_graph = CrateGraphBuilder::default();
    let mut proc_macro_paths = Vec::default();
    for ws in workspaces {
        let (other, mut crate_proc_macros) = ws.to_crate_graph(&mut load, extra_env);

        crate_graph.extend(other, &mut crate_proc_macros);
        proc_macro_paths.push(crate_proc_macros);
    }

    crate_graph.shrink_to_fit();
    proc_macro_paths.shrink_to_fit();
    (crate_graph, proc_macro_paths)
}

pub(crate) fn should_refresh_for_change(
    path: &AbsPath,
    change_kind: ChangeKind,
    additional_paths: &[&str],
) -> bool {
    const IMPLICIT_TARGET_FILES: &[&str] = &["build.rs", "src/main.rs", "src/lib.rs"];
    const IMPLICIT_TARGET_DIRS: &[&str] = &["src/bin", "examples", "tests", "benches"];

    let file_name = match path.file_name() {
        Some(it) => it,
        None => return false,
    };

    if let "Cargo.toml" | "Cargo.lock" = file_name {
        return true;
    }

    if additional_paths.contains(&file_name) {
        return true;
    }

    if change_kind == ChangeKind::Modify {
        return false;
    }

    // .cargo/config{.toml}
    if path.extension().unwrap_or_default() != "rs" {
        let is_cargo_config = matches!(file_name, "config.toml" | "config")
            && path.parent().map(|parent| parent.as_str().ends_with(".cargo")).unwrap_or(false);
        return is_cargo_config;
    }

    if IMPLICIT_TARGET_FILES.iter().any(|it| path.as_str().ends_with(it)) {
        return true;
    }
    let parent = match path.parent() {
        Some(it) => it,
        None => return false,
    };
    if IMPLICIT_TARGET_DIRS.iter().any(|it| parent.as_str().ends_with(it)) {
        return true;
    }
    if file_name == "main.rs" {
        let grand_parent = match parent.parent() {
            Some(it) => it,
            None => return false,
        };
        if IMPLICIT_TARGET_DIRS.iter().any(|it| grand_parent.as_str().ends_with(it)) {
            return true;
        }
    }
    false
}

/// Similar to [`str::eq_ignore_ascii_case`] but instead of ignoring
/// case, we say that `-` and `_` are equal.
fn eq_ignore_underscore(s1: &str, s2: &str) -> bool {
    if s1.len() != s2.len() {
        return false;
    }

    s1.as_bytes().iter().zip(s2.as_bytes()).all(|(c1, c2)| {
        let c1_underscore = c1 == &b'_' || c1 == &b'-';
        let c2_underscore = c2 == &b'_' || c2 == &b'-';

        c1 == c2 || (c1_underscore && c2_underscore)
    })
}
