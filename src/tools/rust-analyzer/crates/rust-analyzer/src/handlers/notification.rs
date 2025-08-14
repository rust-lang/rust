//! This module is responsible for implementing handlers for Language Server
//! Protocol. This module specifically handles notifications.

use std::ops::{Deref, Not as _};

use ide_db::base_db::salsa::Cancelled;
use itertools::Itertools;
use lsp_types::{
    CancelParams, DidChangeConfigurationParams, DidChangeTextDocumentParams,
    DidChangeWatchedFilesParams, DidChangeWorkspaceFoldersParams, DidCloseTextDocumentParams,
    DidOpenTextDocumentParams, DidSaveTextDocumentParams, WorkDoneProgressCancelParams,
};
use paths::Utf8PathBuf;
use triomphe::Arc;
use vfs::{AbsPathBuf, ChangeKind, VfsPath};

use crate::{
    config::{Config, ConfigChange},
    flycheck::Target,
    global_state::{FetchWorkspaceRequest, GlobalState},
    lsp::{from_proto, utils::apply_document_changes},
    lsp_ext::{self, RunFlycheckParams},
    mem_docs::DocumentData,
    reload,
    target_spec::TargetSpec,
    try_default,
};

pub(crate) fn handle_cancel(state: &mut GlobalState, params: CancelParams) -> anyhow::Result<()> {
    let id: lsp_server::RequestId = match params.id {
        lsp_types::NumberOrString::Number(id) => id.into(),
        lsp_types::NumberOrString::String(id) => id.into(),
    };
    state.cancel(id);
    Ok(())
}

pub(crate) fn handle_work_done_progress_cancel(
    state: &mut GlobalState,
    params: WorkDoneProgressCancelParams,
) -> anyhow::Result<()> {
    if let lsp_types::NumberOrString::String(s) = &params.token
        && let Some(id) = s.strip_prefix("rust-analyzer/flycheck/")
        && let Ok(id) = id.parse::<u32>()
        && let Some(flycheck) = state.flycheck.get(id as usize)
    {
        flycheck.cancel();
    }

    // Just ignore this. It is OK to continue sending progress
    // notifications for this token, as the client can't know when
    // we accepted notification.
    Ok(())
}

pub(crate) fn handle_did_open_text_document(
    state: &mut GlobalState,
    params: DidOpenTextDocumentParams,
) -> anyhow::Result<()> {
    let _p = tracing::info_span!("handle_did_open_text_document").entered();

    if let Ok(path) = from_proto::vfs_path(&params.text_document.uri) {
        let already_exists = state
            .mem_docs
            .insert(
                path.clone(),
                DocumentData::new(
                    params.text_document.version,
                    params.text_document.text.clone().into_bytes(),
                ),
            )
            .is_err();
        if already_exists {
            tracing::error!("duplicate DidOpenTextDocument: {}", path);
        }

        if let Some(abs_path) = path.as_path()
            && state.config.excluded().any(|excluded| abs_path.starts_with(&excluded))
        {
            tracing::trace!("opened excluded file {abs_path}");
            state.vfs.write().0.insert_excluded_file(path);
            return Ok(());
        }

        let contents = params.text_document.text.into_bytes();
        state.vfs.write().0.set_file_contents(path, Some(contents));
        if state.config.discover_workspace_config().is_some() {
            tracing::debug!("queuing task");
            let _ = state
                .deferred_task_queue
                .sender
                .send(crate::main_loop::QueuedTask::CheckIfIndexed(params.text_document.uri));
        }
    }
    Ok(())
}

pub(crate) fn handle_did_change_text_document(
    state: &mut GlobalState,
    params: DidChangeTextDocumentParams,
) -> anyhow::Result<()> {
    let _p = tracing::info_span!("handle_did_change_text_document").entered();

    if let Ok(path) = from_proto::vfs_path(&params.text_document.uri) {
        let Some(DocumentData { version, data }) = state.mem_docs.get_mut(&path) else {
            tracing::error!(?path, "unexpected DidChangeTextDocument");
            return Ok(());
        };
        // The version passed in DidChangeTextDocument is the version after all edits are applied
        // so we should apply it before the vfs is notified.
        *version = params.text_document.version;

        let new_contents = apply_document_changes(
            state.config.negotiated_encoding(),
            std::str::from_utf8(data).unwrap(),
            params.content_changes,
        )
        .into_bytes();
        if *data != new_contents {
            data.clone_from(&new_contents);
            state.vfs.write().0.set_file_contents(path, Some(new_contents));
        }
    }
    Ok(())
}

pub(crate) fn handle_did_close_text_document(
    state: &mut GlobalState,
    params: DidCloseTextDocumentParams,
) -> anyhow::Result<()> {
    let _p = tracing::info_span!("handle_did_close_text_document").entered();

    if let Ok(path) = from_proto::vfs_path(&params.text_document.uri) {
        if state.mem_docs.remove(&path).is_err() {
            tracing::error!("orphan DidCloseTextDocument: {}", path);
        }

        // Clear diagnostics also for excluded files, just in case.
        if let Some((file_id, _)) = state.vfs.read().0.file_id(&path) {
            state.diagnostics.clear_native_for(file_id);
        }

        state.semantic_tokens_cache.lock().remove(&params.text_document.uri);

        if let Some(path) = path.as_path() {
            state.loader.handle.invalidate(path.to_path_buf());
        }
    }
    Ok(())
}

pub(crate) fn handle_did_save_text_document(
    state: &mut GlobalState,
    params: DidSaveTextDocumentParams,
) -> anyhow::Result<()> {
    if let Ok(vfs_path) = from_proto::vfs_path(&params.text_document.uri) {
        let snap = state.snapshot();
        let file_id = try_default!(snap.vfs_path_to_file_id(&vfs_path)?);
        let sr = snap.analysis.source_root_id(file_id)?;

        if state.config.script_rebuild_on_save(Some(sr)) && state.build_deps_changed {
            state.build_deps_changed = false;
            state
                .fetch_build_data_queue
                .request_op("build_deps_changed - save notification".to_owned(), ());
        }

        // Re-fetch workspaces if a workspace related file has changed
        if let Some(path) = vfs_path.as_path() {
            let additional_files = &state
                .config
                .discover_workspace_config()
                .map(|cfg| cfg.files_to_watch.iter().map(String::as_str).collect::<Vec<&str>>())
                .unwrap_or_default();

            // FIXME: We should move this check into a QueuedTask and do semantic resolution of
            // the files. There is only so much we can tell syntactically from the path.
            if reload::should_refresh_for_change(path, ChangeKind::Modify, additional_files) {
                state.fetch_workspaces_queue.request_op(
                    format!("workspace vfs file change saved {path}"),
                    FetchWorkspaceRequest {
                        path: Some(path.to_owned()),
                        force_crate_graph_reload: false,
                    },
                );
            } else if state.detached_files.contains(path) {
                state.fetch_workspaces_queue.request_op(
                    format!("detached file saved {path}"),
                    FetchWorkspaceRequest {
                        path: Some(path.to_owned()),
                        force_crate_graph_reload: false,
                    },
                );
            }
        }

        if !state.config.check_on_save(Some(sr)) || run_flycheck(state, vfs_path) {
            return Ok(());
        }
    } else if state.config.check_on_save(None) && state.config.flycheck_workspace(None) {
        // No specific flycheck was triggered, so let's trigger all of them.
        for flycheck in state.flycheck.iter() {
            flycheck.restart_workspace(None);
        }
    }

    Ok(())
}

pub(crate) fn handle_did_change_configuration(
    state: &mut GlobalState,
    _params: DidChangeConfigurationParams,
) -> anyhow::Result<()> {
    // As stated in https://github.com/microsoft/language-server-protocol/issues/676,
    // this notification's parameters should be ignored and the actual config queried separately.
    state.send_request::<lsp_types::request::WorkspaceConfiguration>(
        lsp_types::ConfigurationParams {
            items: vec![lsp_types::ConfigurationItem {
                scope_uri: None,
                section: Some("rust-analyzer".to_owned()),
            }],
        },
        |this, resp| {
            tracing::debug!("config update response: '{:?}", resp);
            let lsp_server::Response { error, result, .. } = resp;

            match (error, result) {
                (Some(err), _) => {
                    tracing::error!("failed to fetch the server settings: {:?}", err)
                }
                (None, Some(mut configs)) => {
                    if let Some(json) = configs.get_mut(0) {
                        let config = Config::clone(&*this.config);
                        let mut change = ConfigChange::default();
                        change.change_client_config(json.take());

                        let (config, e, _) = config.apply_change(change);
                        this.config_errors = e.is_empty().not().then_some(e);

                        // Client config changes necessitates .update_config method to be called.
                        this.update_configuration(config);
                    }
                }
                (None, None) => {
                    tracing::error!("received empty server settings response from the client")
                }
            }
        },
    );

    Ok(())
}

pub(crate) fn handle_did_change_workspace_folders(
    state: &mut GlobalState,
    params: DidChangeWorkspaceFoldersParams,
) -> anyhow::Result<()> {
    let config = Arc::make_mut(&mut state.config);

    for workspace in params.event.removed {
        let Ok(path) = workspace.uri.to_file_path() else { continue };
        let Ok(path) = Utf8PathBuf::from_path_buf(path) else { continue };
        let Ok(path) = AbsPathBuf::try_from(path) else { continue };
        config.remove_workspace(&path);
    }

    let added = params
        .event
        .added
        .into_iter()
        .filter_map(|it| it.uri.to_file_path().ok())
        .filter_map(|it| Utf8PathBuf::from_path_buf(it).ok())
        .filter_map(|it| AbsPathBuf::try_from(it).ok());
    config.add_workspaces(added);

    if !config.has_linked_projects() && config.detached_files().is_empty() {
        config.rediscover_workspaces();

        let req = FetchWorkspaceRequest { path: None, force_crate_graph_reload: false };
        state.fetch_workspaces_queue.request_op("client workspaces changed".to_owned(), req);
    }

    Ok(())
}

pub(crate) fn handle_did_change_watched_files(
    state: &mut GlobalState,
    params: DidChangeWatchedFilesParams,
) -> anyhow::Result<()> {
    for change in params.changes.iter().unique_by(|&it| &it.uri) {
        if let Ok(path) = from_proto::abs_path(&change.uri) {
            state.loader.handle.invalidate(path);
        }
    }
    Ok(())
}

fn run_flycheck(state: &mut GlobalState, vfs_path: VfsPath) -> bool {
    let _p = tracing::info_span!("run_flycheck").entered();

    let file_id = state.vfs.read().0.file_id(&vfs_path);
    if let Some((file_id, vfs::FileExcluded::No)) = file_id {
        let world = state.snapshot();
        let invocation_strategy_once = state.config.flycheck(None).invocation_strategy_once();
        let may_flycheck_workspace = state.config.flycheck_workspace(None);
        let mut updated = false;
        let task = move || -> std::result::Result<(), Cancelled> {
            if invocation_strategy_once {
                let saved_file = vfs_path.as_path().map(|p| p.to_owned());
                world.flycheck[0].restart_workspace(saved_file);
            }

            let target = TargetSpec::for_file(&world, file_id)?.and_then(|it| {
                let tgt_kind = it.target_kind();
                let (tgt_name, root, package) = match it {
                    TargetSpec::Cargo(c) => (c.target, c.workspace_root, c.package),
                    _ => return None,
                };

                let tgt = match tgt_kind {
                    project_model::TargetKind::Bin => Target::Bin(tgt_name),
                    project_model::TargetKind::Example => Target::Example(tgt_name),
                    project_model::TargetKind::Test => Target::Test(tgt_name),
                    project_model::TargetKind::Bench => Target::Benchmark(tgt_name),
                    _ => return Some((None, root, package)),
                };

                Some((Some(tgt), root, package))
            });
            tracing::debug!(?target, "flycheck target");
            // we have a specific non-library target, attempt to only check that target, nothing
            // else will be affected
            if let Some((target, root, package)) = target {
                // trigger a package check if we have a non-library target as that can't affect
                // anything else in the workspace OR if we're not allowed to check the workspace as
                // the user opted into package checks then
                let package_check_allowed = target.is_some() || !may_flycheck_workspace;
                if package_check_allowed {
                    let workspace = world.workspaces.iter().position(|ws| match &ws.kind {
                        project_model::ProjectWorkspaceKind::Cargo { cargo, .. }
                        | project_model::ProjectWorkspaceKind::DetachedFile {
                            cargo: Some((cargo, _, _)),
                            ..
                        } => *cargo.workspace_root() == root,
                        _ => false,
                    });
                    if let Some(idx) = workspace {
                        world.flycheck[idx].restart_for_package(package, target);
                    }
                }
            }

            if !may_flycheck_workspace {
                return Ok(());
            }

            // Trigger flychecks for all workspaces that depend on the saved file
            // Crates containing or depending on the saved file
            let crate_ids = world
                .analysis
                .crates_for(file_id)?
                .into_iter()
                .flat_map(|id| world.analysis.transitive_rev_deps(id))
                .flatten()
                .unique()
                .collect::<Vec<_>>();
            tracing::debug!(?crate_ids, "flycheck crate ids");
            let crate_root_paths: Vec<_> = crate_ids
                .iter()
                .filter_map(|&crate_id| {
                    world
                        .analysis
                        .crate_root(crate_id)
                        .map(|file_id| {
                            world.file_id_to_file_path(file_id).as_path().map(ToOwned::to_owned)
                        })
                        .transpose()
                })
                .collect::<ide::Cancellable<_>>()?;
            let crate_root_paths: Vec<_> = crate_root_paths.iter().map(Deref::deref).collect();
            tracing::debug!(?crate_root_paths, "flycheck crate roots");

            // Find all workspaces that have at least one target containing the saved file
            let workspace_ids =
                world.workspaces.iter().enumerate().filter(|(_, ws)| match &ws.kind {
                    project_model::ProjectWorkspaceKind::Cargo { cargo, .. }
                    | project_model::ProjectWorkspaceKind::DetachedFile {
                        cargo: Some((cargo, _, _)),
                        ..
                    } => cargo.packages().any(|pkg| {
                        cargo[pkg]
                            .targets
                            .iter()
                            .any(|&it| crate_root_paths.contains(&cargo[it].root.as_path()))
                    }),
                    project_model::ProjectWorkspaceKind::Json(project) => project
                        .crates()
                        .any(|(_, krate)| crate_root_paths.contains(&krate.root_module.as_path())),
                    project_model::ProjectWorkspaceKind::DetachedFile { .. } => false,
                });

            let saved_file = vfs_path.as_path().map(|p| p.to_owned());

            // Find and trigger corresponding flychecks
            'flychecks: for flycheck in world.flycheck.iter() {
                for (id, _) in workspace_ids.clone() {
                    if id == flycheck.id() {
                        updated = true;
                        flycheck.restart_workspace(saved_file.clone());
                        continue 'flychecks;
                    }
                }
            }
            // No specific flycheck was triggered, so let's trigger all of them.
            if !updated {
                for flycheck in world.flycheck.iter() {
                    flycheck.restart_workspace(saved_file.clone());
                }
            }
            Ok(())
        };
        state.task_pool.handle.spawn_with_sender(stdx::thread::ThreadIntent::Worker, move |_| {
            if let Err(e) = std::panic::catch_unwind(task) {
                tracing::error!("flycheck task panicked: {e:?}")
            }
        });
        true
    } else {
        false
    }
}

pub(crate) fn handle_cancel_flycheck(state: &mut GlobalState, _: ()) -> anyhow::Result<()> {
    let _p = tracing::info_span!("handle_cancel_flycheck").entered();
    state.flycheck.iter().for_each(|flycheck| flycheck.cancel());
    Ok(())
}

pub(crate) fn handle_clear_flycheck(state: &mut GlobalState, _: ()) -> anyhow::Result<()> {
    let _p = tracing::info_span!("handle_clear_flycheck").entered();
    state.diagnostics.clear_check_all();
    Ok(())
}

pub(crate) fn handle_run_flycheck(
    state: &mut GlobalState,
    params: RunFlycheckParams,
) -> anyhow::Result<()> {
    let _p = tracing::info_span!("handle_run_flycheck").entered();
    if let Some(text_document) = params.text_document
        && let Ok(vfs_path) = from_proto::vfs_path(&text_document.uri)
        && run_flycheck(state, vfs_path)
    {
        return Ok(());
    }
    // No specific flycheck was triggered, so let's trigger all of them.
    if state.config.flycheck_workspace(None) {
        for flycheck in state.flycheck.iter() {
            flycheck.restart_workspace(None);
        }
    }
    Ok(())
}

pub(crate) fn handle_abort_run_test(state: &mut GlobalState, _: ()) -> anyhow::Result<()> {
    if state.test_run_session.take().is_some() {
        state.send_notification::<lsp_ext::EndRunTest>(());
    }
    Ok(())
}
