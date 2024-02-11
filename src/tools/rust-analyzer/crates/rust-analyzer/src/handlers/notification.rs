//! This module is responsible for implementing handlers for Language Server
//! Protocol. This module specifically handles notifications.

use std::ops::Deref;

use itertools::Itertools;
use lsp_types::{
    CancelParams, DidChangeConfigurationParams, DidChangeTextDocumentParams,
    DidChangeWatchedFilesParams, DidChangeWorkspaceFoldersParams, DidCloseTextDocumentParams,
    DidOpenTextDocumentParams, DidSaveTextDocumentParams, WorkDoneProgressCancelParams,
};
use triomphe::Arc;
use vfs::{AbsPathBuf, ChangeKind, VfsPath};

use crate::{
    config::Config,
    global_state::GlobalState,
    lsp::{from_proto, utils::apply_document_changes},
    lsp_ext::RunFlycheckParams,
    mem_docs::DocumentData,
    reload,
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
    if let lsp_types::NumberOrString::String(s) = &params.token {
        if let Some(id) = s.strip_prefix("rust-analyzer/flycheck/") {
            if let Ok(id) = id.parse::<u32>() {
                if let Some(flycheck) = state.flycheck.get(id as usize) {
                    flycheck.cancel();
                }
            }
        }
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
    let _p = tracing::span!(tracing::Level::INFO, "handle_did_open_text_document").entered();

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

        state.vfs.write().0.set_file_contents(path, Some(params.text_document.text.into_bytes()));
        if state.config.notifications().unindexed_project {
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
    let _p = tracing::span!(tracing::Level::INFO, "handle_did_change_text_document").entered();

    if let Ok(path) = from_proto::vfs_path(&params.text_document.uri) {
        let data = match state.mem_docs.get_mut(&path) {
            Some(doc) => {
                // The version passed in DidChangeTextDocument is the version after all edits are applied
                // so we should apply it before the vfs is notified.
                doc.version = params.text_document.version;
                &mut doc.data
            }
            None => {
                tracing::error!("unexpected DidChangeTextDocument: {}", path);
                return Ok(());
            }
        };

        let new_contents = apply_document_changes(
            state.config.position_encoding(),
            std::str::from_utf8(data).unwrap(),
            params.content_changes,
        )
        .into_bytes();
        if *data != new_contents {
            *data = new_contents.clone();
            state.vfs.write().0.set_file_contents(path, Some(new_contents));
        }
    }
    Ok(())
}

pub(crate) fn handle_did_close_text_document(
    state: &mut GlobalState,
    params: DidCloseTextDocumentParams,
) -> anyhow::Result<()> {
    let _p = tracing::span!(tracing::Level::INFO, "handle_did_close_text_document").entered();

    if let Ok(path) = from_proto::vfs_path(&params.text_document.uri) {
        if state.mem_docs.remove(&path).is_err() {
            tracing::error!("orphan DidCloseTextDocument: {}", path);
        }

        if let Some(file_id) = state.vfs.read().0.file_id(&path) {
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
    if state.config.script_rebuild_on_save() && state.proc_macro_changed {
        // reset the flag
        state.proc_macro_changed = false;
        // rebuild the proc macros
        state.fetch_build_data_queue.request_op("ScriptRebuildOnSave".to_owned(), ());
    }

    if let Ok(vfs_path) = from_proto::vfs_path(&params.text_document.uri) {
        // Re-fetch workspaces if a workspace related file has changed
        if let Some(abs_path) = vfs_path.as_path() {
            if reload::should_refresh_for_change(abs_path, ChangeKind::Modify) {
                state
                    .fetch_workspaces_queue
                    .request_op(format!("DidSaveTextDocument {abs_path}"), false);
            }
        }

        if !state.config.check_on_save() || run_flycheck(state, vfs_path) {
            return Ok(());
        }
    } else if state.config.check_on_save() {
        // No specific flycheck was triggered, so let's trigger all of them.
        for flycheck in state.flycheck.iter() {
            flycheck.restart_workspace();
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
                        // Note that json can be null according to the spec if the client can't
                        // provide a configuration. This is handled in Config::update below.
                        let mut config = Config::clone(&*this.config);
                        this.config_errors = config.update(json.take()).err();
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
        let Ok(path) = AbsPathBuf::try_from(path) else { continue };
        config.remove_workspace(&path);
    }

    let added = params
        .event
        .added
        .into_iter()
        .filter_map(|it| it.uri.to_file_path().ok())
        .filter_map(|it| AbsPathBuf::try_from(it).ok());
    config.add_workspaces(added);

    if !config.has_linked_projects() && config.detached_files().is_empty() {
        config.rediscover_workspaces();
        state.fetch_workspaces_queue.request_op("client workspaces changed".to_owned(), false)
    }

    Ok(())
}

pub(crate) fn handle_did_change_watched_files(
    state: &mut GlobalState,
    params: DidChangeWatchedFilesParams,
) -> anyhow::Result<()> {
    for change in params.changes {
        if let Ok(path) = from_proto::abs_path(&change.uri) {
            state.loader.handle.invalidate(path);
        }
    }
    Ok(())
}

fn run_flycheck(state: &mut GlobalState, vfs_path: VfsPath) -> bool {
    let _p = tracing::span!(tracing::Level::INFO, "run_flycheck").entered();

    let file_id = state.vfs.read().0.file_id(&vfs_path);
    if let Some(file_id) = file_id {
        let world = state.snapshot();
        let mut updated = false;
        let task = move || -> std::result::Result<(), ide::Cancelled> {
            // Trigger flychecks for all workspaces that depend on the saved file
            // Crates containing or depending on the saved file
            let crate_ids: Vec<_> = world
                .analysis
                .crates_for(file_id)?
                .into_iter()
                .flat_map(|id| world.analysis.transitive_rev_deps(id))
                .flatten()
                .sorted()
                .unique()
                .collect();

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

            // Find all workspaces that have at least one target containing the saved file
            let workspace_ids = world.workspaces.iter().enumerate().filter_map(|(idx, ws)| {
                let package = match ws {
                    project_model::ProjectWorkspace::Cargo { cargo, .. } => {
                        cargo.packages().find_map(|pkg| {
                            let has_target_with_root = cargo[pkg]
                                .targets
                                .iter()
                                .any(|&it| crate_root_paths.contains(&cargo[it].root.as_path()));
                            has_target_with_root.then(|| cargo[pkg].name.clone())
                        })
                    }
                    project_model::ProjectWorkspace::Json { project, .. } => {
                        if !project
                            .crates()
                            .any(|(c, _)| crate_ids.iter().any(|&crate_id| crate_id == c))
                        {
                            return None;
                        }
                        None
                    }
                    project_model::ProjectWorkspace::DetachedFiles { .. } => return None,
                };
                Some((idx, package))
            });

            // Find and trigger corresponding flychecks
            for flycheck in world.flycheck.iter() {
                for (id, package) in workspace_ids.clone() {
                    if id == flycheck.id() {
                        updated = true;
                        match package.filter(|_| !world.config.flycheck_workspace()) {
                            Some(package) => flycheck.restart_for_package(package),
                            None => flycheck.restart_workspace(),
                        }
                        continue;
                    }
                }
            }
            // No specific flycheck was triggered, so let's trigger all of them.
            if !updated {
                for flycheck in world.flycheck.iter() {
                    flycheck.restart_workspace();
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
    let _p = tracing::span!(tracing::Level::INFO, "handle_stop_flycheck").entered();
    state.flycheck.iter().for_each(|flycheck| flycheck.cancel());
    Ok(())
}

pub(crate) fn handle_clear_flycheck(state: &mut GlobalState, _: ()) -> anyhow::Result<()> {
    let _p = tracing::span!(tracing::Level::INFO, "handle_clear_flycheck").entered();
    state.diagnostics.clear_check_all();
    Ok(())
}

pub(crate) fn handle_run_flycheck(
    state: &mut GlobalState,
    params: RunFlycheckParams,
) -> anyhow::Result<()> {
    let _p = tracing::span!(tracing::Level::INFO, "handle_run_flycheck").entered();
    if let Some(text_document) = params.text_document {
        if let Ok(vfs_path) = from_proto::vfs_path(&text_document.uri) {
            if run_flycheck(state, vfs_path) {
                return Ok(());
            }
        }
    }
    // No specific flycheck was triggered, so let's trigger all of them.
    for flycheck in state.flycheck.iter() {
        flycheck.restart_workspace();
    }
    Ok(())
}
