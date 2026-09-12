//! Entry point of a single LSP session: initialization handshake, then the main loop.

use std::{env, path::PathBuf};

use anyhow::Context;
use lsp_server::Connection;
use paths::Utf8PathBuf;
use vfs::AbsPathBuf;

use crate::{
    config::{Config, ConfigChange, ConfigErrors},
    from_json,
};

/// Handles to the I/O threads shuttling the messages of a [`Connection`], joined when
/// the session ends to surface transport errors.
pub enum IoThreads {
    /// The stdio transport of a standalone server.
    Stdio(lsp_server::IoThreads),
}

impl IoThreads {
    fn join(self) -> anyhow::Result<()> {
        match self {
            IoThreads::Stdio(io_threads) => Ok(io_threads.join()?),
        }
    }
}

/// Runs a full LSP session over `connection`: waits for the client's `initialize`,
/// negotiates capabilities, then runs the main loop until the client disconnects or
/// requests shutdown.
///
/// # Errors
///
/// Returns an error if the connection breaks down or the main loop exits abnormally.
pub fn run_session(
    connection: Connection,
    io_threads: IoThreads,
    startup_notice: Option<String>,
) -> anyhow::Result<()> {
    tracing::info!("server version {} will start", crate::version());

    let (initialize_id, initialize_params) = match connection.initialize_start() {
        Ok(it) => it,
        Err(e) => {
            if e.channel_is_disconnected() {
                io_threads.join()?;
            }
            return Err(e.into());
        }
    };

    tracing::info!("InitializeParams: {}", initialize_params);
    let lsp_types::InitializeParams {
        #[expect(deprecated, reason = "compatibility with old clients")]
        root_uri,
        capabilities,
        workspace_folders_initialize_params,
        initialization_options,
        client_info,
        ..
    } = from_json::<lsp_types::InitializeParams>("InitializeParams", &initialize_params)?;

    let root_path = match root_uri
        .and_then(|it| it.to_file_path().ok())
        .map(patch_path_prefix)
        .and_then(|it| Utf8PathBuf::from_path_buf(it).ok())
        .and_then(|it| AbsPathBuf::try_from(it).ok())
    {
        Some(it) => it,
        None => {
            let cwd = env::current_dir().context("couldn't determine working directory")?;
            AbsPathBuf::assert_utf8(cwd)
        }
    };

    if let Some(client_info) = &client_info {
        tracing::info!(
            "Client '{}' {}",
            client_info.name,
            client_info.version.as_deref().unwrap_or_default()
        );
    }

    let workspace_roots = workspace_folders_initialize_params
        .workspace_folders
        .and_then(|workspaces| match workspaces {
            lsp_types::WorkspaceFolders::WorkspaceFolderList(workspace_folders) => {
                Some(workspace_folders)
            }
            lsp_types::WorkspaceFolders::Null => None,
        })
        .map(|workspaces| {
            workspaces
                .into_iter()
                .filter_map(|it| it.uri.to_file_path().ok())
                .map(patch_path_prefix)
                .filter_map(|it| Utf8PathBuf::from_path_buf(it).ok())
                .filter_map(|it| AbsPathBuf::try_from(it).ok())
                .collect::<Vec<_>>()
        })
        .filter(|workspaces| !workspaces.is_empty())
        .unwrap_or_else(|| vec![root_path.clone()]);
    let mut config = Config::new(root_path, capabilities, workspace_roots, client_info);
    if let Some(json) = initialization_options {
        let mut change = ConfigChange::default();
        change.change_client_config(json);

        let error_sink: ConfigErrors;
        (config, error_sink, _) = config.apply_change(change);

        if !error_sink.is_empty() {
            use lsp_types::{
                MessageType, Notification as _, ShowMessageNotification, ShowMessageParams,
            };
            let not = lsp_server::Notification::new(
                ShowMessageNotification::METHOD.into(),
                ShowMessageParams { kind: MessageType::Warning, message: error_sink.to_string() },
            );
            connection.sender.send(lsp_server::Message::Notification(not)).unwrap();
        }
    }

    let server_capabilities = crate::server_capabilities(&config);

    let initialize_result = lsp_types::InitializeResult {
        capabilities: server_capabilities,
        server_info: Some(lsp_types::ServerInfo {
            name: String::from("rust-analyzer"),
            version: Some(crate::version().to_string()),
        }),
    };

    let initialize_result = serde_json::to_value(initialize_result).unwrap();

    if let Err(e) = connection.initialize_finish(initialize_id, initialize_result) {
        if e.channel_is_disconnected() {
            io_threads.join()?;
        }
        return Err(e.into());
    }

    if let Some(notice) = startup_notice {
        use lsp_types::{
            MessageType, Notification as _, ShowMessageNotification, ShowMessageParams,
        };
        let not = lsp_server::Notification::new(
            ShowMessageNotification::METHOD.into(),
            ShowMessageParams { kind: MessageType::Warning, message: notice },
        );
        connection.sender.send(lsp_server::Message::Notification(not)).unwrap();
    }

    if config.discover_workspace_config().is_none()
        && !config.has_linked_projects()
        && config.detached_files().is_empty()
    {
        config.rediscover_workspaces();
    }

    // If the io_threads have an error, there's usually an error on the main
    // loop too because the channels are closed. Ensure we report both errors.
    match (crate::main_loop(config, connection), io_threads.join()) {
        (Err(loop_e), Err(join_e)) => anyhow::bail!("{loop_e}\n{join_e}"),
        (Ok(_), Err(join_e)) => anyhow::bail!("{join_e}"),
        (Err(loop_e), Ok(_)) => anyhow::bail!("{loop_e}"),
        (Ok(_), Ok(_)) => {}
    }

    tracing::info!("server did shut down");
    Ok(())
}

fn patch_path_prefix(path: PathBuf) -> PathBuf {
    use std::path::{Component, Prefix};
    if cfg!(windows) {
        // VSCode might report paths with the file drive in lowercase, but this can mess
        // with env vars set by tools and build scripts executed by r-a such that it invalidates
        // cargo's compilations unnecessarily. https://github.com/rust-lang/rust-analyzer/issues/14683
        // So we just uppercase the drive letter here unconditionally.
        // (doing it conditionally is a pain because std::path::Prefix always reports uppercase letters on windows)
        let mut comps = path.components();
        match comps.next() {
            Some(Component::Prefix(prefix)) => {
                let prefix = match prefix.kind() {
                    Prefix::Disk(d) => {
                        format!("{}:", d.to_ascii_uppercase() as char)
                    }
                    Prefix::VerbatimDisk(d) => {
                        format!(r"\\?\{}:", d.to_ascii_uppercase() as char)
                    }
                    _ => return path,
                };
                let mut path = PathBuf::new();
                path.push(prefix);
                path.extend(comps);
                path
            }
            _ => path,
        }
    } else {
        path
    }
}

#[test]
#[cfg(windows)]
fn patch_path_prefix_works() {
    assert_eq!(patch_path_prefix(r"c:\foo\bar".into()), PathBuf::from(r"C:\foo\bar"));
    assert_eq!(patch_path_prefix(r"\\?\c:\foo\bar".into()), PathBuf::from(r"\\?\C:\foo\bar"));
}
