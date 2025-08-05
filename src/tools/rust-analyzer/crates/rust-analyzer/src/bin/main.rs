//! Driver for rust-analyzer.
//!
//! Based on cli flags, either spawns an LSP server, or runs a batch analysis

#![allow(clippy::print_stdout, clippy::print_stderr)]
#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]

#[cfg(feature = "in-rust-tree")]
extern crate rustc_driver as _;

mod rustc_wrapper;

use std::{env, fs, path::PathBuf, process::ExitCode, sync::Arc};

use anyhow::Context;
use lsp_server::Connection;
use paths::Utf8PathBuf;
use rust_analyzer::{
    cli::flags,
    config::{Config, ConfigChange, ConfigErrors},
    from_json,
};
use tracing_subscriber::fmt::writer::BoxMakeWriter;
use vfs::AbsPathBuf;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() -> anyhow::Result<ExitCode> {
    if std::env::var("RA_RUSTC_WRAPPER").is_ok() {
        rustc_wrapper::main().map_err(Into::into)
    } else {
        actual_main()
    }
}

fn actual_main() -> anyhow::Result<ExitCode> {
    let flags = flags::RustAnalyzer::from_env_or_exit();

    #[cfg(debug_assertions)]
    if flags.wait_dbg || env::var("RA_WAIT_DBG").is_ok() {
        wait_for_debugger();
    }

    if let Err(e) = setup_logging(flags.log_file.clone()) {
        eprintln!("Failed to setup logging: {e:#}");
    }

    let verbosity = flags.verbosity();

    match flags.subcommand {
        flags::RustAnalyzerCmd::LspServer(cmd) => 'lsp_server: {
            if cmd.print_config_schema {
                println!("{:#}", Config::json_schema());
                break 'lsp_server;
            }
            if cmd.version {
                println!("rust-analyzer {}", rust_analyzer::version());
                break 'lsp_server;
            }

            // rust-analyzer’s “main thread” is actually
            // a secondary latency-sensitive thread with an increased stack size.
            // We use this thread intent because any delay in the main loop
            // will make actions like hitting enter in the editor slow.
            with_extra_thread(
                "LspServer",
                stdx::thread::ThreadIntent::LatencySensitive,
                run_server,
            )?;
        }
        flags::RustAnalyzerCmd::Parse(cmd) => cmd.run()?,
        flags::RustAnalyzerCmd::Symbols(cmd) => cmd.run()?,
        flags::RustAnalyzerCmd::Highlight(cmd) => cmd.run()?,
        flags::RustAnalyzerCmd::AnalysisStats(cmd) => cmd.run(verbosity)?,
        flags::RustAnalyzerCmd::Diagnostics(cmd) => cmd.run()?,
        flags::RustAnalyzerCmd::UnresolvedReferences(cmd) => cmd.run()?,
        flags::RustAnalyzerCmd::Ssr(cmd) => cmd.run()?,
        flags::RustAnalyzerCmd::Search(cmd) => cmd.run()?,
        flags::RustAnalyzerCmd::Lsif(cmd) => {
            cmd.run(&mut std::io::stdout(), Some(project_model::RustLibSource::Discover))?
        }
        flags::RustAnalyzerCmd::Scip(cmd) => cmd.run()?,
        flags::RustAnalyzerCmd::RunTests(cmd) => cmd.run()?,
        flags::RustAnalyzerCmd::RustcTests(cmd) => cmd.run()?,
        flags::RustAnalyzerCmd::PrimeCaches(cmd) => cmd.run()?,
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(debug_assertions)]
fn wait_for_debugger() {
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::System::Diagnostics::Debug::IsDebuggerPresent;
        // SAFETY: WinAPI generated code that is defensively marked `unsafe` but
        // in practice can not be used in an unsafe way.
        while unsafe { IsDebuggerPresent() } == 0 {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        #[allow(unused_mut)]
        let mut d = 4;
        while d == 4 {
            d = 4;
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}

fn setup_logging(log_file_flag: Option<PathBuf>) -> anyhow::Result<()> {
    if cfg!(windows) {
        // This is required so that windows finds our pdb that is placed right beside the exe.
        // By default it doesn't look at the folder the exe resides in, only in the current working
        // directory which we set to the project workspace.
        // https://docs.microsoft.com/en-us/windows-hardware/drivers/debugger/general-environment-variables
        // https://docs.microsoft.com/en-us/windows/win32/api/dbghelp/nf-dbghelp-syminitialize
        if let Ok(path) = env::current_exe()
            && let Some(path) = path.parent()
        {
            // SAFETY: This is safe because this is single-threaded.
            unsafe {
                env::set_var("_NT_SYMBOL_PATH", path);
            }
        }
    }

    if env::var("RUST_BACKTRACE").is_err() {
        // SAFETY: This is safe because this is single-threaded.
        unsafe {
            env::set_var("RUST_BACKTRACE", "short");
        }
    }

    let log_file = env::var("RA_LOG_FILE").ok().map(PathBuf::from).or(log_file_flag);
    let log_file = match log_file {
        Some(path) => {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            Some(
                fs::File::create(&path)
                    .with_context(|| format!("can't create log file at {}", path.display()))?,
            )
        }
        None => None,
    };

    let writer = match log_file {
        Some(file) => BoxMakeWriter::new(Arc::new(file)),
        None => BoxMakeWriter::new(std::io::stderr),
    };

    rust_analyzer::tracing::Config {
        writer,
        // Deliberately enable all `error` logs if the user has not set RA_LOG, as there is usually
        // useful information in there for debugging.
        filter: env::var("RA_LOG").ok().unwrap_or_else(|| "error".to_owned()),
        chalk_filter: env::var("CHALK_DEBUG").ok(),
        profile_filter: env::var("RA_PROFILE").ok(),
        json_profile_filter: std::env::var("RA_PROFILE_JSON").ok(),
    }
    .init()?;

    Ok(())
}

const STACK_SIZE: usize = 1024 * 1024 * 8;

/// Parts of rust-analyzer can use a lot of stack space, and some operating systems only give us
/// 1 MB by default (eg. Windows), so this spawns a new thread with hopefully sufficient stack
/// space.
fn with_extra_thread(
    thread_name: impl Into<String>,
    thread_intent: stdx::thread::ThreadIntent,
    f: impl FnOnce() -> anyhow::Result<()> + Send + 'static,
) -> anyhow::Result<()> {
    let handle =
        stdx::thread::Builder::new(thread_intent, thread_name).stack_size(STACK_SIZE).spawn(f)?;

    handle.join()?;

    Ok(())
}

fn run_server() -> anyhow::Result<()> {
    tracing::info!("server version {} will start", rust_analyzer::version());

    let (connection, io_threads) = Connection::stdio();

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
        root_uri,
        capabilities,
        workspace_folders,
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
            let cwd = env::current_dir()?;
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

    let workspace_roots = workspace_folders
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
                MessageType, ShowMessageParams,
                notification::{Notification, ShowMessage},
            };
            let not = lsp_server::Notification::new(
                ShowMessage::METHOD.to_owned(),
                ShowMessageParams { typ: MessageType::WARNING, message: error_sink.to_string() },
            );
            connection.sender.send(lsp_server::Message::Notification(not)).unwrap();
        }
    }

    let server_capabilities = rust_analyzer::server_capabilities(&config);

    let initialize_result = lsp_types::InitializeResult {
        capabilities: server_capabilities,
        server_info: Some(lsp_types::ServerInfo {
            name: String::from("rust-analyzer"),
            version: Some(rust_analyzer::version().to_string()),
        }),
        offset_encoding: None,
    };

    let initialize_result = serde_json::to_value(initialize_result).unwrap();

    if let Err(e) = connection.initialize_finish(initialize_id, initialize_result) {
        if e.channel_is_disconnected() {
            io_threads.join()?;
        }
        return Err(e.into());
    }

    if config.discover_workspace_config().is_none()
        && !config.has_linked_projects()
        && config.detached_files().is_empty()
    {
        config.rediscover_workspaces();
    }

    // If the io_threads have an error, there's usually an error on the main
    // loop too because the channels are closed. Ensure we report both errors.
    match (rust_analyzer::main_loop(config, connection), io_threads.join()) {
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
