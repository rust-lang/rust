//! Driver for rust-analyzer.
//!
//! Based on cli flags, either spawns an LSP server, or runs a batch analysis

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

mod logger;
mod rustc_wrapper;

use std::{env, fs, path::PathBuf, process};

use anyhow::Context;
use lsp_server::Connection;
use rust_analyzer::{cli::flags, config::Config, from_json};
use vfs::AbsPathBuf;

#[cfg(all(feature = "mimalloc"))]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() -> anyhow::Result<()> {
    if std::env::var("RA_RUSTC_WRAPPER").is_ok() {
        let mut args = std::env::args_os();
        let _me = args.next().unwrap();
        let rustc = args.next().unwrap();
        let code = match rustc_wrapper::run_rustc_skipping_cargo_checking(rustc, args.collect()) {
            Ok(rustc_wrapper::ExitCode(code)) => code.unwrap_or(102),
            Err(err) => {
                eprintln!("{err}");
                101
            }
        };
        process::exit(code);
    }

    let flags = flags::RustAnalyzer::from_env_or_exit();

    #[cfg(debug_assertions)]
    if flags.wait_dbg || env::var("RA_WAIT_DBG").is_ok() {
        #[allow(unused_mut)]
        let mut d = 4;
        while d == 4 {
            d = 4;
        }
    }

    setup_logging(flags.log_file.clone())?;

    let verbosity = flags.verbosity();

    match flags.subcommand {
        flags::RustAnalyzerCmd::LspServer(cmd) => {
            if cmd.print_config_schema {
                println!("{:#}", Config::json_schema());
                return Ok(());
            }
            if cmd.version {
                println!("rust-analyzer {}", rust_analyzer::version());
                return Ok(());
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
        flags::RustAnalyzerCmd::Ssr(cmd) => cmd.run()?,
        flags::RustAnalyzerCmd::Search(cmd) => cmd.run()?,
        flags::RustAnalyzerCmd::Lsif(cmd) => cmd.run()?,
        flags::RustAnalyzerCmd::Scip(cmd) => cmd.run()?,
        flags::RustAnalyzerCmd::RunTests(cmd) => cmd.run()?,
    }
    Ok(())
}

fn setup_logging(log_file_flag: Option<PathBuf>) -> anyhow::Result<()> {
    if cfg!(windows) {
        // This is required so that windows finds our pdb that is placed right beside the exe.
        // By default it doesn't look at the folder the exe resides in, only in the current working
        // directory which we set to the project workspace.
        // https://docs.microsoft.com/en-us/windows-hardware/drivers/debugger/general-environment-variables
        // https://docs.microsoft.com/en-us/windows/win32/api/dbghelp/nf-dbghelp-syminitialize
        if let Ok(path) = env::current_exe() {
            if let Some(path) = path.parent() {
                env::set_var("_NT_SYMBOL_PATH", path);
            }
        }
    }

    if env::var("RUST_BACKTRACE").is_err() {
        env::set_var("RUST_BACKTRACE", "short");
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

    logger::LoggerConfig {
        log_file,
        // Deliberately enable all `error` logs if the user has not set RA_LOG, as there is usually
        // useful information in there for debugging.
        filter: env::var("RA_LOG").ok().unwrap_or_else(|| "error".to_string()),
        // The meaning of CHALK_DEBUG I suspected is to tell chalk crates
        // (i.e. chalk-solve, chalk-ir, chalk-recursive) how to filter tracing
        // logs. But now we can only have just one filter, which means we have to
        // merge chalk filter to our main filter (from RA_LOG env).
        //
        // The acceptable syntax of CHALK_DEBUG is `target[span{field=value}]=level`.
        // As the value should only affect chalk crates, we'd better manually
        // specify the target. And for simplicity, CHALK_DEBUG only accept the value
        // that specify level.
        chalk_filter: env::var("CHALK_DEBUG").ok(),
    }
    .init()?;

    profile::init();

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
    let handle = stdx::thread::Builder::new(thread_intent)
        .name(thread_name.into())
        .stack_size(STACK_SIZE)
        .spawn(f)?;

    handle.join()?;

    Ok(())
}

fn run_server() -> anyhow::Result<()> {
    tracing::info!("server version {} will start", rust_analyzer::version());

    let (connection, io_threads) = Connection::stdio();

    let (initialize_id, initialize_params) = connection.initialize_start()?;
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
        .and_then(|it| AbsPathBuf::try_from(it).ok())
    {
        Some(it) => it,
        None => {
            let cwd = env::current_dir()?;
            AbsPathBuf::assert(cwd)
        }
    };

    let workspace_roots = workspace_folders
        .map(|workspaces| {
            workspaces
                .into_iter()
                .filter_map(|it| it.uri.to_file_path().ok())
                .map(patch_path_prefix)
                .filter_map(|it| AbsPathBuf::try_from(it).ok())
                .collect::<Vec<_>>()
        })
        .filter(|workspaces| !workspaces.is_empty())
        .unwrap_or_else(|| vec![root_path.clone()]);
    let mut config = Config::new(root_path, capabilities, workspace_roots);
    if let Some(json) = initialization_options {
        if let Err(e) = config.update(json) {
            use lsp_types::{
                notification::{Notification, ShowMessage},
                MessageType, ShowMessageParams,
            };
            let not = lsp_server::Notification::new(
                ShowMessage::METHOD.to_string(),
                ShowMessageParams { typ: MessageType::WARNING, message: e.to_string() },
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

    connection.initialize_finish(initialize_id, initialize_result)?;

    if let Some(client_info) = client_info {
        tracing::info!("Client '{}' {}", client_info.name, client_info.version.unwrap_or_default());
    }

    if !config.has_linked_projects() && config.detached_files().is_empty() {
        config.rediscover_workspaces();
    }

    rust_analyzer::main_loop(config, connection)?;

    io_threads.join()?;
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
