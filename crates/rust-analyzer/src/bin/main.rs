//! Driver for rust-analyzer.
//!
//! Based on cli flags, either spawns an LSP server, or runs a batch analysis

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

mod logger;
mod rustc_wrapper;

use std::{env, fs, path::Path, process};

use lsp_server::Connection;
use project_model::ProjectManifest;
use rust_analyzer::{cli::flags, config::Config, from_json, lsp_ext::supports_utf8, Result};
use vfs::AbsPathBuf;

#[cfg(all(feature = "mimalloc"))]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() {
    if std::env::var("RA_RUSTC_WRAPPER").is_ok() {
        let mut args = std::env::args_os();
        let _me = args.next().unwrap();
        let rustc = args.next().unwrap();
        let code = match rustc_wrapper::run_rustc_skipping_cargo_checking(rustc, args.collect()) {
            Ok(rustc_wrapper::ExitCode(code)) => code.unwrap_or(102),
            Err(err) => {
                eprintln!("{}", err);
                101
            }
        };
        process::exit(code);
    }

    if let Err(err) = try_main() {
        tracing::error!("Unexpected error: {}", err);
        eprintln!("{}", err);
        process::exit(101);
    }
}

fn try_main() -> Result<()> {
    let flags = flags::RustAnalyzer::from_env()?;

    #[cfg(debug_assertions)]
    if flags.wait_dbg || env::var("RA_WAIT_DBG").is_ok() {
        #[allow(unused_mut)]
        let mut d = 4;
        while d == 4 {
            d = 4;
        }
    }

    let mut log_file = flags.log_file.as_deref();

    let env_log_file = env::var("RA_LOG_FILE").ok();
    if let Some(env_log_file) = env_log_file.as_deref() {
        log_file = Some(Path::new(env_log_file));
    }

    setup_logging(log_file)?;
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
            if cmd.help {
                println!("{}", flags::RustAnalyzer::HELP);
                return Ok(());
            }
            with_extra_thread("LspServer", run_server)?;
        }
        flags::RustAnalyzerCmd::ProcMacro(flags::ProcMacro) => {
            with_extra_thread("MacroExpander", || proc_macro_srv::cli::run().map_err(Into::into))?;
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
    }
    Ok(())
}

fn setup_logging(log_file: Option<&Path>) -> Result<()> {
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

    let log_file = match log_file {
        Some(path) => {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            Some(fs::File::create(path)?)
        }
        None => None,
    };
    let filter = env::var("RA_LOG").ok();
    // deliberately enable all `error` logs if the user has not set RA_LOG, as there is usually useful
    // information in there for debugging
    logger::Logger::new(log_file, filter.as_deref().or(Some("error"))).install()?;

    profile::init();

    Ok(())
}

const STACK_SIZE: usize = 1024 * 1024 * 8;

/// Parts of rust-analyzer can use a lot of stack space, and some operating systems only give us
/// 1 MB by default (eg. Windows), so this spawns a new thread with hopefully sufficient stack
/// space.
fn with_extra_thread(
    thread_name: impl Into<String>,
    f: impl FnOnce() -> Result<()> + Send + 'static,
) -> Result<()> {
    let handle =
        std::thread::Builder::new().name(thread_name.into()).stack_size(STACK_SIZE).spawn(f)?;
    match handle.join() {
        Ok(res) => res,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

fn run_server() -> Result<()> {
    tracing::info!("server version {} will start", rust_analyzer::version());

    let (connection, io_threads) = Connection::stdio();

    let (initialize_id, initialize_params) = connection.initialize_start()?;
    tracing::info!("InitializeParams: {}", initialize_params);
    let initialize_params =
        from_json::<lsp_types::InitializeParams>("InitializeParams", &initialize_params)?;

    let root_path = match initialize_params
        .root_uri
        .and_then(|it| it.to_file_path().ok())
        .and_then(|it| AbsPathBuf::try_from(it).ok())
    {
        Some(it) => it,
        None => {
            let cwd = env::current_dir()?;
            AbsPathBuf::assert(cwd)
        }
    };

    let mut config = Config::new(root_path, initialize_params.capabilities);
    if let Some(json) = initialize_params.initialization_options {
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
        offset_encoding: if supports_utf8(config.caps()) {
            Some("utf-8".to_string())
        } else {
            None
        },
    };

    let initialize_result = serde_json::to_value(initialize_result).unwrap();

    connection.initialize_finish(initialize_id, initialize_result)?;

    if let Some(client_info) = initialize_params.client_info {
        tracing::info!("Client '{}' {}", client_info.name, client_info.version.unwrap_or_default());
    }

    if config.linked_projects().is_empty() && config.detached_files().is_empty() {
        let workspace_roots = initialize_params
            .workspace_folders
            .map(|workspaces| {
                workspaces
                    .into_iter()
                    .filter_map(|it| it.uri.to_file_path().ok())
                    .filter_map(|it| AbsPathBuf::try_from(it).ok())
                    .collect::<Vec<_>>()
            })
            .filter(|workspaces| !workspaces.is_empty())
            .unwrap_or_else(|| vec![config.root_path().clone()]);

        let discovered = ProjectManifest::discover_all(&workspace_roots);
        tracing::info!("discovered projects: {:?}", discovered);
        if discovered.is_empty() {
            tracing::error!("failed to find any projects in {:?}", workspace_roots);
        }
        config.discovered_projects = Some(discovered);
    }

    rust_analyzer::main_loop(config, connection)?;

    io_threads.join()?;
    tracing::info!("server did shut down");
    Ok(())
}
