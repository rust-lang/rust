//! Driver for rust-analyzer.
//!
//! Based on cli flags, either spawns an LSP server, or runs a batch analysis
mod args;
mod logger;

use std::{convert::TryFrom, env, fs, path::PathBuf, process};

use lsp_server::Connection;
use project_model::ProjectManifest;
use rust_analyzer::{cli, config::Config, from_json, Result};
use vfs::AbsPathBuf;

#[cfg(all(feature = "mimalloc"))]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() {
    if let Err(err) = try_main() {
        log::error!("Unexpected error: {}", err);
        eprintln!("{}", err);
        process::exit(101);
    }
}

fn try_main() -> Result<()> {
    let args = args::Args::parse()?;

    #[cfg(debug_assertions)]
    if args.wait_dbg || env::var("RA_WAIT_DBG").is_ok() {
        #[allow(unused_mut)]
        let mut d = 4;
        while d == 4 {
            d = 4;
        }
    }

    setup_logging(args.log_file, args.no_buffering)?;
    match args.command {
        args::Command::RunServer => run_server()?,
        args::Command::PrintConfigSchema => {
            println!("{:#}", Config::json_schema());
        }
        args::Command::ProcMacro => proc_macro_srv::cli::run()?,

        args::Command::Parse { no_dump } => cli::parse(no_dump)?,
        args::Command::Symbols => cli::symbols()?,
        args::Command::Highlight { rainbow } => cli::highlight(rainbow)?,
        args::Command::AnalysisStats(cmd) => cmd.run(args.verbosity)?,
        args::Command::Bench(cmd) => cmd.run(args.verbosity)?,
        args::Command::Diagnostics { path, load_output_dirs, with_proc_macro } => {
            cli::diagnostics(path.as_ref(), load_output_dirs, with_proc_macro)?
        }
        args::Command::Ssr { rules } => {
            cli::apply_ssr_rules(rules)?;
        }
        args::Command::StructuredSearch { patterns, debug_snippet } => {
            cli::search_for_patterns(patterns, debug_snippet)?;
        }
        args::Command::Version => println!("rust-analyzer {}", env!("REV")),
        args::Command::Help => {}
    }
    Ok(())
}

fn setup_logging(log_file: Option<PathBuf>, no_buffering: bool) -> Result<()> {
    env::set_var("RUST_BACKTRACE", "short");

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
    logger::Logger::new(log_file, no_buffering, filter.as_deref()).install();

    tracing_setup::setup_tracing()?;

    profile::init();

    Ok(())
}

mod tracing_setup {
    use tracing::subscriber;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::Registry;
    use tracing_tree::HierarchicalLayer;

    pub(crate) fn setup_tracing() -> super::Result<()> {
        let filter = EnvFilter::from_env("CHALK_DEBUG");
        let layer = HierarchicalLayer::default()
            .with_indent_lines(true)
            .with_ansi(false)
            .with_indent_amount(2)
            .with_writer(std::io::stderr);
        let subscriber = Registry::default().with(filter).with(layer);
        subscriber::set_global_default(subscriber)?;
        Ok(())
    }
}

fn run_server() -> Result<()> {
    log::info!("server will start");

    let (connection, io_threads) = Connection::stdio();

    let (initialize_id, initialize_params) = connection.initialize_start()?;
    log::info!("InitializeParams: {}", initialize_params);
    let initialize_params =
        from_json::<lsp_types::InitializeParams>("InitializeParams", initialize_params)?;

    let server_capabilities = rust_analyzer::server_capabilities(&initialize_params.capabilities);

    let initialize_result = lsp_types::InitializeResult {
        capabilities: server_capabilities,
        server_info: Some(lsp_types::ServerInfo {
            name: String::from("rust-analyzer"),
            version: Some(String::from(env!("REV"))),
        }),
    };

    let initialize_result = serde_json::to_value(initialize_result).unwrap();

    connection.initialize_finish(initialize_id, initialize_result)?;

    if let Some(client_info) = initialize_params.client_info {
        log::info!("Client '{}' {}", client_info.name, client_info.version.unwrap_or_default());
    }

    let config = {
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
            config.update(json);
        }

        if config.linked_projects().is_empty() {
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
                .unwrap_or_else(|| vec![config.root_path.clone()]);

            let discovered = ProjectManifest::discover_all(&workspace_roots);
            log::info!("discovered projects: {:?}", discovered);
            if discovered.is_empty() {
                log::error!("failed to find any projects in {:?}", workspace_roots);
            }

            config.discovered_projects = Some(discovered);
        }

        config
    };

    rust_analyzer::main_loop(config, connection)?;

    io_threads.join()?;
    log::info!("server did shut down");
    Ok(())
}
