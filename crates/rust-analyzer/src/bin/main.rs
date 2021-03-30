//! Driver for rust-analyzer.
//!
//! Based on cli flags, either spawns an LSP server, or runs a batch analysis
mod flags;
mod logger;

use std::{convert::TryFrom, env, fs, path::Path, process};

use lsp_server::Connection;
use project_model::ProjectManifest;
use rust_analyzer::{
    cli::{self, AnalysisStatsCmd},
    config::Config,
    from_json,
    lsp_ext::supports_utf8,
    Result,
};
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
    let flags = flags::RustAnalyzer::from_env()?;

    #[cfg(debug_assertions)]
    if flags.wait_dbg || env::var("RA_WAIT_DBG").is_ok() {
        #[allow(unused_mut)]
        let mut d = 4;
        while d == 4 {
            d = 4;
        }
    }

    setup_logging(flags.log_file.as_deref(), flags.no_log_buffering)?;
    let verbosity = flags.verbosity();

    match flags.subcommand {
        flags::RustAnalyzerCmd::LspServer(cmd) => {
            if cmd.print_config_schema {
                println!("{:#}", Config::json_schema());
                return Ok(());
            }
            if cmd.version {
                println!("rust-analyzer {}", env!("REV"));
                return Ok(());
            }
            if cmd.help {
                println!("{}", flags::RustAnalyzer::HELP);
                return Ok(());
            }
            run_server()?
        }
        flags::RustAnalyzerCmd::ProcMacro(_) => proc_macro_srv::cli::run()?,
        flags::RustAnalyzerCmd::Parse(cmd) => cli::parse(cmd.no_dump)?,
        flags::RustAnalyzerCmd::Symbols(_) => cli::symbols()?,
        flags::RustAnalyzerCmd::Highlight(cmd) => cli::highlight(cmd.rainbow)?,
        flags::RustAnalyzerCmd::AnalysisStats(cmd) => AnalysisStatsCmd {
            randomize: cmd.randomize,
            parallel: cmd.parallel,
            memory_usage: cmd.memory_usage,
            only: cmd.only,
            with_deps: cmd.with_deps,
            no_sysroot: cmd.no_sysroot,
            path: cmd.path,
            load_output_dirs: cmd.load_output_dirs,
            with_proc_macro: cmd.with_proc_macro,
        }
        .run(verbosity)?,

        flags::RustAnalyzerCmd::Diagnostics(cmd) => {
            cli::diagnostics(&cmd.path, cmd.load_output_dirs, cmd.with_proc_macro)?
        }
        flags::RustAnalyzerCmd::Ssr(cmd) => cli::apply_ssr_rules(cmd.rule)?,
        flags::RustAnalyzerCmd::Search(cmd) => cli::search_for_patterns(cmd.pattern, cmd.debug)?,
    }
    Ok(())
}

fn setup_logging(log_file: Option<&Path>, no_buffering: bool) -> Result<()> {
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
        offset_encoding: if supports_utf8(&initialize_params.capabilities) {
            Some("utf-8".to_string())
        } else {
            None
        },
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
