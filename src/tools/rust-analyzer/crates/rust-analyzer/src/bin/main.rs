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
use rust_analyzer::{cli::flags, config::Config};
use tracing_subscriber::fmt::writer::BoxMakeWriter;

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
                move || run_server(None),
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
        // Deliberately enable all `warn` logs if the user has not set RA_LOG, as there is usually
        // useful information in there for debugging.
        filter: env::var("RA_LOG").ok().unwrap_or_else(|| "warn".to_owned()),
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

fn run_server(startup_notice: Option<String>) -> anyhow::Result<()> {
    let (connection, io_threads) = Connection::stdio();

    rayon::ThreadPoolBuilder::new()
        .thread_name(|ix| format!("RayonWorker{}", ix))
        .build_global()
        .unwrap();

    rust_analyzer::session::run_session(
        connection,
        rust_analyzer::session::IoThreads::Stdio(io_threads),
        startup_notice,
    )
}
