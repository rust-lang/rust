use flexi_logger::{Duplicate, Logger};
use gen_lsp_server::{run_server, stdio_transport};
use serde::Deserialize;

use ra_lsp_server::{Result, ServerConfig};
use ra_prof;

fn main() -> Result<()> {
    std::env::set_var("RUST_BACKTRACE", "short");
    let logger = Logger::with_env_or_str("error").duplicate_to_stderr(Duplicate::All);
    match std::env::var("RA_LOG_DIR") {
        Ok(ref v) if v == "1" => logger.log_to_file().directory("log").start()?,
        _ => logger.start()?,
    };
    ra_prof::set_filter(match std::env::var("RA_PROFILE") {
        Ok(spec) => ra_prof::Filter::from_spec(&spec),
        Err(_) => ra_prof::Filter::disabled(),
    });
    log::info!("lifecycle: server started");
    match std::panic::catch_unwind(main_inner) {
        Ok(res) => {
            log::info!("lifecycle: terminating process with {:?}", res);
            res
        }
        Err(_) => {
            log::error!("server panicked");
            Err("server panicked")?
        }
    }
}

fn main_inner() -> Result<()> {
    let (receiver, sender, threads) = stdio_transport();
    let cwd = std::env::current_dir()?;
    run_server(ra_lsp_server::server_capabilities(), receiver, sender, |params, r, s| {
        let root = params.root_uri.and_then(|it| it.to_file_path().ok()).unwrap_or(cwd);

        let workspace_roots = params
            .workspace_folders
            .map(|workspaces| {
                workspaces
                    .into_iter()
                    .filter_map(|it| it.uri.to_file_path().ok())
                    .collect::<Vec<_>>()
            })
            .filter(|workspaces| !workspaces.is_empty())
            .unwrap_or_else(|| vec![root]);

        let opts = params
            .initialization_options
            .and_then(|v| ServerConfig::deserialize(v).ok())
            .unwrap_or_default();

        ra_lsp_server::main_loop(workspace_roots, params.capabilities, opts, r, s)
    })?;
    log::info!("shutting down IO...");
    threads.join()?;
    log::info!("... IO is down");
    Ok(())
}
