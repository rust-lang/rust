use serde::Deserialize;
use flexi_logger::{Duplicate, Logger};
use gen_lsp_server::{run_server, stdio_transport};

use ra_lsp_server::{Result, InitializationOptions};
use ra_prof;

fn main() -> Result<()> {
    std::env::set_var("RUST_BACKTRACE", "short");
    let logger = Logger::with_env_or_str("error").duplicate_to_stderr(Duplicate::All);
    match std::env::var("RA_INTERNAL_MODE") {
        Ok(ref v) if v == "1" => logger.log_to_file().directory("log").start()?,
        _ => logger.start()?,
    };
    // Filtering syntax
    // env RA_PROFILE=*             // dump everything
    // env RA_PROFILE=foo|bar|baz   // enabled only selected entries
    // env RA_PROFILE=*@3           // dump everything, up to depth 3
    let filter = match std::env::var("RA_PROFILE") {
        Ok(p) => {
            let mut p = p.as_str();
            let depth = if let Some(idx) = p.rfind("@") {
                let depth: usize = p[idx + 1..].parse().expect("invalid profile depth");
                p = &p[..idx];
                depth
            } else {
                999
            };
            let allowed =
                if p == "*" { Vec::new() } else { p.split(";").map(String::from).collect() };
            ra_prof::Filter::new(depth, allowed)
        }
        Err(_) => ra_prof::Filter::disabled(),
    };
    ra_prof::set_filter(filter);
    log::info!("lifecycle: server started");
    match ::std::panic::catch_unwind(main_inner) {
        Ok(res) => {
            log::info!("lifecycle: terminating process with {:?}", res);
            res
        }
        Err(_) => {
            log::error!("server panicked");
            failure::bail!("server panicked")
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
            .and_then(|v| InitializationOptions::deserialize(v).ok())
            .unwrap_or(InitializationOptions::default());

        ra_lsp_server::main_loop(workspace_roots, opts, r, s)
    })?;
    log::info!("shutting down IO...");
    threads.join()?;
    log::info!("... IO is down");
    Ok(())
}
