use serde::Deserialize;
use flexi_logger::{Duplicate, Logger};
use gen_lsp_server::{run_server, stdio_transport};

use ra_lsp_server::Result;

fn main() -> Result<()> {
    ::std::env::set_var("RUST_BACKTRACE", "short");
    let logger = Logger::with_env_or_str("error").duplicate_to_stderr(Duplicate::All);
    match ::std::env::var("RA_INTERNAL_MODE") {
        Ok(ref v) if v == "1" => logger.log_to_file().directory("log").start()?,
        _ => logger.start()?,
    };
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

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct InitializationOptions {
    // Whether the client supports our custom highlighting publishing decorations.
    // This is different to the highlightingOn setting, which is whether the user
    // wants our custom highlighting to be used.
    publish_decorations: Option<bool>,
}

fn main_inner() -> Result<()> {
    let (receiver, sender, threads) = stdio_transport();
    let cwd = ::std::env::current_dir()?;
    run_server(ra_lsp_server::server_capabilities(), receiver, sender, |params, r, s| {
        let root = params.root_uri.and_then(|it| it.to_file_path().ok()).unwrap_or(cwd);
        let supports_decorations = params
            .initialization_options
            .and_then(|v| InitializationOptions::deserialize(v).ok())
            .and_then(|it| it.publish_decorations)
            == Some(true);
        ra_lsp_server::main_loop(false, root, supports_decorations, r, s)
    })?;
    log::info!("shutting down IO...");
    threads.join()?;
    log::info!("... IO is down");
    Ok(())
}
