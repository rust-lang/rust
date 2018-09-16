#[macro_use]
extern crate log;
#[macro_use]
extern crate failure;
extern crate flexi_logger;
extern crate gen_lsp_server;
extern crate ra_lsp_server;

use flexi_logger::{Logger, Duplicate};
use gen_lsp_server::{run_server, stdio_transport};
use ra_lsp_server::Result;

fn main() -> Result<()> {
    ::std::env::set_var("RUST_BACKTRACE", "short");
    Logger::with_env_or_str("error")
        .duplicate_to_stderr(Duplicate::All)
        .log_to_file()
        .directory("log")
        .start()?;
    info!("lifecycle: server started");
    match ::std::panic::catch_unwind(|| main_inner()) {
        Ok(res) => {
            info!("lifecycle: terminating process with {:?}", res);
            res
        }
        Err(_) => {
            error!("server panicked");
            bail!("server panicked")
        }
    }
}

fn main_inner() -> Result<()> {
    let (receiver, sender, threads) = stdio_transport();
    let cwd = ::std::env::current_dir()?;
    run_server(
        ra_lsp_server::server_capabilities(),
        |params, r, s| {
            let root = params.root_uri
                .and_then(|it| it.to_file_path().ok())
                .unwrap_or(cwd);
            ra_lsp_server::main_loop(false, root, r, s)
        },
        receiver,
        sender,
    )?;
    info!("shutting down IO...");
    threads.join()?;
    info!("... IO is down");
    Ok(())
}

