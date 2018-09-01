#[macro_use]
extern crate failure;
#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;
extern crate languageserver_types;
extern crate drop_bomb;
#[macro_use]
extern crate crossbeam_channel;
extern crate threadpool;
#[macro_use]
extern crate log;
extern crate url_serde;
extern crate flexi_logger;
extern crate walkdir;
extern crate libeditor;
extern crate libanalysis;
extern crate libsyntax2;
extern crate gen_lsp_server;
extern crate im;
extern crate relative_path;

mod caps;
mod req;
mod conv;
mod main_loop;
mod vfs;
mod path_map;
mod server_world;

use flexi_logger::{Logger, Duplicate};
use gen_lsp_server::{run_server, stdio_transport};

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;

fn main() -> Result<()> {
    Logger::with_env_or_str("m=error")
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
    let root = ::std::env::current_dir()?;
    run_server(
        caps::server_capabilities(),
        |r, s| main_loop::main_loop(root, r, s),
        receiver,
        sender,
    )?;
    info!("shutting down IO...");
    threads.join()?;
    info!("... IO is down");
    Ok(())
}
