#[macro_use]
extern crate log;
#[macro_use]
extern crate failure;
extern crate serde_derive;
extern crate serde;
extern crate flexi_logger;
extern crate gen_lsp_server;
extern crate ra_lsp_server;

use serde::Deserialize;
use flexi_logger::{Duplicate, Logger};
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
    match ::std::panic::catch_unwind(main_inner) {
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

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct InitializationOptions {
    publish_decorations: bool,
}

fn main_inner() -> Result<()> {
    let (receiver, sender, threads) = stdio_transport();
    let cwd = ::std::env::current_dir()?;
    run_server(
        ra_lsp_server::server_capabilities(),
        receiver,
        sender,
        |params, r, s| {
            let root = params
                .root_uri
                .and_then(|it| it.to_file_path().ok())
                .unwrap_or(cwd);
            let publish_decorations = params
                .initialization_options
                .and_then(|v| InitializationOptions::deserialize(v).ok())
                .map(|it| it.publish_decorations)
                == Some(true);
            ra_lsp_server::main_loop(false, root, publish_decorations, r, s)
        },
    )?;
    info!("shutting down IO...");
    threads.join()?;
    info!("... IO is down");
    Ok(())
}

/*
                    (let ((backend (eglot-xref-backend)))
                      (mapcar
                       (lambda (xref)
                         (let ((loc (xref-item-location xref)))
                           (propertize
                            (concat
                             (when (xref-file-location-p loc)
                               (with-slots (file line column) loc
                                 (format "%s:%s:%s:"
                                         (propertize (file-relative-name file)
                                                     'face 'compilation-info)
                                         (propertize (format "%s" line)
                                                     'face 'compilation-line
                                                     )
                                         column)))
                             (xref-item-summary xref))
                            'xref xref)))
                       (xref-backend-apropos backend "Analysis"))
                      )


*/
