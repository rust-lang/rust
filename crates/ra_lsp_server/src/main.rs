//! `ra_lsp_server` binary

use lsp_server::Connection;
use ra_lsp_server::{from_json, show_message, Result, ServerConfig};
use ra_prof;

fn main() -> Result<()> {
    setup_logging()?;
    match Args::parse()? {
        Args::Version => println!("rust-analyzer {}", env!("REV")),
        Args::Run => run_server()?,
    }
    Ok(())
}

fn setup_logging() -> Result<()> {
    std::env::set_var("RUST_BACKTRACE", "short");

    env_logger::try_init()?;

    ra_prof::set_filter(match std::env::var("RA_PROFILE") {
        Ok(spec) => ra_prof::Filter::from_spec(&spec),
        Err(_) => ra_prof::Filter::disabled(),
    });
    Ok(())
}

enum Args {
    Version,
    Run,
}

impl Args {
    fn parse() -> Result<Args> {
        let res =
            if std::env::args().any(|it| it == "--version") { Args::Version } else { Args::Run };
        Ok(res)
    }
}

fn run_server() -> Result<()> {
    log::info!("lifecycle: server started");

    let (connection, io_threads) = Connection::stdio();
    let server_capabilities = serde_json::to_value(ra_lsp_server::server_capabilities()).unwrap();

    let initialize_params = connection.initialize(server_capabilities)?;
    let initialize_params =
        from_json::<lsp_types::InitializeParams>("InitializeParams", initialize_params)?;

    if let Some(client_info) = initialize_params.client_info {
        log::info!("Client '{}' {}", client_info.name, client_info.version.unwrap_or_default());
    }

    let cwd = std::env::current_dir()?;
    let root = initialize_params.root_uri.and_then(|it| it.to_file_path().ok()).unwrap_or(cwd);

    let workspace_roots = initialize_params
        .workspace_folders
        .map(|workspaces| {
            workspaces.into_iter().filter_map(|it| it.uri.to_file_path().ok()).collect::<Vec<_>>()
        })
        .filter(|workspaces| !workspaces.is_empty())
        .unwrap_or_else(|| vec![root]);

    let server_config = initialize_params
        .initialization_options
        .and_then(|v| {
            from_json::<ServerConfig>("config", v)
                .map_err(|e| {
                    log::error!("{}", e);
                    show_message(lsp_types::MessageType::Error, e.to_string(), &connection.sender);
                })
                .ok()
        })
        .unwrap_or_default();

    ra_lsp_server::main_loop(
        workspace_roots,
        initialize_params.capabilities,
        server_config,
        connection,
    )?;

    log::info!("shutting down IO...");
    io_threads.join()?;
    log::info!("... IO is down");
    Ok(())
}
