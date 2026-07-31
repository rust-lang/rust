use std::io::{self, BufReader, BufWriter};

use emmy_dap_types::prelude::types::Capabilities;
use emmy_dap_types::prelude::{Command, Event, Request, ResponseBody, Server};
use miri::{InterpResult, interp_ok};

use crate::debugger::PrirodaContext;

const MAX_REQUEST_COUNT: usize = 128;
type ServerResult<T = ()> = Result<T, emmy_dap_types::errors::ServerError>;

enum DispatchOutcome {
    Continue,
    Exit,
}

/// Debug Adapter Protocol frontend.
pub(crate) struct Dap;

impl Dap {
    /// Serve DAP requests on stdin/stdout.
    pub(crate) fn run_dap_loop<'tcx>(
        &self,
        _session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx> {
        // FIXME: make this unbounded once Priroda has a full session lifecycle.
        if let Err(err) = DapSession::stdio().run_requests() {
            eprintln!("priroda dap error: {err}");
        }

        interp_ok(())
    }
}

type DapServer = Server<io::StdinLock<'static>, io::StdoutLock<'static>>;

/// Owns the DAP stdio transport and dispatches requests into Priroda handlers.
struct DapSession {
    server: DapServer,
}

impl DapSession {
    fn stdio() -> Self {
        Self {
            server: Server::new(
                BufReader::new(io::stdin().lock()),
                BufWriter::new(io::stdout().lock()),
            ),
        }
    }

    fn run_requests(&mut self) -> ServerResult {
        for _ in 0..MAX_REQUEST_COUNT {
            let Some(request) = self.server.poll_request()? else {
                return Ok(());
            };

            match self.dispatch_request(request)? {
                DispatchOutcome::Continue => {}
                DispatchOutcome::Exit => return Ok(()),
            }
        }

        Ok(())
    }

    fn dispatch_request(&mut self, request: Request) -> ServerResult<DispatchOutcome> {
        match &request.command {
            Command::Initialize(_) =>
                self.handle_initialize(request).map(|()| DispatchOutcome::Continue),
            Command::Launch(_) => self.handle_launch(request).map(|()| DispatchOutcome::Continue),
            Command::ConfigurationDone =>
                self.handle_configuration_done(request).map(|()| DispatchOutcome::Continue),
            _ => self.handle_unsupported_request(request).map(|()| DispatchOutcome::Exit),
        }
    }

    /// FIXME: connect launch arguments to Priroda's session model.
    fn handle_launch(&mut self, request: Request) -> ServerResult {
        let response = request.success(ResponseBody::Launch);
        self.server.respond(response)
    }

    fn handle_configuration_done(&mut self, request: Request) -> ServerResult {
        let response = request.success(ResponseBody::ConfigurationDone);
        self.server.respond(response)
    }

    /// FIXME: grow capabilities as Priroda adds DAP features.
    fn handle_initialize(&mut self, request: Request) -> ServerResult {
        // Advertise configurationDone support ahead of its handler so VS Code
        // completes the full handshake; the handler arrives in a later commit.
        let response = request.success(ResponseBody::Initialize(Capabilities {
            supports_configuration_done_request: Some(true),
            ..Capabilities::default()
        }));
        self.server.respond(response)?;
        self.server.send_event(Event::Initialized)
    }

    fn handle_unsupported_request(&mut self, request: Request) -> ServerResult {
        eprintln!(
            "priroda dap: unsupported request during DAP demo milestone: {}",
            Self::display_command(&request.command)
        );
        let response = request.error("unsupported request in Priroda DAP demo mode");
        self.server.respond(response)
    }

    fn display_command(command: &Command) -> &'static str {
        match command {
            Command::Initialize(_) => "initialize",
            Command::Launch(_) => "launch",
            Command::ConfigurationDone => "configurationDone",
            Command::Threads => "threads",
            Command::StackTrace(_) => "stackTrace",
            Command::Scopes(_) => "scopes",
            Command::Variables(_) => "variables",
            Command::Next(_) => "next",
            Command::StepIn(_) => "stepIn",
            Command::Disconnect(_) => "disconnect",
            _ => "unsupported",
        }
    }
}
