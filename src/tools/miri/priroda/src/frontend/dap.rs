use std::io::{self, BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};

use emmy_dap_types::errors::ServerError;
use emmy_dap_types::prelude::events::{ExitedEventBody, StoppedEventBody};
use emmy_dap_types::prelude::requests::SetBreakpointsArguments;
use emmy_dap_types::prelude::responses::{
    ContinueResponse, ScopesResponse, SetBreakpointsResponse, StackTraceResponse, ThreadsResponse,
    VariablesResponse,
};
use emmy_dap_types::prelude::types::{
    Breakpoint as DapBreakpoint, Capabilities, Scope, ScopePresentationhint, Source, StackFrame,
    StoppedEventReason, Thread, Variable,
};
use emmy_dap_types::prelude::{Command, Event, Request, ResponseBody, Server};
use miri::{InterpErrorInfo, InterpErrorKind, InterpResult, TerminationInfo, bug};

use crate::debugger::{ExecutionResult, LocalDesc, PrirodaContext, StepResult};

// Priroda still exposes one interpreted thread and one selected frame to DAP.
// Keep the ids stable so editor follow-up requests can address the stopped state.
const THREAD_ID: i64 = 1;
const STACK_FRAME_ID: i64 = 1;
const LOCALS_VARIABLES_REFERENCE: i64 = 1;

enum HandlerResponse {
    Success(ResponseBody),
    Error(String),
}

struct HandlerSuccess {
    response: HandlerResponse,
    state: Option<DapState>,
    events: Vec<Event>,
    outcome: HandlerOutcome,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HandlerOutcome {
    Continue,
    Exit,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DapState {
    Fresh,
    Initialized,
    Launched,
    Stopped,
    Terminated,
}

enum ExecutionOutcome {
    Stopped(StepResult),
    Terminated { code: i32 },
    Failed(String),
}

/// Debug Adapter Protocol frontend.
pub(crate) struct Dap {
    pub(crate) port: Option<u16>,
}

impl Dap {
    /// Serve DAP requests on stdin/stdout, or on a TCP socket if `port` is set.
    pub(crate) fn run_dap_loop<'tcx>(
        &self,
        session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx> {
        let result = if let Some(port) = self.port {
            DapSession::tcp(port).run_requests(session)
        } else {
            DapSession::stdio().run_requests(session)
        };

        if let Err(err) = result {
            eprintln!("priroda dap error: {err:?}");
        }

        session.finish_session()
    }
}

/// Owns a DAP transport and dispatches requests into Priroda handlers.
struct DapSession<R: Read, W: Write> {
    server: Server<R, W>,
    state: DapState,
}

impl DapSession<io::StdinLock<'static>, io::StdoutLock<'static>> {
    fn stdio() -> Self {
        Self {
            server: Server::new(
                BufReader::new(io::stdin().lock()),
                BufWriter::new(io::stdout().lock()),
            ),
            state: DapState::Fresh,
        }
    }
}

impl DapSession<TcpStream, TcpStream> {
    fn tcp(port: u16) -> Self {
        let listener = match TcpListener::bind(("127.0.0.1", port)) {
            Ok(listener) => listener,
            Err(err) => fatal(&format!("failed to listen on DAP TCP socket: {err}")),
        };
        eprintln!("priroda dap listening on 127.0.0.1:{port}");
        let (stream, _) = match listener.accept() {
            Ok(conn) => conn,
            Err(err) => fatal(&format!("failed to accept DAP TCP connection: {err}")),
        };
        let reader = match stream.try_clone() {
            Ok(clone) => clone,
            Err(err) => fatal(&format!("failed to clone DAP TCP stream: {err}")),
        };

        Self {
            server: Server::new(BufReader::new(reader), BufWriter::new(stream)),
            state: DapState::Fresh,
        }
    }
}

fn fatal(message: &str) -> ! {
    eprintln!("priroda dap: {message}");
    std::process::exit(1);
}

impl<R: Read, W: Write> DapSession<R, W> {
    fn run_requests<'tcx>(
        &mut self,
        session: &mut PrirodaContext<'tcx>,
    ) -> Result<(), ServerError> {
        loop {
            let request = match self.server.poll_request() {
                Ok(Some(request)) => request,
                Ok(None) => return Ok(()),
                // The message body has already been consumed. js-debug can send
                // commands like `enableNetworking`, which `emmy_dap_types` reports
                // as parse errors because it has no unknown-command variant.
                // FIXME: send a DAP error response once unknown commands are
                // representable.
                Err(ServerError::ParseError(_)) => {
                    eprintln!("priroda dap: skipping request that could not be deserialized");
                    continue;
                }
                Err(err) => return Err(err),
            };

            match self.dispatch_request(&request, session) {
                Ok(s) => {
                    let response = match s.response {
                        HandlerResponse::Success(body) => request.success(body),
                        HandlerResponse::Error(message) => request.error(&message),
                    };
                    self.server.respond(response)?;
                    if let Some(st) = s.state {
                        self.state = st;
                    }
                    for ev in s.events {
                        self.server.send_event(ev)?;
                    }
                    if s.outcome == HandlerOutcome::Exit {
                        return Ok(());
                    }
                }
                Err(msg) => {
                    self.server.respond(request.error(msg))?;
                }
            }
        }
    }

    fn dispatch_request<'tcx>(
        &self,
        request: &Request,
        session: &mut PrirodaContext<'tcx>,
    ) -> Result<HandlerSuccess, &'static str> {
        if self.state == DapState::Fresh && !matches!(&request.command, Command::Initialize(_)) {
            return Err("initialize must be sent first");
        }

        match &request.command {
            Command::Initialize(_) => self.handle_initialize(),
            Command::Launch(_) => self.handle_launch(),
            Command::Attach(_) => self.handle_attach(),
            Command::ConfigurationDone => self.handle_configuration_done(session),
            Command::Threads => self.handle_threads(),
            Command::StackTrace(args) => self.handle_stack_trace(args.thread_id, session),
            Command::Scopes(args) => self.handle_scopes(args.frame_id, session),
            Command::Variables(args) => self.handle_variables(args.variables_reference, session),
            Command::Continue(args) => self.handle_continue(args.thread_id, session),
            Command::SetBreakpoints(args) => self.handle_set_breakpoints(args, session),
            Command::Next(args) => self.handle_step(ResponseBody::Next, args.thread_id, session),
            Command::StepIn(args) =>
                self.handle_step(ResponseBody::StepIn, args.thread_id, session),
            Command::Disconnect(_) => self.handle_disconnect(),
            Command::BreakpointLocations(_)
            | Command::Cancel(_)
            | Command::Completions(_)
            | Command::DataBreakpointInfo(_)
            | Command::Disassemble(_)
            | Command::Evaluate(_)
            | Command::ExceptionInfo(_)
            | Command::Goto(_)
            | Command::GotoTargets(_)
            | Command::LoadedSources
            | Command::Modules(_)
            | Command::Pause(_)
            | Command::ReadMemory(_)
            | Command::Restart(_)
            | Command::RestartFrame(_)
            | Command::ReverseContinue(_)
            | Command::SetDataBreakpoints(_)
            | Command::SetExceptionBreakpoints(_)
            | Command::SetExpression(_)
            | Command::SetFunctionBreakpoints(_)
            | Command::SetInstructionBreakpoints(_)
            | Command::SetVariable(_)
            | Command::Source(_)
            | Command::StepBack(_)
            | Command::StepInTargets(_)
            | Command::StepOut(_)
            | Command::Terminate(_)
            | Command::TerminateThreads(_)
            | Command::WriteMemory(_) => self.handle_unsupported_request(&request.command),
        }
    }

    /// FIXME: connect launch arguments to Priroda's session model.
    fn handle_launch(&self) -> Result<HandlerSuccess, &'static str> {
        self.require_state(DapState::Initialized)?;

        Ok(HandlerSuccess {
            response: HandlerResponse::Success(ResponseBody::Launch),
            state: Some(DapState::Launched),
            events: Vec::new(),
            outcome: HandlerOutcome::Continue,
        })
    }

    fn handle_attach(&self) -> Result<HandlerSuccess, &'static str> {
        self.require_state(DapState::Initialized)?;

        // VS Code's extension-free `debugServer` template uses `attach`.
        // Priroda still starts the same single interpreted session as `launch`.
        Ok(HandlerSuccess {
            response: HandlerResponse::Success(ResponseBody::Attach),
            state: Some(DapState::Launched),
            events: Vec::new(),
            outcome: HandlerOutcome::Continue,
        })
    }

    fn handle_scopes<'tcx>(
        &self,
        frame_id: i64,
        session: &PrirodaContext<'tcx>,
    ) -> Result<HandlerSuccess, &'static str> {
        self.require_stopped()?;
        Self::require_frame_id(frame_id)?;

        let (source, line, column) = match &session.current_location {
            Some(location) => {
                let source = session.local_path(location).as_ref().map(|path| {
                    Source {
                        name: path.file_name().map(|name| name.to_string_lossy().into_owned()),
                        path: Some(path.display().to_string()),
                        source_reference: Some(0),
                        presentation_hint: None,
                        origin: None,
                        sources: None,
                        checksums: None,
                    }
                });
                let line =
                    location.line.try_into().unwrap_or_else(|_| bug!("source line exceeds i64"));
                let column = location
                    .column
                    .try_into()
                    .unwrap_or_else(|_| bug!("source column exceeds i64"));
                (source, Some(line), Some(column))
            }
            None => (None, None, None),
        };
        Ok(HandlerSuccess {
            response: HandlerResponse::Success(ResponseBody::Scopes(ScopesResponse {
                scopes: vec![Scope {
                    name: "Locals".to_string(),
                    presentation_hint: Some(ScopePresentationhint::Locals),
                    variables_reference: LOCALS_VARIABLES_REFERENCE,
                    named_variables: None,
                    indexed_variables: Some(0),
                    expensive: false,
                    source,
                    line,
                    column,
                    end_line: None,
                    end_column: None,
                }],
            })),
            state: None,
            events: Vec::new(),
            outcome: HandlerOutcome::Continue,
        })
    }

    fn handle_variables<'tcx>(
        &self,
        variables_reference: i64,
        session: &PrirodaContext<'tcx>,
    ) -> Result<HandlerSuccess, &'static str> {
        self.require_stopped()?;
        Self::require_variables_reference(variables_reference)?;

        let variables = if variables_reference == LOCALS_VARIABLES_REFERENCE {
            session.list_locals().into_iter().map(Self::local_to_variable).collect()
        } else {
            Vec::new()
        };

        Ok(HandlerSuccess {
            response: HandlerResponse::Success(ResponseBody::Variables(VariablesResponse {
                variables,
            })),
            state: None,
            events: Vec::new(),
            outcome: HandlerOutcome::Continue,
        })
    }

    fn handle_configuration_done<'tcx>(
        &self,
        session: &mut PrirodaContext<'tcx>,
    ) -> Result<HandlerSuccess, &'static str> {
        self.require_state(DapState::Launched)?;

        match Self::execution_outcome(session.stop_at_first_user_location()) {
            ExecutionOutcome::Stopped(result) => {
                // A normal startup stop is an entry event, but an interpreter
                // error before the first user location is an exception stop.
                let stopped = match result {
                    StepResult::Step => Self::stopped_event_body(StoppedEventReason::Entry),
                    result => Self::stopped_event_for(result),
                };
                Ok(HandlerSuccess {
                    response: HandlerResponse::Success(ResponseBody::ConfigurationDone),
                    state: Some(DapState::Stopped),
                    events: vec![Event::Stopped(stopped)],
                    outcome: HandlerOutcome::Continue,
                })
            }
            ExecutionOutcome::Terminated { code } =>
                Ok(HandlerSuccess {
                    response: HandlerResponse::Success(ResponseBody::ConfigurationDone),
                    state: Some(DapState::Terminated),
                    events: vec![
                        Event::Exited(ExitedEventBody { exit_code: code.into() }),
                        Event::Terminated(None),
                    ],
                    outcome: HandlerOutcome::Exit,
                }),
            ExecutionOutcome::Failed(message) =>
                Ok(HandlerSuccess {
                    response: HandlerResponse::Error(message),
                    state: Some(DapState::Terminated),
                    events: vec![Event::Terminated(None)],
                    outcome: HandlerOutcome::Exit,
                }),
        }
    }

    /// FIXME: replace this with Miri thread state once Priroda exposes a
    /// frontend-facing thread model.
    fn handle_threads(&self) -> Result<HandlerSuccess, &'static str> {
        self.reject_after_termination()?;

        Ok(HandlerSuccess {
            response: HandlerResponse::Success(ResponseBody::Threads(ThreadsResponse {
                threads: vec![Thread { id: THREAD_ID, name: "main".to_string() }],
            })),
            state: None,
            events: Vec::new(),
            outcome: HandlerOutcome::Continue,
        })
    }

    /// FIXME: report all frames once Priroda exposes a frontend-facing stack model.
    fn handle_stack_trace<'tcx>(
        &self,
        thread_id: i64,
        session: &PrirodaContext<'tcx>,
    ) -> Result<HandlerSuccess, &'static str> {
        self.require_stopped()?;
        Self::require_thread_id(thread_id)?;

        let stack_frames = match &session.current_location {
            Some(location) => {
                let path = session.local_path(location);
                vec![StackFrame {
                    id: STACK_FRAME_ID,
                    name: session.current_frame_name().unwrap_or_else(|| "<unknown>".to_string()),
                    source: path.as_ref().map(|path| {
                        Source {
                            name: path.file_name().map(|name| name.to_string_lossy().into_owned()),
                            path: Some(path.display().to_string()),
                            source_reference: Some(0),
                            presentation_hint: None,
                            origin: None,
                            sources: None,
                            checksums: None,
                        }
                    }),
                    line: location
                        .line
                        .try_into()
                        .unwrap_or_else(|_| bug!("source line exceeds i64")),
                    column: location
                        .column
                        .try_into()
                        .unwrap_or_else(|_| bug!("source column exceeds i64")),
                    end_line: None,
                    end_column: None,
                    can_restart: None,
                    instruction_pointer_reference: None,
                    module_id: None,
                    presentation_hint: None,
                }]
            }
            None => Vec::new(),
        };
        let total_frames: i64 =
            stack_frames.len().try_into().unwrap_or_else(|_| bug!("frame count exceeds i64"));
        Ok(HandlerSuccess {
            response: HandlerResponse::Success(ResponseBody::StackTrace(StackTraceResponse {
                stack_frames,
                total_frames: Some(total_frames),
            })),
            state: None,
            events: Vec::new(),
            outcome: HandlerOutcome::Continue,
        })
    }

    /// FIXME: grow capabilities as Priroda adds DAP features.
    fn handle_initialize(&self) -> Result<HandlerSuccess, &'static str> {
        if self.state != DapState::Fresh {
            return Err("initialize may only be sent once");
        }

        Ok(HandlerSuccess {
            response: HandlerResponse::Success(ResponseBody::Initialize(Capabilities {
                supports_configuration_done_request: Some(true),
                supports_single_thread_execution_requests: Some(true),
                ..Capabilities::default()
            })),
            state: Some(DapState::Initialized),
            events: vec![Event::Initialized],
            outcome: HandlerOutcome::Continue,
        })
    }

    /// FIXME: distinguish step-over from step-in once Priroda has call-aware stepping.
    fn handle_step<'tcx>(
        &self,
        body: ResponseBody,
        thread_id: i64,
        session: &mut PrirodaContext<'tcx>,
    ) -> Result<HandlerSuccess, &'static str> {
        self.require_stopped()?;
        Self::require_thread_id(thread_id)?;

        match Self::execution_outcome(session.step()) {
            ExecutionOutcome::Stopped(result) =>
                Ok(HandlerSuccess {
                    response: HandlerResponse::Success(body),
                    state: Some(DapState::Stopped),
                    events: vec![Event::Stopped(Self::stopped_event_for(result))],
                    outcome: HandlerOutcome::Continue,
                }),
            ExecutionOutcome::Terminated { code } =>
                Ok(HandlerSuccess {
                    response: HandlerResponse::Success(body),
                    state: Some(DapState::Terminated),
                    events: vec![
                        Event::Exited(ExitedEventBody { exit_code: code.into() }),
                        Event::Terminated(None),
                    ],
                    outcome: HandlerOutcome::Exit,
                }),
            ExecutionOutcome::Failed(message) =>
                Ok(HandlerSuccess {
                    response: HandlerResponse::Error(message),
                    state: Some(DapState::Terminated),
                    events: vec![Event::Terminated(None)],
                    outcome: HandlerOutcome::Exit,
                }),
        }
    }

    fn handle_continue<'tcx>(
        &self,
        thread_id: i64,
        session: &mut PrirodaContext<'tcx>,
    ) -> Result<HandlerSuccess, &'static str> {
        self.require_stopped()?;
        Self::require_thread_id(thread_id)?;

        let body = ResponseBody::Continue(ContinueResponse { all_threads_continued: Some(true) });

        match Self::execution_outcome(session.continue_execution()) {
            ExecutionOutcome::Stopped(result) =>
                Ok(HandlerSuccess {
                    response: HandlerResponse::Success(body),
                    state: Some(DapState::Stopped),
                    events: vec![Event::Stopped(Self::stopped_event_for(result))],
                    outcome: HandlerOutcome::Continue,
                }),
            ExecutionOutcome::Terminated { code } =>
                Ok(HandlerSuccess {
                    response: HandlerResponse::Success(body),
                    state: Some(DapState::Terminated),
                    events: vec![
                        Event::Exited(ExitedEventBody { exit_code: code.into() }),
                        Event::Terminated(None),
                    ],
                    outcome: HandlerOutcome::Exit,
                }),
            ExecutionOutcome::Failed(message) =>
                Ok(HandlerSuccess {
                    response: HandlerResponse::Error(message),
                    state: Some(DapState::Terminated),
                    events: vec![Event::Terminated(None)],
                    outcome: HandlerOutcome::Exit,
                }),
        }
    }

    fn handle_set_breakpoints<'tcx>(
        &self,
        args: &SetBreakpointsArguments,
        session: &mut PrirodaContext<'tcx>,
    ) -> Result<HandlerSuccess, &'static str> {
        self.reject_after_termination()?;

        let Some(ref path_str) = args.source.path else {
            return Err(
                "setBreakpoints requires a source.path; sourceReference loads are not supported",
            );
        };

        let path = std::path::PathBuf::from(path_str);
        let mut breakpoints = Vec::new();
        if let Some(ref req_bps) = args.breakpoints {
            for req_bp in req_bps {
                let line = usize::try_from(req_bp.line).unwrap();
                session.set_breakpoint(path.clone(), line);
                breakpoints.push(DapBreakpoint {
                    verified: true,
                    message: None,
                    source: Some(args.source.clone()),
                    line: Some(req_bp.line),
                    column: req_bp.column,
                    end_line: None,
                    end_column: None,
                    id: None,
                    instruction_reference: None,
                    offset: None,
                });
            }
        }

        Ok(HandlerSuccess {
            response: HandlerResponse::Success(ResponseBody::SetBreakpoints(
                SetBreakpointsResponse { breakpoints },
            )),
            state: None,
            events: Vec::new(),
            outcome: HandlerOutcome::Continue,
        })
    }

    fn handle_disconnect(&self) -> Result<HandlerSuccess, &'static str> {
        Ok(HandlerSuccess {
            response: HandlerResponse::Success(ResponseBody::Disconnect),
            state: Some(DapState::Terminated),
            events: vec![Event::Terminated(None)],
            outcome: HandlerOutcome::Exit,
        })
    }

    fn handle_unsupported_request(
        &self,
        command: &Command,
    ) -> Result<HandlerSuccess, &'static str> {
        Ok(HandlerSuccess {
            response: HandlerResponse::Error(format!(
                "unsupported request in Priroda DAP demo mode: {}",
                Self::display_command(command)
            )),
            state: None,
            events: Vec::new(),
            outcome: HandlerOutcome::Continue,
        })
    }

    fn reject_after_termination(&self) -> Result<(), &'static str> {
        if self.state == DapState::Terminated {
            return Err("request received after termination");
        }
        Ok(())
    }

    fn require_state(&self, expected: DapState) -> Result<(), &'static str> {
        if self.state != expected {
            return Err(match expected {
                DapState::Initialized => "launch or attach requires initialize",
                DapState::Launched => "configurationDone requires launch or attach",
                _ => "invalid session state for request",
            });
        }
        Ok(())
    }

    fn require_stopped(&self) -> Result<(), &'static str> {
        if self.state != DapState::Stopped {
            return Err("request requires a stopped frame");
        }
        Ok(())
    }

    fn require_thread_id(thread_id: i64) -> Result<(), &'static str> {
        if thread_id != THREAD_ID {
            return Err("unknown threadId");
        }
        Ok(())
    }

    fn require_frame_id(frame_id: i64) -> Result<(), &'static str> {
        if frame_id != STACK_FRAME_ID {
            return Err("unknown frameId");
        }
        Ok(())
    }

    fn require_variables_reference(variables_reference: i64) -> Result<(), &'static str> {
        if variables_reference != LOCALS_VARIABLES_REFERENCE {
            return Err("unknown variablesReference");
        }
        Ok(())
    }

    fn execution_outcome<'tcx>(result: InterpResult<'tcx, ExecutionResult>) -> ExecutionOutcome {
        match result.report_err() {
            Ok(ExecutionResult::Stopped(step)) => ExecutionOutcome::Stopped(step),
            Ok(ExecutionResult::ProgramExited { code }) => ExecutionOutcome::Terminated { code },
            Err(err) => Self::interp_error_outcome(err),
        }
    }

    fn interp_error_outcome<'tcx>(err: InterpErrorInfo<'tcx>) -> ExecutionOutcome {
        let kind = err.into_kind();
        if let InterpErrorKind::MachineStop(info) = &kind
            && let Some(TerminationInfo::Exit { code, .. }) = info.downcast_ref::<TerminationInfo>()
        {
            return ExecutionOutcome::Terminated { code: *code };
        }

        ExecutionOutcome::Failed(kind.to_string())
    }

    fn stopped_event_for(result: StepResult) -> StoppedEventBody {
        let (reason, text) = match result {
            StepResult::Step => (StoppedEventReason::Step, None),
            StepResult::Breakpoint => (StoppedEventReason::Breakpoint, None),
            StepResult::Exception { message } => (StoppedEventReason::Exception, Some(message)),
        };
        StoppedEventBody {
            reason,
            description: None,
            thread_id: Some(THREAD_ID),
            preserve_focus_hint: None,
            text,
            all_threads_stopped: Some(true),
            hit_breakpoint_ids: None,
        }
    }

    fn stopped_event_body(reason: StoppedEventReason) -> StoppedEventBody {
        StoppedEventBody {
            reason,
            description: None,
            thread_id: Some(THREAD_ID),
            preserve_focus_hint: None,
            text: None,
            all_threads_stopped: Some(true),
            hit_breakpoint_ids: None,
        }
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
            Command::Attach(_) => "attach",
            Command::BreakpointLocations(_) => "breakpointLocations",
            Command::Cancel(_) => "cancel",
            Command::Completions(_) => "completions",
            Command::Continue(_) => "continue",
            Command::DataBreakpointInfo(_) => "dataBreakpointInfo",
            Command::Disassemble(_) => "disassemble",
            Command::Evaluate(_) => "evaluate",
            Command::ExceptionInfo(_) => "exceptionInfo",
            Command::Goto(_) => "goto",
            Command::GotoTargets(_) => "gotoTargets",
            Command::LoadedSources => "loadedSources",
            Command::Modules(_) => "modules",
            Command::Pause(_) => "pause",
            Command::ReadMemory(_) => "readMemory",
            Command::Restart(_) => "restart",
            Command::RestartFrame(_) => "restartFrame",
            Command::ReverseContinue(_) => "reverseContinue",
            Command::SetBreakpoints(_) => "setBreakpoints",
            Command::SetDataBreakpoints(_) => "setDataBreakpoints",
            Command::SetExceptionBreakpoints(_) => "setExceptionBreakpoints",
            Command::SetExpression(_) => "setExpression",
            Command::SetFunctionBreakpoints(_) => "setFunctionBreakpoints",
            Command::SetInstructionBreakpoints(_) => "setInstructionBreakpoints",
            Command::SetVariable(_) => "setVariable",
            Command::Source(_) => "source",
            Command::StepBack(_) => "stepBack",
            Command::StepInTargets(_) => "stepInTargets",
            Command::StepOut(_) => "stepOut",
            Command::Terminate(_) => "terminate",
            Command::TerminateThreads(_) => "terminateThreads",
            Command::WriteMemory(_) => "writeMemory",
        }
    }

    fn local_to_variable(local: LocalDesc) -> Variable {
        Variable {
            name: Self::local_name(&local),
            value: local.value,
            type_field: Some(local.ty),
            presentation_hint: None,
            evaluate_name: None,
            // FIXME: add child handles once Priroda can identify places across requests.
            variables_reference: 0,
            named_variables: None,
            indexed_variables: None,
            memory_reference: None,
        }
    }

    fn local_name(local: &LocalDesc) -> String {
        let source_projection = local.source_projection_str();

        // Prefer source names when debug info gives us one. If a local only has
        // MIR storage identity, keep that visible so the DAP Variables view
        // still has a stable row for every backing local.
        if let Some(source_name) = local.source_name {
            return format!("{source_name}{source_projection}");
        }

        let local_id = local
            .local
            .map_or_else(|| "<none>".to_string(), |local_idx| format!("_{}", local_idx.index()));
        format!("{local_id}{}", local.storage_projection_str())
    }
}
