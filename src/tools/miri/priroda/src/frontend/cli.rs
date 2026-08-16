use std::io::{self, Write};
use std::num::NonZeroU64;
use std::path::PathBuf;

use miri::{InterpResult, interp_ok};
use rustc_middle::mir::interpret::AllocId;

use crate::debugger::{
    BreakpointSetResult, CommandResult, DebuggerCommand, ExecutionResult, PrirodaContext,
    StepResult,
};

pub(crate) struct Cli;

impl Cli {
    pub(crate) fn run_cli_loop<'tcx>(
        &self,
        session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx> {
        loop {
            print!("(priroda) ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            let bytes_read = io::stdin().read_line(&mut input).unwrap();

            if bytes_read == 0 {
                println!("stdin closed, stopping");
                return interp_ok(());
            }

            if let Some(command) = self.parse_command(&input) {
                let command_res = session.run_command(command)?;
                if !Self::print_command_result(command_res, session)? {
                    return interp_ok(());
                };
            } else {
                println!("no command");
            }

            io::stdout().flush().unwrap();
        }
    }

    fn print_command_result<'tcx>(
        command_res: CommandResult,
        session: &PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx, bool> {
        match command_res {
            CommandResult::Execution(result) =>
                match result {
                    ExecutionResult::Stopped(step) =>
                        match step {
                            StepResult::Step => Self::print_location(session),
                            StepResult::Breakpoint => {
                                println!("Hit breakpoint");
                                Self::print_location(session);
                            }
                        },
                    ExecutionResult::ProgramExited { code } => {
                        println!("program finished with exit code {code}");
                    }
                },
            CommandResult::BreakpointResult(res) =>
                match res {
                    BreakpointSetResult::Added(path, line) => {
                        println!("breakpoint added: {}:{}", path.display(), line)
                    }

                    BreakpointSetResult::Duplicate => println!("Duplicate breakpoint"),
                },
            CommandResult::Locals(locals_desc) =>
                if locals_desc.is_empty() {
                    println!("no locals");
                } else {
                    for local_desc in &locals_desc {
                        let source_projection = local_desc.source_projection_str();

                        let name = local_desc
                            .source_name
                            .map_or_else(|| "<none>".to_string(), |name| name.to_string());

                        let display_name = format!("{name}{source_projection}");

                        let local_id = local_desc.local.map_or_else(
                            || "<none>".to_string(),
                            |local_idx| format!("_{}", local_idx.index()),
                        );

                        let display_local_id =
                            format!("{}{}", local_id, local_desc.storage_projection_str());
                        println!(
                            "Name: {}, Id: {}, Ty: {}, Value: {}",
                            display_name, display_local_id, local_desc.ty, local_desc.value
                        );
                    }
                },
            CommandResult::SingleLocal(local_desc) =>
                match local_desc {
                    Some(local_desc) => {
                        println!(
                            "Id: _{}, Ty: {}, Value: {}",
                            local_desc.local.unwrap().index(),
                            local_desc.ty,
                            local_desc.value
                        );
                    }
                    None => println!("no local for this id"),
                },
            CommandResult::Memory(memory) => println!("{memory}"),
            CommandResult::TerminateSession => {
                println!("quitting");
                return interp_ok(false);
            }
        }
        interp_ok(true)
    }

    fn parse_command(&self, input: &str) -> Option<DebuggerCommand> {
        // TODO: look at the Spanned crate for how to easily produce errors in
        // rustc's style while manually parsing text input.
        // FIXME: we need to distinguish malformed input from the unknown commands by returning useful
        // command error that describes if it malformed or non exist command
        let input = input.trim();
        let mut parts = input.splitn(2, char::is_whitespace);
        let command = parts.next().unwrap_or("");
        let args = parts.next().unwrap_or("").trim();

        match command {
            // FIXME: empty line should repats last command user typed not exeute specific command.
            "" | "si" | "stepi" => Some(DebuggerCommand::StepI),
            "s" | "step" => Some(DebuggerCommand::Step),
            "q" | "quit" => Some(DebuggerCommand::TerminateSession),
            "c" | "continue" => Some(DebuggerCommand::Continue),
            "b" | "break" => self.parse_breakpoint(args),
            "l" | "locals" => Some(DebuggerCommand::ListLocals),
            "p" | "print" => self.parse_print_local(args),
            "f" | "follow" => self.parse_follow(args),
            _ => None,
        }
    }

    fn print_location<'tcx>(session: &PrirodaContext<'tcx>) {
        match &session.current_location {
            Some(location) =>
                if let Some(path) = session.local_path(location) {
                    println!("{}:{}", path.display(), location.line);
                } else {
                    let source_map = session.ecx.tcx.sess.source_map();
                    println!("{}", source_map.span_to_diagnostic_string(location.span));
                },
            None => println!("no-location"),
        }
        io::stdout().flush().unwrap();
    }

    fn parse_breakpoint(&self, input: &str) -> Option<DebuggerCommand> {
        // FIXME: return a typed CommandError so malformed breakpoint input is
        // distinguishable from an unknown command. Semantic validation belongs
        // in PrirodaContext::set_breakpoint so non-CLI frontends cannot bypass it.
        let (path, line) = input.rsplit_once(':')?;
        let line = line.parse().ok()?;

        Some(DebuggerCommand::Breakpoint(PathBuf::from(path), line))
    }

    fn parse_print_local(&self, input: &str) -> Option<DebuggerCommand> {
        let local = input.parse().ok()?;
        Some(DebuggerCommand::Print(local))
    }

    fn parse_follow(&self, input: &str) -> Option<DebuggerCommand> {
        let mut parts = input.split_whitespace();
        let alloc_id = parts.next()?;
        let offset = parts.next()?;
        if parts.next().is_some() {
            return None;
        }

        let alloc_id = alloc_id.strip_prefix("alloc").unwrap_or(alloc_id).parse().ok()?;
        let alloc_id = AllocId(NonZeroU64::new(alloc_id)?);
        let offset = offset.parse().ok()?;
        Some(DebuggerCommand::Follow(alloc_id, offset))
    }
}
