use ipc_channel::ipc;
use nix::sys::{ptrace, signal, wait};
use nix::unistd;

use super::StartFfiInfo;
use super::messages::{Confirmation, MemEvents, TraceRequest};

/// The flags to use when calling `waitid()`.
/// Since bitwise OR on the nix version of these flags is implemented as a trait,
/// we can't use them directly so we do it this way
const WAIT_FLAGS: wait::WaitPidFlag =
    wait::WaitPidFlag::from_bits_truncate(libc::WUNTRACED | libc::WEXITED);

/// A unified event representing something happening on the child process. Wraps
/// `nix`'s `WaitStatus` and our custom signals so it can all be done with one
/// `match` statement.
pub enum ExecEvent {
    /// Child process requests that we begin monitoring it.
    Start(StartFfiInfo),
    /// Child requests that we stop monitoring and pass over the events we
    /// detected.
    End,
    /// The child process with the specified pid was stopped by the given signal.
    Status(unistd::Pid, signal::Signal),
    /// The child process with the specified pid entered or exited a syscall.
    Syscall(unistd::Pid),
    /// A child process exited or was killed; if we have a return code, it is
    /// specified.
    Died(Option<i32>),
}

/// A listener for the FFI start info channel along with relevant state.
pub struct ChildListener {
    /// The matching channel for the child's `Supervisor` struct.
    pub message_rx: ipc::IpcReceiver<TraceRequest>,
    /// Whether an FFI call is currently ongoing.
    pub attached: bool,
    /// If `Some`, overrides the return code with the given value.
    pub override_retcode: Option<i32>,
}

impl Iterator for ChildListener {
    type Item = ExecEvent;

    // Allows us to monitor the child process by just iterating over the listener
    // NB: This should never return None!
    fn next(&mut self) -> Option<Self::Item> {
        // Do not block if the child has nothing to report for `waitid`
        let opts = WAIT_FLAGS | wait::WaitPidFlag::WNOHANG;
        loop {
            // Listen to any child, not just the main one. Important if we want
            // to allow the C code to fork further, along with being a bit of
            // defensive programming since Linux sometimes assigns threads of
            // the same process different PIDs with unpredictable rules...
            match wait::waitid(wait::Id::All, opts) {
                Ok(stat) =>
                    match stat {
                        // Child exited normally with a specific code set
                        wait::WaitStatus::Exited(_, code) => {
                            let code = self.override_retcode.unwrap_or(code);
                            return Some(ExecEvent::Died(Some(code)));
                        }
                        // Child was killed by a signal, without giving a code
                        wait::WaitStatus::Signaled(_, _, _) =>
                            return Some(ExecEvent::Died(self.override_retcode)),
                        // Child entered a syscall. Since we're always technically
                        // tracing, only pass this along if we're actively
                        // monitoring the child
                        wait::WaitStatus::PtraceSyscall(pid) =>
                            if self.attached {
                                return Some(ExecEvent::Syscall(pid));
                            },
                        // Child with the given pid was stopped by the given signal.
                        // It's somewhat dubious when this is returned instead of
                        // WaitStatus::Stopped, but for our purposes they are the
                        // same thing.
                        wait::WaitStatus::PtraceEvent(pid, signal, _) =>
                            if self.attached {
                                // This is our end-of-FFI signal!
                                if signal == signal::SIGUSR1 {
                                    self.attached = false;
                                    return Some(ExecEvent::End);
                                } else {
                                    return Some(ExecEvent::Status(pid, signal));
                                }
                            } else {
                                // Just pass along the signal
                                ptrace::cont(pid, signal).unwrap();
                            },
                        // Child was stopped at the given signal. Same logic as for
                        // WaitStatus::PtraceEvent
                        wait::WaitStatus::Stopped(pid, signal) =>
                            if self.attached {
                                if signal == signal::SIGUSR1 {
                                    self.attached = false;
                                    return Some(ExecEvent::End);
                                } else {
                                    return Some(ExecEvent::Status(pid, signal));
                                }
                            } else {
                                ptrace::cont(pid, signal).unwrap();
                            },
                        _ => (),
                    },
                // This case should only trigger if all children died and we
                // somehow missed that, but it's best we not allow any room
                // for deadlocks
                Err(_) => return Some(ExecEvent::Died(None)),
            }

            // Similarly, do a non-blocking poll of the IPC channel
            if let Ok(req) = self.message_rx.try_recv() {
                match req {
                    TraceRequest::StartFfi(info) =>
                    // Should never trigger - but better to panic explicitly than deadlock!
                        if self.attached {
                            panic!("Attempting to begin FFI multiple times!");
                        } else {
                            self.attached = true;
                            return Some(ExecEvent::Start(info));
                        },
                    TraceRequest::OverrideRetcode(code) => self.override_retcode = Some(code),
                }
            }

            // Not ideal, but doing anything else might sacrifice performance
            std::thread::yield_now();
        }
    }
}

/// An error came up while waiting on the child process to do something.
#[derive(Debug)]
enum ExecError {
    /// The child process died with this return code, if we have one.
    Died(Option<i32>),
}

/// This is the main loop of the supervisor process. It runs in a separate
/// process from the rest of Miri (but because we fork, addresses for anything
/// created before the fork - like statics - are the same).
pub fn sv_loop(
    listener: ChildListener,
    init_pid: unistd::Pid,
    event_tx: ipc::IpcSender<MemEvents>,
    confirm_tx: ipc::IpcSender<Confirmation>,
    _page_size: usize,
) -> Result<!, Option<i32>> {
    // Things that we return to the child process
    let mut acc_events = Vec::new();

    // Memory allocated on the MiriMachine
    let mut _ch_pages = Vec::new();
    let mut _ch_stack = None;

    // The pid of the last process we interacted with, used by default if we don't have a
    // reason to use a different one
    let mut curr_pid = init_pid;

    // There's an initial sigstop we need to deal with
    wait_for_signal(Some(curr_pid), signal::SIGSTOP, false).map_err(|e| {
        match e {
            ExecError::Died(code) => code,
        }
    })?;
    ptrace::cont(curr_pid, None).unwrap();

    for evt in listener {
        match evt {
            // start_ffi was called by the child, so prep memory
            ExecEvent::Start(ch_info) => {
                // All the pages that the child process is "allowed to" access
                _ch_pages = ch_info.page_ptrs;
                // And the fake stack it allocated for us to use later
                _ch_stack = Some(ch_info.stack_ptr);

                // We received the signal and are no longer in the main listener loop,
                // so we can let the child move on to the end of start_ffi where it will
                // raise a SIGSTOP. We need it to be signal-stopped *and waited for* in
                // order to do most ptrace operations!
                confirm_tx.send(Confirmation).unwrap();
                // We can't trust simply calling `Pid::this()` in the child process to give the right
                // PID for us, so we get it this way
                curr_pid = wait_for_signal(None, signal::SIGSTOP, false).unwrap();

                ptrace::syscall(curr_pid, None).unwrap();
            }
            // end_ffi was called by the child
            ExecEvent::End => {
                // Hand over the access info we traced
                event_tx.send(MemEvents { acc_events }).unwrap();
                // And reset our values
                acc_events = Vec::new();
                _ch_stack = None;

                // No need to monitor syscalls anymore, they'd just be ignored
                ptrace::cont(curr_pid, None).unwrap();
            }
            // Child process was stopped by a signal
            ExecEvent::Status(pid, signal) => {
                eprintln!("Process unexpectedly got {signal}; continuing...");
                // In case we're not tracing
                if ptrace::syscall(pid, signal).is_err() {
                    // If *this* fails too, something really weird happened
                    // and it's probably best to just panic
                    signal::kill(pid, signal::SIGCONT).unwrap();
                }
            }
            // Child entered a syscall; we wait for exits inside of this, so it
            // should never trigger on return from a syscall we care about
            ExecEvent::Syscall(pid) => {
                ptrace::syscall(pid, None).unwrap();
            }
            ExecEvent::Died(code) => {
                return Err(code);
            }
        }
    }

    unreachable!()
}

/// Waits for `wait_signal`. If `init_cont`, it will first do a `ptrace::cont`.
/// We want to avoid that in some cases, like at the beginning of FFI.
///
/// If `pid` is `None`, only one wait will be done and `init_cont` should be false.
fn wait_for_signal(
    pid: Option<unistd::Pid>,
    wait_signal: signal::Signal,
    init_cont: bool,
) -> Result<unistd::Pid, ExecError> {
    if init_cont {
        ptrace::cont(pid.unwrap(), None).unwrap();
    }
    // Repeatedly call `waitid` until we get the signal we want, or the process dies
    loop {
        let wait_id = match pid {
            Some(pid) => wait::Id::Pid(pid),
            None => wait::Id::All,
        };
        let stat = wait::waitid(wait_id, WAIT_FLAGS).map_err(|_| ExecError::Died(None))?;
        let (signal, pid) = match stat {
            // Report the cause of death, if we know it
            wait::WaitStatus::Exited(_, code) => {
                return Err(ExecError::Died(Some(code)));
            }
            wait::WaitStatus::Signaled(_, _, _) => return Err(ExecError::Died(None)),
            wait::WaitStatus::Stopped(pid, signal) => (signal, pid),
            wait::WaitStatus::PtraceEvent(pid, signal, _) => (signal, pid),
            // This covers PtraceSyscall and variants that are impossible with
            // the flags set (e.g. WaitStatus::StillAlive)
            _ => {
                ptrace::cont(pid.unwrap(), None).unwrap();
                continue;
            }
        };
        if signal == wait_signal {
            return Ok(pid);
        } else {
            ptrace::cont(pid, None).map_err(|_| ExecError::Died(None))?;
        }
    }
}
