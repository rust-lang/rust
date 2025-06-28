use std::cell::RefCell;
use std::rc::Rc;

use ipc_channel::ipc;
use nix::sys::{ptrace, signal};
use nix::unistd;

use super::CALLBACK_STACK_SIZE;
use super::messages::{Confirmation, MemEvents, StartFfiInfo, TraceRequest};
use super::parent::{ChildListener, sv_loop};
use crate::alloc::isolated_alloc::IsolatedAlloc;

static SUPERVISOR: std::sync::Mutex<Option<Supervisor>> = std::sync::Mutex::new(None);

/// The main means of communication between the child and parent process,
/// allowing the former to send requests and get info from the latter.
pub struct Supervisor {
    /// Sender for FFI-mode-related requests.
    message_tx: ipc::IpcSender<TraceRequest>,
    /// Used for synchronisation, allowing us to receive confirmation that the
    /// parent process has handled the request from `message_tx`.
    confirm_rx: ipc::IpcReceiver<Confirmation>,
    /// Receiver for memory acceses that ocurred during the FFI call.
    event_rx: ipc::IpcReceiver<MemEvents>,
}

/// Marker representing that an error occurred during creation of the supervisor.
#[derive(Debug)]
pub struct SvInitError;

impl Supervisor {
    /// Returns `true` if the supervisor process exists, and `false` otherwise.
    pub fn is_enabled() -> bool {
        SUPERVISOR.lock().unwrap().is_some()
    }

    /// Begins preparations for doing an FFI call. This should be called at
    /// the last possible moment before entering said call. `alloc` points to
    /// the allocator which handed out the memory used for this machine.
    ///
    /// As this locks the supervisor via a mutex, no other threads may enter FFI
    /// until this one returns and its guard is dropped via `end_ffi`. The
    /// pointer returned should be passed to `end_ffi` to avoid a memory leak.
    ///
    /// SAFETY: The resulting guard must be dropped *via `end_ffi`* immediately
    /// after the desired call has concluded.
    pub unsafe fn start_ffi(
        alloc: &Rc<RefCell<IsolatedAlloc>>,
    ) -> (std::sync::MutexGuard<'static, Option<Supervisor>>, Option<*mut [u8; CALLBACK_STACK_SIZE]>)
    {
        let mut sv_guard = SUPERVISOR.lock().unwrap();
        // If the supervisor is not initialised for whatever reason, fast-fail.
        // This might be desired behaviour, as even on platforms where ptracing
        // is not implemented it enables us to enforce that only one FFI call
        // happens at a time.
        let Some(sv) = sv_guard.take() else {
            return (sv_guard, None);
        };

        // Get pointers to all the pages the supervisor must allow accesses in
        // and prepare the callback stack.
        let page_ptrs = alloc.borrow().pages();
        let raw_stack_ptr: *mut [u8; CALLBACK_STACK_SIZE] =
            Box::leak(Box::new([0u8; CALLBACK_STACK_SIZE])).as_mut_ptr().cast();
        let stack_ptr = raw_stack_ptr.expose_provenance();
        let start_info = StartFfiInfo { page_ptrs, stack_ptr };

        // SAFETY: We do not access machine memory past this point until the
        // supervisor is ready to allow it.
        unsafe {
            if alloc.borrow_mut().prepare_ffi().is_err() {
                // Don't mess up unwinding by maybe leaving the memory partly protected
                alloc.borrow_mut().unprep_ffi();
                panic!("Cannot protect memory for FFI call!");
            }
        }

        // Send over the info.
        // NB: if we do not wait to receive a blank confirmation response, it is
        // possible that the supervisor is alerted of the SIGSTOP *before* it has
        // actually received the start_info, thus deadlocking! This way, we can
        // enforce an ordering for these events.
        sv.message_tx.send(TraceRequest::StartFfi(start_info)).unwrap();
        sv.confirm_rx.recv().unwrap();
        *sv_guard = Some(sv);
        // We need to be stopped for the supervisor to be able to make certain
        // modifications to our memory - simply waiting on the recv() doesn't
        // count.
        signal::raise(signal::SIGSTOP).unwrap();
        (sv_guard, Some(raw_stack_ptr))
    }

    /// Undoes FFI-related preparations, allowing Miri to continue as normal, then
    /// gets the memory accesses and changes performed during the FFI call. Note
    /// that this may include some spurious accesses done by `libffi` itself in
    /// the process of executing the function call.
    ///
    /// SAFETY: The `sv_guard` and `raw_stack_ptr` passed must be the same ones
    /// received by a prior call to `start_ffi`, and the allocator must be the
    /// one passed to it also.
    pub unsafe fn end_ffi(
        alloc: &Rc<RefCell<IsolatedAlloc>>,
        mut sv_guard: std::sync::MutexGuard<'static, Option<Supervisor>>,
        raw_stack_ptr: Option<*mut [u8; CALLBACK_STACK_SIZE]>,
    ) -> Option<MemEvents> {
        // We can't use IPC channels here to signal that FFI mode has ended,
        // since they might allocate memory which could get us stuck in a SIGTRAP
        // with no easy way out! While this could be worked around, it is much
        // simpler and more robust to simply use the signals which are left for
        // arbitrary usage. Since this will block until we are continued by the
        // supervisor, we can assume past this point that everything is back to
        // normal.
        signal::raise(signal::SIGUSR1).unwrap();

        // This is safe! It just sets memory to normal expected permissions.
        alloc.borrow_mut().unprep_ffi();

        // If this is `None`, then `raw_stack_ptr` is None and does not need to
        // be deallocated (and there's no need to worry about the guard, since
        // it contains nothing).
        let sv = sv_guard.take()?;
        // SAFETY: Caller upholds that this pointer was allocated as a box with
        // this type.
        unsafe {
            drop(Box::from_raw(raw_stack_ptr.unwrap()));
        }
        // On the off-chance something really weird happens, don't block forever.
        let ret = sv
            .event_rx
            .try_recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|e| {
                match e {
                    ipc::TryRecvError::IpcError(_) => (),
                    ipc::TryRecvError::Empty =>
                        eprintln!("Waiting for accesses from supervisor timed out!"),
                }
            })
            .ok();
        // Do *not* leave the supervisor empty, or else we might get another fork...
        *sv_guard = Some(sv);
        ret
    }
}

/// Initialises the supervisor process. If this function errors, then the
/// supervisor process could not be created successfully; else, the caller
/// is now the child process and can communicate via `start_ffi`/`end_ffi`,
/// receiving back events through `get_events`.
///
/// # Safety
/// The invariants for `fork()` must be upheld by the caller.
pub unsafe fn init_sv() -> Result<(), SvInitError> {
    // FIXME: Much of this could be reimplemented via the mitosis crate if we upstream the
    // relevant missing bits.

    // On Linux, this will check whether ptrace is fully disabled by the Yama module.
    // If Yama isn't running or we're not on Linux, we'll still error later, but
    // this saves a very expensive fork call.
    let ptrace_status = std::fs::read_to_string("/proc/sys/kernel/yama/ptrace_scope");
    if let Ok(stat) = ptrace_status {
        if let Some(stat) = stat.chars().next() {
            // Fast-error if ptrace is fully disabled on the system.
            if stat == '3' {
                return Err(SvInitError);
            }
        }
    }

    // Initialise the supervisor if it isn't already, placing it into SUPERVISOR.
    let mut lock = SUPERVISOR.lock().unwrap();
    if lock.is_some() {
        return Ok(());
    }

    // Prepare the IPC channels we need.
    let (message_tx, message_rx) = ipc::channel().unwrap();
    let (confirm_tx, confirm_rx) = ipc::channel().unwrap();
    let (event_tx, event_rx) = ipc::channel().unwrap();
    // SAFETY: Calling sysconf(_SC_PAGESIZE) is always safe and cannot error.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.try_into().unwrap();
    super::parent::PAGE_SIZE.store(page_size, std::sync::atomic::Ordering::Relaxed);

    unsafe {
        // TODO: Maybe use clone3() instead for better signalling of when the child exits?
        // SAFETY: Caller upholds that only one thread exists.
        match unistd::fork().unwrap() {
            unistd::ForkResult::Parent { child } => {
                // If somehow another thread does exist, prevent it from accessing the lock
                // and thus breaking our safety invariants.
                std::mem::forget(lock);
                // The child process is free to unwind, so we won't to avoid doubly freeing
                // system resources.
                let init = std::panic::catch_unwind(|| {
                    let listener =
                        ChildListener { message_rx, attached: false, override_retcode: None };
                    // Trace as many things as possible, to be able to handle them as needed.
                    let options = ptrace::Options::PTRACE_O_TRACESYSGOOD
                        | ptrace::Options::PTRACE_O_TRACECLONE
                        | ptrace::Options::PTRACE_O_TRACEFORK;
                    // Attach to the child process without stopping it.
                    match ptrace::seize(child, options) {
                        // Ptrace works :D
                        Ok(_) => {
                            let code = sv_loop(listener, child, event_tx, confirm_tx).unwrap_err();
                            // If a return code of 0 is not explicitly given, assume something went
                            // wrong and return 1.
                            std::process::exit(code.0.unwrap_or(1))
                        }
                        // Ptrace does not work and we failed to catch that.
                        Err(_) => {
                            // If we can't ptrace, Miri continues being the parent.
                            signal::kill(child, signal::SIGKILL).unwrap();
                            SvInitError
                        }
                    }
                });
                match init {
                    // The "Ok" case means that we couldn't ptrace.
                    Ok(e) => return Err(e),
                    Err(p) => {
                        eprintln!("Supervisor process panicked!\n{p:?}");
                        std::process::exit(1);
                    }
                }
            }
            unistd::ForkResult::Child => {
                // Make sure we never get orphaned and stuck in SIGSTOP or similar
                // SAFETY: prctl PR_SET_PDEATHSIG is always safe to call.
                let ret = libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                assert_eq!(ret, 0);
                // First make sure the parent succeeded with ptracing us!
                signal::raise(signal::SIGSTOP).unwrap();
                // If we're the child process, save the supervisor info.
                *lock = Some(Supervisor { message_tx, confirm_rx, event_rx });
            }
        }
    }
    Ok(())
}

/// Instruct the supervisor process to return a particular code. Useful if for
/// whatever reason this code fails to be intercepted normally. In the case of
/// `abort_if_errors()` used in `bin/miri.rs`, the return code is erroneously
/// given as a 0 if this is not used.
pub fn register_retcode_sv(code: i32) {
    let mut sv_guard = SUPERVISOR.lock().unwrap();
    if let Some(sv) = sv_guard.take() {
        sv.message_tx.send(TraceRequest::OverrideRetcode(code)).unwrap();
        *sv_guard = Some(sv);
    }
}
