use std::sync::atomic::{AtomicPtr, AtomicUsize};

use ipc_channel::ipc;
use nix::sys::{ptrace, signal, wait};
use nix::unistd;

use super::CALLBACK_STACK_SIZE;
use super::messages::{Confirmation, StartFfiInfo, TraceRequest};
use crate::shims::native_lib::{AccessEvent, AccessRange, MemEvents};

/// The flags to use when calling `waitid()`.
const WAIT_FLAGS: wait::WaitPidFlag =
    wait::WaitPidFlag::WUNTRACED.union(wait::WaitPidFlag::WEXITED);

/// The default word size on a given platform, in bytes.
#[cfg(target_arch = "x86")]
const ARCH_WORD_SIZE: usize = 4;
#[cfg(target_arch = "x86_64")]
const ARCH_WORD_SIZE: usize = 8;

// x86 max instruction length is 15 bytes:
// https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
// See vol. 3B section 24.25.
const ARCH_MAX_INSTR_SIZE: usize = 15;

/// The address of the page set to be edited, initialised to a sentinel null
/// pointer.
static PAGE_ADDR: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());
/// The host pagesize, initialised to a sentinel zero value.
pub static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);
/// How many consecutive pages to unprotect. 1 by default, unlikely to be set
/// higher than 2.
static PAGE_COUNT: AtomicUsize = AtomicUsize::new(1);

/// Allows us to get common arguments from the `user_regs_t` across architectures.
/// Normally this would land us ABI hell, but thankfully all of our usecases
/// consist of functions with a small number of register-sized integer arguments.
/// See <https://man7.org/linux/man-pages/man2/syscall.2.html> for sources.
trait ArchIndependentRegs {
    /// Gets the address of the instruction pointer.
    fn ip(&self) -> usize;
    /// Set the instruction pointer; remember to also set the stack pointer, or
    /// else the stack might get messed up!
    fn set_ip(&mut self, ip: usize);
    /// Set the stack pointer, ideally to a zeroed-out area.
    fn set_sp(&mut self, sp: usize);
}

// It's fine / desirable behaviour for values to wrap here, we care about just
// preserving the bit pattern.
#[cfg(target_arch = "x86_64")]
#[rustfmt::skip]
impl ArchIndependentRegs for libc::user_regs_struct {
    #[inline]
    fn ip(&self) -> usize { self.rip.try_into().unwrap() }
    #[inline]
    fn set_ip(&mut self, ip: usize) { self.rip = ip.try_into().unwrap() }
    #[inline]
    fn set_sp(&mut self, sp: usize) { self.rsp = sp.try_into().unwrap() }
}

#[cfg(target_arch = "x86")]
#[rustfmt::skip]
impl ArchIndependentRegs for libc::user_regs_struct {
    #[inline]
    fn ip(&self) -> usize { self.eip.cast_unsigned().try_into().unwrap() }
    #[inline]
    fn set_ip(&mut self, ip: usize) { self.eip = ip.cast_signed().try_into().unwrap() }
    #[inline]
    fn set_sp(&mut self, sp: usize) { self.esp = sp.cast_signed().try_into().unwrap() }
}

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
    /// The child process with the specified pid entered or existed a syscall.
    Syscall(unistd::Pid),
    /// A child process exited or was killed; if we have a return code, it is
    /// specified.
    Died(Option<i32>),
}

/// A listener for the FFI start info channel along with relevant state.
pub struct ChildListener {
    /// The matching channel for the child's `Supervisor` struct.
    message_rx: ipc::IpcReceiver<TraceRequest>,
    /// ...
    confirm_tx: ipc::IpcSender<Confirmation>,
    /// Whether an FFI call is currently ongoing.
    attached: bool,
    /// If `Some`, overrides the return code with the given value.
    override_retcode: Option<i32>,
    /// Last code obtained from a child exiting.
    last_code: Option<i32>,
}

impl ChildListener {
    pub fn new(
        message_rx: ipc::IpcReceiver<TraceRequest>,
        confirm_tx: ipc::IpcSender<Confirmation>,
    ) -> Self {
        Self { message_rx, confirm_tx, attached: false, override_retcode: None, last_code: None }
    }
}

impl Iterator for ChildListener {
    type Item = ExecEvent;

    // Allows us to monitor the child process by just iterating over the listener.
    // NB: This should never return None!
    fn next(&mut self) -> Option<Self::Item> {
        // Do not block if the child has nothing to report for `waitid`.
        let opts = WAIT_FLAGS | wait::WaitPidFlag::WNOHANG;
        loop {
            // Listen to any child, not just the main one. Important if we want
            // to allow the C code to fork further, along with being a bit of
            // defensive programming since Linux sometimes assigns threads of
            // the same process different PIDs with unpredictable rules...
            match wait::waitid(wait::Id::All, opts) {
                Ok(stat) =>
                    match stat {
                        // Child exited normally with a specific code set.
                        wait::WaitStatus::Exited(_, code) => self.last_code = Some(code),
                        // Child was killed by a signal, without giving a code.
                        wait::WaitStatus::Signaled(_, _, _) => self.last_code = None,
                        // Child entered or exited a syscall.
                        wait::WaitStatus::PtraceSyscall(pid) =>
                            if self.attached {
                                return Some(ExecEvent::Syscall(pid));
                            },
                        // Child with the given pid was stopped by the given signal.
                        // It's somewhat unclear when which of these two is returned;
                        // we just treat them the same.
                        wait::WaitStatus::Stopped(pid, signal)
                        | wait::WaitStatus::PtraceEvent(pid, signal, _) =>
                            if self.attached {
                                // This is our end-of-FFI signal!
                                if signal == signal::SIGUSR1 {
                                    self.attached = false;
                                    return Some(ExecEvent::End);
                                } else {
                                    return Some(ExecEvent::Status(pid, signal));
                                }
                            } else {
                                // Just pass along the signal.
                                ptrace::cont(pid, signal).unwrap();
                            },
                        _ => (),
                    },
                // This case should only trigger when all children died.
                Err(_) => return Some(ExecEvent::Died(self.override_retcode.or(self.last_code))),
            }

            // Similarly, do a non-blocking poll of the IPC channel.
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
                    TraceRequest::OverrideRetcode(code) => {
                        self.override_retcode = Some(code);
                        self.confirm_tx.send(Confirmation).unwrap();
                    }
                }
            }

            // Not ideal, but doing anything else might sacrifice performance.
            std::thread::yield_now();
        }
    }
}

/// An error came up while waiting on the child process to do something.
/// It likely died, with this return code if we have one.
#[derive(Debug)]
pub struct ExecEnd(pub Option<i32>);

/// Whether to call `ptrace::cont()` immediately. Used exclusively by `wait_for_signal`.
enum InitialCont {
    Yes,
    No,
}

/// This is the main loop of the supervisor process. It runs in a separate
/// process from the rest of Miri (but because we fork, addresses for anything
/// created before the fork - like statics - are the same).
pub fn sv_loop(
    listener: ChildListener,
    init_pid: unistd::Pid,
    event_tx: ipc::IpcSender<MemEvents>,
    confirm_tx: ipc::IpcSender<Confirmation>,
) -> Result<!, ExecEnd> {
    // Get the pagesize set and make sure it isn't still on the zero sentinel value!
    let page_size = PAGE_SIZE.load(std::sync::atomic::Ordering::Relaxed);
    assert_ne!(page_size, 0);

    // Things that we return to the child process.
    let mut acc_events = Vec::new();

    // Memory allocated for the MiriMachine.
    let mut ch_pages = Vec::new();
    let mut ch_stack = None;

    // An instance of the Capstone disassembler, so we don't spawn one on every access.
    let cs = get_disasm();

    // The pid of the last process we interacted with, used by default if we don't have a
    // reason to use a different one.
    let mut curr_pid = init_pid;

    // There's an initial sigstop we need to deal with.
    wait_for_signal(Some(curr_pid), signal::SIGSTOP, InitialCont::No)?;
    ptrace::cont(curr_pid, None).unwrap();

    for evt in listener {
        match evt {
            // Child started ffi, so prep memory.
            ExecEvent::Start(ch_info) => {
                // All the pages that the child process is "allowed to" access.
                ch_pages = ch_info.page_ptrs;
                // And the temporary callback stack it allocated for us to use later.
                ch_stack = Some(ch_info.stack_ptr);

                // We received the signal and are no longer in the main listener loop,
                // so we can let the child move on to the end of the ffi prep where it will
                // raise a SIGSTOP. We need it to be signal-stopped *and waited for* in
                // order to do most ptrace operations!
                confirm_tx.send(Confirmation).unwrap();
                // We can't trust simply calling `Pid::this()` in the child process to give the right
                // PID for us, so we get it this way.
                curr_pid = wait_for_signal(None, signal::SIGSTOP, InitialCont::No).unwrap();
                // Continue until next syscall.
                ptrace::syscall(curr_pid, None).unwrap();
            }
            // Child wants to end tracing.
            ExecEvent::End => {
                // Hand over the access info we traced.
                event_tx.send(MemEvents { acc_events }).unwrap();
                // And reset our values.
                acc_events = Vec::new();
                ch_stack = None;

                // No need to monitor syscalls anymore, they'd just be ignored.
                ptrace::cont(curr_pid, None).unwrap();
            }
            // Child process was stopped by a signal
            ExecEvent::Status(pid, signal) =>
                match signal {
                    // If it was a segfault, check if it was an artificial one
                    // caused by it trying to access the MiriMachine memory.
                    signal::SIGSEGV =>
                        handle_segfault(
                            pid,
                            &ch_pages,
                            ch_stack.unwrap(),
                            page_size,
                            &cs,
                            &mut acc_events,
                        )?,
                    // Something weird happened.
                    _ => {
                        eprintln!("Process unexpectedly got {signal}; continuing...");
                        // In case we're not tracing
                        if ptrace::syscall(pid, None).is_err() {
                            // If *this* fails too, something really weird happened
                            // and it's probably best to just panic.
                            signal::kill(pid, signal::SIGCONT).unwrap();
                        }
                    }
                },
            // Child entered or exited a syscall. For now we ignore this and just continue.
            ExecEvent::Syscall(pid) => {
                ptrace::syscall(pid, None).unwrap();
            }
            ExecEvent::Died(code) => {
                return Err(ExecEnd(code));
            }
        }
    }

    unreachable!()
}

/// Spawns a Capstone disassembler for the host architecture.
#[rustfmt::skip]
fn get_disasm() -> capstone::Capstone {
    use capstone::prelude::*;
    let cs_pre = Capstone::new();
    {
        #[cfg(target_arch = "x86_64")]
        {cs_pre.x86().mode(arch::x86::ArchMode::Mode64)}
        #[cfg(target_arch = "x86")]
        {cs_pre.x86().mode(arch::x86::ArchMode::Mode32)}
    }
    .detail(true)
    .build()
    .unwrap()
}

/// Waits for `wait_signal`. If `init_cont`, it will first do a `ptrace::cont`.
/// We want to avoid that in some cases, like at the beginning of FFI.
///
/// If `pid` is `None`, only one wait will be done and `init_cont` should be false.
fn wait_for_signal(
    pid: Option<unistd::Pid>,
    wait_signal: signal::Signal,
    init_cont: InitialCont,
) -> Result<unistd::Pid, ExecEnd> {
    if matches!(init_cont, InitialCont::Yes) {
        ptrace::cont(pid.unwrap(), None).unwrap();
    }
    // Repeatedly call `waitid` until we get the signal we want, or the process dies.
    loop {
        let wait_id = match pid {
            Some(pid) => wait::Id::Pid(pid),
            None => wait::Id::All,
        };
        let stat = wait::waitid(wait_id, WAIT_FLAGS).map_err(|_| ExecEnd(None))?;
        let (signal, pid) = match stat {
            // Report the cause of death, if we know it.
            wait::WaitStatus::Exited(_, code) => {
                return Err(ExecEnd(Some(code)));
            }
            wait::WaitStatus::Signaled(_, _, _) => return Err(ExecEnd(None)),
            wait::WaitStatus::Stopped(pid, signal)
            | wait::WaitStatus::PtraceEvent(pid, signal, _) => (signal, pid),
            // This covers PtraceSyscall and variants that are impossible with
            // the flags set (e.g. WaitStatus::StillAlive).
            _ => {
                ptrace::cont(pid.unwrap(), None).unwrap();
                continue;
            }
        };
        if signal == wait_signal {
            return Ok(pid);
        } else {
            ptrace::cont(pid, signal).map_err(|_| ExecEnd(None))?;
        }
    }
}

/// Add the memory events from `op` being executed while there is a memory access at `addr` to
/// `acc_events`. Return whether this was a memory operand.
fn capstone_find_events(
    addr: usize,
    op: &capstone::arch::ArchOperand,
    acc_events: &mut Vec<AccessEvent>,
) -> bool {
    use capstone::prelude::*;
    match op {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        arch::ArchOperand::X86Operand(x86_operand) => {
            match x86_operand.op_type {
                // We only care about memory accesses
                arch::x86::X86OperandType::Mem(_) => {
                    let push = AccessRange { addr, size: x86_operand.size.into() };
                    // It's called a "RegAccessType" but it also applies to memory
                    let acc_ty = x86_operand.access.unwrap();
                    // The same instruction might do both reads and writes, so potentially add both.
                    // We do not know the order in which they happened, but writing and then reading
                    // makes little sense so we put the read first. That is also the more
                    // conservative choice.
                    if acc_ty.is_readable() {
                        acc_events.push(AccessEvent::Read(push.clone()));
                    }
                    if acc_ty.is_writable() {
                        // FIXME: This could be made certain; either determine all cases where
                        // only reads happen, or have an intermediate mempr_* function to first
                        // map the page(s) as readonly and check if a segfault occurred.

                        // Per https://docs.rs/iced-x86/latest/iced_x86/enum.OpAccess.html,
                        // we know that the possible access types are Read, CondRead, Write,
                        // CondWrite, ReadWrite, and ReadCondWrite. Since we got a segfault
                        // we know some kind of access happened so Cond{Read, Write}s are
                        // certain reads and writes; the only uncertainty is with an RW op
                        // as it might be a ReadCondWrite with the write condition unmet.
                        acc_events.push(AccessEvent::Write(push, !acc_ty.is_readable()));
                    }

                    return true;
                }
                _ => (),
            }
        }
        // FIXME: arm64
        _ => unimplemented!(),
    }

    false
}

/// Extract the events from the given instruction.
fn capstone_disassemble(
    instr: &[u8],
    addr: usize,
    cs: &capstone::Capstone,
    acc_events: &mut Vec<AccessEvent>,
) -> capstone::CsResult<()> {
    // The arch_detail is what we care about, but it relies on these temporaries
    // that we can't drop. 0x1000 is the default base address for Captsone, and
    // we're expecting 1 instruction.
    let insns = cs.disasm_count(instr, 0x1000, 1)?;
    let ins_detail = cs.insn_detail(&insns[0])?;
    let arch_detail = ins_detail.arch_detail();

    let mut found_mem_op = false;

    for op in arch_detail.operands() {
        if capstone_find_events(addr, &op, acc_events) {
            if found_mem_op {
                panic!("more than one memory operand found; we don't know which one accessed what");
            }
            found_mem_op = true;
        }
    }

    Ok(())
}

/// Grabs the access that caused a segfault and logs it down if it's to our memory,
/// or kills the child and returns the appropriate error otherwise.
fn handle_segfault(
    pid: unistd::Pid,
    ch_pages: &[usize],
    ch_stack: usize,
    page_size: usize,
    cs: &capstone::Capstone,
    acc_events: &mut Vec<AccessEvent>,
) -> Result<(), ExecEnd> {
    // Get information on what caused the segfault. This contains the address
    // that triggered it.
    let siginfo = ptrace::getsiginfo(pid).unwrap();
    // All x86 instructions only have at most one memory operand (thankfully!)
    // SAFETY: si_addr is safe to call.
    let addr = unsafe { siginfo.si_addr().addr() };
    let page_addr = addr.strict_sub(addr.strict_rem(page_size));

    if !ch_pages.iter().any(|pg| (*pg..pg.strict_add(page_size)).contains(&addr)) {
        // This was a real segfault (not one of the Miri memory pages), so print some debug info and
        // quit.
        let regs = ptrace::getregs(pid).unwrap();
        eprintln!("Segfault occurred during FFI at {addr:#018x}");
        eprintln!("Expected access on pages: {ch_pages:#018x?}");
        eprintln!("Register dump: {regs:#x?}");
        ptrace::kill(pid).unwrap();
        return Err(ExecEnd(None));
    }

    // Overall structure:
    // - Get the address that caused the segfault
    // - Unprotect the memory: we force the child to execute `mempr_off`, passing parameters via
    //   global atomic variables. This is what we use the temporary callback stack for.
    // - Step 1 instruction
    // - Parse executed code to estimate size & type of access
    // - Reprotect the memory by executing `mempr_on` in the child, using the callback stack again.
    // - Continue

    // Ensure the stack is properly zeroed out!
    for a in (ch_stack..ch_stack.strict_add(CALLBACK_STACK_SIZE)).step_by(ARCH_WORD_SIZE) {
        ptrace::write(pid, std::ptr::with_exposed_provenance_mut(a), 0).unwrap();
    }

    // Guard against both architectures with upwards and downwards-growing stacks.
    let stack_ptr = ch_stack.strict_add(CALLBACK_STACK_SIZE / 2);
    let regs_bak = ptrace::getregs(pid).unwrap();
    let mut new_regs = regs_bak;

    // Read at least one instruction from the ip. It's possible that the instruction
    // that triggered the segfault was short and at the end of the mapped text area,
    // so some of these reads may fail; in that case, just write empty bytes. If all
    // reads failed, the disassembler will report an error.
    let instr = (0..(ARCH_MAX_INSTR_SIZE.div_ceil(ARCH_WORD_SIZE)))
        .flat_map(|ofs| {
            // This reads one word of memory; we divided by `ARCH_WORD_SIZE` above to compensate for that.
            ptrace::read(
                pid,
                std::ptr::without_provenance_mut(
                    regs_bak.ip().strict_add(ARCH_WORD_SIZE.strict_mul(ofs)),
                ),
            )
            .unwrap_or_default()
            .to_ne_bytes()
        })
        .collect::<Vec<_>>();

    // Now figure out the size + type of access and log it down.
    capstone_disassemble(&instr, addr, cs, acc_events).expect("Failed to disassemble instruction");

    // Move the instr ptr into the deprotection code.
    #[expect(clippy::as_conversions)]
    new_regs.set_ip(mempr_off as usize);
    // Don't mess up the stack by accident!
    new_regs.set_sp(stack_ptr);

    // Modify the PAGE_ADDR global on the child process to point to the page
    // that we want unprotected.
    ptrace::write(
        pid,
        (&raw const PAGE_ADDR).cast_mut().cast(),
        libc::c_long::try_from(page_addr.cast_signed()).unwrap(),
    )
    .unwrap();

    // Check if we also own the next page, and if so unprotect it in case
    // the access spans the page boundary.
    let flag = if ch_pages.contains(&page_addr.strict_add(page_size)) { 2 } else { 1 };
    ptrace::write(pid, (&raw const PAGE_COUNT).cast_mut().cast(), flag).unwrap();

    ptrace::setregs(pid, new_regs).unwrap();

    // Our mempr_* functions end with a raise(SIGSTOP).
    wait_for_signal(Some(pid), signal::SIGSTOP, InitialCont::Yes)?;

    // Step 1 instruction.
    ptrace::setregs(pid, regs_bak).unwrap();
    ptrace::step(pid, None).unwrap();
    // Don't use wait_for_signal here since 1 instruction doesn't give room
    // for any uncertainty + we don't want it `cont()`ing randomly by accident
    // Also, don't let it continue with unprotected memory if something errors!
    let _ = wait::waitid(wait::Id::Pid(pid), WAIT_FLAGS).map_err(|_| ExecEnd(None))?;

    // Zero out again to be safe
    for a in (ch_stack..ch_stack.strict_add(CALLBACK_STACK_SIZE)).step_by(ARCH_WORD_SIZE) {
        ptrace::write(pid, std::ptr::with_exposed_provenance_mut(a), 0).unwrap();
    }

    let regs_bak = ptrace::getregs(pid).unwrap();
    new_regs = regs_bak;

    // Reprotect everything and continue.
    #[expect(clippy::as_conversions)]
    new_regs.set_ip(mempr_on as usize);
    new_regs.set_sp(stack_ptr);
    ptrace::setregs(pid, new_regs).unwrap();
    wait_for_signal(Some(pid), signal::SIGSTOP, InitialCont::Yes)?;

    ptrace::setregs(pid, regs_bak).unwrap();
    ptrace::syscall(pid, None).unwrap();
    Ok(())
}

// We only get dropped into these functions via offsetting the instr pointer
// manually, so we *must not ever* unwind from them.

/// Disables protections on the page whose address is currently in `PAGE_ADDR`.
///
/// SAFETY: `PAGE_ADDR` should be set to a page-aligned pointer to an owned page,
/// `PAGE_SIZE` should be the host pagesize, and the range from `PAGE_ADDR` to
/// `PAGE_SIZE` * `PAGE_COUNT` must be owned and allocated memory. No other threads
/// should be running.
pub unsafe extern "C" fn mempr_off() {
    use std::sync::atomic::Ordering;

    // Again, cannot allow unwinds to happen here.
    let len = PAGE_SIZE.load(Ordering::Relaxed).saturating_mul(PAGE_COUNT.load(Ordering::Relaxed));
    // SAFETY: Upheld by "caller".
    unsafe {
        // It's up to the caller to make sure this doesn't actually overflow, but
        // we mustn't unwind from here, so...
        if libc::mprotect(
            PAGE_ADDR.load(Ordering::Relaxed).cast(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
        ) != 0
        {
            // Can't return or unwind, but we can do this.
            std::process::exit(-1);
        }
    }
    // If this fails somehow we're doomed.
    if signal::raise(signal::SIGSTOP).is_err() {
        std::process::exit(-1);
    }
}

/// Reenables protection on the page set by `PAGE_ADDR`.
///
/// SAFETY: See `mempr_off()`.
pub unsafe extern "C" fn mempr_on() {
    use std::sync::atomic::Ordering;

    let len = PAGE_SIZE.load(Ordering::Relaxed).wrapping_mul(PAGE_COUNT.load(Ordering::Relaxed));
    // SAFETY: Upheld by "caller".
    unsafe {
        if libc::mprotect(PAGE_ADDR.load(Ordering::Relaxed).cast(), len, libc::PROT_NONE) != 0 {
            std::process::exit(-1);
        }
    }
    if signal::raise(signal::SIGSTOP).is_err() {
        std::process::exit(-1);
    }
}
