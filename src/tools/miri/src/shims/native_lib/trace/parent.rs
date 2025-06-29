use std::sync::atomic::{AtomicPtr, AtomicUsize};

use ipc_channel::ipc;
use nix::sys::{ptrace, signal, wait};
use nix::unistd;

use super::CALLBACK_STACK_SIZE;
use super::messages::{AccessEvent, Confirmation, MemEvents, StartFfiInfo, TraceRequest};

/// The flags to use when calling `waitid()`.
/// Since bitwise or on the nix version of these flags is implemented as a trait,
/// this cannot be const directly so we do it this way.
const WAIT_FLAGS: wait::WaitPidFlag =
    wait::WaitPidFlag::from_bits_truncate(libc::WUNTRACED | libc::WEXITED);

/// Arch-specific maximum size a single access might perform. x86 value is set
/// assuming nothing bigger than AVX-512 is available.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const ARCH_MAX_ACCESS_SIZE: usize = 64;
/// The largest arm64 simd instruction operates on 16 bytes.
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
const ARCH_MAX_ACCESS_SIZE: usize = 16;

/// The default word size on a given platform, in bytes.
#[cfg(any(target_arch = "x86", target_arch = "arm"))]
const ARCH_WORD_SIZE: usize = 4;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const ARCH_WORD_SIZE: usize = 8;

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
#[expect(clippy::as_conversions)]
#[rustfmt::skip]
impl ArchIndependentRegs for libc::user_regs_struct {
    #[inline]
    fn ip(&self) -> usize { self.rip as _ }
    #[inline]
    fn set_ip(&mut self, ip: usize) { self.rip = ip as _ }
    #[inline]
    fn set_sp(&mut self, sp: usize) { self.rsp = sp as _ }
}

#[cfg(target_arch = "x86")]
#[expect(clippy::as_conversions)]
#[rustfmt::skip]
impl ArchIndependentRegs for libc::user_regs_struct {
    #[inline]
    fn ip(&self) -> usize { self.eip as _ }
    #[inline]
    fn set_ip(&mut self, ip: usize) { self.eip = ip as _ }
    #[inline]
    fn set_sp(&mut self, sp: usize) { self.esp = sp as _ }
}

#[cfg(target_arch = "aarch64")]
#[expect(clippy::as_conversions)]
#[rustfmt::skip]
impl ArchIndependentRegs for libc::user_regs_struct {
    #[inline]
    fn ip(&self) -> usize { self.pc as _ }
    #[inline]
    fn set_ip(&mut self, ip: usize) { self.pc = ip as _ }
    #[inline]
    fn set_sp(&mut self, sp: usize) { self.sp = sp as _ }
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
    pub message_rx: ipc::IpcReceiver<TraceRequest>,
    /// Whether an FFI call is currently ongoing.
    pub attached: bool,
    /// If `Some`, overrides the return code with the given value.
    pub override_retcode: Option<i32>,
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
                        wait::WaitStatus::Exited(_, code) => {
                            let code = self.override_retcode.unwrap_or(code);
                            return Some(ExecEvent::Died(Some(code)));
                        }
                        // Child was killed by a signal, without giving a code.
                        wait::WaitStatus::Signaled(_, _, _) =>
                            return Some(ExecEvent::Died(self.override_retcode)),
                        // Child entered a syscall. Since we're always technically
                        // tracing, only pass this along if we're actively
                        // monitoring the child.
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
                                // Just pass along the signal.
                                ptrace::cont(pid, signal).unwrap();
                            },
                        // Child was stopped at the given signal. Same logic as for
                        // WaitStatus::PtraceEvent.
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
                // for deadlocks.
                Err(_) => return Some(ExecEvent::Died(None)),
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
                    TraceRequest::OverrideRetcode(code) => self.override_retcode = Some(code),
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
    wait_for_signal(Some(curr_pid), signal::SIGSTOP, false)?;
    ptrace::cont(curr_pid, None).unwrap();

    for evt in listener {
        match evt {
            // start_ffi was called by the child, so prep memory.
            ExecEvent::Start(ch_info) => {
                // All the pages that the child process is "allowed to" access.
                ch_pages = ch_info.page_ptrs;
                // And the temporary callback stack it allocated for us to use later.
                ch_stack = Some(ch_info.stack_ptr);

                // We received the signal and are no longer in the main listener loop,
                // so we can let the child move on to the end of start_ffi where it will
                // raise a SIGSTOP. We need it to be signal-stopped *and waited for* in
                // order to do most ptrace operations!
                confirm_tx.send(Confirmation).unwrap();
                // We can't trust simply calling `Pid::this()` in the child process to give the right
                // PID for us, so we get it this way.
                curr_pid = wait_for_signal(None, signal::SIGSTOP, false).unwrap();

                ptrace::syscall(curr_pid, None).unwrap();
            }
            // end_ffi was called by the child.
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
            // Child entered a syscall; we wait for exits inside of this, so it
            // should never trigger on return from a syscall we care about.
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
        #[cfg(target_arch = "aarch64")]
        {cs_pre.arm64().mode(arch::arm64::ArchMode::Arm)}
        #[cfg(target_arch = "arm")]
        {cs_pre.arm().mode(arch::arm::ArchMode::Arm)}
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
    init_cont: bool,
) -> Result<unistd::Pid, ExecEnd> {
    if init_cont {
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
            wait::WaitStatus::Stopped(pid, signal) => (signal, pid),
            wait::WaitStatus::PtraceEvent(pid, signal, _) => (signal, pid),
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
    /// This is just here to not pollute the main namespace with `capstone::prelude::*`.
    #[inline]
    fn capstone_disassemble(
        instr: &[u8],
        addr: usize,
        cs: &capstone::Capstone,
        acc_events: &mut Vec<AccessEvent>,
    ) -> capstone::CsResult<()> {
        use capstone::prelude::*;

        // The arch_detail is what we care about, but it relies on these temporaries
        // that we can't drop. 0x1000 is the default base address for Captsone, and
        // we're expecting 1 instruction.
        let insns = cs.disasm_count(instr, 0x1000, 1)?;
        let ins_detail = cs.insn_detail(&insns[0])?;
        let arch_detail = ins_detail.arch_detail();

        for op in arch_detail.operands() {
            match op {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                arch::ArchOperand::X86Operand(x86_operand) => {
                    match x86_operand.op_type {
                        // We only care about memory accesses
                        arch::x86::X86OperandType::Mem(_) => {
                            let push = addr..addr.strict_add(usize::from(x86_operand.size));
                            // It's called a "RegAccessType" but it also applies to memory
                            let acc_ty = x86_operand.access.unwrap();
                            if acc_ty.is_readable() {
                                acc_events.push(AccessEvent::Read(push.clone()));
                            }
                            if acc_ty.is_writable() {
                                acc_events.push(AccessEvent::Write(push));
                            }
                        }
                        _ => (),
                    }
                }
                #[cfg(target_arch = "aarch64")]
                arch::ArchOperand::Arm64Operand(arm64_operand) => {
                    // Annoyingly, we don't always get the size here, so just be pessimistic for now.
                    match arm64_operand.op_type {
                        arch::arm64::Arm64OperandType::Mem(_) => {
                            // B = 1 byte, H = 2 bytes, S = 4 bytes, D = 8 bytes, Q = 16 bytes.
                            let size = match arm64_operand.vas {
                                // Not an fp/simd instruction.
                                arch::arm64::Arm64Vas::ARM64_VAS_INVALID => ARCH_WORD_SIZE,
                                // 1 byte.
                                arch::arm64::Arm64Vas::ARM64_VAS_1B => 1,
                                // 2 bytes.
                                arch::arm64::Arm64Vas::ARM64_VAS_1H => 2,
                                // 4 bytes.
                                arch::arm64::Arm64Vas::ARM64_VAS_4B
                                | arch::arm64::Arm64Vas::ARM64_VAS_2H
                                | arch::arm64::Arm64Vas::ARM64_VAS_1S => 4,
                                // 8 bytes.
                                arch::arm64::Arm64Vas::ARM64_VAS_8B
                                | arch::arm64::Arm64Vas::ARM64_VAS_4H
                                | arch::arm64::Arm64Vas::ARM64_VAS_2S
                                | arch::arm64::Arm64Vas::ARM64_VAS_1D => 8,
                                // 16 bytes.
                                arch::arm64::Arm64Vas::ARM64_VAS_16B
                                | arch::arm64::Arm64Vas::ARM64_VAS_8H
                                | arch::arm64::Arm64Vas::ARM64_VAS_4S
                                | arch::arm64::Arm64Vas::ARM64_VAS_2D
                                | arch::arm64::Arm64Vas::ARM64_VAS_1Q => 16,
                            };
                            let push = addr..addr.strict_add(size);
                            // FIXME: This now has access type info in the latest
                            // git version of capstone because this pissed me off
                            // and I added it. Change this when it updates.
                            acc_events.push(AccessEvent::Read(push.clone()));
                            acc_events.push(AccessEvent::Write(push));
                        }
                        _ => (),
                    }
                }
                #[cfg(target_arch = "arm")]
                arch::ArchOperand::ArmOperand(arm_operand) =>
                    match arm_operand.op_type {
                        arch::arm::ArmOperandType::Mem(_) => {
                            // We don't get info on the size of the access, but
                            // we're at least told if it's a vector instruction.
                            let size = if arm_operand.vector_index.is_some() {
                                ARCH_MAX_ACCESS_SIZE
                            } else {
                                ARCH_WORD_SIZE
                            };
                            let push = addr..addr.strict_add(size);
                            let acc_ty = arm_operand.access.unwrap();
                            if acc_ty.is_readable() {
                                acc_events.push(AccessEvent::Read(push.clone()));
                            }
                            if acc_ty.is_writable() {
                                acc_events.push(AccessEvent::Write(push));
                            }
                        }
                        _ => (),
                    },
                _ => unimplemented!(),
            }
        }

        Ok(())
    }

    // Get information on what caused the segfault. This contains the address
    // that triggered it.
    let siginfo = ptrace::getsiginfo(pid).unwrap();
    // All x86, ARM, etc. instructions only have at most one memory operand
    // (thankfully!)
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
    // - Reprotect the memory by executing `mempr_on` in the child.
    // - Continue

    // Ensure the stack is properly zeroed out!
    for a in (ch_stack..ch_stack.strict_add(CALLBACK_STACK_SIZE)).step_by(ARCH_WORD_SIZE) {
        ptrace::write(pid, std::ptr::with_exposed_provenance_mut(a), 0).unwrap();
    }

    // Guard against both architectures with upwards and downwards-growing stacks.
    let stack_ptr = ch_stack.strict_add(CALLBACK_STACK_SIZE / 2);
    let regs_bak = ptrace::getregs(pid).unwrap();
    let mut new_regs = regs_bak;
    let ip_prestep = regs_bak.ip();

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
        libc::c_long::try_from(page_addr).unwrap(),
    )
    .unwrap();

    // Check if we also own the next page, and if so unprotect it in case
    // the access spans the page boundary.
    let flag = if ch_pages.contains(&page_addr.strict_add(page_size)) { 2 } else { 1 };
    ptrace::write(pid, (&raw const PAGE_COUNT).cast_mut().cast(), flag).unwrap();

    ptrace::setregs(pid, new_regs).unwrap();

    // Our mempr_* functions end with a raise(SIGSTOP).
    wait_for_signal(Some(pid), signal::SIGSTOP, true)?;

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

    // Save registers and grab the bytes that were executed. This would
    // be really nasty if it was a jump or similar but those thankfully
    // won't do memory accesses and so can't trigger this!
    let regs_bak = ptrace::getregs(pid).unwrap();
    new_regs = regs_bak;
    let ip_poststep = regs_bak.ip();
    // We need to do reads/writes in word-sized chunks.
    let diff = (ip_poststep.strict_sub(ip_prestep)).div_ceil(ARCH_WORD_SIZE);
    let instr = (ip_prestep..ip_prestep.strict_add(diff)).fold(vec![], |mut ret, ip| {
        // This only needs to be a valid pointer in the child process, not ours.
        ret.append(
            &mut ptrace::read(pid, std::ptr::without_provenance_mut(ip))
                .unwrap()
                .to_ne_bytes()
                .to_vec(),
        );
        ret
    });

    // Now figure out the size + type of access and log it down.
    // This will mark down e.g. the same area being read multiple times,
    // since it's more efficient to compress the accesses at the end.
    if capstone_disassemble(&instr, addr, cs, acc_events).is_err() {
        // Read goes first because we need to be pessimistic.
        acc_events.push(AccessEvent::Read(addr..addr.strict_add(ARCH_MAX_ACCESS_SIZE)));
        acc_events.push(AccessEvent::Write(addr..addr.strict_add(ARCH_MAX_ACCESS_SIZE)));
    }

    // Reprotect everything and continue.
    #[expect(clippy::as_conversions)]
    new_regs.set_ip(mempr_on as usize);
    new_regs.set_sp(stack_ptr);
    ptrace::setregs(pid, new_regs).unwrap();
    wait_for_signal(Some(pid), signal::SIGSTOP, true)?;

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
