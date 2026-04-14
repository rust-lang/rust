//! Architecture-specific signal frame injection and delivery.
//!
//! The public entry point is `kernel_post_syscall_signal_check`, which is
//! called from the assembly syscall path on x86_64 immediately before
//! `sysretq`.  It receives a pointer to the saved user trap frame on the
//! kernel stack, checks whether any signal is pending and deliverable, and
//! if so modifies the frame to redirect user execution to the handler.
//!
//! # x86_64 signal frame layout
//!
//! When a handler is invoked, the kernel sets up a `SignalFrame` on the user
//! stack (below the current user RSP, with 128-byte red-zone cleared).  The
//! layout is:
//!
//! ```text
//! lower addresses
//! ┌─────────────────────────────┐ ← new user RSP (16-byte aligned)
//! │ SignalFrame { … }           │
//! │   trampoline[8]             │ ← inline `syscall` for SYS_SIGRETURN
//! │   saved_rip                 │
//! │   saved_rsp                 │
//! │   saved_rflags              │
//! │   saved_rbp                 │
//! │   saved_rcx (user)          │
//! │   saved_rdx (user)          │
//! │   saved_rsi (user)          │
//! │   saved_rdi (user)          │
//! │   saved_rax (user retval)   │
//! │   sig                       │
//! └─────────────────────────────┘
//! higher addresses (original user RSP above)
//! ```
//!
//! The return address pushed by the kernel (i.e., the address that `ret`
//! inside the signal handler would pop) points to `trampoline`, which
//! executes `mov $SYS_SIGRETURN, %eax; syscall` to invoke `sigreturn`.
//!
//! # Other architectures
//!
//! Only x86_64 has the full injection path; other targets use a stub that
//! terminates the process on handler invocation.

use abi::errors::{Errno, SysResult};

// ── x86_64 ───────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
    use super::*;
    use abi::signal::{SIG_DFL, SIG_IGN, SigAction};
    use alloc::sync::Arc;
    use crate::signal::{UNCATCHABLE, default_action, DefaultAction};
    use crate::syscall::validate::validate_user_range;

    /// Layout of the kernel-side saved user register state.
    ///
    /// Must match the push order in `bran/src/arch/x86_64/syscall.rs`:
    /// r15, r14, r13, r12, r11, r10, r9, r8, rbp, rdi, rsi, rdx, rcx,
    /// rbx, rax, error_code, int_no, rip, cs, rflags, rsp, ss
    #[repr(C)]
    pub struct SyscallFrame {
        pub r15: u64,
        pub r14: u64,
        pub r13: u64,
        pub r12: u64,
        pub r11: u64,
        pub r10: u64,
        pub r9: u64,
        pub r8: u64,
        pub rbp: u64,
        pub rdi: u64,
        pub rsi: u64,
        pub rdx: u64,
        pub rcx: u64,
        pub rbx: u64,
        pub rax: u64,
        pub error_code: u64,
        pub int_no: u64,
        pub rip: u64,  // user return instruction pointer
        pub cs: u64,
        pub rflags: u64,
        pub rsp: u64,  // user stack pointer
        pub ss: u64,
    }

    /// Signal frame placed on the *user* stack before invoking a handler.
    ///
    /// Size must be a multiple of 16 for alignment; pad if necessary.
    #[repr(C)]
    pub struct SignalFrame {
        /// Trampoline code: `mov $SYS_SIGRETURN, %eax; syscall; ud2`.
        /// Handler returns here → triggers SYS_SIGRETURN.
        pub trampoline: [u8; 16],
        // ── saved user context ─────────────────────────────────────────
        pub saved_rip: u64,
        pub saved_rsp: u64,
        pub saved_rflags: u64,
        pub saved_rax: u64,
        pub saved_rbx: u64,
        pub saved_rcx: u64,
        pub saved_rdx: u64,
        pub saved_rsi: u64,
        pub saved_rdi: u64,
        pub saved_rbp: u64,
        pub saved_r8: u64,
        pub saved_r9: u64,
        pub saved_r10: u64,
        pub saved_r11: u64,
        pub saved_r12: u64,
        pub saved_r13: u64,
        pub saved_r14: u64,
        pub saved_r15: u64,
        // ── signal info ───────────────────────────────────────────────
        pub signum: u64,
        pub saved_mask: u64,
        /// Padding to reach a multiple of 16 bytes.
        pub _pad: u64,
        pub _pad2: u64,
    }

    const _: () = assert!(core::mem::size_of::<SignalFrame>() % 16 == 0);

    /// Trampoline: `mov $SYS_SIGRETURN, %eax` (5 bytes) + `syscall` (2 bytes)
    /// + `ud2` (2 bytes, fault if accidentally executed again) + padding.
    fn make_trampoline() -> [u8; 16] {
        let n = abi::syscall::SYS_SIGRETURN as u32;
        let nb = n.to_le_bytes();
        [
            0xb8, nb[0], nb[1], nb[2], nb[3], // mov $SYS_SIGRETURN, %eax
            0x0f, 0x05,                        // syscall
            0x0f, 0x0b,                        // ud2 (if return is somehow reached)
            0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, // nop padding
        ]
    }

    /// Check for pending deliverable signals and, if found, set up a handler
    /// frame on the user stack and redirect the trap frame to the handler.
    ///
    /// # Safety
    ///
    /// `frame` must point to a valid `SyscallFrame` on the kernel stack.
    pub unsafe fn post_syscall_signal_check(frame: *mut SyscallFrame) {
        let f = unsafe { &mut *frame };

        // Retrieve process and thread signal state via type-erased hooks.
        let pinfo_arc = match crate::sched::process_info_current() {
            Some(a) => a,
            None => return,
        };

        let thread_mask = crate::sched::hooks::get_signal_mask_current();
        let thread_pending = crate::sched::hooks::get_thread_pending_current();

        let (sig, action) = {
            let mut p = pinfo_arc.lock();
            let combined = thread_pending.union(p.unix_compat.signals.pending);
            let deliverable = combined.difference(thread_mask)
                .union(combined.intersection(UNCATCHABLE));
            let sig = deliverable.lowest();
            if sig == 0 {
                return;
            }
            // Remove from the source queue.
            if thread_pending.contains(sig) {
                drop(p);
                let mut new_pending = thread_pending;
                new_pending.remove(sig);
                crate::sched::hooks::set_thread_pending_current(new_pending);
                let action = pinfo_arc.lock().unix_compat.signals.action(sig);
                (sig, action)
            } else {
                p.unix_compat.signals.pending.remove(sig);
                let action = p.unix_compat.signals.action(sig);
                drop(p);
                (sig, action)
            }
        };

        deliver_signal(f, sig, action, &pinfo_arc);
    }

    fn deliver_signal(
        f: &mut SyscallFrame,
        sig: u8,
        action: SigAction,
        pinfo_arc: &Arc<spin::Mutex<crate::task::Process>>,
    ) {
        use abi::signal::SIG_IGN;
        // SIGKILL and SIGSTOP bypass everything.
        if sig == abi::signal::SIGKILL {
            unsafe { crate::sched::exit_current(-9) };
            return;
        }

        let handler = action.handler;

        // Determine what to do.
        if handler == SIG_IGN {
            // Already removed from pending; nothing to do.
            return;
        }

        if handler == SIG_DFL {
            // Execute default action.
            match default_action(sig) {
                DefaultAction::Ignore => {}
                DefaultAction::Terminate | DefaultAction::CoreDump => {
                    // Encode termination-by-signal exit status.
                    let status = abi::signal::w_term_sig(sig);
                    // Notify parent before exiting.
                    {
                        let p = pinfo_arc.lock();
                        let ppid = p.lifecycle.ppid;
                        let pid = p.pid;
                        drop(p);
                        crate::signal::notify_parent_sigchld(ppid, pid, status);
                    }
                    unsafe { crate::sched::exit_current(status) };
                    return;
                }
                DefaultAction::Stop => {
                    stop_current_process(pinfo_arc, sig);
                    return;
                }
                DefaultAction::Continue => {
                    continue_current_process(pinfo_arc);
                    return;
                }
            }
            return;
        }

        if sig == abi::signal::SIGSTOP {
            // Cannot be caught.
            stop_current_process(pinfo_arc, sig);
            return;
        }

        // Invoke user handler.
        // Build the signal frame on the user stack.
        let user_rsp = f.rsp;
        // Skip the x86_64 128-byte red zone and align to 16 bytes.
        let frame_size = core::mem::size_of::<SignalFrame>() as u64;
        // New RSP: align downward to 16, minus frame size.
        let new_rsp = (user_rsp.wrapping_sub(128).wrapping_sub(frame_size)) & !15u64;

        // Validate that the target user stack range is writable.
        if validate_user_range(new_rsp as usize, frame_size as usize, true).is_err() {
            // Stack fault — terminate with SIGSEGV.
            let status = abi::signal::w_term_sig(abi::signal::SIGSEGV);
            {
                let p = pinfo_arc.lock();
                let ppid = p.lifecycle.ppid;
                let pid = p.pid;
                drop(p);
                crate::signal::notify_parent_sigchld(ppid, pid, status);
            }
            unsafe { crate::sched::exit_current(status) };
            return;
        }

        // Build the frame.
        let sf = SignalFrame {
            trampoline: make_trampoline(),
            saved_rip: f.rip,
            saved_rsp: f.rsp,
            saved_rflags: f.rflags,
            saved_rax: f.rax,
            saved_rbx: f.rbx,
            saved_rcx: f.rcx,
            saved_rdx: f.rdx,
            saved_rsi: f.rsi,
            saved_rdi: f.rdi,
            saved_rbp: f.rbp,
            saved_r8: f.r8,
            saved_r9: f.r9,
            saved_r10: f.r10,
            saved_r11: f.r11,
            saved_r12: f.r12,
            saved_r13: f.r13,
            saved_r14: f.r14,
            saved_r15: f.r15,
            signum: sig as u64,
            saved_mask: crate::sched::hooks::get_signal_mask_current().0,
            _pad: 0,
            _pad2: 0,
        };

        // Write the signal frame into user memory.
        unsafe {
            let dst = new_rsp as *mut SignalFrame;
            core::ptr::write_volatile(dst, sf);
        }

        // The trampoline sits at the base of the frame (lowest address).
        let trampoline_addr = new_rsp;

        // Update signal mask: block `sig` and `action.mask` while handler runs
        // (unless SA_NODEFER is set).
        {
            let mut current_mask = crate::sched::hooks::get_signal_mask_current();
            if action.flags & abi::signal::sa_flags::SA_NODEFER == 0 {
                current_mask.add(sig);
            }
            current_mask = current_mask.union(action.mask);
            current_mask = abi::signal::SigSet(current_mask.0 & !UNCATCHABLE.0);
            crate::sched::hooks::set_signal_mask_current(current_mask);
        }

        // Handle SA_RESETHAND: reset disposition to SIG_DFL after delivery.
        if action.flags & abi::signal::sa_flags::SA_RESETHAND != 0 {
            let mut p = pinfo_arc.lock();
            p.unix_compat.signals.actions[(sig - 1) as usize] = SigAction::default();
        }

        // Redirect the trap frame.
        // Push the trampoline address as the "return address" by adjusting RSP.
        // We do this by pointing RSP to the 8 bytes just below the frame that
        // hold the trampoline address.
        // Actually, the handler will use `ret` which pops the return address.
        // We place the trampoline address at *new_rsp - 8 as a "return address".
        let ret_addr_slot = new_rsp.wrapping_sub(8);
        if validate_user_range(ret_addr_slot as usize, 8, true).is_ok() {
            unsafe {
                core::ptr::write_volatile(ret_addr_slot as *mut u64, trampoline_addr);
            }
            f.rsp = ret_addr_slot;
        } else {
            f.rsp = new_rsp;
        }

        f.rip = handler as u64;
        // Pass signal number as first argument (rdi).
        f.rdi = sig as u64;
        // Zero other argument registers for cleanliness.
        f.rsi = 0;
        f.rdx = 0;
        // Preserve user RFLAGS but clear the direction flag.
        f.rflags = f.rflags & !0x0400;
        // rax will be restored from SignalFrame.saved_rax on sigreturn;
        // leave it alone for now (the handler may clobber it anyway).
    }

    fn stop_current_process(
        pinfo_arc: &Arc<spin::Mutex<crate::task::Process>>,
        sig: u8,
    ) {
        {
            let mut p = pinfo_arc.lock();
            p.unix_compat.signals.stopped = true;
            let ppid = p.lifecycle.ppid;
            let pid = p.pid;
            drop(p);
            crate::signal::notify_parent_sigchld(ppid, pid, abi::signal::w_stop_sig(sig));
        }
        // Block the current thread until SIGCONT is delivered.
        unsafe { crate::sched::block_current_erased() };
    }

    fn continue_current_process(pinfo_arc: &Arc<spin::Mutex<crate::task::Process>>) {
        let mut p = pinfo_arc.lock();
        if !p.unix_compat.signals.stopped {
            return;
        }
        p.unix_compat.signals.stopped = false;
        let tids = p.lifecycle.thread_ids.clone();
        let ppid = p.lifecycle.ppid;
        let pid = p.pid;
        drop(p);
        for tid in tids {
            unsafe { crate::sched::wake_task_erased(tid as u64) };
        }
        crate::signal::notify_parent_sigchld(ppid, pid, abi::signal::w_continued());
    }

    /// Restore user context from the signal frame on the user stack.
    ///
    /// Called by `SYS_SIGRETURN` syscall handler.  The user RSP (from the
    /// trap frame) points to the `SignalFrame` we placed there earlier.
    pub unsafe fn sys_sigreturn_inner(frame: *mut SyscallFrame) -> SysResult<usize> {
        let f = unsafe { &mut *frame };

        // The user RSP at sigreturn entry points to the slot just above the
        // trampoline return address (which the `syscall` instruction advanced
        // past).  We stored `SignalFrame` at `new_rsp`, so:
        //   new_rsp       = frame start (trampoline)
        //   ret_addr_slot = new_rsp - 8  ← that's where f.rsp landed
        // So the frame is at f.rsp + 8.
        let sf_ptr = (f.rsp.wrapping_add(8)) as *const SignalFrame;

        // Validate the frame is readable.
        validate_user_range(sf_ptr as usize, core::mem::size_of::<SignalFrame>(), false)?;

        let sf = unsafe { core::ptr::read_volatile(sf_ptr) };

        // Restore registers.
        f.rip = sf.saved_rip;
        f.rsp = sf.saved_rsp;
        f.rflags = sf.saved_rflags;
        f.rax = sf.saved_rax;
        f.rbx = sf.saved_rbx;
        f.rcx = sf.saved_rcx;
        f.rdx = sf.saved_rdx;
        f.rsi = sf.saved_rsi;
        f.rdi = sf.saved_rdi;
        f.rbp = sf.saved_rbp;
        f.r8 = sf.saved_r8;
        f.r9 = sf.saved_r9;
        f.r10 = sf.saved_r10;
        f.r11 = sf.saved_r11;
        f.r12 = sf.saved_r12;
        f.r13 = sf.saved_r13;
        f.r14 = sf.saved_r14;
        f.r15 = sf.saved_r15;

        // Restore the signal mask.
        let restored_mask = abi::signal::SigSet(sf.saved_mask & !UNCATCHABLE.0);
        crate::sched::hooks::set_signal_mask_current(restored_mask);

        Ok(0)
    }
}

// ── stub for non-x86_64 ───────────────────────────────────────────────────────

#[cfg(not(target_arch = "x86_64"))]
pub mod stub {
    use abi::errors::SysResult;

    /// On non-x86_64, signal delivery is a no-op (stubs pending future work).
    #[inline]
    pub unsafe fn post_syscall_signal_check(_frame: *mut ()) {}

    pub unsafe fn sys_sigreturn_inner(_frame: *mut ()) -> SysResult<usize> {
        Err(abi::errors::Errno::ENOSYS)
    }
}

// ── public dispatch ───────────────────────────────────────────────────────────

/// Called from the flat assembly dispatch (after every syscall) to check
/// and deliver pending signals.
///
/// The `frame` pointer must point to the `SyscallFrame` (trap frame) saved
/// on the kernel stack by the syscall entry stub.
///
/// # Safety
///
/// Must be called from the syscall path with a valid kernel-stack frame.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kernel_post_syscall_signal_check(frame: *mut u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        x86_64::post_syscall_signal_check(
            frame as *mut x86_64::SyscallFrame,
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = frame;
}

/// Called by `SYS_SIGRETURN` to restore a saved user context from the
/// signal frame.  The frame pointer is the same kernel trap frame passed
/// to signal check.
///
/// # Safety
///
/// Must be called from the syscall dispatch path.
pub unsafe fn sys_sigreturn_inner(frame: *mut u8) -> abi::errors::SysResult<usize> {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        x86_64::sys_sigreturn_inner(frame as *mut x86_64::SyscallFrame)
    }
    #[cfg(not(target_arch = "x86_64"))]
    Err(abi::errors::Errno::ENOSYS)
}
