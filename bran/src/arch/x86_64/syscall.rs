use core::arch::{asm, global_asm};

// MSR Constants
const MSR_EFER: u32 = 0xC0000080;
const MSR_STAR: u32 = 0xC0000081;
const MSR_LSTAR: u32 = 0xC0000082;
const MSR_SFMASK: u32 = 0xC0000084;
const MSR_GS_BASE: u32 = 0xC0000101;
const MSR_KERNEL_GS_BASE: u32 = 0xC0000102;

const EFER_SCE: u64 = 1; // Syscall Enable
const EFER_NXE: u64 = 1 << 11; // No-Execute Enable

// GDT Selectors (Must match what we assume in userspace/trampolines)
// Kernel Code: 0x08 (1)
// Kernel Data: 0x10 (2)
// User Code32: 0x18 (3)
// User Data:   0x20 (4)
// User Code64: 0x28 (5) -> Wait, usually Linux uses:
//   32-bit STAR: [31:16] CS (user 32), [15:0] Target CS (kernel)
//   48-bit STAR: [63:48] CS (user 32/64 ret).
// Let's assume:
//  Kernel Code = 0x08
//  Kernel Data = 0x10
//  User Data   = 0x18 | 3  (RPL3)
//  User Code   = 0x20 | 3  (RPL3)
// Limine GDT might differ.
// Just for v0.5, let's hardcode what we use.
#[allow(dead_code)]
const _KERNEL_CS: u16 = 0x08;
#[allow(dead_code)]
const _KERNEL_DS: u16 = 0x10;
// When SYSRET loads CS/SS:
//  CS = (STAR[63:48] + 16) | 3
//  SS = (STAR[63:48] + 8) | 3
// So if STAR[63:48] = 0x08 (Kernel Code), then:
//  CS = 0x18 | 3 = User Code 32? No.
// Let's check AMD manuals.
// SYSRET:
//   CS_Sel = STAR[63:48] + 16.
//   SS_Sel = STAR[63:48] + 8.
// User SS is 0x23 (User Data selector), User CS is 0x2B (User Code64 selector).
//  Diff is 8. So SS should be lower index than CS?
//  Usually User Code is *after* User Data in GDT for automatic SYSRET.
//  If User Data = 0x18, User Code = 0x20.
//  Then STAR[63:48] should be 0x10 (Kernel Data? No).
//  STAR[63:48] = 0x10.
//    CS = 0x10 + 16 = 0x20.
//    SS = 0x10 + 8 = 0x18.
//  Yes. So Base Selector = 0x10.
// SYSCALL:
//   CS = STAR[31:47] = 0x08.
//   SS = STAR[31:47] + 8 = 0x10.
// So STAR = (0x10 << 48) | (0x08 << 32).

const MAX_CPUS: usize = 32;

#[derive(Copy, Clone)]
#[repr(C)]
struct CpuLocal {
    scratch_rsp: u64, // offset 0
    kstack_top: u64,  // offset 8
    cpu_index: u64,   // offset 16
    current_tid: u64, // offset 24
}

static mut CPU_LOCAL: [CpuLocal; MAX_CPUS] = [CpuLocal {
    scratch_rsp: 0,
    kstack_top: 0,
    cpu_index: 0,
    current_tid: 0,
}; MAX_CPUS];

pub unsafe fn init(cpu_index: usize) {
    unsafe {
        CPU_LOCAL[cpu_index].cpu_index = cpu_index as u64;

        // 1. Setup GS Base for this specific CPU
        let gs_base = (&raw mut CPU_LOCAL[cpu_index]) as u64;

        // Keep both GS base MSRs pointing at CPU_LOCAL for now so swapgs is safe.
        wrmsr(MSR_GS_BASE, gs_base);
        wrmsr(MSR_KERNEL_GS_BASE, gs_base);

        // 2. Enable SCE (Syscall) and NXE (No-Execute) in EFER
        let efer = rdmsr(MSR_EFER);
        wrmsr(MSR_EFER, efer | EFER_SCE | EFER_NXE);

        // 3. Setup STAR
        let star = ((crate::arch::x86_64::gdt::KERNEL_CODE_SEL as u64) << 32)
            | (((crate::arch::x86_64::gdt::USER_CODE32_SEL ^ 3) as u64) << 48);
        wrmsr(MSR_STAR, star);

        // 4. Setup LSTAR (Entry point)
        wrmsr(MSR_LSTAR, syscall_entry as *const () as usize as u64);

        // 5. Setup SFMASK (Mask Interrupts 0x200)
        wrmsr(MSR_SFMASK, 0x200);
    }
}

unsafe fn rdmsr(msr: u32) -> u64 {
    let low: u32;
    let high: u32;
    unsafe {
        asm!("rdmsr", in("ecx") msr, out("eax") low, out("edx") high);
    }
    ((high as u64) << 32) | (low as u64)
}

unsafe fn wrmsr(msr: u32, val: u64) {
    let low = val as u32;
    let high = (val >> 32) as u32;
    unsafe {
        asm!("wrmsr", in("ecx") msr, in("eax") low, in("edx") high);
    }
}

unsafe extern "C" {
    fn syscall_entry();
}

global_asm!(
    r#"
.section .text
.global syscall_entry
syscall_entry:
    // Enters with CS=Kernel, SS=Kernel.
    // RCX=User RIP, R11=User RFLAGS.
    // RSP=User Stack.
    
    swapgs
    
    // Save User RSP to scratch (offset 0)
    mov %rsp, %gs:0
    
    // Load Kernel RSP from offset 8
    mov %gs:8, %rsp
    
    // Now on Kernel Stack.
    // Build UserTrapFrame. 
    // Struct layout: r15..r8, rbp, rdi, rsi, rdx, rcx, rbx, rax, error_code, int_no, rip, cs, rflags, rsp, ss
    
    // We need to manufacture SS, RSP, RFLAGS, CS, RIP
    // User SS = 0x23 (Hardcoded matches task.rs) | OR we could just save what we think it is. 
    // But 'syscall' doesn't save SS. We assume standard user SS.
    pushq ${user_ss}        // SS
    pushq %gs:0         // User RSP (from scratch)
    pushq %r11          // RFLAGS
    pushq ${user_cs}        // CS
    pushq %rcx          // RIP
    
    // Error Code / Int No
    pushq $4           // int_no (SYS_DEVICE_CALL=4? No, just using 0x80 or similar marker) -> Let's use 0x80 as "Synch Trap" marker
    pushq $0           // error_code
    
    // GPRs
    pushq %rax
    pushq %rbx
    pushq %rcx  // Note: RCX contains User RIP, but we push it as GPR anyway to match struct
    pushq %rdx
    pushq %rsi
    pushq %rdi
    pushq %rbp
    pushq %r8
    pushq %r9
    pushq %r10
    pushq %r11 // Note: R11 contains User RFLAGS
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    // Arguments for dispatch(n, args)
    // Rust ABI: RDI, RSI.
    // dispatch signature: fn dispatch(n: usize, args: [usize; 6]) -> isize
    
    // Mapping:
    // Syscall ABI (Linux/Standard we chose):
    // RAX = Number
    // RDI = Arg0
    // RSI = Arg1
    // RDX = Arg2
    // R10 = Arg3
    // R8  = Arg4
    // R9  = Arg5
    
    // Rust Call to dispatch(n, a0, a1, a2, a3, a4, a5)
    // RDI = n (from RAX)
    // RSI = a0 (from RDI - wait, conflict)
    // RDX = a1 (from RSI)
    // RCX = a2 (from RDX)
    // R8  = a3 (from R10)
    // R9  = a4 (from R8)
    // Stack = a5 (from R9)
    
    // We need to shuffle.
    // Saved Regs are on stack. We can read from there if needed, or move directly.
    
    // We need to preserve RAX (syscall num) to pass as 1st arg.
    mov %rdi, %r12 // Temp save Arg0 (RDI)
    mov %rsi, %r13 // Temp save Arg1 (RSI)
    mov %rdx, %r14 // Temp save Arg2 (RDX) - Fix: Preserve RDX before overwrite
    
    mov %rax, %rdi // 1st Arg: n
    
    mov %r12, %rsi // 2nd Arg: a0
    mov %r13, %rdx // 3rd Arg: a1
    // RDX (Arg2) needs to go to RCX (4th Arg slot for Rust function)
    mov %r14, %rcx 
    
    // Capture frame_ptr = current RSP (points to base of GPR save area).
    // We use r15 as a temporary register; the user's r15 value is safe on the
    // stack at [rsp+0] and will be restored by popq %r15 below.
    mov %rsp, %r15   // r15 = frame_ptr
    
    // Push stack arguments in right-to-left order:
    // 8th arg (frame_ptr) — pushed first, lands at higher stack address
    pushq %r15
    // 7th arg (a5, from original R9) — pushed second, lands at lower stack address
    pushq %r9
    
    // 5th reg-arg: A4 (R8) → R9
    mov %r8, %r9
    
    // 4th stack-based reg-arg: A3 (R10) → R8
    mov %r10, %r8
    
    // Stack alignment: we pushed 2 × 8 = 16 bytes.
    // Before the pushes RSP was 16-byte aligned; after 2 pushes it is still
    // aligned.  'call' will push RIP (8 bytes) so inside the function RSP
    // will be 16n − 8, satisfying the System V ABI requirement.
    
    call kernel_dispatch_flat
    
    // Cleanup: remove the 2 stack arguments pushed above (frame_ptr + a5).
    add $16, %rsp
    
    // RAX has return value (isize).
    // We need to put it into UserTrapFrame's RAX slot so it gets restored.
    // TrapFrame layout: ... rbx, rax, error_code ...
    // RAX is at top of GPRs (lowest address).
    // Stack top is R15.
    // Struct:
    // R15 (0), R14 (8), R13 (16), R12 (24), R11 (32), R10 (40), R9 (48), R8 (56),
    // RBP (64), RDI (72), RSI (80), RDX (88), RCX (96), RBX (104), RAX (112).
    // So [rsp + 112] is RAX.
    mov %rax, 112(%rsp)
    
    // Restore
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %r11
    popq %r10
    popq %r9
    popq %r8
    popq %rbp
    popq %rdi
    popq %rsi
    popq %rdx
    popq %rcx
    popq %rbx
    popq %rax
    
    // Switch to sysretq for return (faster and assumes consistent GDT)
    // We need to restore RCX (RIP) and R11 (RFLAGS) for sysretq.
    // GPRs popped above restored User RCX/R11 (clobbered/arguments).
    // The "True" RIP/RFLAGS are in the IRET frame on stack.
    // Stack Check: [Error(0), Int(8), RIP(16), CS(24), RFLAGS(32), RSP(40), SS(48)]
    //
    // NOTE: signal delivery in kernel_dispatch_flat may have modified the
    // IRET frame (RIP/RSP/RFLAGS) to redirect to a signal handler.
    // We always load from the saved frame, not from preserved registers,
    // so signal frame redirection is automatically respected here.
    
    mov 16(%rsp), %rcx  // Load RIP into RCX
    mov 32(%rsp), %r11  // Load RFLAGS into R11
    mov 40(%rsp), %rsp  // Restore User RSP from the saved frame instead of GS scratch
    
    cli
    swapgs
    sysretq
"#,
    options(att_syntax),
    user_ss = const crate::arch::x86_64::gdt::USER_DATA_SEL,
    user_cs = const crate::arch::x86_64::gdt::USER_CODE_SEL,
);
