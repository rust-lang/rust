use super::paging::X86_64AddressSpace;
use core::arch::{asm, global_asm};
use kernel::UserTaskSpec;

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct X86_64Context {
    pub sp: usize,
    pub kstack_top: u64,
}

unsafe extern "C" {
    pub fn context_switch(old: *mut usize, new: *const usize);
}

// Trampolines
unsafe extern "C" {
    fn kernel_trampoline();
    fn user_trampoline();
}

global_asm!(
    r#"
.section .text
.global context_switch
context_switch:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov [rdi], rsp
    mov rsp, [rsi]
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

.global kernel_trampoline
kernel_trampoline:
    mov rdi, r13
    call r12
    ud2

.global user_trampoline
user_trampoline:
    // r12 = user_entry, r13 = user_stack, r14 = aspace.0 (cr3), r15 = arg
    swapgs
    mov cr3, r14
    
    push {user_ss}  // User SS (udata selector)
    push r13   // User RSP
    push 0x202 // RFLAGS (IF=1)
    push {user_cs}  // User CS (ucode64 selector)
    push r12   // User RIP
    
    mov rdi, r15
    xor rax, rax
    xor rbx, rbx
    xor rcx, rcx
    xor rdx, rdx
    xor rsi, rsi
    xor rbp, rbp
    xor r8, r8
    xor r9, r9
    xor r10, r10
    xor r11, r11
    
    iretq
"#,
    user_ss = const super::gdt::USER_DATA_SEL,
    user_cs = const super::gdt::USER_CODE_SEL,
);

pub fn init_kernel_context(
    entry: extern "C" fn(arg: usize) -> !,
    kstack_top: u64,
    arg: usize,
) -> X86_64Context {
    let mut sp = kstack_top;
    let mut push = |val: u64| {
        sp -= 8;
        unsafe { (sp as *mut u64).write(val) };
    };

    kernel::kdebug!(
        "INIT KERNEL CTX: entry={:p} kstack_top={:#x} arg={:#x}",
        entry,
        kstack_top,
        arg
    );

    push(kernel_trampoline as *const () as usize as u64);
    push(0); // rbx
    push(0); // rbp
    push(entry as usize as u64); // r12
    push(arg as u64); // r13
    push(0); // r14
    push(0); // r15

    X86_64Context {
        sp: sp as usize,
        kstack_top,
    }
}

pub fn init_user_context(spec: UserTaskSpec<X86_64AddressSpace>, kstack_top: u64) -> X86_64Context {
    let mut sp = kstack_top;
    let mut push = |val: u64| {
        sp -= 8;
        unsafe { (sp as *mut u64).write(val) };
    };

    push(user_trampoline as *const () as usize as u64);
    push(0); // rbx
    push(0); // rbp
    push(spec.entry); // r12
    push(spec.stack_top); // r13
    push(spec.aspace.0); // r14
    push(spec.arg as u64); // r15

    X86_64Context {
        sp: sp as usize,
        kstack_top,
    }
}

pub unsafe fn switch(from: &mut X86_64Context, to: &X86_64Context, to_tid: u64) {
    unsafe {
        // Update the kernel stack in GS via scratch register
        // We assume GS base is already pointing to CpuLocal
        // Offset 8 is kernel_rsp.
        // But wait, we need to know if GS is active.
        // Assuming we set up GS in mod.rs init().

        let kstack = to.kstack_top;
        // Write to GS:8 (assuming CpuLocal layout: user_rsp: u64, kernel_rsp: u64)
        // We do this BEFORE switching, because we are in kernel mode.
        // The NEXT time we enter from user mode (syscall), we want this stack.
        asm!("mov gs:[8], {}", in(reg) kstack);

        // Update current_tid in GS:24
        asm!("mov gs:[24], {}", in(reg) to_tid);

        // Update TSS RSP0 for interrupts/exceptions from Ring 3
        crate::arch::x86_64::gdt::set_rsp0(kstack);

        context_switch(&mut from.sp, &to.sp as *const usize);
    }
}
