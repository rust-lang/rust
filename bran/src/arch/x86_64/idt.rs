use core::mem::size_of;
use core::sync::atomic::{AtomicU64, Ordering};

pub const IRQ_TIMER_VECTOR: u8 = 0x20;
pub const IRQ_RESCHED_VECTOR: u8 = 0x30;
pub const IRQ_TLB_SHOOTDOWN_VECTOR: u8 = 0x41;
use kernel::kinfo;

static IRQ12_COUNT: AtomicU64 = AtomicU64::new(0);
static IRQ1_COUNT: AtomicU64 = AtomicU64::new(0);
static IRQ4_COUNT: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct IdtEntry {
    offset_low: u16,
    selector: u16,
    ist: u8,
    type_attr: u8,
    offset_middle: u16,
    offset_high: u32,
    reserved: u32,
}

impl IdtEntry {
    const fn missing() -> Self {
        Self {
            offset_low: 0,
            selector: 0,
            ist: 0,
            type_attr: 0,
            offset_middle: 0,
            offset_high: 0,
            reserved: 0,
        }
    }

    fn set_handler(&mut self, handler: u64, sel: u16, ist: u8, type_attr: u8) {
        self.offset_low = (handler & 0xFFFF) as u16;
        self.selector = sel;
        self.ist = ist;
        self.type_attr = type_attr;
        self.offset_middle = ((handler >> 16) & 0xFFFF) as u16;
        self.offset_high = (handler >> 32) as u32;
        self.reserved = 0;
    }
}

#[repr(C, align(16))]
pub struct Idt {
    entries: [IdtEntry; 256],
}

pub static mut IDT: Idt = Idt {
    entries: [IdtEntry::missing(); 256],
};

#[repr(C, packed)]
struct IdtDescriptor {
    size: u16,
    offset: u64,
}

unsafe extern "C" {
    fn breakpoint_handler_shim();
    fn double_fault_handler_shim();
    fn gp_handler_shim();
    fn pf_handler_shim();
    fn generic_handler_shim();
    fn irq_common_handler_shim();
    fn irq_timer_handler_shim();
    fn irq_resched_handler_shim();
    fn irq_tlb_shootdown_handler_shim();
    fn irq_keyboard_handler_shim();
    fn irq_mouse_handler_shim();
    fn irq_serial_handler_shim();
    fn invalid_opcode_handler_shim();
    fn div0_handler_shim();
}

core::arch::global_asm!(
    r#"
    .global breakpoint_handler_shim
    breakpoint_handler_shim:
        int3
        iretq

    .global double_fault_handler_shim
    double_fault_handler_shim:
        mov $0x3f8, %dx
        mov $0x44, %al
        out %al, %dx
        mov $0x46, %al
        out %al, %dx
        testb $3, 16(%rsp)
        jz 1f
        swapgs
    1:
        cli
        mov %rsp, %rdi
        call rust_double_fault_handler
    2:  hlt
        jmp 2b

    .global gp_handler_shim
    gp_handler_shim:
        mov $0x3f8, %dx
        mov $0x47, %al
        out %al, %dx
        testb $3, 16(%rsp)
        jz 1f
        swapgs
    1:
        cli
        mov %rsp, %rdi
        call rust_gp_handler
    2:  hlt
        jmp 2b

    .global invalid_opcode_handler_shim
    invalid_opcode_handler_shim:
        push $0
        testb $3, 16(%rsp)
        jz 1f
        swapgs
    1:
        cli
        mov %rsp, %rdi
        call rust_invalid_opcode_handler
    2:  hlt
        jmp 2b

    .global div0_handler_shim
    div0_handler_shim:
        push $0
        testb $3, 16(%rsp)
        jz 1f
        swapgs
    1:
        cli
        mov %rsp, %rdi
        call rust_div0_handler
    2:  hlt
        jmp 2b

    pf_handler_shim:
        // Check if coming from user mode (CS bit 0-1)
        testb $3, 16(%rsp)
        jz 1f
        swapgs
    1:
        // PF pushes error code, so stack has: ERR, RIP, CS, RFLAGS, RSP, SS
        push %r11
        push %r10
        push %r9
        push %r8
        push %rdi
        push %rsi
        push %rdx
        push %rcx
        push %rax

        lea 72(%rsp), %rdi // Pointer to InterruptStackFrame (skipping 9 registers)
        call rust_pf_handler

        pop %rax
        pop %rcx
        pop %rdx
        pop %rsi
        pop %rdi
        pop %r8
        pop %r9
        pop %r10
        pop %r11

        testb $3, 16(%rsp) // CS of frame
        jz 2f
        swapgs
    2:
        add $8, %rsp // Pop error code
        iretq

    .global generic_handler_shim
    generic_handler_shim:
        mov $0x3f8, %dx
        mov $0x3F, %al
        out %al, %dx
    2:  hlt
        jmp 2b

    irq_common_handler_shim:
        // IRQ stubs (generic) don't push error code. 
        // Stack has: RIP, CS, RFLAGS, RSP, SS
        testb $3, 8(%rsp)
        jz 1f
        swapgs
    1:
        push %rax
        push %rcx
        push %rdx
        push %rsi
        push %rdi
        push %r8
        push %r9
        push %r10
        push %r11

        mov $0, %rdi // Vector will be resolved via LAPIC ISR
        call rust_irq_handler

        pop %r11
        pop %r10
        pop %r9
        pop %r8
        pop %rdi
        pop %rsi
        pop %rdx
        pop %rcx
        pop %rax

        testb $3, 8(%rsp)
        jz 2f
        swapgs
    2:
        iretq

    irq_timer_handler_shim:
        testb $3, 8(%rsp)
        jz 1f
        swapgs
    1:
        push %rax
        push %rcx
        push %rdx
        push %rsi
        push %rdi
        push %r8
        push %r9
        push %r10
        push %r11

        mov $0x20, %rdi
        call rust_irq_handler

        pop %r11
        pop %r10
        pop %r9
        pop %r8
        pop %rdi
        pop %rsi
        pop %rdx
        pop %rcx
        pop %rax

        testb $3, 8(%rsp)
        jz 2f
        swapgs
    2:
        iretq

    .global irq_keyboard_handler_shim
    irq_keyboard_handler_shim:
        testb $3, 8(%rsp)
        jz 1f
        swapgs
    1:
        push %rax
        push %rcx
        push %rdx
        push %rsi
        push %rdi
        push %r8
        push %r9
        push %r10
        push %r11

        mov $0x21, %rdi
        call rust_irq_handler

        pop %r11
        pop %r10
        pop %r9
        pop %r8
        pop %rdi
        pop %rsi
        pop %rdx
        pop %rcx
        pop %rax

        testb $3, 8(%rsp)
        jz 2f
        swapgs
    2:
        iretq

    .global irq_resched_handler_shim
    irq_resched_handler_shim:
        testb $3, 8(%rsp)
        jz 1f
        swapgs
    1:
        push %rax
        push %rcx
        push %rdx
        push %rsi
        push %rdi
        push %r8
        push %r9
        push %r10
        push %r11

        mov $0x30, %rdi
        call rust_irq_handler

        pop %r11
        pop %r10
        pop %r9
        pop %r8
        pop %rdi
        pop %rsi
        pop %rdx
        pop %rcx
        pop %rax

        testb $3, 8(%rsp)
        jz 2f
        swapgs
    2:
        iretq
    .global irq_tlb_shootdown_handler_shim
    irq_tlb_shootdown_handler_shim:
        testb $3, 8(%rsp)
        jz 1f
        swapgs
    1:
        push %rax
        push %rcx
        push %rdx
        push %rsi
        push %rdi
        push %r8
        push %r9
        push %r10
        push %r11

        mov $0x41, %rdi
        call rust_irq_handler

        pop %r11
        pop %r10
        pop %r9
        pop %r8
        pop %rdi
        pop %rsi
        pop %rdx
        pop %rcx
        pop %rax

        testb $3, 8(%rsp)
        jz 2f
        swapgs
    2:
        iretq

    .global irq_mouse_handler_shim
    irq_mouse_handler_shim:
        testb $3, 8(%rsp)
        jz 1f
        swapgs
    1:
        push %rax
        push %rcx
        push %rdx
        push %rsi
        push %rdi
        push %r8
        push %r9
        push %r10
        push %r11

        mov $0x2C, %rdi
        call rust_irq_handler

        pop %r11
        pop %r10
        pop %r9
        pop %r8
        pop %rdi
        pop %rsi
        pop %rdx
        pop %rcx
        pop %rax

        testb $3, 8(%rsp)
        jz 2f
        swapgs
    2:
        iretq
    .global irq_serial_handler_shim
    irq_serial_handler_shim:
        testb $3, 8(%rsp)
        jz 1f
        swapgs
    1:
        push %rax
        push %rcx
        push %rdx
        push %rsi
        push %rdi
        push %r8
        push %r9
        push %r10
        push %r11

        mov $0x24, %rdi
        call rust_irq_handler

        pop %r11
        pop %r10
        pop %r9
        pop %r8
        pop %rdi
        pop %rsi
        pop %rdx
        pop %rcx
        pop %rax

        testb $3, 8(%rsp)
        jz 2f
        swapgs
    2:
        iretq
"#,
    options(att_syntax)
);

pub unsafe fn init() {
    let handler = generic_handler_shim as *const () as u64;
    unsafe {
        let base = core::ptr::addr_of_mut!(IDT.entries) as *mut IdtEntry;
        for i in 0..256 {
            (*base.add(i)).set_handler(handler, crate::arch::x86_64::gdt::KERNEL_CODE_SEL, 0, 0x8E);
        }

        IDT.entries[3].set_handler(
            breakpoint_handler_shim as *const () as u64,
            crate::arch::x86_64::gdt::KERNEL_CODE_SEL,
            0,
            0x8E,
        );
        IDT.entries[0].set_handler(
            div0_handler_shim as *const () as u64,
            crate::arch::x86_64::gdt::KERNEL_CODE_SEL,
            0,
            0x8E,
        );
        IDT.entries[6].set_handler(
            invalid_opcode_handler_shim as *const () as u64,
            crate::arch::x86_64::gdt::KERNEL_CODE_SEL,
            0,
            0x8E,
        );
        IDT.entries[8].set_handler(
            double_fault_handler_shim as *const () as u64,
            crate::arch::x86_64::gdt::KERNEL_CODE_SEL,
            1,
            0x8E,
        );
        IDT.entries[13].set_handler(
            gp_handler_shim as *const () as u64,
            crate::arch::x86_64::gdt::KERNEL_CODE_SEL,
            0,
            0x8E,
        );
        IDT.entries[14].set_handler(
            pf_handler_shim as *const () as u64,
            crate::arch::x86_64::gdt::KERNEL_CODE_SEL,
            0,
            0x8E,
        );

        // Hardware IRQs/MSI vectors
        for vector in 0x20..=0xEF {
            IDT.entries[vector as usize].set_handler(
                irq_common_handler_shim as *const () as u64,
                crate::arch::x86_64::gdt::KERNEL_CODE_SEL,
                0,
                0x8E,
            );
        }

        // Dedicated Timer Vector
        IDT.entries[IRQ_TIMER_VECTOR as usize].set_handler(
            irq_timer_handler_shim as *const () as u64,
            crate::arch::x86_64::gdt::KERNEL_CODE_SEL,
            0,
            0x8E,
        );

        // Dedicated Keyboard Vector (0x21) - Bypass Common Shim/ISR lookup
        IDT.entries[0x21].set_handler(
            irq_keyboard_handler_shim as *const () as u64,
            crate::arch::x86_64::gdt::KERNEL_CODE_SEL,
            0,
            0x8E,
        );

        // Dedicated Reschedule IPI Vector
        IDT.entries[IRQ_RESCHED_VECTOR as usize].set_handler(
            irq_resched_handler_shim as *const () as u64,
            crate::arch::x86_64::gdt::KERNEL_CODE_SEL,
            0,
            0x8E,
        );

        // Dedicated TLB Shootdown Vector
        IDT.entries[IRQ_TLB_SHOOTDOWN_VECTOR as usize].set_handler(
            irq_tlb_shootdown_handler_shim as *const () as u64,
            crate::arch::x86_64::gdt::KERNEL_CODE_SEL,
            0,
            0x8E,
        );

        // Dedicated Mouse Vector (0x2C) - Bypass Common Shim/ISR lookup
        IDT.entries[0x2C].set_handler(
            irq_mouse_handler_shim as *const () as u64,
            crate::arch::x86_64::gdt::KERNEL_CODE_SEL,
            0,
            0x8E,
        );

        // Dedicated Serial Vector (0x24) - Bypass Common Shim/ISR lookup
        IDT.entries[0x24].set_handler(
            irq_serial_handler_shim as *const () as u64,
            crate::arch::x86_64::gdt::KERNEL_CODE_SEL,
            0,
            0x8E,
        );

        let idtr = IdtDescriptor {
            size: (size_of::<Idt>() - 1) as u16,
            offset: core::ptr::addr_of!(IDT) as u64,
        };

        core::arch::asm!("lidt [{}]", in(reg) &idtr);
    }
}

/// Load the kernel IDT on secondary CPUs.
/// The IDT entries are already set up by the BSP.
pub unsafe fn load_on_secondary() {
    let idtr = IdtDescriptor {
        size: (size_of::<Idt>() - 1) as u16,
        offset: unsafe { core::ptr::addr_of!(IDT) } as u64,
    };

    unsafe {
        core::arch::asm!("lidt [{}]", in(reg) &idtr);
    }
}

#[repr(C)]
pub struct InterruptStackFrame {
    pub error_code: u64,
    pub rip: u64,
    pub cs: u64,
    pub rflags: u64,
    pub rsp: u64,
    pub ss: u64,
}

/// Hardware IRQ handler - dispatches to kernel and sends EOI
#[unsafe(no_mangle)]
pub extern "C" fn rust_irq_handler(vector: u64) {
    // Dedicated shims pass the exact vector. Generic shared stubs still fall back
    // to LAPIC ISR probing until they are split out into per-vector handlers.
    let resolved = if vector != 0 {
        vector as u8
    } else {
        crate::arch::x86_64::ioapic::lapic_in_service_vector().unwrap_or(0)
    };

    if resolved == 0 {
        return;
    }

    if resolved == 0x21 {
        let count = IRQ1_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        /*
        if count <= 3 || (count % 128 == 0) {
            kinfo!("IRQ1 fired (count={})", count);
        }
        */
    }

    if resolved == 0x2C {
        let count = IRQ12_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        /*
        if count <= 3 || (count % 128 == 0) {
            kinfo!("IRQ12 fired (count={})", count);
        }
        */
    }

    // Send EOI to Local APIC early to avoid wedging during context switch
    crate::arch::x86_64::ioapic::send_eoi();

    if (resolved >= 0x20 && resolved <= 0x2F) || (resolved >= 0xF0) {
        crate::arch::x86_64::pic::send_eoi(resolved);
    }

    if resolved == 0x24 {
        let count = IRQ4_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        /*
        if count <= 3 || (count % 128 == 0) {
            kinfo!("IRQ4 (serial) fired (count={})", count);
        }
        */
    }

    // IRQ_TIMER_VECTOR or IRQ_RESCHED_VECTOR is our preemption heartbeat
    if resolved == IRQ_TIMER_VECTOR {
        kernel::sched::on_tick::<crate::arch::CurrentRuntime>();
        // Boot display path disabled.
        // let now_ticks: u64;
        // unsafe {
        //     let low: u32;
        //     let high: u32;
        //     core::arch::asm!("rdtsc", out("eax") low, out("edx") high, options(nostack, nomem));
        //     now_ticks = ((high as u64) << 32) | (low as u64);
        // }
        // crate::theme::try_tick(now_ticks);
    } else if resolved == 0x24 {
        // Serial interrupt - poll into buffer
        // kernel::contract!("[IRQ] Serial interrupt 0x24 fired");
        crate::RUNTIME.arch.poll_serial();
    } else if resolved == IRQ_RESCHED_VECTOR {
        kernel::sched::on_resched_ipi::<crate::arch::CurrentRuntime>();
    } else if resolved == IRQ_TLB_SHOOTDOWN_VECTOR {
        // Full TLB flush on local CPU (including Global pages)
        crate::arch::x86_64::paging::tlb_flush_all();
    } else {
        kernel::irq::dispatch_irq(resolved);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_pf_handler(frame: &InterruptStackFrame) {
    let cr2: u64;
    unsafe {
        core::arch::asm!("mov {}, cr2", out(reg) cr2);
    }

    if frame.cs & 3 == 3 {
        unsafe {
            unsafe extern "C" {
                fn kernel_handle_page_fault(rip: u64, addr: u64, err: u64);
            }
            kernel_handle_page_fault(frame.rip, cr2, frame.error_code);
        }
        // If we handled it (e.g. stack growth), return to user mode
        return;
    }

    panic!(
        "PAGE FAULT at 0x{:x} RIP=0x{:x} CS=0x{:x} ERR=0x{:x}",
        cr2, frame.rip, frame.cs, frame.error_code
    );
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_gp_handler(frame: &InterruptStackFrame) -> ! {
    if frame.cs & 3 == 3 {
        unsafe {
            unsafe extern "C" {
                fn kernel_handle_exception(rip: u64, error_code: u64, rsp: u64, cs: u64, kind: u64);
            }
            kernel_handle_exception(frame.rip, frame.error_code, frame.rsp, frame.cs, 13);
            loop {
                core::arch::asm!("hlt");
            }
        }
    } else {
        panic!(
            "KERNEL GPF at RIP=0x{:x} CS=0x{:x} ERR=0x{:x} RSP=0x{:x}",
            frame.rip, frame.cs, frame.error_code, frame.rsp
        );
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_invalid_opcode_handler(frame: &InterruptStackFrame) -> ! {
    if frame.cs & 3 == 3 {
        // Diagnostic: Print the bytes at the faulting RIP
        unsafe {
            let rip = frame.rip as *const u8;
            let mut bytes = [0u8; 8];
            core::ptr::copy_nonoverlapping(rip, bytes.as_mut_ptr(), 8);
            kernel::kdebug!("USER-UD: rip=0x{:x} bytes={:02x?}", frame.rip, bytes);
        }

        unsafe {
            unsafe extern "C" {
                fn kernel_handle_exception(rip: u64, error_code: u64, rsp: u64, cs: u64, kind: u64);
            }
            kernel_handle_exception(frame.rip, frame.error_code, frame.rsp, frame.cs, 6);
            loop {
                core::arch::asm!("hlt");
            }
        }
    } else {
        panic!(
            "KERNEL INVALID OPCODE at RIP=0x{:x} CS=0x{:x} RSP=0x{:x}",
            frame.rip, frame.cs, frame.rsp
        );
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_div0_handler(frame: &InterruptStackFrame) -> ! {
    if frame.cs & 3 == 3 {
        unsafe {
            unsafe extern "C" {
                fn kernel_handle_exception(rip: u64, error_code: u64, rsp: u64, cs: u64, kind: u64);
            }
            kernel_handle_exception(frame.rip, frame.error_code, frame.rsp, frame.cs, 0);
            loop {
                core::arch::asm!("hlt");
            }
        }
    } else {
        panic!(
            "KERNEL DIVIDE BY ZERO at RIP=0x{:x} CS=0x{:x} RSP=0x{:x}",
            frame.rip, frame.cs, frame.rsp
        );
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_double_fault_handler(frame: &InterruptStackFrame) -> ! {
    panic!(
        "DOUBLE FAULT at RIP=0x{:x} CS=0x{:x} ERR=0x{:x} RSP=0x{:x}",
        frame.rip, frame.cs, frame.error_code, frame.rsp
    );
}
