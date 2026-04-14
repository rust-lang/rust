use core::arch::global_asm;
use core::sync::atomic::AtomicU64;

// Trampoline symbols
unsafe extern "C" {
    pub static trampoline_start: u8;
    pub static trampoline_end: u8;
}

// Data area in the trampoline (relative to start)
// At offset 0x500 from trampoline start:
// 0x500 + 0x00: flag (atomic, AP sets to 1 when done)
// 0x500 + 0x08: cr3
// 0x500 + 0x10: stack_top
// 0x500 + 0x18: entry_point
// 0x500 + 0x20: cpu_index
// 0x500 + 0x28: hhdm

const TRAMPOLINE_DATA_OFFSET: usize = 0x500; // Data at offset 0x500
const TRAMPOLINE_SIZE: usize = 4096;

global_asm!(
    r#"
    .section .trampoline, "ax", @progbits
    .intel_syntax noprefix
    .code16
    .global trampoline_start
    .global trampoline_end

trampoline_start:
    cli
    cld
    
    // Setup segments for Real Mode
    mov ax, cs
    mov ds, ax
    xor sp, sp
    
    // GDT=0x8100, GDTR=0x8148
    mov word ptr [0x148], 0x47          // Limit
    mov dword ptr [0x14a], 0x8100       // Base
    
    lgdt [0x148]
    
    // Enable Protected Mode
    mov eax, cr0
    or eax, 1
    mov cr0, eax

    // Far Jump to Protected Mode (0x8150)
    .byte 0x66, 0xea
    .long 0x8150
    .word 0x38

    // Padding to 0x100 for GDT
    .skip 0x100 - (. - trampoline_start)
gdt:
    .quad 0x0000000000000000 // 0x00: Null
    .quad 0x00a09a0000000000 // 0x08: Kernel Code 64
    .quad 0x00c0920000000000 // 0x10: Kernel Data
    .quad 0x0000000000000000 // 0x18: Unused
    .quad 0x0000000000000000 // 0x20: Unused
    .quad 0x0000000000000000 // 0x28: Unused
    .quad 0x0000000000000000 // 0x30: Unused
    .quad 0x00cf9a000000ffff // 0x38: Temp Code 32
    .quad 0x00cf92000000ffff // 0x40: Temp Data 32

    // Padding to 0x148 for GDTR
    .skip 0x148 - (. - trampoline_start)
gdtr:
    .word 0x47
    .long 0x8100

    // Padding to 0x150 for Protected Mode
    .skip 0x150 - (. - trampoline_start)
    .code32
protected_mode:
    mov ax, 0x40   // Temp Data 32
    mov ds, ax
    mov es, ax
    mov ss, ax
    xor esp, esp
    
    // Enable PAE
    mov eax, cr4
    or eax, 1 << 5
    mov cr4, eax

    // Load CR3 from physical 0x8508
    mov eax, [0x8508]
    mov cr3, eax

    // Enable Long Mode and NX in EFER
    mov ecx, 0xC0000080
    rdmsr
    or eax, (1 << 8) | (1 << 11) // LME | NXE
    wrmsr

    // Enable Paging
    mov eax, cr0
    or eax, 0x80000000
    mov cr0, eax

    // Far Jump to 64-bit mode (0x8200)
    .byte 0xea
    .long 0x8200
    .word 0x08

    // Padding to 0x200 for Long Mode
    .skip 0x200 - (. - trampoline_start)
    .code64
long_mode:
    mov ax, 0x10   // Kernel Data
    mov ds, ax
    mov es, ax
    mov ss, ax

    // Load kernel parameters from data area
    mov rbx, [0x8518]   // Entry

    // Signals to BSP
    mov rax, 1
    mov [0x8500], rax

    // Load kernel parameters from data area
    mov rsp, [0x8510]   // Stack
    mov rdi, [0x8520]   // CPU Index
    mov rsi, [0x8528]   // HHDM

    // 1. Load kernel GDT (at offset 0x30 from data area)
    lgdt [0x8530]

    // 2. Load kernel IDT (at offset 0x40 from data area)
    lidt [0x8540]

    // 3. Reload segments for the new GDT
    push 0x08   // KERNEL_CODE_SEL
    lea rax, [rip + 1f]
    push rax
    retfq
1:
    mov ax, 0x10 // KERNEL_DATA_SEL
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    // rbx already has entry point

    call rbx

halt_loop:
    hlt
    jmp halt_loop

    // Final padding to ensure section size
    .skip 0x600 - (. - trampoline_start)
trampoline_end:
    "#
);

#[repr(C)]
pub struct TrampolineData {
    pub flag: AtomicU64,    // 0x00
    pub cr3: u64,           // 0x08
    pub stack_top: u64,     // 0x10
    pub entry_point: u64,   // 0x18
    pub cpu_index: u64,     // 0x20
    pub hhdm: u64,          // 0x28
    pub gdt_desc: [u8; 16], // 0x30 (10 bytes used)
    pub idt_desc: [u8; 16], // 0x40 (10 bytes used)
}
