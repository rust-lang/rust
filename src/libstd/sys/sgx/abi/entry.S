/*  This symbol is used at runtime to figure out the virtual address that the */
/*  enclave is loaded at. */
.section absolute
.global IMAGE_BASE
IMAGE_BASE:

.section ".note.x86_64-fortanix-unknown-sgx", "", @note
    .align 4
    .long 1f - 0f              /* name length (not including padding) */
    .long 3f - 2f              /* desc length (not including padding) */
    .long 1                    /* type = NT_VERSION */
0:  .asciz "toolchain-version" /* name */
1:  .align 4
2:  .long 0                    /* desc - toolchain version number, 32-bit LE */
3:  .align 4

.section .rodata
/*  The XSAVE area needs to be a large chunk of readable memory, but since we are */
/*  going to restore everything to its initial state (XSTATE_BV=0), only certain */
/*  parts need to have a defined value. In particular: */
/*  */
/*    * MXCSR in the legacy area. This register is always restored if RFBM[1] or */
/*      RFBM[2] is set, regardless of the value of XSTATE_BV */
/*    * XSAVE header */
.align 64
.Lxsave_clear:
.org .+24
.Lxsave_mxcsr:
    .int 0

/*  We can store a bunch of data in the gap between MXCSR and the XSAVE header */

/*  The following symbols point at read-only data that will be filled in by the */
/*  post-linker. */

/*  When using this macro, don't forget to adjust the linker version script! */
.macro globvar name:req size:req
    .global \name
    .protected \name
    .align \size
    .size \name , \size
    \name :
        .org .+\size
.endm
    /*  The base address (relative to enclave start) of the heap area */
    globvar HEAP_BASE 8
    /*  The heap size in bytes */
    globvar HEAP_SIZE 8
    /*  Value of the RELA entry in the dynamic table */
    globvar RELA 8
    /*  Value of the RELACOUNT entry in the dynamic table */
    globvar RELACOUNT 8
    /*  The enclave size in bytes */
    globvar ENCLAVE_SIZE 8
    /*  The base address (relative to enclave start) of the enclave configuration area */
    globvar CFGDATA_BASE 8
    /*  Non-zero if debugging is enabled, zero otherwise */
    globvar DEBUG 1
    /*  The base address (relative to enclave start) of the enclave text section */
    globvar TEXT_BASE 8
    /*  The size in bytes of enclacve text section */
    globvar TEXT_SIZE 8
    /*  The base address (relative to enclave start) of the enclave EH_FRM_HDR section */
    globvar EH_FRM_HDR_BASE 8
    /*  The size in bytes of enclacve EH_FRM_HDR section */
    globvar EH_FRM_HDR_SIZE 8

.org .Lxsave_clear+512
.Lxsave_header:
    .int 0, 0 /*  XSTATE_BV */
    .int 0, 0 /*  XCOMP_BV */
    .org .+48 /*  reserved bits */

.data
.Laborted:
    .byte 0

/*  TCS local storage section */
.equ tcsls_tos,                 0x00 /*  initialized by loader to *offset* from image base to TOS */
.equ tcsls_flags,               0x08 /*  initialized by loader */
.equ tcsls_flag_secondary,      0    /*  initialized by loader; 0 = standard TCS, 1 = secondary TCS */
.equ tcsls_flag_init_once,      1    /*  initialized by loader to 0 */
/*  14 unused bits */
.equ tcsls_user_fcw,            0x0a
.equ tcsls_user_mxcsr,          0x0c
.equ tcsls_last_rsp,            0x10 /*  initialized by loader to 0 */
.equ tcsls_panic_last_rsp,      0x18 /*  initialized by loader to 0 */
.equ tcsls_debug_panic_buf_ptr, 0x20 /*  initialized by loader to 0 */
.equ tcsls_user_rsp,            0x28
.equ tcsls_user_retip,          0x30
.equ tcsls_user_rbp,            0x38
.equ tcsls_user_r12,            0x40
.equ tcsls_user_r13,            0x48
.equ tcsls_user_r14,            0x50
.equ tcsls_user_r15,            0x58
.equ tcsls_tls_ptr,             0x60
.equ tcsls_tcs_addr,            0x68

.macro load_tcsls_flag_secondary_bool reg:req comments:vararg
    .ifne tcsls_flag_secondary /* to convert to a bool, must be the first bit */
    .abort
    .endif
        mov $(1<<tcsls_flag_secondary),%e\reg
        and %gs:tcsls_flags,%\reg
.endm

.text
.global sgx_entry
.type sgx_entry,function
sgx_entry:
/*  save user registers */
    mov %rcx,%gs:tcsls_user_retip
    mov %rsp,%gs:tcsls_user_rsp
    mov %rbp,%gs:tcsls_user_rbp
    mov %r12,%gs:tcsls_user_r12
    mov %r13,%gs:tcsls_user_r13
    mov %r14,%gs:tcsls_user_r14
    mov %r15,%gs:tcsls_user_r15
    mov %rbx,%gs:tcsls_tcs_addr
    stmxcsr %gs:tcsls_user_mxcsr
    fnstcw %gs:tcsls_user_fcw
/*  reset user state */
    cld /* x86-64 ABI requires DF to be unset at function entry/exit */
/*  check for debug buffer pointer */
    testb  $0xff,DEBUG(%rip)
    jz .Lskip_debug_init
    mov %r10,%gs:tcsls_debug_panic_buf_ptr
.Lskip_debug_init:
/*  check for abort */
    bt $0,.Laborted(%rip)
    jc .Lreentry_panic
/*  check if returning from usercall */
    mov %gs:tcsls_last_rsp,%r11
    test %r11,%r11
    jnz .Lusercall_ret
/*  setup stack */
    mov %gs:tcsls_tos,%rsp /*  initially, RSP is not set to the correct value */
                           /*  here. This is fixed below under "adjust stack". */
/*  check for thread init */
    bts $tcsls_flag_init_once,%gs:tcsls_flags
    jc .Lskip_init
/*  adjust stack */
    lea IMAGE_BASE(%rip),%rax
    add %rax,%rsp
    mov %rsp,%gs:tcsls_tos
/*  call tcs_init */
/*  store caller-saved registers in callee-saved registers */
    mov %rdi,%rbx
    mov %rsi,%r12
    mov %rdx,%r13
    mov %r8,%r14
    mov %r9,%r15
    load_tcsls_flag_secondary_bool di /* RDI = tcs_init() argument: secondary: bool */
    call tcs_init
/*  reload caller-saved registers */
    mov %rbx,%rdi
    mov %r12,%rsi
    mov %r13,%rdx
    mov %r14,%r8
    mov %r15,%r9
.Lskip_init:
/*  call into main entry point */
    load_tcsls_flag_secondary_bool cx /* RCX = entry() argument: secondary: bool */
    call entry /* RDI, RSI, RDX, R8, R9 passed in from userspace */
    mov %rax,%rsi  /* RSI = return value */
    /* NOP: mov %rdx,%rdx */ /*  RDX = return value */
    xor %rdi,%rdi  /* RDI = normal exit */
.Lexit:
/*  clear general purpose register state */
    /*  RAX overwritten by ENCLU */
    /*  RBX set later */
    /*  RCX overwritten by ENCLU */
    /*  RDX contains return value */
    /*  RSP set later */
    /*  RBP set later */
    /*  RDI contains exit mode */
    /*  RSI contains return value */
    xor %r8,%r8
    xor %r9,%r9
    xor %r10,%r10
    xor %r11,%r11
    /*  R12 ~ R15 set by sgx_exit */
.Lsgx_exit:
/*  clear extended register state */
    mov %rdx, %rcx /*  save RDX */
    mov $-1, %rax
    mov %rax, %rdx
    xrstor .Lxsave_clear(%rip)
    mov %rcx, %rdx /*  restore RDX */
/*  clear flags */
    pushq $0
    popfq
/*  restore user registers */
    mov %gs:tcsls_user_r12,%r12
    mov %gs:tcsls_user_r13,%r13
    mov %gs:tcsls_user_r14,%r14
    mov %gs:tcsls_user_r15,%r15
    mov %gs:tcsls_user_retip,%rbx
    mov %gs:tcsls_user_rsp,%rsp
    mov %gs:tcsls_user_rbp,%rbp
    fldcw %gs:tcsls_user_fcw
    ldmxcsr %gs:tcsls_user_mxcsr
/*  exit enclave */
    mov $0x4,%eax /*  EEXIT */
    enclu
/*  end sgx_entry */

.Lreentry_panic:
    orq $8,%rsp
    jmp abort_reentry

/*  This *MUST* be called with 6 parameters, otherwise register information */
/*  might leak! */
.global usercall
usercall:
    test %rcx,%rcx            /* check `abort` function argument */
    jnz .Lusercall_abort      /* abort is set, jump to abort code (unlikely forward conditional) */
    jmp .Lusercall_save_state /* non-aborting usercall */
.Lusercall_abort:
/* set aborted bit */
    movb $1,.Laborted(%rip)
/* save registers in DEBUG mode, so that debugger can reconstruct the stack */
    testb $0xff,DEBUG(%rip)
    jz .Lusercall_noreturn
.Lusercall_save_state:
/*  save callee-saved state */
    push %r15
    push %r14
    push %r13
    push %r12
    push %rbp
    push %rbx
    sub $8, %rsp
    fstcw 4(%rsp)
    stmxcsr (%rsp)
    movq %rsp,%gs:tcsls_last_rsp
.Lusercall_noreturn:
/*  clear general purpose register state */
    /*  RAX overwritten by ENCLU */
    /*  RBX set by sgx_exit */
    /*  RCX overwritten by ENCLU */
    /*  RDX contains parameter */
    /*  RSP set by sgx_exit */
    /*  RBP set by sgx_exit */
    /*  RDI contains parameter */
    /*  RSI contains parameter */
    /*  R8 contains parameter */
    /*  R9 contains parameter */
    xor %r10,%r10
    xor %r11,%r11
    /*  R12 ~ R15 set by sgx_exit */
/*  extended registers/flags cleared by sgx_exit */
/*  exit */
    jmp .Lsgx_exit
.Lusercall_ret:
    movq $0,%gs:tcsls_last_rsp
/*  restore callee-saved state, cf. "save" above */
    mov %r11,%rsp
    ldmxcsr (%rsp)
    fldcw 4(%rsp)
    add $8, %rsp
    pop %rbx
    pop %rbp
    pop %r12
    pop %r13
    pop %r14
    pop %r15
/*  return */
    mov %rsi,%rax /*  RAX = return value */
    /* NOP: mov %rdx,%rdx */ /*  RDX = return value */
    ret

/*
The following functions need to be defined externally:
```
// Called by entry code on re-entry after exit
extern "C" fn abort_reentry() -> !;

// Called once when a TCS is first entered
extern "C" fn tcs_init(secondary: bool);

// Standard TCS entrypoint
extern "C" fn entry(p1: u64, p2: u64, p3: u64, secondary: bool, p4: u64, p5: u64) -> (u64, u64);
```
*/

.global get_tcs_addr
get_tcs_addr:
    mov %gs:tcsls_tcs_addr,%rax
    ret

.global get_tls_ptr
get_tls_ptr:
    mov %gs:tcsls_tls_ptr,%rax
    ret

.global set_tls_ptr
set_tls_ptr:
    mov %rdi,%gs:tcsls_tls_ptr
    ret

.global take_debug_panic_buf_ptr
take_debug_panic_buf_ptr:
    xor %rax,%rax
    xchg %gs:tcsls_debug_panic_buf_ptr,%rax
    ret
