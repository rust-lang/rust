#include "macros.S"

// Mark stack as non-executable
#if defined(__linux__) && defined(__ELF__)
	.section	.note.GNU-stack, "", %progbits
#endif

func	rust_swap_registers
	str x0, [x0, #0]
	stp x2, x3, [x0, #16]
	stp x4, x5, [x0, #32]
	stp x6, x7, [x0, #48]
	stp x8, x9, [x0, #64]
	stp x10, x11, [x0, #80]
	stp x12, x13, [x0, #96]
	stp x14, x15, [x0, #112]
	stp x16, x17, [x0, #128]
	stp x18, x19, [x0, #144]
	stp x20, x21, [x0, #160]
	stp x22, x23, [x0, #176]
	stp x24, x25, [x0, #192]
	stp x26, x27, [x0, #208]
	stp x28, x29, [x0, #224]
	str x30, [x0, #240]
	
	mov x2, sp
	str x2, [x0, #248]
	
	stp q0, q1, [x0, #256]
	stp q2, q3, [x0, #288]
	stp q4, q5, [x0, #320]
	stp q6, q7, [x0, #352]
	stp q8, q9, [x0, #384]
	stp q10, q11, [x0, #416]
	stp q12, q13, [x0, #448]
	stp q14, q15, [x0, #480]
	stp q16, q17, [x0, #512]
	stp q18, q19, [x0, #544]
	stp q20, q21, [x0, #576]
	stp q22, q23, [x0, #608]
	stp q24, q25, [x0, #640]
	stp q26, q27, [x0, #672]
	stp q28, q29, [x0, #704]
	stp q30, q31, [x0, #736]
	
	mrs x2, fpcr
	str x2, [x0, #768]
	mrs x2, fpsr
	str x2, [x0, #776]
	mrs x2, nzcv
	str x2, [x0, #784]

	ldr x0, [x1, #0]
	ldp x2, x3, [x1, #16]
	ldp x4, x5, [x1, #32]
	ldp x6, x7, [x1, #48]
	ldp x8, x9, [x1, #64]
	ldp x10, x11, [x1, #80]
	ldp x12, x13, [x1, #96]
	ldp x14, x15, [x1, #112]
	ldp x16, x17, [x1, #128]
	ldp x18, x19, [x1, #144]
	ldp x20, x21, [x1, #160]
	ldp x22, x23, [x1, #176]
	ldp x24, x25, [x1, #192]
	ldp x26, x27, [x1, #208]
	ldp x28, x29, [x1, #224]
	ldr x30, [x1, #240]
	
	ldr x2, [x1, #248]
	mov sp, x2
	
	ldp q0, q1, [x1, #256]
	ldp q2, q3, [x1, #288]
	ldp q4, q5, [x1, #320]
	ldp q6, q7, [x1, #352]
	ldp q8, q9, [x1, #384]
	ldp q10, q11, [x1, #416]
	ldp q12, q13, [x1, #448]
	ldp q14, q15, [x1, #480]
	ldp q16, q17, [x1, #512]
	ldp q18, q19, [x1, #544]
	ldp q20, q21, [x1, #576]
	ldp q22, q23, [x1, #608]
	ldp q24, q25, [x1, #640]
	ldp q26, q27, [x1, #672]
	ldp q28, q29, [x1, #704]
	ldp q30, q31, [x1, #736]
	
	ldr x2, [x1, #768]
	msr fpcr, x2
	ldr x2, [x1, #776]
	msr fpsr, x2
	ldr x2, [x1, #784]
	msr nzcv, x2

	ret
endfunc	rust_swap_registers

// For reasons of this existence, see the comments in x86_64/_context.S
func	rust_bootstrap_green_task
	mov x0, x0
	mov x1, x3
	mov x2, x4
	br x5
endfunc	rust_bootstrap_green_task
