declare fastcc i32 @"\01rust_native_rust_local_copy"(i32, i32)

module asm "\09.globl rust_native_rust_local_copy"
module asm "\09.balign 4"
module asm "rust_native_rust_local_copy:"
module asm "\09.cfi_startproc"
module asm "\09pushl %ebp"
module asm "\09.cfi_def_cfa_offset 8"
module asm "\09.cfi_offset %ebp, -8"
module asm "\09pushl %edi"
module asm "\09.cfi_def_cfa_offset 12"
module asm "\09pushl %esi"
module asm "\09.cfi_def_cfa_offset 16"
module asm "\09pushl %ebx"
module asm "\09.cfi_def_cfa_offset 20"
module asm "\09movl  %esp, %ebp     # ebp = rust_sp"
module asm "\09.cfi_def_cfa_register %ebp"
module asm "\09movl  %esp, 16(%edx)"
module asm "\09movl  12(%edx), %esp"
module asm "\09subl  $4, %esp   # esp -= args"
module asm "\09andl  $~0xf, %esp    # align esp down"
module asm "\09movl  %edx, (%esp)"
module asm "\09movl  %edx, %edi     # save task from edx to edi"
module asm "\09call  *%ecx          # call *%ecx"
module asm "\09movl  %edi, %edx     # restore edi-saved task to edx"
module asm "\09movl  16(%edx), %esp"
module asm "\09popl  %ebx"
module asm "\09popl  %esi"
module asm "\09popl  %edi"
module asm "\09popl  %ebp"
module asm "\09ret"
module asm "\09.cfi_endproc"


declare i32 @upcall_exit(i32)

define void @rust_new_exit_task_glue(i32, i32, i32, i32, i32) {
entry:
  %5 = inttoptr i32 %0 to void (i32, i32, i32, i32)*
  tail call fastcc void %5(i32 %1, i32 %2, i32 %3, i32 %4)
  %6 = tail call fastcc i32 @"\01rust_native_rust_local_copy"(i32 ptrtoint (i32 (i32)* @upcall_exit to i32), i32 %2)
  ret void
}
