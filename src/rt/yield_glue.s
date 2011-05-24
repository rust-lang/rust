/* More glue code, this time the 'bottom half' of yielding.
 *
 * We arrived here because an native call decided to deschedule the
 * running task. So the native call's return address got patched to the
 * first instruction of this glue code.
 *
 * When the native call does 'ret' it will come here, and its esp will be
 * pointing to the last argument pushed on the C stack before making
 * the native call: the 0th argument to the native call, which is always
 * the task ptr performing the native call. That's where we take over.
 *
 * Our goal is to complete the descheduling
 *
 *   - Switch over to the task stack temporarily.
 *
 *   - Save the task's callee-saves onto the task stack.
 *     (the task is now 'descheduled', safe to set aside)
 *
 *   - Switch *back* to the C stack.
 *
 *   - Restore the C-stack callee-saves.
 *
 *   - Return to the caller on the C stack that activated the task.
 *
 */

	.globl new_rust_yield_glue
	.balign 4
new_rust_yield_glue:
	movl  0(%esp), %ecx    # ecx = rust_task
	movl  16(%ecx), %esp
	pushl %ebp
	pushl %edi
	pushl %esi
	pushl %ebx
	movl  %esp, 16(%ecx)
	movl  12(%ecx), %esp
	popl  %ebx
	popl  %esi
	popl  %edi
	popl  %ebp
	ret
