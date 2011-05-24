/*
 * This is a bit of glue-code.
 *
 *   - save regs on C stack
 *   - save sp to task.runtime_sp (runtime_sp is thus always aligned)
 *   - load saved task sp (switch stack)
 *   - restore saved task regs
 *   - return to saved task pc
 *
 * Our incoming stack looks like this:
 *
 *   *esp+4        = [arg1   ] = task ptr
 *   *esp          = [retpc  ]
 */

	.globl new_rust_activate_glue
	.balign 4
new_rust_activate_glue:
	movl  4(%esp), %ecx    # ecx = rust_task
	pushl %ebp
	pushl %edi
	pushl %esi
	pushl %ebx
	movl  %esp, 12(%ecx)
	movl  16(%ecx), %esp

        /*
         * There are two paths we can arrive at this code from:
         *
         *
         *   1. We are activating a task for the first time. When we switch
         *      into the task stack and 'ret' to its first instruction, we'll
         *      start doing whatever the first instruction says. Probably
         *      saving registers and starting to establish a frame. Harmless
         *      stuff, doesn't look at task->rust_sp again except when it
         *      clobbers it during a later native call.
         *
         *
         *   2. We are resuming a task that was descheduled by the yield glue
         *      below.  When we switch into the task stack and 'ret', we'll be
         *      ret'ing to a very particular instruction:
         *
         *              "esp <- task->rust_sp"
         *
         *      this is the first instruction we 'ret' to after this glue,
         *      because it is the first instruction following *any* native
         *      call, and the task we are activating was descheduled
         *      mid-native-call.
         *
         *      Unfortunately for us, we have already restored esp from
         *      task->rust_sp and are about to eat the 5 words off the top of
         *      it.
         *
         *
         *      | ...    | <-- where esp will be once we restore + ret, below,
         *      | retpc  |     and where we'd *like* task->rust_sp to wind up.
         *      | ebp    |
         *      | edi    |
         *      | esi    |
         *      | ebx    | <-- current task->rust_sp == current esp
         *
         *
         *      This is a problem. If we return to "esp <- task->rust_sp" it
         *      will push esp back down by 5 words. This manifests as a rust
         *      stack that grows by 5 words on each yield/reactivate. Not
         *      good.
         *
         *      So what we do here is just adjust task->rust_sp up 5 words as
         *      well, to mirror the movement in esp we're about to
         *      perform. That way the "esp <- task->rust_sp" we 'ret' to below
         *      will be a no-op. Esp won't move, and the task's stack won't
         *      grow.
         */
	addl  $20, 16(%ecx)

        /*
         * In most cases, the function we're returning to (activating)
         * will have saved any caller-saves before it yielded via native call,
         * so no work to do here. With one exception: when we're initially
         * activating, the task needs to be in the fastcall 2nd parameter
         * expected by the rust main function. That's edx.
         */
        mov  %ecx, %edx

	popl  %ebx
	popl  %esi
	popl  %edi
	popl  %ebp
	ret
