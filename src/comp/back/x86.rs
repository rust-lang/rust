import lib.llvm.llvm;
import lib.llvm.llvm.ModuleRef;
import std._str;
import std._vec;
import std.os.target_os;
import util.common.istr;

const int wordsz = 4;

fn wstr(int i) -> str {
    ret istr(i * wordsz);
}

fn save_callee_saves() -> vec[str] {
    ret vec("pushl %ebp",
            "pushl %edi",
            "pushl %esi",
            "pushl %ebx");
}

fn restore_callee_saves() -> vec[str] {
    ret vec("popl  %ebx",
            "popl  %esi",
            "popl  %edi",
            "popl  %ebp");
}

fn load_esp_from_rust_sp() -> vec[str] {
    ret vec("movl  " + wstr(abi.task_field_rust_sp) + "(%ecx), %esp");
}

fn load_esp_from_runtime_sp() -> vec[str] {
    ret vec("movl  " + wstr(abi.task_field_runtime_sp) + "(%ecx), %esp");
}

fn store_esp_to_rust_sp() -> vec[str] {
    ret vec("movl  %esp, " + wstr(abi.task_field_rust_sp) + "(%ecx)");
}

fn store_esp_to_runtime_sp() -> vec[str] {
    ret vec("movl  %esp, " + wstr(abi.task_field_runtime_sp) + "(%ecx)");
}

/*
 * This is a bit of glue-code. It should be emitted once per
 * compilation unit.
 *
 *   - save regs on C stack
 *   - align sp on a 16-byte boundary
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

fn rust_activate_glue() -> vec[str] {
    ret vec("movl  4(%esp), %ecx    # ecx = rust_task")
        + save_callee_saves()
        + store_esp_to_runtime_sp()
        + load_esp_from_rust_sp()

        /*
         * There are two paths we can arrive at this code from:
         *
         *
         *   1. We are activating a task for the first time. When we switch
         *      into the task stack and 'ret' to its first instruction, we'll
         *      start doing whatever the first instruction says. Probably
         *      saving registers and starting to establish a frame. Harmless
         *      stuff, doesn't look at task->rust_sp again except when it
         *      clobbers it during a later upcall.
         *
         *
         *   2. We are resuming a task that was descheduled by the yield glue
         *      below.  When we switch into the task stack and 'ret', we'll be
         *      ret'ing to a very particular instruction:
         *
         *              "esp <- task->rust_sp"
         *
         *      this is the first instruction we 'ret' to after this glue,
         *      because it is the first instruction following *any* upcall,
         *      and the task we are activating was descheduled mid-upcall.
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
        + vec("addl  $20, " + wstr(abi.task_field_rust_sp) + "(%ecx)")


        /*
         * In most cases, the function we're returning to (activating)
         * will have saved any caller-saves before it yielded via upcalling,
         * so no work to do here. With one exception: when we're initially
         * activating, the task needs to be in the fastcall 2nd parameter
         * expected by the rust main function. That's edx.
         */
        + vec("mov  %ecx, %edx")

        + restore_callee_saves()
        + vec("ret");
}

/* More glue code, this time the 'bottom half' of yielding.
 *
 * We arrived here because an upcall decided to deschedule the
 * running task. So the upcall's return address got patched to the
 * first instruction of this glue code.
 *
 * When the upcall does 'ret' it will come here, and its esp will be
 * pointing to the last argument pushed on the C stack before making
 * the upcall: the 0th argument to the upcall, which is always the
 * task ptr performing the upcall. That's where we take over.
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

fn rust_yield_glue() -> vec[str] {
    ret vec("movl  0(%esp), %ecx    # ecx = rust_task")
        + load_esp_from_rust_sp()
        + save_callee_saves()
        + store_esp_to_rust_sp()
        + load_esp_from_runtime_sp()
        + restore_callee_saves()
        + vec("ret");
}

fn upcall_glue(int n_args) -> vec[str] {

    /*
     * 0, 4, 8, 12 are callee-saves
     * 16 is retpc
     * 20 .. (5+i) * 4 are args
     *
     * ecx is taskptr
     * edx is callee
     *
     */

    fn copy_arg(uint i) -> str {
        auto src_off = wstr(5 + (i as int));
        auto dst_off = wstr(1 + (i as int));
        auto m = vec("movl  " + src_off + "(%ebp),%eax",
                     "movl  %eax," + dst_off + "(%esp)");
        ret _str.connect(m, "\n\t");
    }

    auto carg = copy_arg;

    ret
        save_callee_saves()

        + vec("movl  %esp, %ebp     # ebp = rust_sp")

        + store_esp_to_rust_sp()
        + load_esp_from_runtime_sp()

        + vec("subl  $" + wstr(n_args + 1) + ", %esp   # esp -= args",
              "andl  $~0xf, %esp    # align esp down",
              "movl  %ecx, (%esp)   # arg[0] = rust_task ")

        + _vec.init_fn[str](carg, n_args as uint)

        +  vec("movl  %ecx, %edi     # save task from ecx to edi",
               "call  *%edx          # call *%edx",
               "movl  %edi, %ecx     # restore edi-saved task to ecx")

        + load_esp_from_rust_sp()
        + restore_callee_saves()
        + vec("ret");

}


fn decl_glue(int align, str prefix, str name, vec[str] insns) -> str {
    auto sym = prefix + name;
    ret "\t.globl " + sym + "\n" +
        "\t.balign " + istr(align) + "\n" +
        sym + ":\n" +
        "\t" + _str.connect(insns, "\n\t");
}


fn decl_upcall_glue(int align, str prefix, uint n) -> str {
    let int i = n as int;
    ret decl_glue(align, prefix,
                  abi.upcall_glue_name(i),
                  upcall_glue(i));
}

fn get_symbol_prefix() -> str {
    if (_str.eq(target_os(), "macos") ||
        _str.eq(target_os(), "win32")) {
        ret "_";
    } else {
        ret "";
    }
}

fn get_module_asm() -> str {
    auto align = 4;

    auto prefix = get_symbol_prefix();

    auto glues =
        vec(decl_glue(align, prefix,
                      abi.activate_glue_name(),
                      rust_activate_glue()),

            decl_glue(align, prefix,
                      abi.yield_glue_name(),
                      rust_yield_glue()))

        + _vec.init_fn[str](bind decl_upcall_glue(align, prefix, _),
                            abi.n_upcall_glues as uint);

    ret _str.connect(glues, "\n\n");
}

fn get_data_layout() -> str {
    if (_str.eq(target_os(), "macos")) {
      ret "e-p:32:32-f64:32:64-i64:32:64-f80:128:128-n8:16:32";
    }
    if (_str.eq(target_os(), "win32")) {
      ret "e-p:32:32-f64:64:64-i64:64:64-f80:32:32-n8:16:32";
    }
    ret "e-p:32:32-f64:32:64-i64:32:64-f80:32:32-n8:16:32";
}

fn get_target_triple() -> str {
    if (_str.eq(target_os(), "macos")) {
        ret "i686-apple-darwin";
    }
    if (_str.eq(target_os(), "win32")) {
        ret "i686-pc-mingw32";
    }
    ret "i686-pc-linux-gnu";
}


//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
