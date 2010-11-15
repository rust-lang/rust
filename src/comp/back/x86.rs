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

fn rust_activate_glue() -> vec[str] {
    ret vec("movl  4(%esp), %ecx    # ecx = rust_task")
        + save_callee_saves()
        + store_esp_to_runtime_sp()
        + load_esp_from_rust_sp()

        // This 'add' instruction is a bit surprising.
        // See lengthy comment in boot/be/x86.ml activate_glue.
        + vec("addl  $20, " + wstr(abi.task_field_rust_sp) + "(%ecx)")

        + restore_callee_saves()
        + vec("ret");
}

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
