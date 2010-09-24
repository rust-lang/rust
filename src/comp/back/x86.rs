import lib.llvm.llvm;
import lib.llvm.llvm.ModuleRef;
import std._str;
import std._vec;
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
    ret vec("movl  " + wstr(abi.task_field_rust_sp) + "(%edx), %esp");
}

fn load_esp_from_runtime_sp() -> vec[str] {
    ret vec("movl  " + wstr(abi.task_field_runtime_sp) + "(%edx), %esp");
}

fn store_esp_to_rust_sp() -> vec[str] {
    ret vec("movl  %esp, " + wstr(abi.task_field_rust_sp) + "(%edx)");
}

fn store_esp_to_runtime_sp() -> vec[str] {
    ret vec("movl  %esp, " + wstr(abi.task_field_runtime_sp) + "(%edx)");
}

fn rust_activate_glue() -> vec[str] {
    ret vec("movl  4(%esp), %edx    # edx = rust_task")
        + save_callee_saves()
        + store_esp_to_runtime_sp()
        + load_esp_from_rust_sp()

        // This 'add' instruction is a bit surprising.
        // See lengthy comment in boot/be/x86.ml activate_glue.
        + vec("addl  $20, " + wstr(abi.task_field_rust_sp) + "(%edx)")

        + restore_callee_saves()
        + vec("ret");
}

fn rust_yield_glue() -> vec[str] {
    ret vec("movl  0(%esp), %edx    # edx = rust_task")
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
     * 20 is taskptr
     * 24 is callee
     * 28 .. (7+i) * 4 are args
     */

    fn copy_arg(uint i) -> str {
        auto off = wstr(7 + (i as int));
        auto m = vec("movl  " + off + "(%ebp),%edx",
                     "movl  %edx," + off + "(%esp)");
        ret _str.connect(m, "\n\t");
    }

    auto carg = copy_arg;

    ret
        save_callee_saves()

        + vec("movl  %esp, %ebp     # ebp = rust_sp",
              "movl  20(%esp), %edx # edx = rust_task")

        + store_esp_to_rust_sp()
        + load_esp_from_runtime_sp()

        + vec("subl  $" + wstr(n_args + 1) + ", %esp   # esp -= args",
              "andl  $~0xf, %esp    # align esp down",
              "movl  %edx, (%esp)   # arg[0] = rust_task ")

        + _vec.init_fn[str](carg, n_args as uint)

        +  vec("movl  24(%ebp), %edx # edx = callee",
               "call  *%edx          # call *%edx",
               "movl  20(%ebp), %edx # edx = rust_task")

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
                  "rust_upcall_" + istr(i),
                  upcall_glue(i));
}

fn get_module_asm() -> str {
    auto align = 4;
    auto prefix = "";

    auto glues =
        vec(decl_glue(align, prefix,
                      "rust_activate_glue",
                      rust_activate_glue()),

            decl_glue(align, prefix,
                      "rust_yield_glue",
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
