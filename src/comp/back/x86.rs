import lib::llvm::llvm;
import lib::llvm::llvm::ModuleRef;
import std::str;
import std::vec;
import std::os::target_os;
import util::common::istr;

const int wordsz = 4;

fn wstr(int i) -> str {
    ret istr(i * wordsz);
}

fn start() -> vec[str] {
    ret [".cfi_startproc"];
}

fn end() -> vec[str] {
    ret [".cfi_endproc"];
}

fn save_callee_saves() -> vec[str] {
    ret ["pushl %ebp",
            "pushl %edi",
            "pushl %esi",
            "pushl %ebx"];
}

fn save_callee_saves_with_cfi() -> vec[str] {
    auto offset = 8;
    auto t;
    t  = ["pushl %ebp"];
    t += [".cfi_def_cfa_offset " + istr(offset)];
    t += [".cfi_offset %ebp, -" + istr(offset)];

    t += ["pushl %edi"];
    offset += 4;
    t += [".cfi_def_cfa_offset " + istr(offset)];

    t += ["pushl %esi"];
    offset += 4;
    t += [".cfi_def_cfa_offset " + istr(offset)];

    t += ["pushl %ebx"];
    offset += 4;
    t += [".cfi_def_cfa_offset " + istr(offset)];
    ret t;
}

fn restore_callee_saves() -> vec[str] {
    ret ["popl  %ebx",
            "popl  %esi",
            "popl  %edi",
            "popl  %ebp"];
}

fn load_esp_from_rust_sp_first_arg() -> vec[str] {
    ret ["movl  " + wstr(abi::task_field_rust_sp) + "(%ecx), %esp"];
}

fn load_esp_from_runtime_sp_first_arg() -> vec[str] {
    ret ["movl  " + wstr(abi::task_field_runtime_sp) + "(%ecx), %esp"];
}

fn store_esp_to_rust_sp_first_arg() -> vec[str] {
    ret ["movl  %esp, " + wstr(abi::task_field_rust_sp) + "(%ecx)"];
}

fn store_esp_to_runtime_sp_first_arg() -> vec[str] {
    ret ["movl  %esp, " + wstr(abi::task_field_runtime_sp) + "(%ecx)"];
}

fn load_esp_from_rust_sp_second_arg() -> vec[str] {
    ret ["movl  " + wstr(abi::task_field_rust_sp) + "(%edx), %esp"];
}

fn load_esp_from_runtime_sp_second_arg() -> vec[str] {
    ret ["movl  " + wstr(abi::task_field_runtime_sp) + "(%edx), %esp"];
}

fn store_esp_to_rust_sp_second_arg() -> vec[str] {
    ret ["movl  %esp, " + wstr(abi::task_field_rust_sp) + "(%edx)"];
}

fn store_esp_to_runtime_sp_second_arg() -> vec[str] {
    ret ["movl  %esp, " + wstr(abi::task_field_runtime_sp) + "(%edx)"];
}

fn native_glue(int n_args, abi::native_glue_type ngt) -> vec[str] {

    let bool pass_task;
    alt (ngt) {
        case (abi::ngt_rust)         { pass_task = true; }
        case (abi::ngt_pure_rust)    { pass_task = true; }
        case (abi::ngt_cdecl)        { pass_task = false; }
    }

    /*
     * 0, 4, 8, 12 are callee-saves
     * 16 is retpc
     * 20 .. (5+i) * 4 are args
     *
     * ecx is taskptr
     * edx is callee
     *
     */

    fn copy_arg(bool pass_task, uint i) -> str {
        if (i == 0u && pass_task) {
            ret "movl  %edx, (%esp)";
        }
        auto dst_off = wstr(0 + (i as int));
        auto src_off;
        if (pass_task) {
            src_off = wstr(4 + (i as int));
        } else {
            src_off = wstr(5 + (i as int));
        }
        auto m = ["movl  " + src_off + "(%ebp),%eax",
                     "movl  %eax," + dst_off + "(%esp)"];
        ret str::connect(m, "\n\t");
    }

    auto carg = bind copy_arg(pass_task, _);

    ret
        start()
        + save_callee_saves_with_cfi()

        + ["movl  %esp, %ebp     # ebp = rust_sp"]
        + [".cfi_def_cfa_register %ebp"]

        + store_esp_to_rust_sp_second_arg()
        + load_esp_from_runtime_sp_second_arg()

        + ["subl  $" + wstr(n_args) + ", %esp   # esp -= args",
              "andl  $~0xf, %esp    # align esp down"]

        + vec::init_fn[str](carg, (n_args) as uint)

        +  ["movl  %edx, %edi     # save task from edx to edi",
               "call  *%ecx          # call *%ecx",
               "movl  %edi, %edx     # restore edi-saved task to edx"]

        + load_esp_from_rust_sp_second_arg()
        + restore_callee_saves()
        + ["ret"]
        + end();

}


fn decl_glue(int align, str prefix, str name, vec[str] insns) -> str {
    auto sym = prefix + name;
    ret "\t.globl " + sym + "\n" +
        "\t.balign " + istr(align) + "\n" +
        sym + ":\n" +
        "\t" + str::connect(insns, "\n\t");
}


fn decl_native_glue(int align, str prefix, abi::native_glue_type ngt, uint n)
        -> str {
    let int i = n as int;
    ret decl_glue(align, prefix,
                  abi::native_glue_name(i, ngt),
                  native_glue(i, ngt));
}

fn get_symbol_prefix() -> str {
    if (str::eq(target_os(), "macos") ||
        str::eq(target_os(), "win32")) {
        ret "_";
    } else {
        ret "";
    }
}

fn get_module_asm() -> str {
    auto align = 4;

    auto prefix = get_symbol_prefix();

    let vec[str] glues =
        []
        + vec::init_fn[str](bind decl_native_glue(align, prefix,
            abi::ngt_rust, _), (abi::n_native_glues + 1) as uint)
        + vec::init_fn[str](bind decl_native_glue(align, prefix,
            abi::ngt_pure_rust, _), (abi::n_native_glues + 1) as uint)
        + vec::init_fn[str](bind decl_native_glue(align, prefix,
            abi::ngt_cdecl, _), (abi::n_native_glues + 1) as uint);


    ret str::connect(glues, "\n\n");
}

fn get_meta_sect_name() -> str {
    if (str::eq(target_os(), "macos")) {
        ret "__DATA,__note.rustc";
    }
    if (str::eq(target_os(), "win32")) {
        ret ".note.rustc";
    }
    ret ".note.rustc";
}

fn get_data_layout() -> str {
    if (str::eq(target_os(), "macos")) {
      ret "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64" +
        "-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128" +
        "-n8:16:32";
    }
    if (str::eq(target_os(), "win32")) {
      ret "e-p:32:32-f64:64:64-i64:64:64-f80:32:32-n8:16:32";
    }
    ret "e-p:32:32-f64:32:64-i64:32:64-f80:32:32-n8:16:32";
}

fn get_target_triple() -> str {
    if (str::eq(target_os(), "macos")) {
        ret "i686-apple-darwin";
    }
    if (str::eq(target_os(), "win32")) {
        ret "i686-pc-mingw32";
    }
    ret "i686-unknown-linux-gnu";
}


//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
