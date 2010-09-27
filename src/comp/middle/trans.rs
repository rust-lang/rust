import std._str;
import std._vec;
import std._str.rustrt.sbuf;
import std._vec.rustrt.vbuf;
import std.map.hashmap;

import front.ast;
import driver.session;
import back.x86;
import back.abi;

import util.common.istr;
import util.common.new_str_hash;

import lib.llvm.llvm;
import lib.llvm.builder;
import lib.llvm.llvm.ModuleRef;
import lib.llvm.llvm.ValueRef;
import lib.llvm.llvm.TypeRef;
import lib.llvm.llvm.BuilderRef;
import lib.llvm.llvm.BasicBlockRef;

import lib.llvm.False;
import lib.llvm.True;

type glue_fns = rec(ValueRef activate_glue,
                    ValueRef yield_glue,
                    vec[ValueRef] upcall_glues);

type trans_ctxt = rec(session.session sess,
                      ModuleRef llmod,
                      hashmap[str,ValueRef] upcalls,
                      hashmap[str,ValueRef] fns,
                      @glue_fns glues,
                      str path);

type fn_ctxt = rec(ValueRef llfn,
                   ValueRef lloutptr,
                   ValueRef lltaskptr,
                   @trans_ctxt tcx);

type terminator = fn(@fn_ctxt cx, builder build);

type block_ctxt = rec(BasicBlockRef llbb,
                      builder build,
                      terminator term,
                      @fn_ctxt fcx);


// LLVM type constructors.

fn T_nil() -> TypeRef {
    ret llvm.LLVMVoidType();
}

fn T_int() -> TypeRef {
    ret llvm.LLVMInt32Type();
}

fn T_fn(vec[TypeRef] inputs, TypeRef output) -> TypeRef {
    ret llvm.LLVMFunctionType(output,
                              _vec.buf[TypeRef](inputs),
                              _vec.len[TypeRef](inputs),
                              False);
}

fn T_ptr(TypeRef t) -> TypeRef {
    ret llvm.LLVMPointerType(t, 0u);
}

fn T_struct(vec[TypeRef] elts) -> TypeRef {
    ret llvm.LLVMStructType(_vec.buf[TypeRef](elts),
                            _vec.len[TypeRef](elts),
                            False);
}

fn T_opaque() -> TypeRef {
    ret llvm.LLVMOpaqueType();
}

fn T_task() -> TypeRef {
    ret T_struct(vec(T_int(),      // Refcount
                     T_int(),      // Delegate pointer
                     T_int(),      // Stack segment pointer
                     T_int(),      // Runtime SP
                     T_int(),      // Rust SP
                     T_int(),      // GC chain
                     T_int(),      // Domain pointer
                     T_int()       // Crate cache pointer
                     ));
}

fn T_crate() -> TypeRef {
    ret T_struct(vec(T_int(),      // ptrdiff_t image_base_off
                     T_int(),      // uintptr_t self_addr
                     T_int(),      // ptrdiff_t debug_abbrev_off
                     T_int(),      // size_t debug_abbrev_sz
                     T_int(),      // ptrdiff_t debug_info_off
                     T_int(),      // size_t debug_info_sz
                     T_int(),      // size_t activate_glue_off
                     T_int(),      // size_t yield_glue_off
                     T_int(),      // size_t unwind_glue_off
                     T_int(),      // size_t gc_glue_off
                     T_int(),      // size_t main_exit_task_glue_off
                     T_int(),      // int n_rust_syms
                     T_int(),      // int n_c_syms
                     T_int()       //int n_libs
                     ));
}

fn T_double() -> TypeRef {
    ret llvm.LLVMDoubleType();
}

fn T_taskptr() -> TypeRef {
    ret T_ptr(T_task());
}


// LLVM constant constructors.

fn C_null(TypeRef t) -> ValueRef {
    ret llvm.LLVMConstNull(t);
}

fn C_int(int i) -> ValueRef {
    // FIXME. We can't use LLVM.ULongLong with our existing minimal native
    // API, which only knows word-sized args.  Lucky for us LLVM has a "take a
    // string encoding" version.  Hilarious. Please fix to handle:
    //
    // ret llvm.LLVMConstInt(T_int(), t as LLVM.ULongLong, False);
    //
    ret llvm.LLVMConstIntOfString(T_int(),
                                  _str.buf(istr(i)), 10);
}

fn C_str(str s) -> ValueRef {
    ret llvm.LLVMConstString(_str.buf(s), _str.byte_len(s), False);
}

fn C_struct(vec[ValueRef] elts) -> ValueRef {
    ret llvm.LLVMConstStruct(_vec.buf[ValueRef](elts),
                             _vec.len[ValueRef](elts),
                             False);
}

fn decl_cdecl_fn(ModuleRef llmod, str name,
                 vec[TypeRef] inputs, TypeRef output) -> ValueRef {
    let TypeRef llty = T_fn(inputs, output);
    log "declaring " + name + " with type " + lib.llvm.type_to_str(llty);
    let ValueRef llfn =
        llvm.LLVMAddFunction(llmod, _str.buf(name), llty);
    llvm.LLVMSetFunctionCallConv(llfn, lib.llvm.LLVMCCallConv);
    ret llfn;
}

fn decl_glue(ModuleRef llmod, str s) -> ValueRef {
    ret decl_cdecl_fn(llmod, s, vec(T_taskptr()), T_nil());
}

fn decl_upcall(ModuleRef llmod, uint _n) -> ValueRef {
    // It doesn't actually matter what type we come up with here, at the
    // moment, as we cast the upcall function pointers to int before passing
    // them to the indirect upcall-invocation glue.  But eventually we'd like
    // to call them directly, once we have a calling convention worked out.
    let int n = _n as int;
    let str s = abi.upcall_glue_name(n);
    let vec[TypeRef] args =
        vec(T_taskptr(), // taskptr
            T_int())     // callee
        + _vec.init_elt[TypeRef](T_int(), n as uint);

    ret decl_cdecl_fn(llmod, s, args, T_int());
}

fn get_upcall(@trans_ctxt cx, str name, int n_args) -> ValueRef {
    if (cx.upcalls.contains_key(name)) {
        ret cx.upcalls.get(name);
    }
    auto inputs = vec(T_taskptr());
    inputs += _vec.init_elt[TypeRef](T_int(), n_args as uint);
    auto output = T_nil();
    auto f = decl_cdecl_fn(cx.llmod, name, inputs, output);
    cx.upcalls.insert(name, f);
    ret f;
}

fn trans_upcall(@block_ctxt cx, str name, vec[ValueRef] args) -> ValueRef {
    let int n = _vec.len[ValueRef](args) as int;
    let ValueRef llupcall = get_upcall(cx.fcx.tcx, name, n);
    llupcall = llvm.LLVMConstPointerCast(llupcall, T_int());

    let ValueRef llglue = cx.fcx.tcx.glues.upcall_glues.(n);
    let vec[ValueRef] call_args = vec(cx.fcx.lltaskptr, llupcall) + args;
    log "emitting indirect-upcall via " + abi.upcall_glue_name(n);
    for (ValueRef v in call_args) {
        log "arg: " + lib.llvm.type_to_str(llvm.LLVMTypeOf(v));
    }
    log "emitting call to callee of type: " +
        lib.llvm.type_to_str(llvm.LLVMTypeOf(llglue));
    ret cx.build.Call(llglue, call_args);
}

fn trans_log(@block_ctxt cx, &ast.atom a) {
    alt (a) {
        case (ast.atom_lit(?lit)) {
            alt (*lit) {
                case (ast.lit_int(?i)) {
                    trans_upcall(cx, "upcall_log_int", vec(C_int(i)));
                }
                case (_) {
                    cx.fcx.tcx.sess.unimpl("literal variant in trans_log");
                }
            }
        }
        case (_) {
            cx.fcx.tcx.sess.unimpl("atom variant in trans_log");
        }
    }
}

fn trans_stmt(@block_ctxt cx, &ast.stmt s) {
    alt (s) {
        case (ast.stmt_log(?a)) {
            trans_log(cx, *a);
        }
        case (_) {
            cx.fcx.tcx.sess.unimpl("stmt variant");
        }
    }
}

fn default_terminate(@fn_ctxt cx, builder build) {
    build.RetVoid();
}

fn new_builder(BasicBlockRef llbb) -> builder {
    let BuilderRef llbuild = llvm.LLVMCreateBuilder();
    llvm.LLVMPositionBuilderAtEnd(llbuild, llbb);
    ret builder(llbuild);
}

fn trans_block(@fn_ctxt cx, &ast.block b, terminator term) {
    let BasicBlockRef llbb =
        llvm.LLVMAppendBasicBlock(cx.llfn, _str.buf(""));
    auto bcx = @rec(llbb=llbb,
                    build=new_builder(llbb),
                    term=term,
                    fcx=cx);
    for (@ast.stmt s in b) {
        trans_stmt(bcx, *s);
    }
    bcx.term(cx, bcx.build);
}

fn trans_fn(@trans_ctxt cx, &ast._fn f) {
    let vec[TypeRef] args = vec(T_ptr(T_int()), // outptr.
                                T_taskptr()     // taskptr
                                );
    let ValueRef llfn = decl_cdecl_fn(cx.llmod, cx.path, args, T_nil());
    cx.fns.insert(cx.path, llfn);
    let ValueRef lloutptr = llvm.LLVMGetParam(llfn, 0u);
    let ValueRef lltaskptr = llvm.LLVMGetParam(llfn, 1u);
    auto fcx = @rec(llfn=llfn,
                    lloutptr=lloutptr,
                    lltaskptr=lltaskptr,
                    tcx=cx);
    auto term = default_terminate;
    trans_block(fcx, f.body, term);
}

fn trans_item(@trans_ctxt cx, &str name, &ast.item item) {
    auto sub_cx = @rec(path=cx.path + "." + name with *cx);
    alt (item) {
        case (ast.item_fn(?f)) {
            trans_fn(sub_cx, *f);
        }
        case (ast.item_mod(?m)) {
            trans_mod(sub_cx, *m);
        }
    }
}

fn trans_mod(@trans_ctxt cx, &ast._mod m) {
    for each (tup(str, ast.item) pair in m.items()) {
        trans_item(cx, pair._0, pair._1);
    }
}


fn p2i(ValueRef v) -> ValueRef {
    ret llvm.LLVMConstPtrToInt(v, T_int());
}

fn crate_constant(@trans_ctxt cx) -> ValueRef {

    let ValueRef crate_ptr =
        llvm.LLVMAddGlobal(cx.llmod, T_crate(),
                           _str.buf("rust_crate"));

    let ValueRef crate_addr = p2i(crate_ptr);

    let ValueRef activate_glue_off =
        llvm.LLVMConstSub(p2i(cx.glues.activate_glue), crate_addr);

    let ValueRef yield_glue_off =
        llvm.LLVMConstSub(p2i(cx.glues.yield_glue), crate_addr);

    // FIXME: we aren't generating the exit-task glue yet.
    // llvm.LLVMConstSub(p2i(cx.glues.exit_task_glue), crate_addr);
    let ValueRef exit_task_glue_off = C_null(T_int());

    let ValueRef crate_val =
        C_struct(vec(C_null(T_int()),     // ptrdiff_t image_base_off
                     p2i(crate_ptr),      // uintptr_t self_addr
                     C_null(T_int()),     // ptrdiff_t debug_abbrev_off
                     C_null(T_int()),     // size_t debug_abbrev_sz
                     C_null(T_int()),     // ptrdiff_t debug_info_off
                     C_null(T_int()),     // size_t debug_info_sz
                     activate_glue_off,   // size_t activate_glue_off
                     yield_glue_off,      // size_t yield_glue_off
                     C_null(T_int()),     // size_t unwind_glue_off
                     C_null(T_int()),     // size_t gc_glue_off
                     exit_task_glue_off,  // size_t main_exit_task_glue_off
                     C_null(T_int()),     // int n_rust_syms
                     C_null(T_int()),     // int n_c_syms
                     C_null(T_int())      // int n_libs
                     ));

    llvm.LLVMSetInitializer(crate_ptr, crate_val);
    ret crate_ptr;
}

fn trans_main_fn(@trans_ctxt cx, ValueRef llcrate) {
    auto T_main_args = vec(T_int(), T_int());
    auto T_rust_start_args = vec(T_int(), T_int(), T_int(), T_int());

    auto llmain =
        decl_cdecl_fn(cx.llmod, "main", T_main_args, T_int());

    auto llrust_start =
        decl_cdecl_fn(cx.llmod, "rust_start", T_rust_start_args, T_int());

    auto llargc = llvm.LLVMGetParam(llmain, 0u);
    auto llargv = llvm.LLVMGetParam(llmain, 1u);
    auto llrust_main = cx.fns.get("_rust.main");

    //
    // Emit the moral equivalent of:
    //
    // main(int argc, char **argv) {
    //     rust_start(&_rust.main, &crate, argc, argv);
    // }
    //

    let BasicBlockRef llbb =
        llvm.LLVMAppendBasicBlock(llmain, _str.buf(""));
    auto b = new_builder(llbb);

    auto start_args = vec(p2i(llrust_main), p2i(llcrate), llargc, llargv);

    b.Ret(b.Call(llrust_start, start_args));

}

fn trans_crate(session.session sess, ast.crate crate) {
    auto llmod =
        llvm.LLVMModuleCreateWithNameInContext(_str.buf("rust_out"),
                                               llvm.LLVMGetGlobalContext());

    llvm.LLVMSetModuleInlineAsm(llmod, _str.buf(x86.get_module_asm()));

    auto glues = @rec(activate_glue = decl_glue(llmod,
                                                abi.activate_glue_name()),
                      yield_glue = decl_glue(llmod, abi.yield_glue_name()),
                      upcall_glues =
                      _vec.init_fn[ValueRef](bind decl_upcall(llmod, _),
                                             abi.n_upcall_glues as uint));

    auto cx = @rec(sess = sess,
                   llmod = llmod,
                   upcalls = new_str_hash[ValueRef](),
                   fns = new_str_hash[ValueRef](),
                   glues = glues,
                   path = "_rust");

    trans_mod(cx, crate.module);

    trans_main_fn(cx, crate_constant(cx));

    llvm.LLVMWriteBitcodeToFile(llmod, _str.buf("rust_out.bc"));
    llvm.LLVMDisposeModule(llmod);
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
