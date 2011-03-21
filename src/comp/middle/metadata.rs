import std._str;
import std._vec;
import std.option;

import front.ast;
import middle.trans;
import middle.ty;
import back.x86;
import util.common;

import lib.llvm.llvm;
import lib.llvm.llvm.ValueRef;
import lib.llvm.False;

// Type encoding

// Compact string representation for ty.t values. API ty_str & parse_from_str.
// (The second has to be authed pure.) Extra parameters are for converting
// to/from def_ids in the string rep. Whatever format you choose should not
// contain pipe characters.

// Callback to translate defs to strs or back.
type def_str = fn(ast.def_id) -> str;

fn ty_str(@ty.t t, def_str ds) -> str {
    ret sty_str(t.struct, ds);
}

fn mt_str(&ty.mt mt, def_str ds) -> str {
    auto mut_str;
    alt (mt.mut) {
        case (ast.imm)       { mut_str = "";  }
        case (ast.mut)       { mut_str = "m"; }
        case (ast.maybe_mut) { mut_str = "?"; }
    }
    ret mut_str + ty_str(mt.ty, ds);
}

fn sty_str(ty.sty st, def_str ds) -> str {
    alt (st) {
        case (ty.ty_nil) {ret "n";}
        case (ty.ty_bool) {ret "b";}
        case (ty.ty_int) {ret "i";}
        case (ty.ty_uint) {ret "u";}
        case (ty.ty_machine(?mach)) {
            alt (mach) {
                case (common.ty_u8) {ret "Mb";}
                case (common.ty_u16) {ret "Mw";}
                case (common.ty_u32) {ret "Ml";}
                case (common.ty_u64) {ret "Md";}
                case (common.ty_i8) {ret "MB";}
                case (common.ty_i16) {ret "MW";}
                case (common.ty_i32) {ret "ML";}
                case (common.ty_i64) {ret "MD";}
                case (common.ty_f32) {ret "Mf";}
                case (common.ty_f64) {ret "MF";}
            }
        }
        case (ty.ty_char) {ret "c";}
        case (ty.ty_str) {ret "s";}
        case (ty.ty_tag(?def,?tys)) { // TODO restore def_id
            auto acc = "t[" + ds(def) + "|";
            for (@ty.t t in tys) {acc += ty_str(t, ds);}
            ret acc + "]";
        }
        case (ty.ty_box(?mt)) {ret "@" + mt_str(mt, ds);}
        case (ty.ty_vec(?mt)) {ret "V" + mt_str(mt, ds);}
        case (ty.ty_port(?t)) {ret "P" + ty_str(t, ds);}
        case (ty.ty_chan(?t)) {ret "C" + ty_str(t, ds);}
        case (ty.ty_tup(?mts)) {
            auto acc = "T[";
            for (ty.mt mt in mts) {acc += mt_str(mt, ds);}
            ret acc + "]";
        }
        case (ty.ty_rec(?fields)) {
            auto acc = "R[";
            for (ty.field field in fields) {
                acc += field.ident + "=";
                acc += mt_str(field.mt, ds);
            }
            ret acc + "]";
        }
        case (ty.ty_fn(?proto,?args,?out)) {
            ret proto_str(proto) + ty_fn_str(args, out, ds);
        }
        case (ty.ty_native_fn(?abi,?args,?out)) {
            auto abistr;
            alt (abi) {
                case (ast.native_abi_rust) {abistr = "r";}
                case (ast.native_abi_cdecl) {abistr = "c";}
            }
            ret "N" + abistr + ty_fn_str(args, out, ds);
        }
        case (ty.ty_obj(?methods)) {
            auto acc = "O[";
            for (ty.method m in methods) {
                acc += proto_str(m.proto);
                acc += m.ident;
                acc += ty_fn_str(m.inputs, m.output, ds);
            }
            ret acc + "]";
        }
        case (ty.ty_var(?id)) {ret "X" + common.istr(id);}
        case (ty.ty_native) {ret "E";}
        // TODO (maybe?)   ty_param(ast.def_id), ty_type;
    }
}

fn proto_str(ast.proto proto) -> str {
    alt (proto) {
        case (ast.proto_iter) {ret "W";}
        case (ast.proto_fn) {ret "F";}
    }
}

fn ty_fn_str(vec[ty.arg] args, @ty.t out, def_str ds) -> str {
    auto acc = "[";
    for (ty.arg arg in args) {
        if (arg.mode == ast.alias) {acc += "&";}
        acc += ty_str(arg.ty, ds);
    }
    ret acc + "]" + ty_str(out, ds);
}


// Returns a Plain Old LLVM String.
fn C_postr(str s) -> ValueRef {
    ret llvm.LLVMConstString(_str.buf(s), _str.byte_len(s), False);
}

fn collect_meta_directives(@trans.crate_ctxt cx, @ast.crate crate)
        -> ValueRef {
    ret C_postr("Hello world!");    // TODO
}

fn write_metadata(@trans.crate_ctxt cx, @ast.crate crate) {
    auto llmeta = collect_meta_directives(cx, crate);

    auto llconst = trans.C_struct(vec(llmeta));
    auto llglobal = llvm.LLVMAddGlobal(cx.llmod, trans.val_ty(llconst),
                                       _str.buf("rust_metadata"));
    llvm.LLVMSetInitializer(llglobal, llconst);
    llvm.LLVMSetSection(llglobal, _str.buf(x86.get_meta_sect_name()));
}

