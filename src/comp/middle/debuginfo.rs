import std::{vec, str, map, option, unsafe};
import std::vec::to_ptr;
import std::map::hashmap;
import lib::llvm::llvm;
import lib::llvm::llvm::{ModuleRef, ValueRef};
import middle::trans_common::*;
import syntax::{ast, codemap};

const LLVMDebugVersion: int = 0x80000;

const DW_LANG_RUST: int = 0x9000;
const DW_VIRTUALITY_none: int = 0;

const CompileUnitTag: int = 17;
const FileDescriptorTag: int = 41;
const SubprogramTag: int = 46;

fn as_buf(s: str) -> str::sbuf {
    str::as_buf(s, {|sbuf| sbuf})
}
fn llstr(s: str) -> ValueRef {
    llvm::LLVMMDString(as_buf(s), str::byte_len(s))
}

fn lltag(lltag: int) -> ValueRef {
    lli32(0x80000 + lltag)
}
fn lli32(val: int) -> ValueRef {
    C_i32(val as i32)
}
fn lli1(bval: bool) -> ValueRef {
    C_bool(bval)
}
fn llmdnode(elems: [ValueRef]) -> ValueRef unsafe {
    llvm::LLVMMDNode(vec::unsafe::to_ptr(elems),
                     vec::len(elems))
}
fn llunused() -> ValueRef {
    lli32(0x0)
}

fn update_cache(cache: metadata_cache, mdtag: int, val: debug_metadata) {
    let existing = if cache.contains_key(mdtag) {
        cache.get(mdtag)
    } else {
        []
    };
    cache.insert(mdtag, existing + [val]);
}

////////////////

type metadata<T> = {node: ValueRef, data: T};

type file_md = {path: str};
type compile_unit_md = {path: str};
type subprogram_md = {name: str, file: str};

type metadata_cache = hashmap<int, [debug_metadata]>;

tag debug_metadata {
    file_metadata(@metadata<file_md>);
    compile_unit_metadata(@metadata<compile_unit_md>);
    subprogram_metadata(@metadata<subprogram_md>);
}

fn md_from_metadata<T>(val: debug_metadata) -> T unsafe {
    alt val {
      file_metadata(md) { unsafe::reinterpret_cast(md) }
      compile_unit_metadata(md) { unsafe::reinterpret_cast(md) }
      subprogram_metadata(md) { unsafe::reinterpret_cast(md) }
    }
}

fn cached_metadata<T>(cache: metadata_cache, mdtag: int,
                      eq: block(md: T) -> bool) -> option::t<T> {
    if cache.contains_key(mdtag) {
        let items = cache.get(mdtag);
        for item in items {
            let md: T = md_from_metadata::<T>(item);
            if eq(md) {
                ret option::some(md);
            }
        }
    }
    ret option::none;
}

fn get_compile_unit_metadata(cx: @crate_ctxt, full_path: str)
    -> @metadata<compile_unit_md> {
    let cache = cx.llmetadata;
    alt cached_metadata::<@metadata<compile_unit_md>>(cache, CompileUnitTag,
                        {|md| md.data.path == full_path}) {
      option::some(md) { ret md; }
      option::none. {}
    }
    let sep = str::rindex(full_path, '/' as u8) as uint;
    let fname = str::slice(full_path, sep + 1u,
                           str::byte_len(full_path));
    let path = str::slice(full_path, 0u, sep + 1u);
    let unit_metadata = [lltag(CompileUnitTag),
                         llunused(),
                         lli32(DW_LANG_RUST),
                         llstr(fname),
                         llstr(path),
                         llstr(#env["CFG_VERSION"]),
                         lli1(false), // main compile unit
                         lli1(cx.sess.get_opts().optimize != 0u),
                         llstr(""), // flags (???)
                         lli32(0) // runtime version (???)
                         // list of enum types
                         // list of retained values
                         // list of subprograms
                         // list of global variables
                        ];
    let unit_node = llmdnode(unit_metadata);
    llvm::LLVMAddNamedMetadataOperand(cx.llmod, as_buf("llvm.dbg.cu"),
                                  str::byte_len("llvm.dbg.cu"),
                                  unit_node);
    let mdval = @{node: unit_node, data: {path: full_path}};
    update_cache(cache, CompileUnitTag, compile_unit_metadata(mdval));
    ret mdval;
}

//        let kind_id = llvm::LLVMGetMDKindID(as_buf("dbg"),
//                                            str::byte_len("dbg"));


fn get_file_metadata(cx: @crate_ctxt, full_path: str) -> @metadata<file_md> {
    let cache = cx.llmetadata;
    alt cached_metadata::<@metadata<file_md>>(
        cache, FileDescriptorTag, {|md| md.data.path == full_path}) {
      option::some(md) { ret md; }
      option::none. {}
    }
    let sep = str::rindex(full_path, '/' as u8) as uint;
    let fname = str::slice(full_path, sep + 1u,
                           str::byte_len(full_path));
    let path = str::slice(full_path, 0u, sep + 1u);
    let unit_node = get_compile_unit_metadata(cx, path).node;
    let file_md = [lltag(FileDescriptorTag),
                   llstr(fname),
                   llstr(path),
                   unit_node];
    let val = llmdnode(file_md);
    let mdval = @{node: val, data: {path: full_path}};
    update_cache(cache, FileDescriptorTag, file_metadata(mdval));
    ret mdval;
}

fn get_function_metadata(cx: @crate_ctxt, item: @ast::item,
                         llfndecl: ValueRef) -> @metadata<subprogram_md> {
    let cache = cx.llmetadata;
    alt cached_metadata::<@metadata<subprogram_md>>(
        cache, SubprogramTag, {|md| md.data.name == item.ident &&
                                    /*sub.path == ??*/ true}) {
      option::some(md) { ret md; }
      option::none. {}
    }
    let loc = codemap::lookup_char_pos(cx.sess.get_codemap(),
                                           item.span.lo);
        let file_node = get_file_metadata(cx, loc.filename).node;
        let fn_metadata = [lltag(SubprogramTag),
                           llunused(),
                           file_node,
                           llstr(item.ident),
                           llstr(item.ident), //XXX fully-qualified C++ name
                           llstr(item.ident), //XXX MIPS name?????
                           file_node,
                           lli32(loc.line as int),
                           C_null(T_ptr(T_nil())), // XXX reference to tydesc
                           lli1(false), //XXX static
                           lli1(true), // not extern
                           lli32(DW_VIRTUALITY_none), // virtual-ness
                           lli32(0i), //index into virt func
                           C_null(T_ptr(T_nil())), // base type with vtbl
                           lli1(false), // artificial
                           lli1(cx.sess.get_opts().optimize != 0u),
                           llfndecl
                           //list of template params
                           //func decl descriptor
                           //list of func vars
                          ];
        let val = llmdnode(fn_metadata);
        llvm::LLVMAddNamedMetadataOperand(cx.llmod, as_buf("llvm.dbg.sp"),
                                          str::byte_len("llvm.dbg.sp"),
                                          val);
        let mdval = @{node: val, data: {name: item.ident,
                                        file: loc.filename}};
        update_cache(cache, SubprogramTag, subprogram_metadata(mdval));
        ret mdval;
}