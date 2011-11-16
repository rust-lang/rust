import std::{vec, str, option, unsafe, fs};
import std::map::hashmap;
import lib::llvm::llvm;
import lib::llvm::llvm::ValueRef;
import middle::trans_common::*;
import middle::ty;
import ast::ty;
import syntax::{ast, codemap};

const LLVMDebugVersion: int = (9 << 16); //& 0xffff0000; // 0x80000 ?

const DW_LANG_RUST: int = 0x9000;
const DW_VIRTUALITY_none: int = 0;

const CompileUnitTag: int = 17;
const FileDescriptorTag: int = 41;
const SubprogramTag: int = 46;
const BasicTypeDescriptorTag: int = 36;
const AutoVariableTag: int = 256;
const ArgVariableTag: int = 257;
const ReturnVariableTag: int = 258;
const LexicalBlockTag: int = 11;

const DW_ATE_boolean: int = 0x02;
const DW_ATE_float: int = 0x04;
const DW_ATE_signed: int = 0x05;
const DW_ATE_signed_char: int = 0x06;
const DW_ATE_unsigned: int = 0x07;
const DW_ATE_unsigned_char: int = 0x08;

fn as_buf(s: str) -> str::sbuf {
    str::as_buf(s, {|sbuf| sbuf})
}
fn llstr(s: str) -> ValueRef {
    llvm::LLVMMDString(as_buf(s), str::byte_len(s))
}

fn lltag(lltag: int) -> ValueRef {
    lli32(LLVMDebugVersion | lltag)
}
fn lli32(val: int) -> ValueRef {
    C_i32(val as i32)
}
fn lli64(val: int) -> ValueRef {
    C_i64(val as i64)
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
fn llnull() -> ValueRef {
    C_null(T_ptr(T_nil()))
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
type local_var_md = {id: ast::node_id};
type tydesc_md = {hash: uint};
type block_md = {start: codemap::loc, end: codemap::loc};

type metadata_cache = hashmap<int, [debug_metadata]>;

tag debug_metadata {
    file_metadata(@metadata<file_md>);
    compile_unit_metadata(@metadata<compile_unit_md>);
    subprogram_metadata(@metadata<subprogram_md>);
    local_var_metadata(@metadata<local_var_md>);
    tydesc_metadata(@metadata<tydesc_md>);
    block_metadata(@metadata<block_md>);
}

fn cast_safely<T, U>(val: T) -> U unsafe {
    let val2 = val;
    let val3 = unsafe::reinterpret_cast(val2);
    unsafe::leak(val2);
    ret val3;
}

fn md_from_metadata<T>(val: debug_metadata) -> T unsafe {
    alt val {
      file_metadata(md) { cast_safely(md) }
      compile_unit_metadata(md) { cast_safely(md) }
      subprogram_metadata(md) { cast_safely(md) }
      local_var_metadata(md) { cast_safely(md) }
      tydesc_metadata(md) { cast_safely(md) }
      block_metadata(md) { cast_safely(md) }
    }
}

fn cached_metadata<T>(cache: metadata_cache, mdtag: int,
                      eq: block(md: T) -> bool) -> option::t<T> unsafe {
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
    let fname = fs::basename(full_path);
    let path = fs::dirname(full_path);
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

fn get_file_metadata(cx: @crate_ctxt, full_path: str) -> @metadata<file_md> {
    let cache = cx.llmetadata;
    alt cached_metadata::<@metadata<file_md>>(
        cache, FileDescriptorTag,
        {|md|
         md.data.path == full_path}) {
      option::some(md) { ret md; }
      option::none. {}
    }
    let fname = fs::basename(full_path);
    let path = fs::dirname(full_path);
    let unit_node = get_compile_unit_metadata(cx, full_path).node;
    let file_md = [lltag(FileDescriptorTag),
                   llstr(fname),
                   llstr(path),
                   unit_node];
    let val = llmdnode(file_md);
    let mdval = @{node: val, data: {path: full_path}};
    update_cache(cache, FileDescriptorTag, file_metadata(mdval));
    ret mdval;
}

fn get_block_metadata(cx: @block_ctxt) -> @metadata<block_md> {
    let cache = bcx_ccx(cx).llmetadata;
    let start = codemap::lookup_char_pos(bcx_ccx(cx).sess.get_codemap(),
                                         cx.sp.lo);
    let fname = start.filename;
    let end = codemap::lookup_char_pos(bcx_ccx(cx).sess.get_codemap(),
                                       cx.sp.hi);
    alt cached_metadata::<@metadata<block_md>>(
        cache, LexicalBlockTag,
        {|md| start == md.data.start && end == md.data.end}) {
      option::some(md) { ret md; }
      option::none. {}
    }
    let parent = alt cx.parent {
      trans_common::parent_none. { llnull() }
      trans_common::parent_some(bcx) {
        get_block_metadata(bcx).node
      }
    };
    let file_node = get_file_metadata(bcx_ccx(cx), fname);
    let unique_id = alt cache.find(LexicalBlockTag) {
      option::some(v) { vec::len(v) as int }
      option::none. { 0 }
    };
    let lldata = [lltag(LexicalBlockTag),
                  parent,
                  lli32(start.line as int),
                  lli32(start.col as int),
                  file_node.node,
                  lli32(unique_id)
                 ];
      let val = llmdnode(lldata);
      let mdval = @{node: val, data: {start: start, end: end}};
      update_cache(cache, LexicalBlockTag, block_metadata(mdval));
      ret mdval;
}

fn get_ty_metadata(cx: @crate_ctxt, t: ty::t, ty: @ast::ty) -> @metadata<tydesc_md> {
    let cache = cx.llmetadata;
    alt cached_metadata::<@metadata<tydesc_md>>(
        cache, BasicTypeDescriptorTag,
        {|md| ty::hash_ty(t) == ty::hash_ty(md.data.hash)}) {
      option::some(md) { ret md; }
      option::none. {}
    }
    let (name, size, flags) = alt ty.node {
      ast::ty_bool. { ("bool", 1, DW_ATE_boolean) }
      ast::ty_int. { ("int", 32, DW_ATE_signed) } //XXX machine-dependent?
      ast::ty_uint. { ("uint", 32, DW_ATE_unsigned) } //XXX machine-dependent?
      ast::ty_float. { ("float", 32, DW_ATE_float) } //XXX machine-dependent?
      ast::ty_machine(m) { alt m {
        ast::ty_i8. { ("i8", 1, DW_ATE_signed_char) }
        ast::ty_i16. { ("i16", 16, DW_ATE_signed) }
        ast::ty_i32. { ("i32", 32, DW_ATE_signed) }
        ast::ty_i64. { ("i64", 64, DW_ATE_signed) }
        ast::ty_u8. { ("u8", 8, DW_ATE_unsigned_char) }
        ast::ty_u16. { ("u16", 16, DW_ATE_unsigned) }
        ast::ty_u32. { ("u32", 32, DW_ATE_unsigned) }
        ast::ty_u64. { ("u64", 64, DW_ATE_unsigned) }
        ast::ty_f32. { ("f32", 32, DW_ATE_float) }
        ast::ty_f64. { ("f64", 64, DW_ATE_float) }
      } }
      ast::ty_char. { ("char", 32, DW_ATE_unsigned) }
    };
    let lldata = [lltag(BasicTypeDescriptorTag),
                  llunused(), //XXX scope context
                  llstr(name),
                  llnull(), //XXX basic types only
                  lli32(0), //XXX basic types only
                  lli64(size),
                  lli64(32), //XXX alignment?
                  lli64(0), //XXX offset?
                  lli32(flags)];
    let llnode = llmdnode(lldata);
    let mdval = @{node: llnode, data: {hash: ty::hash_ty(t)}};
    update_cache(cache, BasicTypeDescriptorTag, tydesc_metadata(mdval));
    llvm::LLVMAddNamedMetadataOperand(cx.llmod, as_buf("llvm.dbg.ty"),
                                      str::byte_len("llvm.dbg.ty"),
                                      llnode);
    ret mdval;
}

fn get_local_var_metadata(bcx: @block_ctxt, local: @ast::local)
    -> @metadata<local_var_md> unsafe {
    let cx = bcx_ccx(bcx);
    let cache = cx.llmetadata;
    alt cached_metadata::<@metadata<local_var_md>>(
        cache, AutoVariableTag, {|md| md.data.id == local.node.id}) {
      option::some(md) { ret md; }
      option::none. {}
    }
    let name = alt local.node.pat.node {
      ast::pat_bind(ident) { ident }
    };
    let loc = codemap::lookup_char_pos(cx.sess.get_codemap(),
                                       local.span.lo);
    let ty = trans::node_id_type(cx, local.node.id);
    let tymd = get_ty_metadata(cx, ty, local.node.ty);
    let filemd = get_file_metadata(cx, loc.filename);
    let blockmd = get_block_metadata(bcx);
    let lldata = [lltag(AutoVariableTag),
                  blockmd.node, //XXX block context (maybe subprogram if possible?)
                  llstr(name), // name
                  filemd.node,
                  lli32(loc.line as int), // line
                  tymd.node,
                  lli32(0), //XXX flags
                  llnull() // inline loc reference
                 ];
    let mdnode = llmdnode(lldata);
    let mdval = @{node: mdnode, data: {id: local.node.id}};
    update_cache(cache, AutoVariableTag, local_var_metadata(mdval));
    let llptr = alt bcx.fcx.lllocals.find(local.node.id) {
      option::some(local_mem(v)) { v }
      option::none. {
        alt bcx.fcx.lllocals.get(local.node.pat.id) {
          local_imm(v) { v }
        }
      }
    };
    let declargs = [llmdnode([llptr]), mdnode];
    trans_build::Call(bcx, cx.intrinsics.get("llvm.dbg.declare"),
                      declargs);
    llvm::LLVMAddNamedMetadataOperand(cx.llmod, as_buf("llvm.dbg.vars"),
                                      str::byte_len("llvm.dbg.vars"),
                                      mdnode);
    ret mdval;
}

fn update_source_pos(cx: @block_ctxt, s: codemap::span) {
    if !bcx_ccx(cx).sess.get_opts().debuginfo {
        ret;
    }
    cx.source_pos = option::some(
        codemap::lookup_char_pos(bcx_ccx(cx).sess.get_codemap(),
                                 s.lo)); //XXX maybe hi

}

fn reset_source_pos(cx: @block_ctxt) {
    cx.source_pos = option::none;
}

fn add_line_info(cx: @block_ctxt, llinstr: ValueRef) {
    if !bcx_ccx(cx).sess.get_opts().debuginfo ||
        option::is_none(cx.source_pos) {
        ret;
    }
    let loc = option::get(cx.source_pos);
    let blockmd = get_block_metadata(cx);
    let kind_id = llvm::LLVMGetMDKindID(as_buf("dbg"), str::byte_len("dbg"));
    let scopedata = [lli32(loc.line as int),
                     lli32(loc.col as int),
                     blockmd.node,
                     llnull()];
    let dbgscope = llmdnode(scopedata);
    llvm::LLVMSetMetadata(llinstr, kind_id, dbgscope);    
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
    let mangled = cx.item_symbols.get(item.id);
    let ret_ty = alt item.node {
      ast::item_fn(f, _) { f.decl.output }
    };
    let ty_node = alt ret_ty.node {
      ast::ty_nil. { llnull() }
      _ { get_ty_metadata(cx, ty::node_id_to_type(ccx_tcx(cx), item.id),
                          ret_ty).node }
    };
    let fn_metadata = [lltag(SubprogramTag),
                       llunused(),
                       file_node,
                       llstr(item.ident),
                       llstr(item.ident), //XXX fully-qualified C++ name
                       llstr(mangled), //XXX MIPS name?????
                       file_node,
                       lli32(loc.line as int),
                       ty_node,
                       lli1(false), //XXX static (check export)
                       lli1(true), // not extern
                       lli32(DW_VIRTUALITY_none), // virtual-ness
                       lli32(0i), //index into virt func
                       llnull(), // base type with vtbl
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