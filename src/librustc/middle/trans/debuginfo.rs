// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use driver::session;
use lib::llvm::llvm;
use lib::llvm::{ValueRef, ModuleRef, ContextRef};
use lib::llvm::debuginfo::*;
use middle::trans::common::*;
use middle::trans::machine;
use middle::trans::type_of;
use middle::trans;
use middle::ty;
use util::ppaux::ty_to_str;

use core::hashmap::HashMap;
use core::libc;
use core::libc::c_uint;
use core::str::as_c_str;
use syntax::codemap::span;
use syntax::parse::token::ident_interner;
use syntax::{ast, codemap, ast_util, ast_map};

static LLVMDebugVersion: int = (12 << 16);

static DW_LANG_RUST: int = 12; //0x9000;

static CompileUnitTag: int = 17;
static FileDescriptorTag: int = 41;
static SubprogramTag: int = 46;
static SubroutineTag: int = 21;
static BasicTypeDescriptorTag: int = 36;
static AutoVariableTag: int = 256;
static ArgVariableTag: int = 257;
static ReturnVariableTag: int = 258;
static LexicalBlockTag: int = 11;
static PointerTypeTag: int = 15;
static StructureTypeTag: int = 19;
static MemberTag: int = 13;
static ArrayTypeTag: int = 1;
static SubrangeTag: int = 33;

static DW_ATE_boolean: int = 0x02;
static DW_ATE_float: int = 0x04;
static DW_ATE_signed: int = 0x05;
static DW_ATE_signed_char: int = 0x06;
static DW_ATE_unsigned: int = 0x07;
static DW_ATE_unsigned_char: int = 0x08;

////////////////

pub struct DebugContext {
    //llmetadata: metadata_cache,
    names: namegen,
    crate_file: ~str,
    builder: DIBuilderRef,
    
    created_files: @mut HashMap<~str, DIFile>,
    created_functions: @mut HashMap<ast::node_id, DISubprogram>,
    created_blocks: @mut HashMap<ast::node_id, DILexicalBlock>,
    created_types: @mut HashMap<uint, DIType>
}

/** Create new DebugContext */
pub fn mk_ctxt(llmod: ModuleRef, crate: ~str, intr: @ident_interner) -> DebugContext {
    debug!("mk_ctxt");
    let builder = unsafe { llvm::DIBuilder_new(llmod) };
    DebugContext {
        //llmetadata: @mut HashMap::new(),
        names: new_namegen(intr),
        crate_file: crate,
        builder: builder,
        created_files: @mut HashMap::new(),
        created_functions: @mut HashMap::new(),
        created_blocks: @mut HashMap::new(),
        created_types: @mut HashMap::new(),
}
}

#[inline(always)]
fn get_builder(cx: @CrateContext) -> DIBuilderRef {
    let dbg_cx = cx.dbg_cx.get_ref();
    return dbg_cx.builder;
}

fn create_DIArray(builder: DIBuilderRef, arr: &[DIDescriptor]) -> DIArray {
    return unsafe { 
        llvm::DIBuilder_getOrCreateArray(builder, vec::raw::to_ptr(arr), arr.len() as u32) 
    };
}

/** Create any deferred debug metadata nodes */
pub fn finalize(cx: @CrateContext) {
    debug!("finalize");
    create_compile_unit(cx);
    unsafe {
        llvm::DIBuilder_finalize(get_builder(cx));
        llvm::DIBuilder_delete(get_builder(cx));
    };
}

fn filename_from_span(cx: @CrateContext, sp: codemap::span) -> ~str {
    /*bad*/copy cx.sess.codemap.lookup_char_pos(sp.lo).file.name
}

//fn filename_from_span<'cx>(cx: &'cx CrateContext, sp: codemap::span) -> &'cx str {
//    let fname: &str = cx.sess.codemap.lookup_char_pos(sp.lo).file.name;
//  return fname;
//}

fn get_file_path_and_dir(work_dir: &str, full_path: &str) -> (~str, ~str) {
    let full_path = 
        if str::starts_with(full_path, work_dir) {
            str::slice(full_path, str::len(work_dir) + 1u,
                       str::len(full_path)).to_owned()
        } else {
            full_path.to_owned()
    };
    
    return (full_path, work_dir.to_owned());
}

fn create_compile_unit(cx: @CrateContext) {
    let crate_name: &str = cx.dbg_cx.get_ref().crate_file;

    let (_, work_dir) = get_file_path_and_dir(
        cx.sess.working_dir.to_str(), crate_name);
        
    let producer = fmt!("rustc version %s", env!("CFG_VERSION"));

    do as_c_str(crate_name) |crate_name| {
    do as_c_str(work_dir) |work_dir| {
    do as_c_str(producer) |producer| {
    do as_c_str("") |flags| {
    do as_c_str("") |split_name| { unsafe {
        llvm::DIBuilder_createCompileUnit(get_builder(cx),
            DW_LANG_RUST as c_uint, crate_name, work_dir, producer,
            cx.sess.opts.optimize != session::No,
            flags, 0, split_name);
    }}}}}};
}

fn create_file(cx: @CrateContext, full_path: &str) -> DIFile {
    let mut dbg_cx = cx.dbg_cx.get_ref();

    match dbg_cx.created_files.find(&full_path.to_owned()) {
        Some(file_md) => return *file_md,
        None => ()
    }

    debug!("create_file: %s", full_path);

    let (file_path, work_dir) =
        get_file_path_and_dir(cx.sess.working_dir.to_str(),
                              full_path);

    let file_md =
        do as_c_str(file_path) |file_path| {
        do as_c_str(work_dir) |work_dir| { unsafe {
            llvm::DIBuilder_createFile(get_builder(cx), file_path, work_dir)
        }}};

    dbg_cx.created_files.insert(full_path.to_owned(), file_md);
    return file_md;
}

fn line_from_span(cm: @codemap::CodeMap, sp: span) -> uint {
    cm.lookup_char_pos(sp.lo).line
}

fn create_block(bcx: block) -> DILexicalBlock {
    let mut bcx = bcx;
    let mut dbg_cx = bcx.ccx().dbg_cx.get_ref();    

    while bcx.node_info.is_none() {
        match bcx.parent {
          Some(b) => bcx = b,
          None => fail!()
        }
    }
    let sp = bcx.node_info.get().span;
    let id = bcx.node_info.get().id;

    match dbg_cx.created_blocks.find(&id) {
        Some(block) => return *block,
        None => ()
    }

    debug!("create_block: %s", bcx.sess().codemap.span_to_str(sp));

    let start = bcx.sess().codemap.lookup_char_pos(sp.lo);
    let end = bcx.sess().codemap.lookup_char_pos(sp.hi);
    
    let parent = match bcx.parent {
        None => create_function(bcx.fcx),
        Some(b) => create_block(b)
    };

    let file_md = create_file(bcx.ccx(), start.file.name);
    
    let block_md = unsafe {
        llvm::LLVMDIBuilderCreateLexicalBlock(
            dcx.builder,
            parent, file_md,
            start.line.to_int() as c_uint, start.col.to_int() as c_uint) 
    };

    dbg_cx.created_blocks.insert(id, block_md);

    return block_md;
}

fn size_and_align_of(cx: @CrateContext, t: ty::t) -> (uint, uint) {
    let llty = type_of::type_of(cx, t);
    (machine::llsize_of_real(cx, llty),
     machine::llalign_of_pref(cx, llty))
}

fn create_basic_type(cx: @CrateContext, t: ty::t, span: span) -> DIType{
    let mut dbg_cx = cx.dbg_cx.get_ref();
    let ty_id = ty::type_id(t);
    match dbg_cx.created_types.find(&ty_id) {
        Some(ty_md) => return *ty_md,
        None => ()
    }

    debug!("create_basic_type: %?", ty::get(t));

    let (name, encoding) = match ty::get(t).sty {
        ty::ty_nil | ty::ty_bot => (~"uint", DW_ATE_unsigned),
        ty::ty_bool => (~"bool", DW_ATE_boolean),
        ty::ty_int(int_ty) => match int_ty {
            ast::ty_i => (~"int", DW_ATE_signed),
            ast::ty_char => (~"char", DW_ATE_signed_char),
            ast::ty_i8 => (~"i8", DW_ATE_signed),
            ast::ty_i16 => (~"i16", DW_ATE_signed),
            ast::ty_i32 => (~"i32", DW_ATE_signed),
            ast::ty_i64 => (~"i64", DW_ATE_signed)
        },
        ty::ty_uint(uint_ty) => match uint_ty {
            ast::ty_u => (~"uint", DW_ATE_unsigned),
            ast::ty_u8 => (~"u8", DW_ATE_unsigned),
            ast::ty_u16 => (~"u16", DW_ATE_unsigned),
            ast::ty_u32 => (~"u32", DW_ATE_unsigned),
            ast::ty_u64 => (~"u64", DW_ATE_unsigned)
        },
        ty::ty_float(float_ty) => match float_ty {
            ast::ty_f => (~"float", DW_ATE_float),
            ast::ty_f32 => (~"f32", DW_ATE_float),
            ast::ty_f64 => (~"f64", DW_ATE_float)
        },
        _ => cx.sess.bug(~"debuginfo::create_basic_type - t is invalid type")
    };

    let (size, align) = size_and_align_of(cx, t);
    let ty_md = do as_c_str(name) |name| { unsafe {
            llvm::LLVMDIBuilderCreateBasicType(
                dcx.builder, name,
                size * 8 as u64, align * 8 as u64, encoding as c_uint)
        }};

    dbg_cx.created_types.insert(ty_id, ty_md);
    return ty_md;
}

fn create_pointer_type(cx: @CrateContext, t: ty::t, span: span, pointee: DIType) -> DIType {
    let (size, align) = size_and_align_of(cx, t);
    let name = ty_to_str(cx.tcx, t);
    let ptr_md = do as_c_str(name) |name| { unsafe {
        llvm::DIBuilder_createPointerType(get_builder(cx), 
                pointee, size * 8 as u64, align * 8 as u64, name)
    }};
    return ptr_md;
}

struct StructContext {
    cx: @CrateContext,
    file: DIFile,
    name: ~str,
    line: uint,
    members: ~[DIDerivedType],
    total_size: uint,
    align: uint
}

impl StructContext {
    fn create(cx: @CrateContext, file: DIFile, name: ~str, line: uint) -> ~StructContext {
        let scx = ~StructContext {
            cx: cx,
            file: file,
            name: name,
            line: line,
            members: ~[],
            total_size: 0,
            align: 64 //XXX different alignment per arch?
        };
        return scx;
    }

    fn add_member(&mut self, name: &str, line: uint, size: uint, align: uint, ty: DIType) {
        let mem_t = do as_c_str(name) |name| { unsafe {
            llvm::DIBuilder_createMemberType(get_builder(self.cx), 
                ptr::null(), name, self.file, line as c_uint,
                size * 8 as u64, align * 8 as u64, self.total_size as u64, 
                0, ty)
            }};
        // XXX What about member alignment???
        self.members.push(mem_t);
        self.total_size += size * 8;
    }

    fn finalize(&self) -> DICompositeType {
        let members_md = create_DIArray(get_builder(self.cx), self.members);

        let struct_md =
            do as_c_str(self.name) |name| { unsafe {
                llvm::LLVMDIBuilderCreateStructType(
                    dcx.builder, ptr::null(), name, 
                    self.file, self.line as c_uint,
                    self.total_size as u64, self.align as u64, 0, ptr::null(),
                    members_md, 0, ptr::null())
            }};
        return struct_md;
    }
}

fn create_struct(cx: @CrateContext, t: ty::t, fields: ~[ty::field], span: span) -> DICompositeType {
    let fname = filename_from_span(cx, span);
    let file_md = create_file(cx, fname);
    let line = line_from_span(cx.sess.codemap, span);

    let mut scx = StructContext::create(cx, file_md, ty_to_str(cx.tcx, t), line);
    for fields.each |field| {
        let field_t = field.mt.ty;
        let ty_md = create_ty(cx, field_t, span);
        let (size, align) = size_and_align_of(cx, field_t);
        scx.add_member(cx.sess.str_of(field.ident),
                   line_from_span(cx.sess.codemap, span),
                   size, align, ty_md);
    }
    return scx.finalize();
}

// returns (void* type as a ValueRef, size in bytes, align in bytes)
fn voidptr() -> (DIDerivedType, uint, uint) {
    let size = sys::size_of::<ValueRef>();
    let align = sys::min_align_of::<ValueRef>();
    let vp = ptr::null();
    /*
    let vp = create_derived_type(PointerTypeTag, null, ~"", 0,
                                 size, align, 0, null);
    */
    return (vp, size, align);
}

fn create_tuple(cx: @CrateContext, t: ty::t, elements: &[ty::t], span: span) -> DICompositeType {
    let fname = filename_from_span(cx, span);
    let file_md = create_file(cx, fname);

    let name = (cx.sess.str_of((dcx.names)("tuple"))).to_owned();
    let mut scx = StructContext::create(cx, file_md, name, loc.line);

    for elements.each |element| {
        let ty_md = create_ty(cx, *element, span);
        let (size, align) = size_and_align_of(cx, *element);
        scx.add_member("", line_from_span(cx.sess.codemap, span),
                   size, align, ty_md);
    }
    return scx.finalize();
}

fn create_boxed_type(cx: @CrateContext, contents: ty::t,
                     span: span, boxed: DIType) -> DICompositeType {
    let fname = filename_from_span(cx, span);
    let file_md = create_file(cx, fname);
    let int_t = ty::mk_int();
    let refcount_type = create_basic_type(cx, int_t, span);
    let name = ty_to_str(cx.tcx, contents);

    let mut scx = StructContext::create(cx, file_md, fmt!("box<%s>", name), 0);
    scx.add_member("refcnt", 0, sys::size_of::<uint>(),
               sys::min_align_of::<uint>(), refcount_type);
    // the tydesc and other pointers should be irrelevant to the
    // debugger, so treat them as void* types
    let (vp, vpsize, vpalign) = voidptr();
    scx.add_member("tydesc", 0, vpsize, vpalign, vp);
    scx.add_member("prev", 0, vpsize, vpalign, vp);
    scx.add_member("next", 0, vpsize, vpalign, vp);
    let (size, align) = size_and_align_of(cx, contents);
    scx.add_member("boxed", 0, size, align, boxed);
    return scx.finalize();
}

fn create_fixed_vec(cx: @CrateContext, vec_t: ty::t, elem_t: ty::t,
                    len: uint, span: span) -> DIType {
    let elem_ty_md = create_ty(cx, elem_t, span);
    let fname = filename_from_span(cx, span);
    let file_md = create_file(cx, fname);
    let (size, align) = size_and_align_of(cx, elem_t);

    let subrange = unsafe {
        llvm::DIBuilder_getOrCreateSubrange(get_builder(cx), 0_i64, (len-1) as i64) };

    let subscripts = create_DIArray(get_builder(cx), [subrange]);
    return unsafe {
        llvm::DIBuilder_createVectorType(get_builder(cx), 
            size * len as u64, align as u64, elem_ty_md, subscripts) 
    };
}

fn create_boxed_vec(cx: @CrateContext, vec_t: ty::t, elem_t: ty::t,
                    vec_ty_span: codemap::span) -> DICompositeType {
    let fname = filename_from_span(cx, vec_ty_span);
    let file_md = create_file(cx, fname);
    let elem_ty_md = create_ty(cx, elem_t, vec_ty_span);

    let mut vec_scx = StructContext::create(cx, file_md, ty_to_str(cx.tcx, vec_t), 0);

    let size_t_type = create_basic_type(cx, ty::mk_uint(), vec_ty_span);
    vec_scx.add_member("fill", 0, sys::size_of::<libc::size_t>(),
               sys::min_align_of::<libc::size_t>(), size_t_type);
    vec_scx.add_member("alloc", 0, sys::size_of::<libc::size_t>(),
               sys::min_align_of::<libc::size_t>(), size_t_type);
    let subrange = unsafe { llvm::DIBuilder_getOrCreateSubrange(get_builder(cx), 0_i64, 0_i64) };
    let (arr_size, arr_align) = size_and_align_of(cx, elem_t);
    let name = fmt!("[%s]", ty_to_str(cx.tcx, elem_t));

    let subscripts = create_DIArray(get_builder(cx), [subrange]);
    let data_ptr = unsafe { llvm::DIBuilder_createVectorType(get_builder(cx), 
                arr_size as u64, arr_align as u64, elem_ty_md, subscripts) };
    vec_scx.add_member("data", 0, 0, // clang says the size should be 0
               sys::min_align_of::<u8>(), data_ptr);
    let vec_md = vec_scx.finalize();

    let mut box_scx = StructContext::create(cx, file_md, fmt!("box<%s>", name), 0);
    let int_t = ty::mk_int();
    let refcount_type = create_basic_type(cx, int_t, vec_ty_span);
    box_scx.add_member("refcnt", 0, sys::size_of::<uint>(),
               sys::min_align_of::<uint>(), refcount_type);
    let (vp, vpsize, vpalign) = voidptr();
    box_scx.add_member("tydesc", 0, vpsize, vpalign, vp);
    box_scx.add_member("prev", 0, vpsize, vpalign, vp);
    box_scx.add_member("next", 0, vpsize, vpalign, vp);
    let size = 2 * sys::size_of::<int>();
    let align = sys::min_align_of::<int>();
    box_scx.add_member("boxed", 0, size, align, vec_md);
    let mdval = box_scx.finalize();
    return mdval;
}

fn create_vec_slice(cx: @CrateContext, vec_t: ty::t, elem_t: ty::t, span: span) -> DICompositeType {
    let fname = filename_from_span(cx, span);
    let file_md = create_file(cx, fname);
    let elem_ty_md = create_ty(cx, elem_t, span);
    let uint_type = create_basic_type(cx, ty::mk_uint(), span);
    let elem_ptr = create_pointer_type(cx, elem_t, span, elem_ty_md);

    let mut scx = StructContext::create(cx, file_md, ty_to_str(cx.tcx, vec_t), 0);
    let (_, ptr_size, ptr_align) = voidptr();
    scx.add_member("vec", 0, ptr_size, ptr_align, elem_ptr);
    scx.add_member("length", 0, sys::size_of::<uint>(),
               sys::min_align_of::<uint>(), uint_type);
    return scx.finalize();
}

fn create_fn_ty(cx: @CrateContext, fn_ty: ty::t, inputs: ~[ty::t], output: ty::t,
                span: span) -> DICompositeType {
    let fname = filename_from_span(cx, span);
    let file_md = create_file(cx, fname);
    let (vp, _, _) = voidptr();
    let output_md = create_ty(cx, output, span);
    let output_ptr_md = create_pointer_type(cx, output, span, output_md);
    let inputs_vals = do inputs.map |arg| { create_ty(cx, *arg, span) };
    let members = ~[output_ptr_md, vp] + inputs_vals;

    return unsafe {
        llvm::DIBuilder_createSubroutineType(get_builder(cx), file_md, 
            create_DIArray(get_builder(cx), members)) 
    };
}

fn create_ty(cx: @CrateContext, t: ty::t, span: span) -> DIType {
    let mut dbg_cx = cx.dbg_cx.get_ref();
    let ty_id = ty::type_id(t);
    match dbg_cx.created_types.find(&ty_id) {
        Some(ty_md) => return *ty_md,
        None => ()
    }

    debug!("create_ty: %?", ty::get(t));

    let sty = copy ty::get(t).sty;
    let ty_md = match sty {
        ty::ty_nil | ty::ty_bot | ty::ty_bool | ty::ty_int(_) | ty::ty_uint(_)
        | ty::ty_float(_) => create_basic_type(cx, t, span),
        ty::ty_estr(ref vstore) => {
            let i8_t = ty::mk_i8();
            match *vstore {
                ty::vstore_fixed(len) => {
                    create_fixed_vec(cx, t, i8_t, len + 1, span)
                },
                ty::vstore_uniq | ty::vstore_box => {
                    let box_md = create_boxed_vec(cx, t, i8_t, span);
                    create_pointer_type(cx, t, span, box_md)
                }
                ty::vstore_slice(_region) => {
                    create_vec_slice(cx, t, i8_t, span)
                }
            }
        },
        ty::ty_enum(_did, ref _substs) => {
            cx.sess.span_bug(span, "debuginfo for enum NYI")
        }
        ty::ty_box(ref mt) | ty::ty_uniq(ref mt) => {
            let boxed = create_ty(cx, mt.ty, span);
            let box_md = create_boxed_type(cx, mt.ty, span, boxed);
            create_pointer_type(cx, t, span, box_md)
        },
        ty::ty_evec(ref mt, ref vstore) => {
            match *vstore {
                ty::vstore_fixed(len) => {
                    create_fixed_vec(cx, t, mt.ty, len, span)
                },
                ty::vstore_uniq | ty::vstore_box => {
                    let box_md = create_boxed_vec(cx, t, mt.ty, span);
                    create_pointer_type(cx, t, span, box_md)
                },
                ty::vstore_slice(_region) => {
                    create_vec_slice(cx, t, mt.ty, span)
                }
            }
        },
        ty::ty_ptr(ref mt) => {
            let pointee = create_ty(cx, mt.ty, span);
            create_pointer_type(cx, t, span, pointee)
        },
        ty::ty_rptr(ref _region, ref _mt) => {
            cx.sess.span_bug(span, "debuginfo for rptr NYI")
        },
        ty::ty_bare_fn(ref barefnty) => {
            let inputs = barefnty.sig.inputs.map(|a| *a);
            let output = barefnty.sig.output;
            create_fn_ty(cx, t, inputs, output, span)
        },
        ty::ty_closure(ref _closurety) => {
            cx.sess.span_bug(span, "debuginfo for closure NYI")
        },
        ty::ty_trait(_did, ref _substs, ref _vstore, _) => {
            cx.sess.span_bug(span, "debuginfo for trait NYI")
        },
        ty::ty_struct(did, ref substs) => {
            let fields = ty::struct_fields(cx.tcx, did, substs);
            create_struct(cx, t, fields, span)
        },
        ty::ty_tup(ref elements) => {
            create_tuple(cx, t, *elements, span)
        },
        _ => cx.sess.bug(~"debuginfo: unexpected type in create_ty")
    };

    dbg_cx.created_types.insert(ty_id, ty_md);
    return ty_md;
}

pub fn create_local_var(bcx: block, local: @ast::local) -> DIVariable {
    debug!("create_local_var");
    let cx = bcx.ccx();
    /*
    let cache = get_cache(cx);
    let tg = AutoVariableTag;
    match cached_metadata::<@Metadata<LocalVarMetadata>>(
        cache, tg, |md| md.data.id == local.node.id) {
      option::Some(md) => return md,
      option::None => ()
    }
    */

    let name = match local.node.pat.node {
      ast::pat_ident(_, pth, _) => ast_util::path_to_ident(pth),
      // FIXME this should be handled (#2533)
      _ => fail!("no single variable name for local")
    };
    let name: &str = cx.sess.str_of(ident);
    debug!("create_local_var: %s", name);

    let loc = span_start(cx, local.span);
    let ty = node_id_type(bcx, local.node.id);
    let tymd = create_ty(cx, ty, local.node.ty.span);
    let filemd = create_file(cx, /*bad*/copy loc.file.name);
    let context = match bcx.parent {
        None => create_function(bcx.fcx),
        Some(_) => create_block(bcx)
    };

    let mdval = do as_c_str(*cx.sess.str_of(name)) |name| { unsafe {
        llvm::DIBuilder_createLocalVariable(get_builder(cx), AutoVariableTag as u32,
                 ptr::null(), name, filemd, loc.line as c_uint, tymd, false, 0, 0)
        }};

    let llptr = match bcx.fcx.lllocals.find(&local.node.id) {
      option::Some(&local_mem(v)) => v,
      option::Some(_) => {
        bcx.tcx().sess.span_bug(local.span, "local is bound to something weird");
        }
      option::None => {
        match bcx.fcx.lllocals.get_copy(&local.node.pat.id) {
          local_imm(v) => v,
          _ => bcx.tcx().sess.span_bug(local.span, "local is bound to something weird")
    }
      }
    };
    /*
    llvm::DIBuilder_insertDeclare(get_builder(cx), llptr, mdval, 

    let declargs = ~[llmdnode(~[llptr]), mdnode];
    trans::build::Call(bcx, *cx.intrinsics.get(&~"llvm.dbg.declare"),
                       declargs);
    */
    return mdval;
    }

pub fn create_arg(bcx: block, arg: ast::arg, sp: span) -> Option<DIVariable> {
    debug!("create_arg");
    let fcx = bcx.fcx, cx = *fcx.ccx;
    /*
    let cache = get_cache(cx);
    let tg = ArgVariableTag;
    match cached_metadata::<@Metadata<ArgumentMetadata>>(
        cache, ArgVariableTag, |md| md.data.id == arg.id) {
      option::Some(md) => return Some(md),
      option::None => ()
    }
    */

    let loc = cx.sess.codemap.lookup_char_pos(sp.lo);
    if "<intrinsic>" == loc.file.name {
        return None;
    }
    let ty = node_id_type(bcx, arg.id);
    let tymd = create_ty(cx, ty, arg.ty.span);
    let filemd = create_file(cx, /*bad*/copy loc.file.name);
    let context = create_function(bcx.fcx);

    match arg.pat.node {
        ast::pat_ident(_, path, _) => {
            // XXX: This is wrong; it should work for multiple bindings.
            let ident = path.idents.last();
            let name: &str = cx.sess.str_of(*ident);
            let mdnode = do as_c_str(name) |name| { unsafe {
                llvm::LLVMDIBuilderCreateLocalVariable(dcx.builder,
                    ArgVariableTag as u32, context, name,
                    filemd, loc.line as c_uint, tymd, false, 0, 0)
                    // XXX need to pass a real argument number
            }};

            let llptr = match fcx.llargs.get_copy(&arg.id) {
              local_mem(v) | local_imm(v) => v,
            };
            
            /*
            llvm::DIBuilder_insertDeclare(get_builder(cx), mdnode, llptr, mdnode
            */
            
            return Some(mdnode);
        }
        _ => {
            return None;
        }
    }
}

fn create_debug_loc(line: int, col: int, scope: DIScope) -> DILocation {
    let elems = ~[C_i32(line as i32), C_i32(col as i32), scope, ptr::null()];
    unsafe {
        return llvm::LLVMMDNode(vec::raw::to_ptr(elems), elems.len() as libc::c_uint);
    }
}

    let cm = bcx.sess().codemap;
    let blockmd = create_block(bcx);
    let loc = cm.lookup_char_pos(sp.lo);
    let dbgscope = create_debug_loc(loc.line.to_int(), loc.col.to_int(), blockmd);
    unsafe {
        llvm::LLVMSetCurrentDebugLocation(trans::build::B(bcx), dbgscope);
    }
}

pub fn create_function(fcx: fn_ctxt) -> DISubprogram {
    let cx = *fcx.ccx;
    let mut dbg_cx = cx.dbg_cx.get_ref();
    let fcx = &mut *fcx;
    let sp = fcx.span.get();

    let (ident, ret_ty, id) = match cx.tcx.items.get_copy(&fcx.id) {
      ast_map::node_item(item, _) => {
        match item.node {
          ast::item_fn(ref decl, _, _, _, _) => {
            (item.ident, decl.output, item.id)
          }
          _ => fcx.ccx.sess.span_bug(item.span, "create_function: item bound to non-function")
        }
      }
      ast_map::node_method(method, _, _) => {
          (method.ident, method.decl.output, method.id)
      }
      ast_map::node_expr(expr) => {
        match expr.node {
          ast::expr_fn_block(ref decl, _) => {
            ((dbg_cx.names)("fn"), decl.output, expr.id)
          }
          _ => fcx.ccx.sess.span_bug(expr.span,
                  "create_function: expected an expr_fn_block here")
        }
      }
      _ => fcx.ccx.sess.bug("create_function: unexpected sort of node")
    };

    match dbg_cx.created_functions.find(&id) {
        Some(fn_md) => return *fn_md,
        None => ()
    }

    debug!("create_function: %s, %s", cx.sess.str_of(ident), cx.sess.codemap.span_to_str(span));

    let loc = cx.sess.codemap.lookup_char_pos(sp.lo);
    let file_md = create_file(cx, loc.file.name);

    let ret_ty_md = if cx.sess.opts.extra_debuginfo {
        match ret_ty.node {
          ast::ty_nil => ptr::null(),
          _ => create_ty(cx, ty::node_id_to_type(cx.tcx, id),
                         ret_ty.span)
        }
    } else {
        ptr::null()
    };

    let fn_ty = unsafe {
        llvm::DIBuilder_createSubroutineType(get_builder(cx),
            file_md, create_DIArray(get_builder(cx), [ret_ty_md]))
        };

    let fn_md =
        do as_c_str(cx.sess.str_of(ident)) |name| {
        do as_c_str(cx.sess.str_of(ident)) |linkage| { unsafe {
            llvm::LLVMDIBuilderCreateFunction(
                dcx.builder,
                file_md,
                name, linkage,
                file_md, loc.line as c_uint,
                fn_ty, false, true,
                loc.line as c_uint,
                FlagPrototyped as c_uint,
                cx.sess.opts.optimize != session::No,
                fcx.llfn, ptr::null(), ptr::null())
            }}};

    dbg_cx.created_functions.insert(id, fn_md);
    return fn_md;
}
