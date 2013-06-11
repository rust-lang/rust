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
use core::ptr;
use core::str::as_c_str;
use core::sys;
use core::vec;
use syntax::codemap::span;
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

pub type DebugContext = @mut _DebugContext;

struct _DebugContext {
    names: namegen,
    crate_file: ~str,
    builder: DIBuilderRef,
    curr_loc: (uint, uint),
    created_files: HashMap<~str, DIFile>,
    created_functions: HashMap<ast::node_id, DISubprogram>,
    created_blocks: HashMap<ast::node_id, DILexicalBlock>,
    created_types: HashMap<uint, DIType>
}

/** Create new DebugContext */
pub fn mk_ctxt(llmod: ModuleRef, crate: ~str) -> DebugContext {
    debug!("mk_ctxt");
    let builder = unsafe { llvm::LLVMDIBuilderCreate(llmod) };
    let dcx = @mut _DebugContext {
        names: new_namegen(),
        crate_file: crate,
        builder: builder,
        curr_loc: (0, 0),
        created_files: HashMap::new(),
        created_functions: HashMap::new(),
        created_blocks: HashMap::new(),
        created_types: HashMap::new(),
    };
    return dcx;
}

#[inline(always)]
fn dbg_cx(cx: &CrateContext) -> DebugContext
{
    return cx.dbg_cx.get();
}

fn create_DIArray(builder: DIBuilderRef, arr: &[DIDescriptor]) -> DIArray {
    return unsafe {
        llvm::LLVMDIBuilderGetOrCreateArray(builder, vec::raw::to_ptr(arr), arr.len() as u32)
    };
}

/** Create any deferred debug metadata nodes */
pub fn finalize(cx: @CrateContext) {
    debug!("finalize");
    create_compile_unit(cx);
    let dcx = dbg_cx(cx);
    unsafe {
        llvm::LLVMDIBuilderFinalize(dcx.builder);
        llvm::LLVMDIBuilderDispose(dcx.builder);
    };
}

fn create_compile_unit(cx: @CrateContext) {
    let crate_name: &str = dbg_cx(cx).crate_file;
    let work_dir = cx.sess.working_dir.to_str();
    let producer = fmt!("rustc version %s", env!("CFG_VERSION"));

    do as_c_str(crate_name) |crate_name| {
    do as_c_str(work_dir) |work_dir| {
    do as_c_str(producer) |producer| {
    do as_c_str("") |flags| {
    do as_c_str("") |split_name| { unsafe {
        llvm::LLVMDIBuilderCreateCompileUnit(dbg_cx(cx).builder,
            DW_LANG_RUST as c_uint, crate_name, work_dir, producer,
            cx.sess.opts.optimize != session::No,
            flags, 0, split_name);
    }}}}}};
}

fn create_file(cx: @CrateContext, full_path: &str) -> DIFile {
    let dcx = dbg_cx(cx);

    match dcx.created_files.find_equiv(&full_path) {
        Some(file_md) => return *file_md,
        None => ()
    }

    debug!("create_file: %s", full_path);

    let work_dir = cx.sess.working_dir.to_str();
    let file_name =
        if full_path.starts_with(work_dir) {
            full_path.slice(work_dir.len() + 1u, full_path.len())
        } else {
            full_path
        };

    let file_md =
        do as_c_str(file_name) |file_name| {
        do as_c_str(work_dir) |work_dir| { unsafe {
            llvm::LLVMDIBuilderCreateFile(dcx.builder, file_name, work_dir)
        }}};

    dcx.created_files.insert(full_path.to_owned(), file_md);
    return file_md;
}

fn span_start(cx: @CrateContext, span: span) -> codemap::Loc {
    return cx.sess.codemap.lookup_char_pos(span.lo);
}

fn create_block(bcx: block) -> DILexicalBlock {
    let mut bcx = bcx;
    let cx = bcx.ccx();
    let dcx = dbg_cx(cx);

    while bcx.node_info.is_none() {
        match bcx.parent {
          Some(b) => bcx = b,
          None => fail!()
        }
    }
    let span = bcx.node_info.get().span;
    let id = bcx.node_info.get().id;

    match dcx.created_blocks.find(&id) {
        Some(block) => return *block,
        None => ()
    }

    debug!("create_block: %s", bcx.sess().codemap.span_to_str(span));

    let parent = match bcx.parent {
        None => create_function(bcx.fcx),
        Some(b) => create_block(b)
    };
    let cx = bcx.ccx();
    let loc = span_start(cx, span);
    let file_md = create_file(cx, loc.file.name);

    let block_md = unsafe {
        llvm::LLVMDIBuilderCreateLexicalBlock(
            dcx.builder,
            parent, file_md,
            loc.line as c_uint, loc.col.to_uint() as c_uint)
    };

    dcx.created_blocks.insert(id, block_md);

    return block_md;
}

fn size_and_align_of(cx: @CrateContext, t: ty::t) -> (uint, uint) {
    let llty = type_of::type_of(cx, t);
    (machine::llsize_of_real(cx, llty),
     machine::llalign_of_pref(cx, llty))
}

fn create_basic_type(cx: @CrateContext, t: ty::t, span: span) -> DIType{
    let dcx = dbg_cx(cx);
    let ty_id = ty::type_id(t);
    match dcx.created_types.find(&ty_id) {
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

    dcx.created_types.insert(ty_id, ty_md);
    return ty_md;
}

fn create_pointer_type(cx: @CrateContext, t: ty::t, span: span, pointee: DIType) -> DIType {
    let (size, align) = size_and_align_of(cx, t);
    let name = ty_to_str(cx.tcx, t);
    let ptr_md = do as_c_str(name) |name| { unsafe {
        llvm::LLVMDIBuilderCreatePointerType(dbg_cx(cx).builder,
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
            llvm::LLVMDIBuilderCreateMemberType(dbg_cx(self.cx).builder,
                ptr::null(), name, self.file, line as c_uint,
                size * 8 as u64, align * 8 as u64, self.total_size as u64,
                0, ty)
            }};
        // XXX What about member alignment???
        self.members.push(mem_t);
        self.total_size += size * 8;
    }

    fn finalize(&self) -> DICompositeType {
        let dcx = dbg_cx(self.cx);
        let members_md = create_DIArray(dcx.builder, self.members);

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
    let loc = span_start(cx, span);
    let file_md = create_file(cx, loc.file.name);

    let mut scx = StructContext::create(cx, file_md, ty_to_str(cx.tcx, t), loc.line);
    for fields.each |field| {
        let field_t = field.mt.ty;
        let ty_md = create_ty(cx, field_t, span);
        let (size, align) = size_and_align_of(cx, field_t);
        scx.add_member(cx.sess.str_of(field.ident),
                   loc.line, size, align, ty_md);
    }
    return scx.finalize();
}

// returns (void* type as a ValueRef, size in bytes, align in bytes)
fn voidptr(cx: @CrateContext) -> (DIDerivedType, uint, uint) {
    let size = sys::size_of::<ValueRef>();
    let align = sys::min_align_of::<ValueRef>();
    let vp = do as_c_str("*void") |name| { unsafe {
            llvm::LLVMDIBuilderCreatePointerType(dbg_cx(cx).builder, ptr::null(),
                size*8 as u64, align*8 as u64, name)
        }};
    return (vp, size, align);
}

fn create_tuple(cx: @CrateContext, t: ty::t, elements: &[ty::t], span: span) -> DICompositeType {
    let dcx = dbg_cx(cx);
    let loc = span_start(cx, span);
    let file_md = create_file(cx, loc.file.name);

    let name = (cx.sess.str_of((dcx.names)("tuple"))).to_owned();
    let mut scx = StructContext::create(cx, file_md, name, loc.line);

    for elements.each |element| {
        let ty_md = create_ty(cx, *element, span);
        let (size, align) = size_and_align_of(cx, *element);
        scx.add_member("", loc.line, size, align, ty_md);
    }
    return scx.finalize();
}

fn create_boxed_type(cx: @CrateContext, contents: ty::t,
                     span: span, boxed: DIType) -> DICompositeType {
    let loc = span_start(cx, span);
    let file_md = create_file(cx, loc.file.name);
    let int_t = ty::mk_int();
    let refcount_type = create_basic_type(cx, int_t, span);
    let name = ty_to_str(cx.tcx, contents);

    let mut scx = StructContext::create(cx, file_md, fmt!("box<%s>", name), 0);
    scx.add_member("refcnt", 0, sys::size_of::<uint>(),
               sys::min_align_of::<uint>(), refcount_type);
    // the tydesc and other pointers should be irrelevant to the
    // debugger, so treat them as void* types
    let (vp, vpsize, vpalign) = voidptr(cx);
    scx.add_member("tydesc", 0, vpsize, vpalign, vp);
    scx.add_member("prev", 0, vpsize, vpalign, vp);
    scx.add_member("next", 0, vpsize, vpalign, vp);
    let (size, align) = size_and_align_of(cx, contents);
    scx.add_member("boxed", 0, size, align, boxed);
    return scx.finalize();
}

fn create_fixed_vec(cx: @CrateContext, vec_t: ty::t, elem_t: ty::t,
                    len: uint, span: span) -> DIType {
    let dcx = dbg_cx(cx);
    let elem_ty_md = create_ty(cx, elem_t, span);
    let loc = span_start(cx, span);
    let file_md = create_file(cx, loc.file.name);
    let (size, align) = size_and_align_of(cx, elem_t);

    let subrange = unsafe {
        llvm::LLVMDIBuilderGetOrCreateSubrange(dcx.builder, 0_i64, (len-1) as i64) };

    let subscripts = create_DIArray(dcx.builder, [subrange]);
    return unsafe {
        llvm::LLVMDIBuilderCreateVectorType(dcx.builder,
            size * len as u64, align as u64, elem_ty_md, subscripts)
    };
}

fn create_boxed_vec(cx: @CrateContext, vec_t: ty::t, elem_t: ty::t,
                    vec_ty_span: span) -> DICompositeType {
    let dcx = dbg_cx(cx);
    let loc = span_start(cx, vec_ty_span);
    let file_md = create_file(cx, loc.file.name);
    let elem_ty_md = create_ty(cx, elem_t, vec_ty_span);

    let mut vec_scx = StructContext::create(cx, file_md, ty_to_str(cx.tcx, vec_t), 0);

    let size_t_type = create_basic_type(cx, ty::mk_uint(), vec_ty_span);
    vec_scx.add_member("fill", 0, sys::size_of::<libc::size_t>(),
               sys::min_align_of::<libc::size_t>(), size_t_type);
    vec_scx.add_member("alloc", 0, sys::size_of::<libc::size_t>(),
               sys::min_align_of::<libc::size_t>(), size_t_type);
    let subrange = unsafe { llvm::LLVMDIBuilderGetOrCreateSubrange(dcx.builder, 0_i64, 0_i64) };
    let (arr_size, arr_align) = size_and_align_of(cx, elem_t);
    let name = fmt!("[%s]", ty_to_str(cx.tcx, elem_t));

    let subscripts = create_DIArray(dcx.builder, [subrange]);
    let data_ptr = unsafe { llvm::LLVMDIBuilderCreateVectorType(dcx.builder,
                arr_size as u64, arr_align as u64, elem_ty_md, subscripts) };
    vec_scx.add_member("data", 0, 0, // clang says the size should be 0
               sys::min_align_of::<u8>(), data_ptr);
    let vec_md = vec_scx.finalize();

    let mut box_scx = StructContext::create(cx, file_md, fmt!("box<%s>", name), 0);
    let int_t = ty::mk_int();
    let refcount_type = create_basic_type(cx, int_t, vec_ty_span);
    box_scx.add_member("refcnt", 0, sys::size_of::<uint>(),
               sys::min_align_of::<uint>(), refcount_type);
    let (vp, vpsize, vpalign) = voidptr(cx);
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
    let loc = span_start(cx, span);
    let file_md = create_file(cx, loc.file.name);
    let elem_ty_md = create_ty(cx, elem_t, span);
    let uint_type = create_basic_type(cx, ty::mk_uint(), span);
    let elem_ptr = create_pointer_type(cx, elem_t, span, elem_ty_md);

    let mut scx = StructContext::create(cx, file_md, ty_to_str(cx.tcx, vec_t), 0);
    let (_, ptr_size, ptr_align) = voidptr(cx);
    scx.add_member("vec", 0, ptr_size, ptr_align, elem_ptr);
    scx.add_member("length", 0, sys::size_of::<uint>(),
               sys::min_align_of::<uint>(), uint_type);
    return scx.finalize();
}

fn create_fn_ty(cx: @CrateContext, fn_ty: ty::t, inputs: ~[ty::t], output: ty::t,
                span: span) -> DICompositeType {
    let dcx = dbg_cx(cx);
    let loc = span_start(cx, span);
    let file_md = create_file(cx, loc.file.name);
    let (vp, _, _) = voidptr(cx);
    let output_md = create_ty(cx, output, span);
    let output_ptr_md = create_pointer_type(cx, output, span, output_md);
    let inputs_vals = do inputs.map |arg| { create_ty(cx, *arg, span) };
    let members = ~[output_ptr_md, vp] + inputs_vals;

    return unsafe {
        llvm::LLVMDIBuilderCreateSubroutineType(dcx.builder, file_md,
            create_DIArray(dcx.builder, members))
    };
}

fn create_ty(cx: @CrateContext, t: ty::t, span: span) -> DIType {
    let dcx = dbg_cx(cx);
    let ty_id = ty::type_id(t);
    match dcx.created_types.find(&ty_id) {
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

    dcx.created_types.insert(ty_id, ty_md);
    return ty_md;
}

pub fn create_local_var(bcx: block, local: @ast::local) -> DIVariable {
    let cx = bcx.ccx();
    let dcx = dbg_cx(cx);

    let ident = match local.node.pat.node {
      ast::pat_ident(_, pth, _) => ast_util::path_to_ident(pth),
      // FIXME this should be handled (#2533)
      _ => fail!("no single variable name for local")
    };
    let name: &str = cx.sess.str_of(ident);
    debug!("create_local_var: %s", name);

    let loc = span_start(cx, local.span);
    let ty = node_id_type(bcx, local.node.id);
    let tymd = create_ty(cx, ty, local.node.ty.span);
    let filemd = create_file(cx, loc.file.name);
    let context = match bcx.parent {
        None => create_function(bcx.fcx),
        Some(_) => create_block(bcx)
    };

    let var_md = do as_c_str(name) |name| { unsafe {
        llvm::LLVMDIBuilderCreateLocalVariable(
            dcx.builder, AutoVariableTag as u32,
            context, name, filemd,
            loc.line as c_uint, tymd, false, 0, 0)
        }};

    // FIXME(#6814) Should use `pat_util::pat_bindings` for pats like (a, b) etc
    let llptr = match bcx.fcx.lllocals.find_copy(&local.node.pat.id) {
        Some(v) => v,
        None => {
            bcx.tcx().sess.span_bug(
                local.span,
                fmt!("No entry in lllocals table for %?", local.node.id));
        }
    };

    set_debug_location(bcx, loc.line, loc.col.to_uint());
    unsafe {
        let instr = llvm::LLVMDIBuilderInsertDeclareAtEnd(dcx.builder, llptr, var_md, bcx.llbb);
        llvm::LLVMSetInstDebugLocation(trans::build::B(bcx), instr);
    }

    return var_md;
}

pub fn create_arg(bcx: block, arg: ast::arg, span: span) -> Option<DIVariable> {
    debug!("create_arg");
    if true {
        // FIXME(5848) create_arg disabled for now because "node_id_type(bcx, arg.id)" below blows
        // up: "error: internal compiler error: node_id_to_type: no type for node `arg (id=10)`"
        return None;
    }

    let fcx = bcx.fcx;
    let cx = *fcx.ccx;
    let dcx = dbg_cx(cx);

    let loc = span_start(cx, span);
    if "<intrinsic>" == loc.file.name {
        return None;
    }

    let ty = node_id_type(bcx, arg.id);
    let tymd = create_ty(cx, ty, arg.ty.span);
    let filemd = create_file(cx, loc.file.name);
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
                    // FIXME need to pass a real argument number
            }};

            let llptr = fcx.llargs.get_copy(&arg.id);
            unsafe {
                llvm::LLVMDIBuilderInsertDeclareAtEnd(dcx.builder, llptr, mdnode, bcx.llbb);
            }
            return Some(mdnode);
        }
        _ => {
            return None;
        }
    }
}

fn set_debug_location(bcx: block, line: uint, col: uint) {
    let blockmd = create_block(bcx);
    let elems = ~[C_i32(line as i32), C_i32(col as i32), blockmd, ptr::null()];
    unsafe {
        let dbg_loc = llvm::LLVMMDNode(vec::raw::to_ptr(elems), elems.len() as libc::c_uint);
        llvm::LLVMSetCurrentDebugLocation(trans::build::B(bcx), dbg_loc);
    }
}

pub fn update_source_pos(bcx: block, span: span) {
    if !bcx.sess().opts.debuginfo || (*span.lo == 0 && *span.hi == 0) {
        return;
    }

    debug!("update_source_pos: %s", bcx.sess().codemap.span_to_str(span));

    let cx = bcx.ccx();
    let loc = span_start(cx, span);
    let dcx = dbg_cx(cx);

    let loc = (loc.line, loc.col.to_uint());
    if  loc == dcx.curr_loc {
        return;
    }
    debug!("setting_location to %u %u", loc.first(), loc.second());
    dcx.curr_loc = loc;
    set_debug_location(bcx, loc.first(), loc.second());
}

pub fn create_function(fcx: fn_ctxt) -> DISubprogram {
    let cx = *fcx.ccx;
    let dcx = dbg_cx(cx);
    let fcx = &mut *fcx;
    let span = fcx.span.get();

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
            ((dcx.names)("fn"), decl.output, expr.id)
          }
          _ => fcx.ccx.sess.span_bug(expr.span,
                  "create_function: expected an expr_fn_block here")
        }
      }
      _ => fcx.ccx.sess.bug("create_function: unexpected sort of node")
    };

    match dcx.created_functions.find(&id) {
        Some(fn_md) => return *fn_md,
        None => ()
    }

    debug!("create_function: %s, %s", cx.sess.str_of(ident), cx.sess.codemap.span_to_str(span));

    let loc = span_start(cx, span);
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
        llvm::LLVMDIBuilderCreateSubroutineType(dcx.builder,
            file_md, create_DIArray(dcx.builder, [ret_ty_md]))
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

    dcx.created_functions.insert(id, fn_md);
    return fn_md;
}
