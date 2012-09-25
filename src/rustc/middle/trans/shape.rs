// A "shape" is a compact encoding of a type that is used by interpreted glue.
// This substitutes for the runtime tags used by e.g. MLs.

use lib::llvm::llvm;
use lib::llvm::{True, False, ModuleRef, TypeRef, ValueRef};
use driver::session;
use driver::session::session;
use trans::base;
use middle::trans::common::*;
use middle::trans::machine::*;
use back::abi;
use middle::ty;
use middle::ty::field;
use syntax::ast;
use syntax::ast_util::dummy_sp;
use syntax::util::interner;
use util::ppaux::ty_to_str;
use syntax::codemap::span;
use dvec::DVec;

use std::map::HashMap;
use option::is_some;

use ty_ctxt = middle::ty::ctxt;

type nominal_id_ = {did: ast::def_id, parent_id: Option<ast::def_id>,
                    tps: ~[ty::t]};
type nominal_id = @nominal_id_;

impl nominal_id_ : core::cmp::Eq {
    pure fn eq(other: &nominal_id_) -> bool {
        if self.did != other.did ||
            self.parent_id != other.parent_id {
            false
        } else {
            do vec::all2(self.tps, other.tps) |m_tp, n_tp| {
                ty::type_id(m_tp) == ty::type_id(n_tp)
            }
        }
    }
    pure fn ne(other: &nominal_id_) -> bool {
        ! (self == *other)
    }
}

impl nominal_id_ : to_bytes::IterBytes {
    pure fn iter_bytes(lsb0: bool, f: to_bytes::Cb) {
        to_bytes::iter_bytes_2(&self.did, &self.parent_id, lsb0, f);
        for self.tps.each |t| {
            ty::type_id(*t).iter_bytes(lsb0, f);
        }
    }
}

fn mk_nominal_id(tcx: ty::ctxt, did: ast::def_id,
                 parent_id: Option<ast::def_id>,
                 tps: ~[ty::t]) -> nominal_id {
    let tps_norm = tps.map(|t| ty::normalize_ty(tcx, *t));
    @{did: did, parent_id: parent_id, tps: tps_norm}
}

fn new_nominal_id_hash<T: Copy>() -> HashMap<nominal_id, T> {
    return HashMap();
}

type enum_data = {did: ast::def_id, substs: ty::substs};

type ctxt =
    {mut next_tag_id: u16,
     pad: u16,
     tag_id_to_index: HashMap<nominal_id, u16>,
     tag_order: DVec<enum_data>,
     resources: interner::interner<nominal_id>,
     llshapetablesty: TypeRef,
     llshapetables: ValueRef};

const shape_u8: u8 = 0u8;
const shape_u16: u8 = 1u8;
const shape_u32: u8 = 2u8;
const shape_u64: u8 = 3u8;
const shape_i8: u8 = 4u8;
const shape_i16: u8 = 5u8;
const shape_i32: u8 = 6u8;
const shape_i64: u8 = 7u8;
const shape_f32: u8 = 8u8;
const shape_f64: u8 = 9u8;
const shape_box: u8 = 10u8;
const shape_enum: u8 = 12u8;
const shape_struct: u8 = 17u8;
const shape_box_fn: u8 = 18u8;
const shape_res: u8 = 20u8;
const shape_uniq: u8 = 22u8;
const shape_opaque_closure_ptr: u8 = 23u8; // the closure itself.
const shape_uniq_fn: u8 = 25u8;
const shape_stack_fn: u8 = 26u8;
const shape_bare_fn: u8 = 27u8;
const shape_tydesc: u8 = 28u8;
const shape_send_tydesc: u8 = 29u8;
const shape_rptr: u8 = 31u8;
const shape_fixedvec: u8 = 32u8;
const shape_slice: u8 = 33u8;
const shape_unboxed_vec: u8 = 34u8;

fn mk_global(ccx: @crate_ctxt, name: ~str, llval: ValueRef, internal: bool) ->
   ValueRef {
    let llglobal =
        str::as_c_str(name,
                      |buf| {
                        lib::llvm::llvm::LLVMAddGlobal(ccx.llmod,
                                                       val_ty(llval), buf)
                    });
    lib::llvm::llvm::LLVMSetInitializer(llglobal, llval);
    lib::llvm::llvm::LLVMSetGlobalConstant(llglobal, True);

    if internal {
        lib::llvm::SetLinkage(llglobal, lib::llvm::InternalLinkage);
    }

    return llglobal;
}


// Computes a set of variants of a enum that are guaranteed to have size and
// alignment at least as large as any other variant of the enum. This is an
// important performance optimization.

fn round_up(size: u16, align: u8) -> u16 {
    assert (align >= 1u8);
    let alignment = align as u16;
    return size - 1u16 + alignment & !(alignment - 1u16);
}

type size_align = {size: u16, align: u8};

enum enum_kind {
    tk_unit,    // 1 variant, no data
    tk_enum,    // N variants, no data
    tk_newtype, // 1 variant, data
    tk_complex  // N variants, no data
}

fn enum_kind(ccx: @crate_ctxt, did: ast::def_id) -> enum_kind {
    let variants = ty::enum_variants(ccx.tcx, did);
    if vec::any(*variants, |v| vec::len(v.args) > 0u) {
        if vec::len(*variants) == 1u { tk_newtype }
        else { tk_complex }
    } else {
        if vec::len(*variants) <= 1u { tk_unit }
        else { tk_enum }
    }
}

// Returns the code corresponding to the pointer size on this architecture.
fn s_int(tcx: ty_ctxt) -> u8 {
    return match tcx.sess.targ_cfg.arch {
        session::arch_x86 => shape_i32,
        session::arch_x86_64 => shape_i64,
        session::arch_arm => shape_i32
    };
}

fn s_uint(tcx: ty_ctxt) -> u8 {
    return match tcx.sess.targ_cfg.arch {
        session::arch_x86 => shape_u32,
        session::arch_x86_64 => shape_u64,
        session::arch_arm => shape_u32
    };
}

fn s_float(tcx: ty_ctxt) -> u8 {
    return match tcx.sess.targ_cfg.arch {
        session::arch_x86 => shape_f64,
        session::arch_x86_64 => shape_f64,
        session::arch_arm => shape_f64
    };
}

fn s_variant_enum_t(tcx: ty_ctxt) -> u8 {
    return s_int(tcx);
}

fn s_tydesc(_tcx: ty_ctxt) -> u8 {
    return shape_tydesc;
}

fn s_send_tydesc(_tcx: ty_ctxt) -> u8 {
    return shape_send_tydesc;
}

fn mk_ctxt(llmod: ModuleRef) -> ctxt {
    let llshapetablesty = trans::common::T_named_struct(~"shapes");
    let llshapetables = str::as_c_str(~"shapes", |buf| {
        lib::llvm::llvm::LLVMAddGlobal(llmod, llshapetablesty, buf)
    });

    return {mut next_tag_id: 0u16,
         pad: 0u16,
         tag_id_to_index: new_nominal_id_hash(),
         tag_order: DVec(),
         resources: interner::mk(),
         llshapetablesty: llshapetablesty,
         llshapetables: llshapetables};
}

fn add_bool(&dest: ~[u8], val: bool) {
    dest += ~[if val { 1u8 } else { 0u8 }];
}

fn add_u16(&dest: ~[u8], val: u16) {
    dest += ~[(val & 0xffu16) as u8, (val >> 8u16) as u8];
}

fn add_substr(&dest: ~[u8], src: ~[u8]) {
    add_u16(dest, vec::len(src) as u16);
    dest += src;
}

