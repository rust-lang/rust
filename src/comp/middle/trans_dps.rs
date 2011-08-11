// Translates individual functions in the completed AST to the LLVM IR, using
// destination-passing style.

import back::abi;
import back::link;
import lib::llvm::llvm;
import llvm::TypeRef;
import llvm::ValueRef;
import middle::trans_common;
import middle::ty;
import syntax::ast;
import syntax::codemap::span;
import util::ppaux;
import trans_common::*;
import std::ivec;
import std::option::none;
import std::option::some;
import std::str;
import std::uint;

import LLFalse = lib::llvm::False;
import LLTrue = lib::llvm::True;
import ll = lib::llvm;
import lltype_of = trans_common::val_ty;
import option = std::option::t;
import tc = trans_common;
import type_of_node = trans::node_id_type;


// LLVM utilities

fn llelement_type(llty: TypeRef) -> TypeRef {
    lib::llvm::llvm::LLVMGetElementType(llty)
}

fn llalign_of(ccx: &@crate_ctxt, llty: TypeRef) -> uint {
    ret llvm::LLVMPreferredAlignmentOfType(ccx.td.lltd, llty);
}

fn llsize_of(ccx: &@crate_ctxt, llty: TypeRef) -> uint {
    ret llvm::LLVMStoreSizeOfType(ccx.td.lltd, llty);
}

fn mk_const(ccx: &@crate_ctxt, name: &str, exported: bool, llval: ValueRef) ->
   ValueRef {
    let llglobal =
        llvm::LLVMAddGlobal(ccx.llmod, tc::val_ty(llval), str::buf(name));

    llvm::LLVMSetInitializer(llglobal, llval);
    llvm::LLVMSetGlobalConstant(llglobal, LLTrue);

    if !exported {
        llvm::LLVMSetLinkage(llglobal,
                             lib::llvm::LLVMInternalLinkage as llvm::Linkage);
    }

    ret llglobal;
}


// Type utilities

fn size_of(ccx: &@crate_ctxt, sp: &span, t: ty::t) -> uint {
    if ty::type_has_dynamic_size(ccx.tcx, t) {
        ccx.sess.bug("trans_dps::size_of() called on a type with dynamic " +
                         "size");
    }
    ret llsize_of(ccx, trans::type_of_inner(ccx, sp, t));
}


// Destination utilities

tag dest {
    dst_nil; // Unit destination; ignore.

    dst_imm(@mutable option[ValueRef]); // Fill with an immediate value.

    dst_alias(@mutable option[ValueRef]); // Fill with an alias pointer.

    dst_copy(ValueRef); // Copy to the given address.

    dst_move(ValueRef); // Move to the given address.
}

fn dest_imm(tcx: &ty::ctxt, t: ty::t) -> dest {
    if ty::type_is_nil(tcx, t) { dst_nil } else { dst_imm(@mutable none) }
}

fn dest_alias(tcx: &ty::ctxt, t: ty::t) -> dest {
    if ty::type_is_nil(tcx, t) { dst_nil } else { dst_alias(@mutable none) }
}

fn dest_copy(tcx: &ty::ctxt, llptr: ValueRef, t: ty::t) -> dest {
    if ty::type_is_nil(tcx, t) { dst_nil } else { dst_copy(llptr) }
}

fn dest_move(tcx: &ty::ctxt, llptr: ValueRef, t: ty::t) -> dest {
    if ty::type_is_nil(tcx, t) { dst_nil } else { dst_move(llptr) }
}

// Invariant: the type of the destination must be structural (non-immediate).
fn dest_ptr(dest: &dest) -> ValueRef {
    alt dest {
      dst_nil. { fail "nil dest in dest_ptr" }
      dst_imm(_) { fail "immediate dest in dest_ptr" }
      dst_alias(box) {
        alt *box {
          none. { fail "alias wasn't filled in prior to dest_ptr" }
          some(llval) { llval }
        }
      }
      dst_copy(llptr) { llptr }
      dst_move(llptr) { llptr }
    }
}

fn dest_llval(dest: &dest) -> ValueRef {
    alt dest {
      dst_nil. { ret tc::C_nil(); }
      dst_imm(box) {
        alt *box {
          none. { fail "immediate wasn't filled in prior to dest_llval"; }
          some(llval) { ret llval; }
        }
      }
      dst_alias(box) {
        alt *box {
          none. { fail "alias wasn't filled in prior to dest_llval"; }
          some(llval) { ret llval; }
        }
      }
      dst_copy(llptr) { ret llptr; }
      dst_move(llptr) { ret llptr; }
    }
}

fn dest_is_alias(dest: &dest) -> bool {
    alt dest { dst_alias(_) { true } _ { false } }
}


// Common operations

fn memmove(bcx: &@block_ctxt, lldestptr: ValueRef, llsrcptr: ValueRef,
           llsz: ValueRef) {
    let lldestty = llelement_type(tc::val_ty(lldestptr));
    let llsrcty = llelement_type(tc::val_ty(llsrcptr));
    let dest_align = llalign_of(bcx_ccx(bcx), lldestty);
    let src_align = llalign_of(bcx_ccx(bcx), llsrcty);
    let align = uint::min(dest_align, src_align);
    let llfn = bcx_ccx(bcx).intrinsics.get("llvm.memmove.p0i8.p0i8.i32");
    let lldestptr_i8 =
        bcx.build.PointerCast(lldestptr, tc::T_ptr(tc::T_i8()));
    let llsrcptr_i8 = bcx.build.PointerCast(llsrcptr, tc::T_ptr(tc::T_i8()));
    bcx.build.Call(llfn,
                   ~[lldestptr_i8, llsrcptr_i8, llsz, tc::C_uint(align),
                     tc::C_bool(false)]);
}

// If "cast" is true, casts dest appropriately before the store.
fn store_imm(bcx: &@block_ctxt, dest: &dest, llsrc: ValueRef, cast: bool) ->
   @block_ctxt {
    alt dest {
      dst_nil. {/* no-op */ }
      dst_imm(box) {
        assert (std::option::is_none(*box));
        *box = some(llsrc);
      }
      dst_alias(box) {
        bcx_ccx(bcx).sess.unimpl("dst_alias spill in store_imm");
      }
      dst_copy(lldestptr_orig) | dst_move(lldestptr_orig) {
        let lldestptr = lldestptr_orig;
        if cast {
            lldestptr =
                bcx.build.PointerCast(lldestptr, tc::T_ptr(lltype_of(llsrc)));
        }
        bcx.build.Store(llsrc, lldestptr);
      }
    }
    ret bcx;
}

fn store_ptr(bcx: &@block_ctxt, dest: &dest, llsrcptr: ValueRef) ->
   @block_ctxt {
    alt dest {
      dst_nil. {/* no-op */ }
      dst_imm(box) {
        assert (std::option::is_none(*box));
        *box = some(bcx.build.Load(llsrcptr));
      }
      dst_alias(box) {
        assert (std::option::is_none(*box));
        *box = some(llsrcptr);
      }
      dst_copy(lldestptr) | dst_move(lldestptr) {
        let llsrcty = llelement_type(tc::val_ty(llsrcptr));
        let llsz = tc::C_uint(llsize_of(bcx_ccx(bcx), llsrcty));
        memmove(bcx, lldestptr, llsrcptr, llsz);
        ret bcx;
      }
    }
    ret bcx;
}

// Allocates a value of the given LLVM size on either the task heap or the
// shared heap.
//
// TODO: This should *not* use destination-passing style, because doing so
// makes callers incur an extra load.
tag heap { hp_task; hp_shared; }
fn malloc(bcx: &@block_ctxt, lldest: ValueRef, heap: heap,
          llcustom_size_opt: option[ValueRef]) -> @block_ctxt {
    let llptrty = llelement_type(lltype_of(lldest));
    let llty = llelement_type(llptrty);

    let lltydescptr = tc::C_null(tc::T_ptr(bcx_ccx(bcx).tydesc_type));

    let llsize;
    alt llcustom_size_opt {
      none. { llsize = trans::llsize_of(llty); }
      some(llcustom_size) { llsize = llcustom_size; }
    }

    let llupcall;
    alt heap {
      hp_task. { llupcall = bcx_ccx(bcx).upcalls.malloc; }
      hp_shared. { llupcall = bcx_ccx(bcx).upcalls.shared_malloc; }
    }

    let llresult =
        bcx.build.Call(llupcall,
                       ~[bcx_fcx(bcx).lltaskptr, llsize, lltydescptr]);
    llresult = bcx.build.PointerCast(llresult, llptrty);
    bcx.build.Store(llresult, lldest);
    ret bcx;
}

// If the supplied destination is an alias, spills to a temporary. Returns the
// new destination.
fn spill_alias(cx: &@block_ctxt, dest: &dest, t: ty::t) ->
   {bcx: @block_ctxt, dest: dest} {
    let bcx = cx;
    alt dest {
      dst_alias(box) {
        // TODO: Mark the alias as needing a cleanup.
        assert (std::option::is_none(*box));
        let r = trans::alloc_ty(cx, t);
        bcx = r.bcx;
        let llptr = r.val;
        *box = some(llptr);
        ret {bcx: bcx, dest: dst_move(llptr)};
      }
      _ { ret {bcx: bcx, dest: dest}; }
    }
}

fn mk_temp(cx: &@block_ctxt, t: ty::t) -> {bcx: @block_ctxt, dest: dest} {
    let bcx = cx;
    if ty::type_is_nil(bcx_tcx(bcx), t) { ret {bcx: bcx, dest: dst_nil}; }
    if trans::type_is_immediate(bcx_ccx(bcx), t) {
        ret {bcx: bcx, dest: dst_imm(@mutable none)};
    }

    let r = trans::alloc_ty(cx, t);
    bcx = r.bcx;
    let llptr = r.val;
    ret {bcx: bcx, dest: dst_copy(llptr)};
}


// AST substructure translation, with destinations

fn trans_lit(cx: &@block_ctxt, dest: &dest, lit: &ast::lit) -> @block_ctxt {
    let bcx = cx;
    alt lit.node {
      ast::lit_str(s, ast::sk_unique.) {
        let r = trans_lit_str_common(bcx_ccx(bcx), s, dest_is_alias(dest));
        let llstackpart = r.stack;
        let llheappartopt = r.heap;
        bcx = store_ptr(bcx, dest, llstackpart);
        alt llheappartopt {
          none. {/* no-op */ }
          some(llheappart) {
            let lldestptrptr =
                bcx.build.InBoundsGEP(dest_ptr(dest),
                                      ~[tc::C_int(0),
                                        tc::C_uint(abi::ivec_elt_elems)]);
            let llheappartty = lltype_of(llheappart);
            lldestptrptr =
                bcx.build.PointerCast(lldestptrptr,
                                      tc::T_ptr(tc::T_ptr(llheappartty)));
            malloc(bcx, lldestptrptr, hp_shared, none);
            let lldestptr = bcx.build.Load(lldestptrptr);
            store_ptr(bcx, dst_copy(lldestptr), llheappart);
          }
        }
      }
      _ {
        bcx =
            store_imm(bcx, dest, trans_lit_common(bcx_ccx(bcx), lit), false);
      }
    }

    ret bcx;
}

fn trans_binary(cx: &@block_ctxt, dest: &dest, sp: &span, op: ast::binop,
                lhs: &@ast::expr, rhs: &@ast::expr) -> @block_ctxt {
    let bcx = cx;
    alt op {
      ast::add. {
        bcx =
            trans_vec::trans_concat(bcx, dest, sp,
                                    ty::expr_ty(bcx_tcx(bcx), rhs), lhs, rhs);
      }
    }
    // TODO: Many more to add here.
    ret bcx;
}

fn trans_log(cx: &@block_ctxt, sp: &span, level: int, expr: &@ast::expr) ->
   @block_ctxt {
    fn trans_log_level(lcx: &@local_ctxt) -> ValueRef {
        let modname = str::connect_ivec(lcx.module_path, "::");

        if lcx_ccx(lcx).module_data.contains_key(modname) {
            ret lcx_ccx(lcx).module_data.get(modname);
        }

        let s =
            link::mangle_internal_name_by_path_and_seq(lcx_ccx(lcx),
                                                       lcx.module_path,
                                                       "loglevel");
        let lllevelptr =
            llvm::LLVMAddGlobal(lcx.ccx.llmod, tc::T_int(), str::buf(s));
        llvm::LLVMSetGlobalConstant(lllevelptr, LLFalse);
        llvm::LLVMSetInitializer(lllevelptr, tc::C_int(0));
        llvm::LLVMSetLinkage(lllevelptr,
                             lib::llvm::LLVMInternalLinkage as llvm::Linkage);
        lcx_ccx(lcx).module_data.insert(modname, lllevelptr);
        ret lllevelptr;
    }

    let bcx = cx;

    let lllevelptr = trans_log_level(bcx_lcx(bcx));

    let log_bcx = trans::new_scope_block_ctxt(bcx, "log");
    let next_bcx = trans::new_scope_block_ctxt(bcx, "next_log");

    let should_log =
        bcx.build.ICmp(ll::LLVMIntSGE, bcx.build.Load(lllevelptr),
                       tc::C_int(level));
    bcx.build.CondBr(should_log, log_bcx.llbb, next_bcx.llbb);

    let expr_t = ty::expr_ty(bcx_tcx(log_bcx), expr);
    let arg_dest = dest_alias(bcx_tcx(log_bcx), expr_t);
    log_bcx = trans_expr(log_bcx, arg_dest, expr);

    let llarg = dest_llval(arg_dest);
    let llarg_i8 = bcx.build.PointerCast(llarg, T_ptr(T_i8()));

    let ti = none;
    let r2 = trans::get_tydesc(bcx, expr_t, false, ti);
    bcx = r2.bcx;
    let lltydesc = r2.val;

    log_bcx.build.Call(bcx_ccx(log_bcx).upcalls.log_type,
                       ~[bcx_fcx(bcx).lltaskptr, lltydesc, llarg_i8,
                         tc::C_int(level)]);

    log_bcx =
        trans::trans_block_cleanups(log_bcx, tc::find_scope_cx(log_bcx));
    log_bcx.build.Br(next_bcx.llbb);
    ret next_bcx;
}

fn trans_path(bcx: &@block_ctxt, dest: &dest, path: &ast::path,
              id: ast::node_id) -> @block_ctxt {
    alt bcx_tcx(bcx).def_map.get(id) {
      ast::def_local(def_id) {
        alt bcx_fcx(bcx).lllocals.find(def_id.node) {
          none. { bcx_ccx(bcx).sess.unimpl("upvar in trans_path"); }
          some(llptr) {
            // TODO: Copy hooks.
            store_ptr(bcx, dest, llptr);
          }
        }
      }
      _ { bcx_ccx(bcx).sess.unimpl("def variant in trans_dps::trans_path"); }
    }
    ret bcx;
}

fn trans_expr(bcx: &@block_ctxt, dest: &dest, expr: &@ast::expr) ->
   @block_ctxt {
    alt expr.node {
      ast::expr_lit(lit) { trans_lit(bcx, dest, *lit); ret bcx; }
      ast::expr_log(level, operand) {
        ret trans_log(bcx, expr.span, level, operand);
      }
      ast::expr_binary(op, lhs, rhs) {
        ret trans_binary(bcx, dest, expr.span, op, lhs, rhs);
      }
      ast::expr_path(path) { ret trans_path(bcx, dest, path, expr.id); }
      _ { fail "unhandled expr type in trans_expr"; }
    }
}

fn trans_recv(bcx: &@block_ctxt, dest: &dest, expr: &@ast::expr) ->
   @block_ctxt {
    ret bcx; // TODO
}

fn trans_block(cx: &@block_ctxt, dest: &dest, blk: &ast::blk) -> @block_ctxt {
    let bcx = cx;
    for each local: @ast::local  in trans::block_locals(blk) {
        bcx = trans::alloc_local(bcx, local).bcx;
    }

    for stmt: @ast::stmt  in blk.node.stmts {
        bcx = trans_stmt(bcx, stmt);


        // If we hit a terminator, control won't go any further so
        // we're in dead-code land. Stop here.
        if trans::is_terminated(bcx) { ret bcx; }
    }

    alt blk.node.expr {
      some(e) { bcx = trans_expr(bcx, dest, e); }
      none. {/* no-op */ }
    }

    bcx = trans::trans_block_cleanups(bcx, tc::find_scope_cx(bcx));
    ret bcx;
}



// AST substructure translation, without destinations

// Common setup code shared between the crate-constant literal string case and
// the block-local literal string case. We don't use destination-passing style
// since that doesn't work for crate constants.
//
// If |expand| is true, we never spill to the heap. This should be used
// whenever the destination size isn't fixed.
fn trans_lit_str_common(ccx: &@crate_ctxt, s: &str, expand: bool) ->
   {stack: ValueRef, heap: option[ValueRef]} {
    let llstackpart;
    let llheappartopt;

    let len = str::byte_len(s);

    let array = ~[];
    for ch: u8  in s { array += ~[tc::C_u8(ch as uint)]; }
    array += ~[tc::C_u8(0u)];

    if expand {
        llstackpart =
            tc::C_struct(~[tc::C_uint(len + 1u), tc::C_uint(len + 1u),
                           tc::C_array(tc::T_i8(), array)]);
        llheappartopt = none;
    } else if (len < abi::ivec_default_length - 1u)
     { // minus one for the null
        while ivec::len(array) < abi::ivec_default_length {
            array += ~[tc::C_u8(0u)];
        }

        llstackpart =
            tc::C_struct(~[tc::C_uint(len + 1u),
                           tc::C_uint(abi::ivec_default_length),
                           tc::C_array(tc::T_i8(), array)]);
        llheappartopt = none;
    } else {
        let llheappart =
            tc::C_struct(~[tc::C_uint(len), tc::C_array(tc::T_i8(), array)]);
        llstackpart =
            tc::C_struct(~[tc::C_uint(0u),
                           tc::C_uint(abi::ivec_default_length),
                           tc::C_null(tc::T_ptr(lltype_of(llheappart)))]);
        llheappartopt =
            some(mk_const(ccx, "const_istr_heap", false, llheappart));
    }

    ret {stack: mk_const(ccx, "const_istr_stack", false, llstackpart),
         heap: llheappartopt};
}

// As above, we don't use destination-passing style here.
fn trans_lit_common(ccx: &@crate_ctxt, lit: &ast::lit) -> ValueRef {
    alt lit.node {
      ast::lit_int(i) { ret tc::C_int(i); }
      ast::lit_uint(u) { ret tc::C_int(u as int); }
      ast::lit_mach_int(tm, i) {
        // FIXME: the entire handling of mach types falls apart
        // if target int width is larger than host, at the moment;
        // re-do the mach-int types using 'big' when that works.

        let t = tc::T_int();
        let s = LLTrue;
        alt tm {
          ast::ty_u8. { t = tc::T_i8(); s = LLFalse; }
          ast::ty_u16. { t = tc::T_i16(); s = LLFalse; }
          ast::ty_u32. { t = tc::T_i32(); s = LLFalse; }
          ast::ty_u64. { t = tc::T_i64(); s = LLFalse; }
          ast::ty_i8. { t = tc::T_i8(); }
          ast::ty_i16. { t = tc::T_i16(); }
          ast::ty_i32. { t = tc::T_i32(); }
          ast::ty_i64. { t = tc::T_i64(); }
        }
        ret tc::C_integral(t, i as uint, s);
      }
      ast::lit_float(fs) { ret tc::C_float(fs); }
      ast::lit_mach_float(tm, s) {
        let t = tc::T_float();
        alt tm {
          ast::ty_f32. { t = tc::T_f32(); }
          ast::ty_f64. { t = tc::T_f64(); }
        }
        ret tc::C_floating(s, t);
      }
      ast::lit_char(c) {
        ret tc::C_integral(tc::T_char(), c as uint, LLFalse);
      }
      ast::lit_bool(b) { ret tc::C_bool(b); }
      ast::lit_nil. { ret tc::C_nil(); }
      ast::lit_str(s, ast::sk_rc.) { ret tc::C_str(ccx, s); }
      ast::lit_str(s, ast::sk_unique.) {
        fail "unique str in trans_lit_common";
      }
    }
}

fn trans_init_local(bcx: &@block_ctxt, local: &@ast::local) -> @block_ctxt {
    let llptr = bcx_fcx(bcx).lllocals.get(local.node.pat.id); // FIXME DESTR

    let t = type_of_node(bcx_ccx(bcx), local.node.pat.id);
    tc::add_clean(bcx, llptr, t);


    alt local.node.init {
      some(init) {
        alt init.op {
          ast::init_assign. {
            ret trans_expr(bcx, dest_copy(bcx_tcx(bcx), llptr, t), init.expr);
          }
          ast::init_move. {
            ret trans_expr(bcx, dest_move(bcx_tcx(bcx), llptr, t), init.expr);
          }
        }
      }
      none. { ret bcx; }
    }
}

fn trans_stmt(cx: &@block_ctxt, stmt: &@ast::stmt) -> @block_ctxt {
    let bcx = cx;
    alt stmt.node {
      ast::stmt_expr(e, _) {
        let tmp = dest_alias(bcx_tcx(bcx), ty::expr_ty(bcx_tcx(bcx), e));
        ret trans_expr(bcx, tmp, e);
      }
      ast::stmt_decl(d, _) {
        alt d.node {
          ast::decl_local(locals) {
            for local: @ast::local  in locals {
                bcx = trans_init_local(bcx, local);
            }
          }
          ast::decl_item(item) { trans::trans_item(bcx_lcx(bcx), *item); }
        }
        ret bcx;
      }
    }
}

