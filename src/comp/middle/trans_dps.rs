// Translates individual functions in the completed AST to the LLVM IR, using
// destination-passing style.

import back::abi;
import back::link;
import lib::llvm::llvm;
import llvm::TypeRef;
import llvm::ValueRef;
import middle::trans;
import middle::ty;
import syntax::ast;
import syntax::codemap::span;
import trans::block_ctxt;
import trans::crate_ctxt;
import trans::fn_ctxt;
import trans::local_ctxt;
import util::ppaux;

import std::ivec;
import std::option::none;
import std::option::some;
import std::str;
import std::uint;

import LLFalse = lib::llvm::False;
import LLTrue = lib::llvm::True;
import ll = lib::llvm;
import lltype_of = trans::val_ty;
import option = std::option::t;
import tc = trans_common;
import type_of_node = trans::node_id_type;


// LLVM utilities

fn llelement_type(TypeRef llty) -> TypeRef {
    lib::llvm::llvm::LLVMGetElementType(llty)
}

fn llalign_of(&@crate_ctxt ccx, TypeRef llty) -> uint {
    ret llvm::LLVMPreferredAlignmentOfType(ccx.td.lltd, llty);
}

fn llsize_of(&@crate_ctxt ccx, TypeRef llty) -> uint {
    ret llvm::LLVMStoreSizeOfType(ccx.td.lltd, llty);
}

fn mk_const(&@crate_ctxt ccx, &str name, bool exported, ValueRef llval)
        -> ValueRef {
    auto llglobal = llvm::LLVMAddGlobal(ccx.llmod, trans::val_ty(llval),
                                        str::buf(name));

    llvm::LLVMSetInitializer(llglobal, llval);
    llvm::LLVMSetGlobalConstant(llglobal, LLTrue);

    if !exported {
        llvm::LLVMSetLinkage(llglobal,
                             lib::llvm::LLVMInternalLinkage as llvm::Linkage);
    }

    ret llglobal;
}


// Type utilities

fn size_of(&@crate_ctxt ccx, &span sp, ty::t t) -> uint {
    if ty::type_has_dynamic_size(ccx.tcx, t) {
        ccx.sess.bug("trans_dps::size_of() called on a type with dynamic " +
                     "size");
    }
    ret llsize_of(ccx, trans::type_of_inner(ccx, sp, t));
}


// Destination utilities

tag dest {
    dst_nil;                                // Unit destination; ignore.
    dst_imm(@mutable option[ValueRef]);     // Fill with an immediate value.
    dst_alias(@mutable option[ValueRef]);   // Fill with an alias pointer.
    dst_copy(ValueRef);                     // Copy to the given address.
    dst_move(ValueRef);                     // Move to the given address.
}

fn dest_imm(&ty::ctxt tcx, ty::t t) -> dest {
    if ty::type_is_nil(tcx, t) { dst_nil } else { dst_imm(@mutable none) }
}

fn dest_alias(&ty::ctxt tcx, ty::t t) -> dest {
    if ty::type_is_nil(tcx, t) { dst_nil } else { dst_alias(@mutable none) }
}

fn dest_copy(&ty::ctxt tcx, ValueRef llptr, ty::t t) -> dest {
    if ty::type_is_nil(tcx, t) { dst_nil } else { dst_copy(llptr) }
}

fn dest_move(&ty::ctxt tcx, ValueRef llptr, ty::t t) -> dest {
    if ty::type_is_nil(tcx, t) { dst_nil } else { dst_move(llptr) }
}

// Invariant: the type of the destination must be structural (non-immediate).
fn dest_ptr(&dest dest) -> ValueRef {
    alt (dest) {
      dst_nil { fail "nil dest in dest_ptr" }
      dst_imm(_) { fail "immediate dest in dest_ptr" }
      dst_alias(?box) {
        alt (*box) {
          none { fail "alias wasn't filled in prior to dest_ptr" }
          some(?llval) { llval }
        }
      }
      dst_copy(?llptr) { llptr }
      dst_move(?llptr) { llptr }
    }
}

fn dest_llval(&dest dest) -> ValueRef {
    alt (dest) {
      dst_nil { ret tc::C_nil(); }
      dst_imm(?box) {
        alt (*box) {
          none { fail "immediate wasn't filled in prior to dest_llval"; }
          some(?llval) { ret llval; }
        }
      }
      dst_alias(?box) {
        alt (*box) {
          none { fail "alias wasn't filled in prior to dest_llval"; }
          some(?llval) { ret llval; }
        }
      }
      dst_copy(?llptr) { ret llptr; }
      dst_move(?llptr) { ret llptr; }
    }
}

fn dest_is_alias(&dest dest) -> bool {
    alt (dest) { dst_alias(_) { true } _ { false } }
}


// Accessors
// TODO: When we have overloading, simplify these names!

fn bcx_tcx(&@block_ctxt bcx) -> ty::ctxt { ret bcx.fcx.lcx.ccx.tcx; }
fn bcx_ccx(&@block_ctxt bcx) -> @crate_ctxt { ret bcx.fcx.lcx.ccx; }
fn bcx_lcx(&@block_ctxt bcx) -> @local_ctxt { ret bcx.fcx.lcx; }
fn bcx_fcx(&@block_ctxt bcx) -> @fn_ctxt { ret bcx.fcx; }
fn lcx_ccx(&@local_ctxt lcx) -> @crate_ctxt { ret lcx.ccx; }
fn ccx_tcx(&@crate_ctxt ccx) -> ty::ctxt { ret ccx.tcx; }


// Common operations

fn memmove(&@block_ctxt bcx, ValueRef lldestptr, ValueRef llsrcptr,
           ValueRef llsz) {
    auto lldestty = llelement_type(trans::val_ty(lldestptr));
    auto llsrcty = llelement_type(trans::val_ty(llsrcptr));
    auto dest_align = llalign_of(bcx_ccx(bcx), lldestty);
    auto src_align = llalign_of(bcx_ccx(bcx), llsrcty);
    auto align = uint::min(dest_align, src_align);
    auto llfn = bcx_ccx(bcx).intrinsics.get("llvm.memmove.p0i8.p0i8.i32");
    auto lldestptr_i8 = bcx.build.PointerCast(lldestptr,
                                              tc::T_ptr(tc::T_i8()));
    auto llsrcptr_i8 = bcx.build.PointerCast(llsrcptr,
                                             tc::T_ptr(tc::T_i8()));
    bcx.build.Call(llfn, ~[lldestptr_i8, llsrcptr_i8, llsz, tc::C_uint(align),
                           tc::C_bool(false)]);
}

// If "cast" is true, casts dest appropriately before the store.
fn store_imm(&@block_ctxt bcx, &dest dest, ValueRef llsrc, bool cast)
        -> @block_ctxt {
    alt (dest) {
      dst_nil { /* no-op */ }
      dst_imm(?box) {
        assert (std::option::is_none(*box));
        *box = some(llsrc);
      }
      dst_alias(?box) {
        bcx_ccx(bcx).sess.unimpl("dst_alias spill in store_imm");
      }
      dst_copy(?lldestptr_orig) | dst_move(?lldestptr_orig) {
        auto lldestptr = lldestptr_orig;
        if cast {
            lldestptr = bcx.build.PointerCast(lldestptr,
                                              tc::T_ptr(lltype_of(llsrc)));
        }
        bcx.build.Store(llsrc, lldestptr);
      }
    }
    ret bcx;
}

fn store_ptr(&@block_ctxt bcx, &dest dest, ValueRef llsrcptr) -> @block_ctxt {
    alt (dest) {
      dst_nil { /* no-op */ }
      dst_imm(?box) { fail "dst_imm in store_ptr"; }
      dst_alias(?box) {
        assert (std::option::is_none(*box));
        *box = some(llsrcptr);
      }
      dst_copy(?lldestptr) | dst_move(?lldestptr) {
        auto llsrcty = llelement_type(trans::val_ty(llsrcptr));
        auto llsz = tc::C_uint(llsize_of(bcx_ccx(bcx), llsrcty));
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
fn malloc(&@block_ctxt bcx, ValueRef lldest, heap heap,
          option[ValueRef] llcustom_size_opt) -> @block_ctxt {
    auto llptrty = llelement_type(lltype_of(lldest));
    auto llty = llelement_type(llptrty);

    auto lltydescptr = tc::C_null(tc::T_ptr(bcx_ccx(bcx).tydesc_type));

    auto llsize;
    alt (llcustom_size_opt) {
      none { llsize = trans::llsize_of(llty); }
      some(?llcustom_size) { llsize = llcustom_size; }
    }

    auto llupcall;
    alt (heap) {
      hp_task { llupcall = bcx_ccx(bcx).upcalls.malloc; }
      hp_shared { llupcall = bcx_ccx(bcx).upcalls.shared_malloc; }
    }

    auto llresult = bcx.build.Call(llupcall, ~[bcx_fcx(bcx).lltaskptr, llsize,
                                               lltydescptr]);
    llresult = bcx.build.PointerCast(llresult, llptrty);
    bcx.build.Store(llresult, lldest);
    ret bcx;
}

// If the supplied destination is an alias, spills to a temporary. Returns the
// new destination.
fn spill_alias(&@block_ctxt cx, &dest dest, ty::t t)
        -> tup(@block_ctxt, dest) {
    auto bcx = cx;
    alt (dest) {
      dst_alias(?box) {
        // TODO: Mark the alias as needing a cleanup.
        assert (std::option::is_none(*box));
        auto r = trans::alloc_ty(cx, t);
        bcx = r.bcx; auto llptr = r.val;
        *box = some(llptr);
        ret tup(bcx, dst_move(llptr));
      }
      _ { ret tup(bcx, dest); }
    }
}

fn mk_temp(&@block_ctxt cx, ty::t t) -> tup(@block_ctxt, dest) {
    auto bcx = cx;
    if ty::type_is_nil(bcx_tcx(bcx), t) { ret tup(bcx, dst_nil); }
    if trans::type_is_immediate(bcx_ccx(bcx), t) {
        ret tup(bcx, dst_imm(@mutable none));
    }

    auto r = trans::alloc_ty(cx, t);
    bcx = r.bcx; auto llptr = r.val;
    ret tup(bcx, dst_copy(llptr));
}


// AST substructure translation, with destinations

fn trans_lit(&@block_ctxt cx, &dest dest, &ast::lit lit) -> @block_ctxt {
    auto bcx = cx;
    alt (lit.node) {
      ast::lit_str(?s, ast::sk_unique) {
        auto r = trans_lit_str_common(bcx_ccx(bcx), s, dest_is_alias(dest));
        auto llstackpart = r._0; auto llheappartopt = r._1;
        bcx = store_ptr(bcx, dest, llstackpart);
        alt (llheappartopt) {
          none { /* no-op */ }
          some(?llheappart) {
            auto lldestptrptr =
                bcx.build.InBoundsGEP(dest_ptr(dest),
                                      ~[tc::C_int(0),
                                        tc::C_uint(abi::ivec_elt_elems)]);
            auto llheappartty = lltype_of(llheappart);
            lldestptrptr =
                bcx.build.PointerCast(lldestptrptr,
                                      tc::T_ptr(tc::T_ptr(llheappartty)));
            malloc(bcx, lldestptrptr, hp_shared, none);
            auto lldestptr = bcx.build.Load(lldestptrptr);
            store_ptr(bcx, dst_copy(lldestptr), llheappart);
          }
        }
      }
      _ {
        bcx = store_imm(bcx, dest, trans_lit_common(bcx_ccx(bcx), lit),
                        false);
      }
    }

    ret bcx;
}

fn trans_binary(&@block_ctxt cx, &dest dest, &span sp, ast::binop op,
                &@ast::expr lhs, &@ast::expr rhs) -> @block_ctxt {
    auto bcx = cx;
    alt (op) {
      ast::add {
        bcx = trans_vec::trans_concat(bcx, dest, sp,
                                      ty::expr_ty(bcx_tcx(bcx), rhs), lhs,
                                      rhs);
      }
      // TODO: Many more to add here.
    }
    ret bcx;
}

fn trans_log(&@block_ctxt cx, &span sp, int level, &@ast::expr expr)
        -> @block_ctxt {
    fn trans_log_level(&@local_ctxt lcx) -> ValueRef {
        auto modname = str::connect_ivec(lcx.module_path, "::");

        if (lcx_ccx(lcx).module_data.contains_key(modname)) {
            ret lcx_ccx(lcx).module_data.get(modname);
        }

        auto s =
            link::mangle_internal_name_by_path_and_seq(lcx_ccx(lcx),
                                                       lcx.module_path,
                                                       "loglevel");
        auto lllevelptr = llvm::LLVMAddGlobal(lcx.ccx.llmod, tc::T_int(),
                                              str::buf(s));
        llvm::LLVMSetGlobalConstant(lllevelptr, LLFalse);
        llvm::LLVMSetInitializer(lllevelptr, tc::C_int(0));
        llvm::LLVMSetLinkage(lllevelptr, lib::llvm::LLVMInternalLinkage as
                             llvm::Linkage);
        lcx_ccx(lcx).module_data.insert(modname, lllevelptr);
        ret lllevelptr;
    }

    tag upcall_style { us_imm; us_imm_i32_zext; us_alias; us_alias_istr; }
    fn get_upcall(&@crate_ctxt ccx, &span sp, ty::t t)
            -> tup(ValueRef, upcall_style) {
        alt (ty::struct(ccx_tcx(ccx), t)) {
          ty::ty_machine(ast::ty_f32) {
            ret tup(ccx.upcalls.log_float, us_imm);
          }
          ty::ty_machine(ast::ty_f64) | ty::ty_float {
            // TODO: We have to spill due to legacy calling conventions that
            // should probably be modernized.
            ret tup(ccx.upcalls.log_double, us_alias);
          }
          ty::ty_bool | ty::ty_machine(ast::ty_i8) |
                ty::ty_machine(ast::ty_i16) | ty::ty_machine(ast::ty_u8) |
                ty::ty_machine(ast::ty_u16) {
            ret tup(ccx.upcalls.log_int, us_imm_i32_zext);
          }
          ty::ty_int | ty::ty_machine(ast::ty_i32) |
                ty::ty_machine(ast::ty_u32) {
            ret tup(ccx.upcalls.log_int, us_imm);
          }
          ty::ty_istr {
            ret tup(ccx.upcalls.log_istr, us_alias_istr);
          }
          _ {
            ccx.sess.span_unimpl(sp, "logging for values of type " +
                                 ppaux::ty_to_str(ccx_tcx(ccx), t));
          }
        }
    }

    auto bcx = cx;

    auto lllevelptr = trans_log_level(bcx_lcx(bcx));

    auto log_bcx = trans::new_scope_block_ctxt(bcx, "log");
    auto next_bcx = trans::new_scope_block_ctxt(bcx, "next_log");

    auto should_log = bcx.build.ICmp(ll::LLVMIntSGE,
                                     bcx.build.Load(lllevelptr),
                                     tc::C_int(level));
    bcx.build.CondBr(should_log, log_bcx.llbb, next_bcx.llbb);

    auto expr_t = ty::expr_ty(bcx_tcx(log_bcx), expr);
    auto r = get_upcall(bcx_ccx(bcx), sp, expr_t);
    auto llupcall = r._0; auto style = r._1;

    auto arg_dest;
    alt (style) {
      us_imm | us_imm_i32_zext {
        arg_dest = dest_imm(bcx_tcx(log_bcx), expr_t);
      }
      us_alias | us_alias_istr {
        arg_dest = dest_alias(bcx_tcx(log_bcx), expr_t);
      }
    }
    log_bcx = trans_expr(log_bcx, arg_dest, expr);

    auto llarg = dest_llval(arg_dest);
    alt (style) {
      us_imm | us_alias { /* no-op */ }
      us_imm_i32_zext { llarg = log_bcx.build.ZExt(llarg, tc::T_i32()); }
      us_alias_istr {
        llarg = log_bcx.build.PointerCast(llarg,
                                          tc::T_ptr(tc::T_ivec(tc::T_i8())));
      }
    }

    log_bcx.build.Call(llupcall,
                       ~[bcx_fcx(bcx).lltaskptr, tc::C_int(level), llarg]);

    log_bcx = trans::trans_block_cleanups(log_bcx,
                                          trans::find_scope_cx(log_bcx));
    log_bcx.build.Br(next_bcx.llbb);
    ret next_bcx;
}

fn trans_expr(&@block_ctxt bcx, &dest dest, &@ast::expr expr) -> @block_ctxt {
    alt (expr.node) {
      ast::expr_lit(?lit) { trans_lit(bcx, dest, *lit); ret bcx; }
      ast::expr_log(?level, ?operand) {
        ret trans_log(bcx, expr.span, level, operand);
      }
      ast::expr_binary(?op, ?lhs, ?rhs) {
        ret trans_binary(bcx, dest, expr.span, op, lhs, rhs);
      }
      _ { fail "unhandled expr type in trans_expr"; }
    }
}

fn trans_recv(&@block_ctxt bcx, &dest dest, &@ast::expr expr) -> @block_ctxt {
    ret bcx;    // TODO
}

fn trans_block(&@block_ctxt cx, &dest dest, &ast::block block)
        -> @block_ctxt {
    auto bcx = cx;
    for each (@ast::local local in trans::block_locals(block)) {
        bcx = trans::alloc_local(bcx, local).bcx;
    }

    for (@ast::stmt stmt in block.node.stmts) {
        bcx = trans_stmt(bcx, stmt);

        // If we hit a terminator, control won't go any further so
        // we're in dead-code land. Stop here.
        if trans::is_terminated(bcx) { ret bcx; }
    }

    alt (block.node.expr) {
      some(?e) { bcx = trans_expr(bcx, dest, e); }
      none { /* no-op */ }
    }

    bcx = trans::trans_block_cleanups(bcx, trans::find_scope_cx(bcx));
    ret bcx;
}



// AST substructure translation, without destinations

// Common setup code shared between the crate-constant literal string case and
// the block-local literal string case. We don't use destination-passing style
// since that doesn't work for crate constants.
//
// If |expand| is true, we never spill to the heap. This should be used
// whenever the destination size isn't fixed.
fn trans_lit_str_common(&@crate_ctxt ccx, &str s, bool expand)
        -> tup(ValueRef, option[ValueRef]) {
    auto llstackpart; auto llheappartopt;

    auto len = str::byte_len(s);

    auto array = ~[];
    for (u8 ch in s) { array += ~[tc::C_u8(ch as uint)]; }
    array += ~[tc::C_u8(0u)];

    if expand {
        llstackpart = tc::C_struct(~[tc::C_uint(len + 1u),
                                     tc::C_uint(len + 1u),
                                     tc::C_array(tc::T_i8(), array)]);
        llheappartopt = none;
    } else if len < abi::ivec_default_length - 1u { // minus one for the null
        while (ivec::len(array) < abi::ivec_default_length) {
            array += ~[tc::C_u8(0u)];
        }

        llstackpart = tc::C_struct(~[tc::C_uint(len + 1u),
                                     tc::C_uint(abi::ivec_default_length),
                                     tc::C_array(tc::T_i8(), array)]);
        llheappartopt = none;
    } else {
        auto llheappart = tc::C_struct(~[tc::C_uint(len),
                                         tc::C_array(tc::T_i8(), array)]);
        llstackpart =
            tc::C_struct(~[tc::C_uint(0u),
                           tc::C_uint(abi::ivec_default_length),
                           tc::C_null(tc::T_ptr(lltype_of(llheappart)))]);
        llheappartopt = some(mk_const(ccx, "const_istr_heap", false,
                                      llheappart));
    }

    ret tup(mk_const(ccx, "const_istr_stack", false, llstackpart),
            llheappartopt);
}

// As above, we don't use destination-passing style here.
fn trans_lit_common(&@crate_ctxt ccx, &ast::lit lit) -> ValueRef {
    alt (lit.node) {
      ast::lit_int(?i) { ret tc::C_int(i); }
      ast::lit_uint(?u) { ret tc::C_int(u as int); }
      ast::lit_mach_int(?tm, ?i) {
        // FIXME: the entire handling of mach types falls apart
        // if target int width is larger than host, at the moment;
        // re-do the mach-int types using 'big' when that works.

        auto t = tc::T_int();
        auto s = LLTrue;
        alt (tm) {
          ast::ty_u8 { t = tc::T_i8(); s = LLFalse; }
          ast::ty_u16 { t = tc::T_i16(); s = LLFalse; }
          ast::ty_u32 { t = tc::T_i32(); s = LLFalse; }
          ast::ty_u64 { t = tc::T_i64(); s = LLFalse; }
          ast::ty_i8 { t = tc::T_i8(); }
          ast::ty_i16 { t = tc::T_i16(); }
          ast::ty_i32 { t = tc::T_i32(); }
          ast::ty_i64 { t = tc::T_i64(); }
        }
        ret tc::C_integral(t, i as uint, s);
      }
      ast::lit_float(?fs) { ret tc::C_float(fs); }
      ast::lit_mach_float(?tm, ?s) {
        auto t = tc::T_float();
        alt (tm) {
          ast::ty_f32 { t = tc::T_f32(); }
          ast::ty_f64 { t = tc::T_f64(); }
        }
        ret tc::C_floating(s, t);
      }
      ast::lit_char(?c) {
        ret tc::C_integral(tc::T_char(), c as uint, LLFalse);
      }
      ast::lit_bool(?b) { ret tc::C_bool(b); }
      ast::lit_nil { ret tc::C_nil(); }
      ast::lit_str(?s, ast::sk_rc) { ret tc::C_str(ccx, s); }
      ast::lit_str(?s, ast::sk_unique) {
        fail "unique str in trans_lit_common";
      }
    }
}

fn trans_init_local(&@block_ctxt bcx, &@ast::local local) -> @block_ctxt {
    auto llptr = bcx_fcx(bcx).lllocals.get(local.node.id);

    auto t = type_of_node(bcx_ccx(bcx), local.node.id);
    trans::add_clean(bcx, llptr, t);

    alt (local.node.init) {
      some(?init) {
        alt (init.op) {
          ast::init_assign {
            ret trans_expr(bcx, dest_copy(bcx_tcx(bcx), llptr, t), init.expr);
          }
          ast::init_move {
            ret trans_expr(bcx, dest_move(bcx_tcx(bcx), llptr, t), init.expr);
          }
          ast::init_recv {
            ret trans_recv(bcx, dest_copy(bcx_tcx(bcx), llptr, t), init.expr);
          }
        }
      }
      none { ret bcx; }
    }
}

fn trans_stmt(&@block_ctxt cx, &@ast::stmt stmt) -> @block_ctxt {
    auto bcx = cx;
    alt (stmt.node) {
      ast::stmt_expr(?e, _) {
        auto tmp = dest_alias(bcx_tcx(bcx), ty::expr_ty(bcx_tcx(bcx), e));
        ret trans_expr(bcx, tmp, e);
      }
      ast::stmt_decl(?d, _) {
        alt (d.node) {
          ast::decl_local(?local) { ret trans_init_local(bcx, local); }
          ast::decl_item(?item) {
            trans::trans_item(bcx_lcx(bcx), *item);
            ret bcx;
          }
        }
      }
    }
}

