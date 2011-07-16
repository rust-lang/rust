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


// Destination utilities

tag dest_slot {
    dst_nil;
    dst_val(ValueRef);
}

tag dest_mode { dm_copy; dm_move; dm_alias; }

type dest = rec(dest_slot slot, dest_mode mode);

fn dest_slot_for_ptr(&ty::ctxt tcx, ValueRef llptr, ty::t t) -> dest_slot {
    if ty::type_is_nil(tcx, t) { dst_nil } else { dst_val(llptr) }
}

fn dest_copy(&ty::ctxt tcx, ValueRef llptr, ty::t t) -> dest {
    ret rec(slot=dest_slot_for_ptr(tcx, llptr, t), mode=dm_copy);
}

fn dest_move(&ty::ctxt tcx, ValueRef llptr, ty::t t) -> dest {
    ret rec(slot=dest_slot_for_ptr(tcx, llptr, t), mode=dm_move);
}

fn dest_alias(&ty::ctxt tcx, ValueRef llptr, ty::t t) -> dest {
    ret rec(slot=dest_slot_for_ptr(tcx, llptr, t), mode=dm_alias);
}

fn dest_tmp(&@block_ctxt bcx, ty::t t, bool alias) -> tup(@block_ctxt, dest) {
    auto mode = if alias { dm_alias } else { dm_move };
    if ty::type_is_nil(bcx_tcx(bcx), t) {
        ret tup(bcx, rec(slot=dst_nil, mode=mode));
    }
    auto r = trans::alloc_ty(bcx, t);
    trans::add_clean(bcx, r.val, t);
    ret tup(r.bcx, rec(slot=dest_slot_for_ptr(bcx_tcx(bcx), r.val, t),
                       mode=mode));
}

fn dest_ptr(&dest dest) -> ValueRef {
    alt (dest.slot) {
      dst_nil { tc::C_null(tc::T_ptr(tc::T_i8())) }
      dst_val(?llptr) { llptr }
    }
}


// Accessors
// TODO: When we have overloading, simplify these names!

fn bcx_tcx(&@block_ctxt bcx) -> ty::ctxt { ret bcx.fcx.lcx.ccx.tcx; }
fn bcx_ccx(&@block_ctxt bcx) -> @crate_ctxt { ret bcx.fcx.lcx.ccx; }
fn bcx_lcx(&@block_ctxt bcx) -> @local_ctxt { ret bcx.fcx.lcx; }
fn bcx_fcx(&@block_ctxt bcx) -> @fn_ctxt { ret bcx.fcx; }
fn lcx_ccx(&@local_ctxt lcx) -> @crate_ctxt { ret lcx.ccx; }


// Common operations

// If "cast" is true, casts dest appropriately before the store.
fn store(&@block_ctxt bcx, &dest dest, ValueRef llsrc, bool cast)
        -> @block_ctxt {
    alt (dest.slot) {
      dst_nil { /* no-op */ }
      dst_val(?lldestptr_orig) {
        auto lldestptr = lldestptr_orig;
        if (cast) {
            lldestptr = bcx.build.PointerCast(lldestptr,
                                              tc::T_ptr(lltype_of(llsrc)));
        }

        bcx.build.Store(llsrc, lldestptr);
      }
    }
    ret bcx;
}

tag heap { hp_task; hp_shared; }

// Allocates a value of the given LLVM size on either the task heap or the
// shared heap.
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


// AST substructure translation, with destinations

fn trans_lit(&@block_ctxt cx, &dest dest, &ast::lit lit) -> @block_ctxt {
    auto bcx = cx;
    alt (lit.node) {
      ast::lit_str(?s, ast::sk_unique) {
        auto r = trans_lit_str_common(bcx_ccx(bcx), s);
        auto llstackpart = r._0; auto llheappartopt = r._1;
        bcx = store(bcx, dest, llstackpart, true);
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
            bcx.build.Store(llheappart, lldestptr);
          }
        }
      }
      _ {
        bcx = store(bcx, dest, trans_lit_common(bcx_ccx(bcx), lit), false);
      }
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

    fn trans_log_upcall(&@block_ctxt bcx, &span sp, ValueRef in_llval,
                        int level, ty::t t) {
        auto llval = in_llval;
        auto by_val; auto llupcall;
        alt (ty::struct(bcx_tcx(bcx), t)) {
          ty::ty_machine(ast::ty_f32) {
            by_val = true; llupcall = bcx_ccx(bcx).upcalls.log_float;
          }
          ty::ty_machine(ast::ty_f64) | ty::ty_float {
            by_val = false; llupcall = bcx_ccx(bcx).upcalls.log_double;
          }
          ty::ty_bool | ty::ty_machine(ast::ty_i8) |
                ty::ty_machine(ast::ty_i16) | ty::ty_machine(ast::ty_u8) |
                ty::ty_machine(ast::ty_u16) {
            by_val = true; llupcall = bcx_ccx(bcx).upcalls.log_int;
            llval = bcx.build.ZExt(llval, tc::T_i32());
          }
          ty::ty_int | ty::ty_machine(ast::ty_i32) |
                ty::ty_machine(ast::ty_u32) {
            by_val = true; llupcall = bcx_ccx(bcx).upcalls.log_int;
          }
          ty::ty_istr {
            by_val = false; llupcall = bcx_ccx(bcx).upcalls.log_istr;
          }
          _ {
            bcx_ccx(bcx).sess.span_unimpl(sp, "logging for values of type " +
                ppaux::ty_to_str(bcx_tcx(bcx), t));
          }
        }

        if by_val { llval = bcx.build.Load(llval); }
        bcx.build.Call(llupcall,
                       ~[bcx_fcx(bcx).lltaskptr, tc::C_int(level), llval]);
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
    auto r = dest_tmp(log_bcx, expr_t, true);
    log_bcx = r._0; auto tmp = r._1;
    log_bcx = trans_expr(log_bcx, tmp, expr);

    trans_log_upcall(log_bcx, sp, dest_ptr(tmp), level, expr_t);

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
fn trans_lit_str_common(&@crate_ctxt ccx, &str s)
        -> tup(ValueRef, option[ValueRef]) {
    auto len = str::byte_len(s);

    auto array = ~[];
    for (u8 ch in s) { array += ~[tc::C_u8(ch as uint)]; }
    array += ~[tc::C_u8(0u)];

    if len < abi::ivec_default_length - 1u {    // minus 1 because of the \0
        while (ivec::len(array) < abi::ivec_default_length) {
            array += ~[tc::C_u8(0u)];
        }

        ret tup(tc::C_struct(~[tc::C_uint(len + 1u),
                               tc::C_uint(abi::ivec_default_length),
                               tc::C_array(tc::T_i8(), array)]),
                none);
    }

    auto llheappart = tc::C_struct(~[tc::C_uint(len),
                                     tc::C_array(tc::T_i8(), array)]);
    ret tup(tc::C_struct(~[tc::C_uint(0u),
                           tc::C_uint(abi::ivec_default_length),
                           tc::C_null(tc::T_ptr(lltype_of(llheappart)))]),
            some(llheappart));
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
      none {
        ret store(bcx, dest_copy(bcx_tcx(bcx), llptr, t),
                  tc::C_null(llelement_type(trans::val_ty(llptr))), false);
      }
    }
}

fn trans_stmt(&@block_ctxt cx, &@ast::stmt stmt) -> @block_ctxt {
    auto bcx = cx;
    alt (stmt.node) {
      ast::stmt_expr(?e, _) {
        auto tmp_r = dest_tmp(bcx, ty::expr_ty(bcx_tcx(bcx), e), true);
        bcx = tmp_r._0; auto tmp = tmp_r._1;
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

