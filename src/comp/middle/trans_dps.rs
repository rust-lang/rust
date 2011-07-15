// Translates individual functions in the completed AST to the LLVM IR, using
// destination-passing style.

import syntax::ast;
import middle::trans;
import middle::ty;
import trans::block_ctxt;
import trans::crate_ctxt;
import trans::fn_ctxt;
import trans::local_ctxt;
import lib::llvm::llvm::TypeRef;
import lib::llvm::llvm::ValueRef;
import std::option::none;
import std::option::some;

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

type dest = rec(dest_slot slot, bool move);

fn dest_slot_for_ptr(&ty::ctxt tcx, ValueRef llptr, ty::t t) -> dest_slot {
    if ty::type_is_nil(tcx, t) { dst_nil } else { dst_val(llptr) }
}

fn dest_copy(&ty::ctxt tcx, ValueRef llptr, ty::t t) -> dest {
    ret rec(slot=dest_slot_for_ptr(tcx, llptr, t), move=false);
}

fn dest_move(&ty::ctxt tcx, ValueRef llptr, ty::t t) -> dest {
    ret rec(slot=dest_slot_for_ptr(tcx, llptr, t), move=true);
}

fn dest_tmp(&@block_ctxt bcx, ty::t t) -> tup(@block_ctxt, dest) {
    if ty::type_is_nil(bcx_tcx(bcx), t) {
        ret tup(bcx, rec(slot=dst_nil, move=true));
    }
    auto r = trans::alloc_ty(bcx, t);
    ret tup(r.bcx, dest_move(bcx_tcx(bcx), r.val, t));
}


// Accessors
// TODO: When we have overloading, simplify these names!

fn bcx_tcx(&@block_ctxt bcx) -> ty::ctxt { ret bcx.fcx.lcx.ccx.tcx; }
fn bcx_ccx(&@block_ctxt bcx) -> @crate_ctxt { ret bcx.fcx.lcx.ccx; }
fn bcx_lcx(&@block_ctxt bcx) -> @local_ctxt { ret bcx.fcx.lcx; }
fn bcx_fcx(&@block_ctxt bcx) -> @fn_ctxt { ret bcx.fcx; }


// Common operations

fn store(&@block_ctxt bcx, &dest dest, ValueRef llsrc) -> @block_ctxt {
    alt (dest.slot) {
      dst_nil { /* no-op */ }
      dst_val(?lldest) { bcx.build.Store(llsrc, lldest); }
    }
    ret bcx;
}


// AST substructure translation, with destinations

fn trans_expr(&@block_ctxt bcx, &dest dest, &@ast::expr expr) -> @block_ctxt {
    ret bcx;    // TODO
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
      some(?e) { ret trans_expr(bcx, dest, e); }
      none { ret bcx; }
    }
}


// AST substructure translation, without destinations

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
                  trans_common::C_null(llelement_type(trans::val_ty(llptr))));
      }
    }
}

fn trans_stmt(&@block_ctxt cx, &@ast::stmt stmt) -> @block_ctxt {
    auto bcx = cx;
    alt (stmt.node) {
      ast::stmt_expr(?e, _) {
        auto tmp_r = dest_tmp(bcx, ty::expr_ty(bcx_tcx(bcx), e));
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

