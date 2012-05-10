import syntax::ast;
import syntax::ast::{m_mutbl, m_imm, m_const};
import syntax::visit;
import syntax::ast_util;
import syntax::codemap::span;
import util::ppaux::{ty_to_str, region_to_str};
import driver::session::session;
import std::map::{int_hash, hashmap, set};
import std::list;
import std::list::{list, cons, nil};
import result::{result, ok, err, extensions};
import syntax::print::pprust;
import util::common::indenter;

export check_crate, root_map;

fn check_crate(tcx: ty::ctxt,
               method_map: typeck::method_map,
               crate: @ast::crate) -> root_map {

    // big hack to keep this off except when I want it on
    let msg_level = alt os::getenv("RUST_BORROWCK") {
      none {tcx.sess.opts.borrowck}
      some(v) {option::get(uint::from_str(v))}
    };

    let bccx = @{tcx: tcx,
                 method_map: method_map,
                 msg_level: msg_level,
                 root_map: int_hash(),
                 in_ctor: none};

    if msg_level > 0u {
        let req_loan_map = gather_loans(bccx, crate);
        check_loans(bccx, req_loan_map, crate);
    }
    ret bccx.root_map;
}

const TREAT_CONST_AS_IMM: bool = true;

// ----------------------------------------------------------------------
// Type definitions

type borrowck_ctxt = @{tcx: ty::ctxt,
                       method_map: typeck::method_map,
                       msg_level: uint,
                       root_map: root_map,

                       // Keep track of whether we're inside a ctor, so as to
                       // allow mutating immutable fields in the same class if
                       // we are in a ctor, we track the self id
                       in_ctor: option<ast::node_id>};

// a map mapping id's of expressions of task-local type (@T, []/@, etc) where
// the box needs to be kept live to the id of the scope for which they must
// stay live.
type root_map = hashmap<ast::node_id, ast::node_id>;

enum bckerr_code {
    err_mutbl(ast::mutability, ast::mutability),
    err_mut_uniq,
    err_mut_variant,
    err_preserve_gc
}

type bckerr = {cmt: cmt, code: bckerr_code};

type bckres<T> = result<T, bckerr>;

enum categorization {
    cat_rvalue(rvalue_kind),    // result of eval'ing some rvalue
    cat_local(ast::node_id),    // local variable
    cat_arg(ast::node_id),      // formal argument
    cat_deref(cmt, ptr_kind),   // deref of a ptr
    cat_comp(cmt, comp_kind),   // adjust to locate an internal component
}

// different kinds of pointers:
enum ptr_kind {uniq_ptr, gc_ptr, region_ptr, unsafe_ptr}

// I am coining the term "components" to mean "pieces of a data
// structure accessible without a dereference":
enum comp_kind {comp_tuple, comp_res, comp_variant,
                comp_field(str), comp_index}

// We pun on *T to mean both actual deref of a ptr as well
// as accessing of components:
enum deref_kind {deref_ptr(ptr_kind), deref_comp(comp_kind)}

// different kinds of expressions we might evaluate
enum rvalue_kind {
    rv_method,
    rv_static_item,
    rv_upvar,
    rv_misc,
    rv_self
}

// a complete categorization of a value indicating where it originated
// and how it is located, as well as the mutability of the memory in
// which the value is stored.
type cmt = @{id: ast::node_id,        // id of expr/pat producing this value
             span: span,              // span of same expr/pat
             cat: categorization,     // categorization of expr
             lp: option<@loan_path>,  // loan path for expr, if any
             mutbl: ast::mutability,  // mutability of expr as lvalue
             ty: ty::t};              // type of the expr

// a loan path is like a category, but it exists only when the data is
// interior to the stack frame.  loan paths are used as the key to a
// map indicating what is borrowed at any point in time.
enum loan_path {
    lp_local(ast::node_id),
    lp_arg(ast::node_id),
    lp_deref(@loan_path, ptr_kind),
    lp_comp(@loan_path, comp_kind)
}

// a complete record of a loan that was granted
type loan = {lp: @loan_path, cmt: cmt, mutbl: ast::mutability};

fn sup_mutbl(req_m: ast::mutability,
             act_m: ast::mutability) -> bool {
    alt (req_m, act_m) {
      (m_const, _) |
      (m_imm, m_imm) |
      (m_mutbl, m_mutbl) {
        true
      }

      (_, m_const) |
      (m_imm, m_mutbl) |
      (m_mutbl, m_imm) {
        false
      }
    }
}

fn check_sup_mutbl(req_m: ast::mutability,
                   cmt: cmt) -> bckres<()> {
    if sup_mutbl(req_m, cmt.mutbl) {
        ok(())
    } else {
        err({cmt:cmt, code:err_mutbl(req_m, cmt.mutbl)})
    }
}

// ----------------------------------------------------------------------
// Gathering loans
//
// The borrow check proceeds in two phases. In phase one, we gather the full
// set of loans that are required at any point.  These are sorted according to
// their associated scopes.  In phase two, checking loans, we will then make
// sure that all of these loans are honored.

// Maps a scope to a list of loans that were issued within that scope.
type req_loan_map = hashmap<ast::node_id, @mut [@const [loan]]>;

enum gather_loan_ctxt = @{bccx: borrowck_ctxt, req_loan_map: req_loan_map};

fn gather_loans(bccx: borrowck_ctxt, crate: @ast::crate) -> req_loan_map {
    let glcx = gather_loan_ctxt(@{bccx: bccx, req_loan_map: int_hash()});
    let v = visit::mk_vt(@{visit_expr: req_loans_in_expr
                           with *visit::default_visitor()});
    visit::visit_crate(*crate, glcx, v);
    ret glcx.req_loan_map;
}

fn req_loans_in_expr(ex: @ast::expr,
                     &&self: gather_loan_ctxt,
                     vt: visit::vt<gather_loan_ctxt>) {
    let bccx = self.bccx;
    let tcx = bccx.tcx;

    // If this expression is borrowed, have to ensure it remains valid:
    for tcx.borrowings.find(ex.id).each { |scope_id|
        let cmt = self.bccx.cat_borrow_of_expr(ex);
        self.guarantee_valid(cmt, m_const, ty::re_scope(scope_id));
    }

    // Special checks for various kinds of expressions:
    alt ex.node {
      ast::expr_addr_of(mutbl, base) {
        let base_cmt = self.bccx.cat_expr(base);

        // make sure that if we taking an &mut or &imm ptr, the thing it
        // points at is mutable or immutable respectively:
        self.bccx.report_if_err(
            check_sup_mutbl(mutbl, base_cmt).chain { |_ok|
                // make sure that the thing we are pointing out stays valid
                // for the lifetime `scope_r`:
                let scope_r =
                    alt check ty::get(tcx.ty(ex)).struct {
                      ty::ty_rptr(r, _) { r }
                    };
                self.guarantee_valid(base_cmt, mutbl, scope_r);
                ok(())
            });
      }

      ast::expr_call(f, args, _) {
        let arg_tys = ty::ty_fn_args(ty::expr_ty(self.tcx(), f));
        vec::iter2(args, arg_tys) { |arg, arg_ty|
            alt ty::resolved_mode(self.tcx(), arg_ty.mode) {
              ast::by_mutbl_ref {
                let arg_cmt = self.bccx.cat_expr(arg);
                self.guarantee_valid(arg_cmt, m_mutbl, ty::re_scope(ex.id));
              }
              ast::by_ref {
                let arg_cmt = self.bccx.cat_expr(arg);
                if TREAT_CONST_AS_IMM {
                    self.guarantee_valid(arg_cmt, m_imm,
                                         ty::re_scope(ex.id));
                } else {
                    self.guarantee_valid(arg_cmt, m_const,
                                         ty::re_scope(ex.id));
                }
              }
              ast::by_move | ast::by_copy | ast::by_val {}
            }
        }
      }

      ast::expr_alt(ex_v, arms, _) {
        let cmt = self.bccx.cat_expr(ex_v);
        for arms.each { |arm|
            for arm.pats.each { |pat|
                self.gather_pat(cmt, pat, arm.body.node.id);
            }
        }
      }

      _ { /*ok*/ }
    }

    // Check any contained expressions:
    visit::visit_expr(ex, self, vt);
}

impl methods for gather_loan_ctxt {
    fn tcx() -> ty::ctxt { self.bccx.tcx }

    // guarantees that addr_of(cmt) will be valid for the duration of
    // `scope_r`, or reports an error.  This may entail taking out loans,
    // which will be added to the `req_loan_map`.
    fn guarantee_valid(cmt: cmt,
                       mutbl: ast::mutability,
                       scope_r: ty::region) {

        #debug["guarantee_valid(cmt=%s, mutbl=%s, scope_r=%s)",
               self.bccx.cmt_to_repr(cmt),
               self.bccx.mut_to_str(mutbl),
               region_to_str(self.tcx(), scope_r)];
        let _i = indenter();

        alt cmt.lp {
          // If this expression is a loanable path, we MUST take out a loan.
          // This is somewhat non-obvious.  You might think, for example, that
          // if we have an immutable local variable `x` whose value is being
          // borrowed, we could rely on `x` not to change.  This is not so,
          // however, because even immutable locals can be moved.  So we take
          // out a loan on `x`, guaranteeing that it remains immutable for the
          // duration of the reference: if there is an attempt to move it
          // within that scope, the loan will be detected and an error will be
          // reported.
          some(_) {
            alt scope_r {
              ty::re_scope(scope_id) {
                alt self.bccx.loan(cmt, mutbl) {
                  ok(loans) { self.add_loans(scope_id, loans); }
                  err(e) { self.bccx.report(e); }
                }
              }
              _ {
                self.bccx.span_err(
                    cmt.span,
                    #fmt["Cannot guarantee the stability \
                          of this expression for the entirety of \
                          its lifetime, %s",
                         region_to_str(self.tcx(), scope_r)]);
              }
            }
          }

          // The path is not loanable: in that case, we must try and preserve
          // it dynamically (or see that it is preserved by virtue of being
          // rooted in some immutable path)
          none {
            self.bccx.report_if_err(
                check_sup_mutbl(mutbl, cmt).chain { |_ok|
                    let opt_scope_id = alt scope_r {
                      ty::re_scope(scope_id) { some(scope_id) }
                      _ { none }
                    };

                    self.bccx.preserve(cmt, opt_scope_id)
                })
          }
        }
    }

    fn add_loans(scope_id: ast::node_id, loans: @const [loan]) {
        alt self.req_loan_map.find(scope_id) {
          some(l) {
            *l += [loans];
          }
          none {
            self.req_loan_map.insert(scope_id, @mut [loans]);
          }
        }
    }

    fn gather_pat(cmt: cmt, pat: @ast::pat, alt_id: ast::node_id) {

        // Here, `cmt` is the categorization for the value being
        // matched and pat is the pattern it is being matched against.
        //
        // In general, the way that this works is that we

        #debug["gather_pat: id=%d pat=%s cmt=%s alt_id=%d",
               pat.id, pprust::pat_to_str(pat),
               self.bccx.cmt_to_repr(cmt), alt_id];
        let _i = indenter();

        let tcx = self.tcx();
        alt pat.node {
          ast::pat_wild {
            // _
          }

          ast::pat_enum(_, none) {
            // variant(*)
          }
          ast::pat_enum(_, some(subpats)) {
            // variant(x, y, z)
            for subpats.each { |subpat|
                let subcmt = self.bccx.cat_variant(pat, cmt, subpat);
                self.gather_pat(subcmt, subpat, alt_id);
            }
          }

          ast::pat_ident(_, none) if self.pat_is_variant(pat) {
            // nullary variant
            #debug["nullary variant"];
          }
          ast::pat_ident(id, o_pat) {
            // x or x @ p --- `x` must remain valid for the scope of the alt
            #debug["defines identifier %s", pprust::path_to_str(id)];
            self.guarantee_valid(cmt, m_const, ty::re_scope(alt_id));
            for o_pat.each { |p| self.gather_pat(cmt, p, alt_id); }
          }

          ast::pat_rec(field_pats, _) {
            // {f1: p1, ..., fN: pN}
            for field_pats.each { |fp|
                let cmt_field = self.bccx.cat_field(pat, cmt, fp.ident,
                                                    tcx.ty(fp.pat));
                self.gather_pat(cmt_field, fp.pat, alt_id);
            }
          }

          ast::pat_tup(subpats) {
            // (p1, ..., pN)
            for subpats.each { |subpat|
                let subcmt = self.bccx.cat_tuple_elt(pat, cmt, subpat);
                self.gather_pat(subcmt, subpat, alt_id);
            }
          }

          ast::pat_box(subpat) | ast::pat_uniq(subpat) {
            // @p1, ~p1
            alt self.bccx.cat_deref(pat, cmt, true) {
              some(subcmt) { self.gather_pat(subcmt, subpat, alt_id); }
              none { tcx.sess.span_bug(pat.span, "Non derefable type"); }
            }
          }

          ast::pat_lit(_) | ast::pat_range(_, _) { /*always ok*/ }
        }
    }

    fn pat_is_variant(pat: @ast::pat) -> bool {
        pat_util::pat_is_variant(self.bccx.tcx.def_map, pat)
    }
}

// ----------------------------------------------------------------------
// Checking loans
//
// Phase 2 of check: we walk down the tree and check that:
// 1. assignments are always made to mutable locations;
// 2. loans made in overlapping scopes do not conflict
// 3. assignments do not affect things loaned out as immutable
// 4. moves to dnot affect things loaned out in any way

enum check_loan_ctxt = @{
    bccx: borrowck_ctxt,
    req_loan_map: req_loan_map
};

fn check_loans(bccx: borrowck_ctxt,
               req_loan_map: req_loan_map,
               crate: @ast::crate) {
    let clcx = check_loan_ctxt(@{bccx: bccx, req_loan_map: req_loan_map});
    let vt = visit::mk_vt(@{visit_expr: check_loans_in_expr,
                            visit_block: check_loans_in_block
                            with *visit::default_visitor()});
    visit::visit_crate(*crate, clcx, vt);
}

impl methods for check_loan_ctxt {
    fn tcx() -> ty::ctxt { self.bccx.tcx }

    fn walk_loans(scope_id: ast::node_id,
                  f: fn(loan) -> bool) {
        let mut scope_id = scope_id;
        let parents = self.tcx().region_map.parents;
        let req_loan_map = self.req_loan_map;

        loop {
            for req_loan_map.find(scope_id).each { |loanss|
                for (*loanss).each { |loans|
                    for (*loans).each { |loan|
                        if !f(loan) { ret; }
                    }
                }
            }

            alt parents.find(scope_id) {
              none { ret; }
              some(next_scope_id) { scope_id = next_scope_id; }
            }
        }
    }

    fn walk_loans_of(scope_id: ast::node_id,
                     lp: @loan_path,
                     f: fn(loan) -> bool) {
        for self.walk_loans(scope_id) { |loan|
            if loan.lp == lp {
                if !f(loan) { ret; }
            }
        }
    }

    fn check_for_conflicting_loans(scope_id: ast::node_id) {
        let new_loanss = alt self.req_loan_map.find(scope_id) {
            none { ret; }
            some(loanss) { loanss }
        };

        let par_scope_id = self.tcx().region_map.parents.get(scope_id);
        for self.walk_loans(par_scope_id) { |old_loan|
            for (*new_loanss).each { |new_loans|
                for (*new_loans).each { |new_loan|
                    if old_loan.lp != new_loan.lp { cont; }
                    alt (old_loan.mutbl, new_loan.mutbl) {
                      (m_const, _) | (_, m_const) |
                      (m_mutbl, m_mutbl) | (m_imm, m_imm) {
                        /*ok*/
                      }

                      (m_mutbl, m_imm) | (m_imm, m_mutbl) {
                        self.bccx.span_err(
                            new_loan.cmt.span,
                            #fmt["loan of %s as %s \
                                  conflicts with prior loan",
                                 self.bccx.cmt_to_str(new_loan.cmt),
                                 self.bccx.mut_to_str(new_loan.mutbl)]);
                        self.bccx.span_note(
                            old_loan.cmt.span,
                            #fmt["prior loan as %s granted here",
                                 self.bccx.mut_to_str(old_loan.mutbl)]);
                      }
                    }
                }
            }
        }
    }

    fn check_assignment(ex: @ast::expr) {
        let cmt = self.bccx.cat_expr(ex);

        #debug["check_assignment(cmt=%s)",
               self.bccx.cmt_to_repr(cmt)];

        alt cmt.mutbl {
          m_mutbl { /*ok*/ }
          m_const | m_imm {
            self.bccx.span_err(
                ex.span,
                #fmt["Cannot assign to %s", self.bccx.cmt_to_str(cmt)]);
            ret;
          }
        }

        // check for a conflicting loan as well
        let lp = alt cmt.lp {
          none { ret; }
          some(lp) { lp }
        };
        for self.walk_loans_of(ex.id, lp) { |loan|
            alt loan.mutbl {
              m_mutbl | m_const { /*ok*/ }
              m_imm {
                self.bccx.span_err(
                    ex.span,
                    #fmt["cannot assign to %s due to outstanding loan",
                         self.bccx.cmt_to_str(cmt)]);
                self.bccx.span_note(
                    loan.cmt.span,
                    #fmt["loan of %s granted here",
                         self.bccx.cmt_to_str(loan.cmt)]);
                ret;
              }
            }
        }
    }

    fn check_move_out(ex: @ast::expr) {
        let cmt = self.bccx.cat_expr(ex);

        alt cmt.cat {
          // Rvalues and locals can be moved:
          cat_rvalue(_) | cat_local(_) { /*ok*/ }

          // Owned arguments can be moved:
          cat_arg(_) if cmt.mutbl == m_mutbl { /*ok*/ }

          // Nothing else.
          _ {
            self.bccx.span_err(
                ex.span,
                #fmt["Cannot move from %s", self.bccx.cmt_to_str(cmt)]);
            ret;
          }
        }

        // check for a conflicting loan:
        let lp = alt cmt.lp {
          none { ret; }
          some(lp) { lp }
        };
        for self.walk_loans_of(ex.id, lp) { |loan|
            self.bccx.span_err(
                ex.span,
                #fmt["Cannot move from %s due to outstanding loan",
                     self.bccx.cmt_to_str(cmt)]);
            self.bccx.span_note(
                loan.cmt.span,
                #fmt["Loan of %s granted here",
                     self.bccx.cmt_to_str(loan.cmt)]);
            ret;
        }
    }
}

fn check_loans_in_expr(expr: @ast::expr,
                       &&self: check_loan_ctxt,
                       vt: visit::vt<check_loan_ctxt>) {
    self.check_for_conflicting_loans(expr.id);
    alt expr.node {
      ast::expr_swap(l, r) {
        self.check_assignment(l);
        self.check_assignment(r);
      }
      ast::expr_move(dest, src) {
        self.check_assignment(dest);
        self.check_move_out(src);
      }
      ast::expr_assign(dest, _) |
      ast::expr_assign_op(_, dest, _) {
        self.check_assignment(dest);
      }
      ast::expr_call(f, args, _) {
        let arg_tys = ty::ty_fn_args(ty::expr_ty(self.tcx(), f));
        vec::iter2(args, arg_tys) { |arg, arg_ty|
            alt ty::resolved_mode(self.tcx(), arg_ty.mode) {
              ast::by_move {
                self.check_move_out(arg);
              }
              ast::by_mutbl_ref |
              ast::by_ref {
                // these are translated into loans
              }
              ast::by_copy | ast::by_val {
              }
            }
        }
      }
      _ { }
    }
    visit::visit_expr(expr, self, vt);
}

fn check_loans_in_block(blk: ast::blk,
                        &&self: check_loan_ctxt,
                        vt: visit::vt<check_loan_ctxt>) {
    self.check_for_conflicting_loans(blk.node.id);
    visit::visit_block(blk, self, vt);
}

// ----------------------------------------------------------------------
// Categorization
//
// Imagine a routine ToAddr(Expr) that evaluates an expression and returns an
// address where the result is to be found.  If Expr is an lvalue, then this
// is the address of the lvalue.  If Expr is an rvalue, this is the address of
// some temporary spot in memory where the result is stored.
//
// Now, cat_expr() classies the expression Expr and the address A=ToAddr(Expr)
// as follows:
//
// - cat: what kind of expression was this?  This is a subset of the
//   full expression forms which only includes those that we care about
//   for the purpose of the analysis.
// - mutbl: mutability of the address A
// - ty: the type of data found at the address A
//
// The resulting categorization tree differs somewhat from the expressions
// themselves.  For example, auto-derefs are explicit.  Also, an index a[b] is
// decomposed into two operations: a derefence to reach the array data and
// then an index to jump forward to the relevant item.

// Categorizes a derefable type.  Note that we include vectors and strings as
// derefable (we model an index as the combination of a deref and then a
// pointer adjustment).
fn deref_kind(tcx: ty::ctxt, t: ty::t) -> deref_kind {
    alt ty::get(t).struct {
      ty::ty_uniq(*) | ty::ty_vec(*) | ty::ty_str |
      ty::ty_evec(_, ty::vstore_uniq) |
      ty::ty_estr(ty::vstore_uniq) {
        deref_ptr(uniq_ptr)
      }

      ty::ty_rptr(*) |
      ty::ty_evec(_, ty::vstore_slice(_)) |
      ty::ty_estr(ty::vstore_slice(_)) {
        deref_ptr(region_ptr)
      }

      ty::ty_box(*) |
      ty::ty_evec(_, ty::vstore_box) |
      ty::ty_estr(ty::vstore_box) {
        deref_ptr(gc_ptr)
      }

      ty::ty_ptr(*) {
        deref_ptr(unsafe_ptr)
      }

      ty::ty_enum(*) {
        deref_comp(comp_variant)
      }

      ty::ty_res(*) {
        deref_comp(comp_res)
      }

      _ {
        tcx.sess.bug(
            #fmt["deref_cat() invoked on non-derefable type %s",
                 ty_to_str(tcx, t)]);
      }
    }
}

iface ast_node {
    fn id() -> ast::node_id;
    fn span() -> span;
}

impl of ast_node for @ast::expr {
    fn id() -> ast::node_id { self.id }
    fn span() -> span { self.span }
}

impl of ast_node for @ast::pat {
    fn id() -> ast::node_id { self.id }
    fn span() -> span { self.span }
}

impl methods for ty::ctxt {
    fn ty<N: ast_node>(node: N) -> ty::t {
        ty::node_id_to_type(self, node.id())
    }
}

impl categorize_methods for borrowck_ctxt {
    fn cat_borrow_of_expr(expr: @ast::expr) -> cmt {
        // a borrowed expression must be either an @, ~, or a vec/@, vec/~
        let expr_ty = ty::expr_ty(self.tcx, expr);
        alt ty::get(expr_ty).struct {
          ty::ty_vec(*) | ty::ty_evec(*) |
          ty::ty_str | ty::ty_estr(*) {
            self.cat_index(expr, expr)
          }

          ty::ty_uniq(*) | ty::ty_box(*) | ty::ty_rptr(*) {
            let cmt = self.cat_expr(expr);
            self.cat_deref(expr, cmt, true).get()
          }

          _ {
            self.tcx.sess.span_bug(
                expr.span,
                #fmt["Borrowing of non-derefable type `%s`",
                     ty_to_str(self.tcx, expr_ty)]);
          }
        }
    }

    fn cat_expr(expr: @ast::expr) -> cmt {
        let tcx = self.tcx;
        let expr_ty = tcx.ty(expr);

        #debug["cat_expr: id=%d expr=%s",
               expr.id, pprust::expr_to_str(expr)];

        if self.method_map.contains_key(expr.id) {
            ret @{id:expr.id, span:expr.span,
                  cat:cat_rvalue(rv_method), lp:none,
                  mutbl:m_imm, ty:expr_ty};
        }

        alt expr.node {
          ast::expr_unary(ast::deref, e_base) {
            let base_cmt = self.cat_expr(e_base);
            alt self.cat_deref(expr, base_cmt, true) {
              some(cmt) { ret cmt; }
              none {
                tcx.sess.span_bug(
                    e_base.span,
                    #fmt["Explicit deref of non-derefable type `%s`",
                         ty_to_str(tcx, tcx.ty(e_base))]);
              }
            }
          }

          ast::expr_field(base, f_name, _) {
            let base_cmt = self.cat_autoderef(expr, base);
            self.cat_field(expr, base_cmt, f_name, expr_ty)
          }

          ast::expr_index(base, _) {
            self.cat_index(expr, base)
          }

          ast::expr_path(_) {
            let def = self.tcx.def_map.get(expr.id);
            self.cat_def(expr, expr_ty, def)
          }

          ast::expr_addr_of(*) | ast::expr_call(*) | ast::expr_bind(*) |
          ast::expr_swap(*) | ast::expr_move(*) | ast::expr_assign(*) |
          ast::expr_assign_op(*) | ast::expr_fn(*) | ast::expr_fn_block(*) |
          ast::expr_assert(*) | ast::expr_check(*) | ast::expr_ret(*) |
          ast::expr_be(*) | ast::expr_loop_body(*) | ast::expr_unary(*) |
          ast::expr_copy(*) | ast::expr_cast(*) | ast::expr_fail(*) |
          ast::expr_vstore(*) | ast::expr_vec(*) | ast::expr_tup(*) |
          ast::expr_if_check(*) | ast::expr_if(*) | ast::expr_log(*) |
          ast::expr_new(*) | ast::expr_binary(*) | ast::expr_while(*) |
          ast::expr_block(*) | ast::expr_loop(*) | ast::expr_alt(*) |
          ast::expr_lit(*) | ast::expr_break | ast::expr_mac(*) |
          ast::expr_cont | ast::expr_rec(*) {
            @{id:expr.id, span:expr.span,
              cat:cat_rvalue(rv_misc), lp:none,
              mutbl:m_imm, ty:expr_ty}
          }
        }
    }

    fn cat_field<N:ast_node>(node: N, base_cmt: cmt,
                             f_name: str, f_ty: ty::t) -> cmt {
        let f_mutbl = alt field_mutbl(self.tcx, base_cmt.ty, f_name) {
          some(f_mutbl) { f_mutbl }
          none {
            self.tcx.sess.span_bug(
                node.span(),
                #fmt["Cannot find field `%s` in type `%s`",
                     f_name, ty_to_str(self.tcx, base_cmt.ty)]);
          }
        };
        let m = alt f_mutbl {
          m_imm { base_cmt.mutbl } // imm: as mutable as the container
          m_mutbl | m_const { f_mutbl }
        };
        let lp = base_cmt.lp.map { |lp|
            @lp_comp(lp, comp_field(f_name))
        };
        @{id: node.id(), span: node.span(),
          cat: cat_comp(base_cmt, comp_field(f_name)), lp:lp,
          mutbl: m, ty: f_ty}
    }

    fn cat_deref<N:ast_node>(node: N, base_cmt: cmt,
                             expl: bool) -> option<cmt> {
        ty::deref(self.tcx, base_cmt.ty, expl).map { |mt|
            alt deref_kind(self.tcx, base_cmt.ty) {
              deref_ptr(ptr) {
                let lp = base_cmt.lp.chain { |l|
                    // Given that the ptr itself is loanable, we can
                    // loan out deref'd uniq ptrs as the data they are
                    // the only way to reach the data they point at.
                    // Other ptr types admit aliases and are therefore
                    // not loanable.
                    alt ptr {
                      uniq_ptr {some(@lp_deref(l, ptr))}
                      gc_ptr | region_ptr | unsafe_ptr {none}
                    }
                };
                @{id:node.id(), span:node.span(),
                  cat:cat_deref(base_cmt, ptr), lp:lp,
                  mutbl:mt.mutbl, ty:mt.ty}
              }

              deref_comp(comp) {
                let lp = base_cmt.lp.map { |l| @lp_comp(l, comp) };
                @{id:node.id(), span:node.span(),
                  cat:cat_comp(base_cmt, comp), lp:lp,
                  mutbl:mt.mutbl, ty:mt.ty}
              }
            }
        }
    }

    fn cat_autoderef(expr: @ast::expr, base: @ast::expr) -> cmt {
        // Creates a string of implicit derefences so long as base is
        // dereferencable.  n.b., it is important that these dereferences are
        // associated with the field/index that caused the autoderef (expr).
        // This is used later to adjust ref counts and so forth in trans.

        // Given something like base.f where base has type @m1 @m2 T, we want
        // to yield the equivalent categories to (**base).f.
        let mut cmt = self.cat_expr(base);
        loop {
            alt self.cat_deref(expr, cmt, false) {
              none { ret cmt; }
              some(cmt1) { cmt = cmt1; }
            }
        }
    }

    fn cat_index(expr: @ast::expr, base: @ast::expr) -> cmt {
        let base_cmt = self.cat_autoderef(expr, base);

        let mt = alt ty::index(self.tcx, base_cmt.ty) {
          some(mt) { mt }
          none {
            self.tcx.sess.span_bug(
                expr.span,
                #fmt["Explicit index of non-index type `%s`",
                     ty_to_str(self.tcx, base_cmt.ty)]);
          }
        };

        let ptr = alt deref_kind(self.tcx, base_cmt.ty) {
          deref_ptr(ptr) { ptr }
          deref_comp(_) {
            self.tcx.sess.span_bug(
                expr.span,
                "Deref of indexable type yielded comp kind");
          }
        };

        // make deref of vectors explicit, as explained in the comment at
        // the head of this section
        let deref_lp = base_cmt.lp.map { |lp| @lp_deref(lp, ptr) };
        let deref_cmt = @{id:expr.id, span:expr.span,
                          cat:cat_deref(base_cmt, ptr), lp:deref_lp,
                          mutbl:mt.mutbl, ty:mt.ty};
        let index_lp = deref_lp.map { |lp| @lp_comp(lp, comp_index) };
        @{id:expr.id, span:expr.span,
          cat:cat_comp(deref_cmt, comp_index), lp:index_lp,
          mutbl:mt.mutbl, ty:mt.ty}
    }

    fn cat_variant<N: ast_node>(variant: N, cmt: cmt, arg: N) -> cmt {
        @{id: variant.id(), span: variant.span(),
          cat: cat_comp(cmt, comp_variant),
          lp: cmt.lp.map { |l| @lp_comp(l, comp_variant) },
          mutbl: cmt.mutbl, // imm iff in an immutable context
          ty: self.tcx.ty(arg)}
    }

    fn cat_tuple_elt<N: ast_node>(pat: N, cmt: cmt, elt: N) -> cmt {
        @{id: pat.id(), span: pat.span(),
          cat: cat_comp(cmt, comp_tuple),
          lp: cmt.lp.map { |l| @lp_comp(l, comp_tuple) },
          mutbl: cmt.mutbl, // imm iff in an immutable context
          ty: self.tcx.ty(elt)}
    }

    fn cat_def(expr: @ast::expr,
               expr_ty: ty::t,
               def: ast::def) -> cmt {
        alt def {
          ast::def_fn(_, _) | ast::def_mod(_) |
          ast::def_native_mod(_) | ast::def_const(_) |
          ast::def_use(_) | ast::def_variant(_, _) |
          ast::def_ty(_) | ast::def_prim_ty(_) |
          ast::def_ty_param(_, _) | ast::def_class(_) |
          ast::def_region(_) {
            @{id:expr.id, span:expr.span,
              cat:cat_rvalue(rv_static_item), lp:none,
              mutbl:m_imm, ty:expr_ty}
          }

          ast::def_arg(vid, mode) {
            // Idea: make this could be rewritten to model by-ref
            // stuff as `&const` and `&mut`?

            // m: mutability of the argument
            // lp: loan path, must be none for aliasable things
            let {m,lp} = alt ty::resolved_mode(self.tcx, mode) {
              ast::by_mutbl_ref {
                {m:m_mutbl, lp:none}
              }
              ast::by_move | ast::by_copy {
                {m:m_mutbl, lp:some(@lp_arg(vid))}
              }
              ast::by_ref {
                if TREAT_CONST_AS_IMM {
                    {m:m_imm, lp:some(@lp_arg(vid))}
                } else {
                    {m:m_const, lp:none}
                }
              }
              ast::by_val {
                {m:m_imm, lp:some(@lp_arg(vid))}
              }
            };
            @{id:expr.id, span:expr.span,
              cat:cat_arg(vid), lp:lp,
              mutbl:m, ty:expr_ty}
          }

          ast::def_self(_) {
            @{id:expr.id, span:expr.span,
              cat:cat_rvalue(rv_self), lp:none,
              mutbl:m_imm, ty:expr_ty}
          }

          ast::def_upvar(upvid, inner, fn_node_id) {
            let ty = ty::node_id_to_type(self.tcx, fn_node_id);
            let proto = ty::ty_fn_proto(ty);
            alt proto {
              ast::proto_any | ast::proto_block {
                self.cat_def(expr, expr_ty, *inner)
              }
              ast::proto_bare | ast::proto_uniq | ast::proto_box {
                // FIXME #2152 allow mutation of moved upvars
                @{id:expr.id, span:expr.span,
                  cat:cat_rvalue(rv_upvar), lp:none,
                  mutbl:m_imm, ty:expr_ty}
              }
            }
          }

          ast::def_local(vid, mutbl) {
            let m = if mutbl {m_mutbl} else {m_imm};
            @{id:expr.id, span:expr.span,
              cat:cat_local(vid), lp:some(@lp_local(vid)),
              mutbl:m, ty:expr_ty}
          }

          ast::def_binding(vid) {
            // no difference between a binding and any other local variable
            // from out point of view, except that they are always immutable
            @{id:expr.id, span:expr.span,
              cat:cat_local(vid), lp:some(@lp_local(vid)),
              mutbl:m_imm, ty:expr_ty}
          }
        }
    }

    fn cat_to_str(mutbl: str, cat: categorization) -> str {
        alt cat {
          cat_rvalue(rv_method) { "method" }
          cat_rvalue(rv_static_item) { "static item" }
          cat_rvalue(rv_upvar) { "upvar" }
          cat_rvalue(rv_misc) { "non-lvalue" }
          cat_rvalue(rv_self) { "self reference" }
          cat_local(_) { mutbl + " local variable" }
          cat_arg(_) { mutbl + " argument" }
          cat_deref(_, _) { mutbl + " dereference" }
          cat_comp(_, comp_field(_)) { mutbl + " field" }
          cat_comp(_, comp_index) { mutbl + " vector/string contents" }
          cat_comp(_, comp_tuple) { mutbl + " tuple contents" }
          cat_comp(_, comp_res) { mutbl + " resource contents" }
          cat_comp(_, comp_variant) { mutbl + " enum contents" }
        }
    }

    fn cat_to_repr(cat: categorization) -> str {
        alt cat {
          cat_rvalue(rv_method) { "method" }
          cat_rvalue(rv_static_item) { "static_item" }
          cat_rvalue(rv_upvar) { "upvar" }
          cat_rvalue(rv_misc) { "rvalue" }
          cat_rvalue(rv_self) { "self" }
          cat_local(node_id) { #fmt["local(%d)", node_id] }
          cat_arg(node_id) { #fmt["arg(%d)", node_id] }
          cat_deref(cmt, ptr) {
            #fmt["%s->(%s)", self.cat_to_repr(cmt.cat), self.ptr_sigil(ptr)]
          }
          cat_comp(cmt, comp) {
            #fmt["%s.%s", self.cat_to_repr(cmt.cat), self.comp_to_repr(comp)]
          }
        }
    }

    fn mut_to_str(mutbl: ast::mutability) -> str {
        alt mutbl {
          m_mutbl { "mutable" }
          m_const { "const" }
          m_imm { "immutable" }
        }
    }

    fn ptr_sigil(ptr: ptr_kind) -> str {
        alt ptr {
          uniq_ptr { "~" }
          gc_ptr { "@" }
          region_ptr { "&" }
          unsafe_ptr { "*" }
        }
    }

    fn comp_to_repr(comp: comp_kind) -> str {
        alt comp {
          comp_field(fld) { fld }
          comp_index { "[]" }
          comp_tuple { "()" }
          comp_res { "<res>" }
          comp_variant { "<enum>" }
        }
    }

    fn lp_to_str(lp: @loan_path) -> str {
        alt *lp {
          lp_local(node_id) {
            #fmt["local(%d)", node_id]
          }
          lp_arg(node_id) {
            #fmt["arg(%d)", node_id]
          }
          lp_deref(lp, ptr) {
            #fmt["%s->(%s)", self.lp_to_str(lp),
                 self.ptr_sigil(ptr)]
          }
          lp_comp(lp, comp) {
            #fmt["%s.%s", self.lp_to_str(lp),
                 self.comp_to_repr(comp)]
          }
        }
    }

    fn cmt_to_repr(cmt: cmt) -> str {
        #fmt["{%s id:%d m:%s lp:%s ty:%s}",
             self.cat_to_repr(cmt.cat),
             cmt.id,
             self.mut_to_str(cmt.mutbl),
             cmt.lp.map_default("none", { |p| self.lp_to_str(p) }),
             ty_to_str(self.tcx, cmt.ty)]
    }

    fn cmt_to_str(cmt: cmt) -> str {
        let mut_str = self.mut_to_str(cmt.mutbl);
        self.cat_to_str(mut_str, cmt.cat)
    }

    fn bckerr_code_to_str(code: bckerr_code) -> str {
        alt code {
          err_mutbl(req, act) {
            #fmt["mutability mismatch, required %s but found %s",
                 self.mut_to_str(req), self.mut_to_str(act)]
          }
          err_mut_uniq {
            "unique value in aliasable, mutable location"
          }
          err_mut_variant {
            "enum variant in aliasable, mutable location"
          }
          err_preserve_gc {
            "GC'd value would have to be preserved for longer \
                 than the scope of the function"
          }
        }
    }

    fn report_if_err(bres: bckres<()>) {
        alt bres {
          ok(()) { }
          err(e) { self.report(e); }
        }
    }

    fn report(err: bckerr) {
        self.span_err(
            err.cmt.span,
            #fmt["illegal borrow: %s",
                 self.bckerr_code_to_str(err.code)]);
    }

    fn span_err(s: span, m: str) {
        if self.msg_level == 1u {
            self.tcx.sess.span_warn(s, m);
        } else {
            self.tcx.sess.span_err(s, m);
        }
    }

    fn span_note(s: span, m: str) {
        self.tcx.sess.span_note(s, m);
    }
}

fn field_mutbl(tcx: ty::ctxt,
               base_ty: ty::t,
               f_name: str) -> option<ast::mutability> {
    // Need to refactor so that records/class fields can be treated uniformly.
    alt ty::get(base_ty).struct {
      ty::ty_rec(fields) {
        for fields.each { |f|
            if f.ident == f_name {
                ret some(f.mt.mutbl);
            }
        }
      }
      ty::ty_class(did, substs) {
        for ty::lookup_class_fields(tcx, did).each { |fld|
            if fld.ident == f_name {
                let m = alt fld.mutability {
                  ast::class_mutable { ast::m_mutbl }
                  ast::class_immutable { ast::m_imm }
                };
                ret some(m);
            }
        }
      }
      _ { }
    }

    ret none;
}

// ----------------------------------------------------------------------
// Preserve(Ex, S) holds if ToAddr(Ex) will remain valid for the entirety of
// the scope S.

enum preserve_ctxt = @{
    bccx: borrowck_ctxt, opt_scope_id: option<ast::node_id>
};

impl preserve_methods for borrowck_ctxt {
    fn preserve(cmt: cmt,
                opt_scope_id: option<ast::node_id>) -> bckres<()> {
        preserve_ctxt(@{bccx:self, opt_scope_id:opt_scope_id}).preserve(cmt)
    }
}

impl preserve_methods for preserve_ctxt {
    fn preserve(cmt: cmt) -> bckres<()> {
        #debug["preserve(%s)", self.bccx.cmt_to_repr(cmt)];
        let _i = indenter();

        alt cmt.cat {
          cat_rvalue(_) {
            ok(())
          }
          cat_local(_) {
            // This should never happen.  Local variables are always lendable,
            // so either `loan()` should be called or there must be some
            // intermediate @ or &---they are not lendable but do not recurse.
            self.bccx.tcx.sess.span_bug(
                cmt.span,
                "preserve() called with local");
          }
          cat_arg(_) {
            // This can happen as not all args are lendable (e.g., &&
            // modes).  In that case, the caller guarantees stability.
            // This is basically a deref of a region ptr.
            ok(())
          }
          cat_comp(cmt_base, comp_field(_)) |
          cat_comp(cmt_base, comp_index) |
          cat_comp(cmt_base, comp_tuple) |
          cat_comp(cmt_base, comp_res) {
            // Most embedded components: if the base is stable, the
            // type never changes.
            self.preserve(cmt_base)
          }
          cat_comp(cmt1, comp_variant) {
            self.require_imm(cmt, cmt1, err_mut_variant)
          }
          cat_deref(cmt1, uniq_ptr) {
            self.require_imm(cmt, cmt1, err_mut_uniq)
          }
          cat_deref(_, region_ptr) {
            // References are always "stable" by induction (when the
            // reference of type &MT was created, the memory must have
            // been stable)
            ok(())
          }
          cat_deref(_, unsafe_ptr) {
            // Unsafe pointers are the user's problem
            ok(())
          }
          cat_deref(_, gc_ptr) {
            // GC'd pointers of type @MT: always stable because we can inc
            // the ref count or keep a GC root as necessary.  We need to
            // insert this id into the root_map, however.
            alt self.opt_scope_id {
              some(scope_id) {
                self.bccx.root_map.insert(cmt.id, scope_id);
                ok(())
              }
              none {
                err({cmt:cmt, code:err_preserve_gc})
              }
            }
          }
        }
    }

    fn require_imm(cmt: cmt, cmt1: cmt, code: bckerr_code) -> bckres<()> {
        // Variant contents and unique pointers: must be immutably
        // rooted to a preserved address.
        alt cmt1.mutbl {
          m_mutbl | m_const { err({cmt:cmt, code:code}) }
          m_imm { self.preserve(cmt1) }
        }
    }
}

// ----------------------------------------------------------------------
// Loan(Ex, M, S) = Ls holds if ToAddr(Ex) will remain valid for the entirety
// of the scope S, presuming that the returned set of loans `Ls` are honored.

type loan_ctxt = @{
    bccx: borrowck_ctxt,
    loans: @mut [loan]
};

impl loan_methods for borrowck_ctxt {
    fn loan(cmt: cmt,
            mutbl: ast::mutability) -> bckres<@const [loan]> {
        let lc = @{bccx: self, loans: @mut []};
        alt lc.loan(cmt, mutbl) {
          ok(()) { ok(lc.loans) }
          err(e) { err(e) }
        }
    }
}

impl loan_methods for loan_ctxt {
    fn ok_with_loan_of(cmt: cmt,
                       mutbl: ast::mutability) -> bckres<()> {
        // Note: all cmt's that we deal with will have a non-none lp, because
        // the entry point into this routine, `borrowck_ctxt::loan()`, rejects
        // any cmt with a none-lp.
        *self.loans += [{lp:option::get(cmt.lp),
                         cmt:cmt,
                         mutbl:mutbl}];
        ok(())
    }

    fn loan(cmt: cmt, req_mutbl: ast::mutability) -> bckres<()> {

        #debug["loan(%s, %s)",
               self.bccx.cmt_to_repr(cmt),
               self.bccx.mut_to_str(req_mutbl)];
        let _i = indenter();

        // see stable() above; should only be called when `cmt` is lendable
        if cmt.lp.is_none() {
            self.bccx.tcx.sess.span_bug(
                cmt.span,
                "loan() called with non-lendable value");
        }

        alt cmt.cat {
          cat_rvalue(_) {
            self.bccx.tcx.sess.span_bug(
                cmt.span,
                "rvalue with a non-none lp");
          }
          cat_local(_) | cat_arg(_) {
            self.ok_with_loan_of(cmt, req_mutbl)
          }
          cat_comp(cmt_base, comp_field(_)) |
          cat_comp(cmt_base, comp_index) |
          cat_comp(cmt_base, comp_tuple) |
          cat_comp(cmt_base, comp_res) {
            // For most components, the type of the embedded data is
            // stable.  Therefore, the base structure need only be
            // const---unless the component must be immutable.  In
            // that case, it must also be embedded in an immutable
            // location, or else the whole structure could be
            // overwritten and the component along with it.
            let base_mutbl = alt req_mutbl {
              m_imm { m_imm }
              m_const | m_mutbl { m_const }
            };

            self.loan(cmt_base, base_mutbl).chain { |_ok|
                self.ok_with_loan_of(cmt, req_mutbl)
            }
          }
          cat_comp(cmt1, comp_variant) |
          cat_deref(cmt1, uniq_ptr) {
            // Variant components: the base must be immutable, because
            // if it is overwritten, the types of the embedded data
            // could change.
            //
            // Unique pointers: the base must be immutable, because if
            // it is overwritten, the unique content will be freed.
            self.loan(cmt1, m_imm).chain { |_ok|
                self.ok_with_loan_of(cmt, req_mutbl)
            }
          }
          cat_deref(cmt1, unsafe_ptr) |
          cat_deref(cmt1, gc_ptr) |
          cat_deref(cmt1, region_ptr) {
            // Aliased data is simply not lendable.
            self.bccx.tcx.sess.span_bug(
                cmt.span,
                "aliased ptr with a non-none lp");
          }
        }
    }
}
