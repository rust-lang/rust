import syntax::ast;
import syntax::ast::{m_mutbl, m_imm, m_const};
import syntax::visit;
import syntax::ast_util;
import syntax::ast_map;
import syntax::codemap::span;
import util::ppaux::{ty_to_str, region_to_str};
import driver::session::session;
import std::map::{int_hash, hashmap, set};
import std::list;
import std::list::{list, cons, nil};
import result::{result, ok, err, extensions};
import syntax::print::pprust;
import util::common::indenter;
import ast_util::op_expr_callee_id;

export check_crate, root_map, mutbl_map;

fn check_crate(tcx: ty::ctxt,
               method_map: typeck::method_map,
               crate: @ast::crate) -> (root_map, mutbl_map) {

    // big hack to keep this off except when I want it on
    let msg_level = if tcx.sess.opts.borrowck != 0u {
        tcx.sess.opts.borrowck
    } else {
        os::getenv("RUST_BORROWCK").map_default(0u) { |v|
            option::get(uint::from_str(v))
        }
    };

    let bccx = @{tcx: tcx,
                 method_map: method_map,
                 msg_level: msg_level,
                 root_map: root_map(),
                 mutbl_map: int_hash()};

    let req_loan_map = if msg_level > 0u {
        gather_loans(bccx, crate)
    } else {
        int_hash()
    };
    check_loans(bccx, req_loan_map, crate);
    ret (bccx.root_map, bccx.mutbl_map);
}

const TREAT_CONST_AS_IMM: bool = true;

// ----------------------------------------------------------------------
// Type definitions

type borrowck_ctxt = @{tcx: ty::ctxt,
                       method_map: typeck::method_map,
                       msg_level: uint,
                       root_map: root_map,
                       mutbl_map: mutbl_map};

// a map mapping id's of expressions of task-local type (@T, []/@, etc) where
// the box needs to be kept live to the id of the scope for which they must
// stay live.
type root_map = hashmap<root_map_key, ast::node_id>;

// the keys to the root map combine the `id` of the expression with
// the number of types that it is autodereferenced.  So, for example,
// if you have an expression `x.f` and x has type ~@T, we could add an
// entry {id:x, derefs:0} to refer to `x` itself, `{id:x, derefs:1}`
// to refer to the deref of the unique pointer, and so on.
type root_map_key = {id: ast::node_id, derefs: uint};

// set of ids of local vars / formal arguments that are modified / moved.
// this is used in trans for optimization purposes.
type mutbl_map = std::map::hashmap<ast::node_id, ()>;

enum bckerr_code {
    err_mutbl(ast::mutability, ast::mutability),
    err_mut_uniq,
    err_mut_variant,
    err_preserve_gc
}

type bckerr = {cmt: cmt, code: bckerr_code};

type bckres<T> = result<T, bckerr>;

enum categorization {
    cat_rvalue,                     // result of eval'ing some misc expr
    cat_special(special_kind),      //
    cat_local(ast::node_id),        // local variable
    cat_arg(ast::node_id),          // formal argument
    cat_stack_upvar(cmt),           // upvar in stack closure
    cat_deref(cmt, uint, ptr_kind), // deref of a ptr
    cat_comp(cmt, comp_kind),       // adjust to locate an internal component
    cat_discr(cmt, ast::node_id),   // alt discriminant (see preserve())
}

// different kinds of pointers:
enum ptr_kind {uniq_ptr, gc_ptr, region_ptr, unsafe_ptr}

// I am coining the term "components" to mean "pieces of a data
// structure accessible without a dereference":
enum comp_kind {comp_tuple, comp_res, comp_variant,
                comp_field(str), comp_index(ty::t)}

// We pun on *T to mean both actual deref of a ptr as well
// as accessing of components:
enum deref_kind {deref_ptr(ptr_kind), deref_comp(comp_kind)}

// different kinds of expressions we might evaluate
enum special_kind {
    sk_method,
    sk_static_item,
    sk_self,
    sk_heap_upvar
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

fn save_and_restore<T:copy,U>(&t: T, f: fn() -> U) -> U {
    let old_t = t;
    let u <- f();
    t = old_t;
    ret u;
}

fn root_map() -> root_map {
    ret hashmap(root_map_key_hash, root_map_key_eq);

    fn root_map_key_eq(k1: root_map_key, k2: root_map_key) -> bool {
        k1.id == k2.id && k1.derefs == k2.derefs
    }

    fn root_map_key_hash(k: root_map_key) -> uint {
        (k.id << 4) as uint | k.derefs
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
        let scope_r = ty::re_scope(scope_id);
        self.guarantee_valid(cmt, m_const, scope_r);
    }

    // Special checks for various kinds of expressions:
    alt ex.node {
      ast::expr_addr_of(mutbl, base) {
        let base_cmt = self.bccx.cat_expr(base);

        // make sure that the thing we are pointing out stays valid
        // for the lifetime `scope_r` of the resulting ptr:
        let scope_r =
            alt check ty::get(tcx.ty(ex)).struct {
              ty::ty_rptr(r, _) { r }
            };
        self.guarantee_valid(base_cmt, mutbl, scope_r);
      }

      ast::expr_call(f, args, _) {
        let arg_tys = ty::ty_fn_args(ty::expr_ty(self.tcx(), f));
        let scope_r = ty::re_scope(ex.id);
        vec::iter2(args, arg_tys) { |arg, arg_ty|
            alt ty::resolved_mode(self.tcx(), arg_ty.mode) {
              ast::by_mutbl_ref {
                let arg_cmt = self.bccx.cat_expr(arg);
                self.guarantee_valid(arg_cmt, m_mutbl, scope_r);
              }
              ast::by_ref {
                let arg_cmt = self.bccx.cat_expr(arg);
                if TREAT_CONST_AS_IMM {
                    self.guarantee_valid(arg_cmt, m_imm,  scope_r);
                } else {
                    self.guarantee_valid(arg_cmt, m_const, scope_r);
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
                self.gather_pat(cmt, pat, arm.body.node.id, ex.id);
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
    // `static_scope_r`, or reports an error.  This may entail taking
    // out loans, which will be added to the `req_loan_map`.  This can
    // also entail "rooting" GC'd pointers, which means ensuring
    // dynamically that they are not freed.
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
                    #fmt["cannot guarantee the stability \
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

    fn gather_pat(cmt: cmt, pat: @ast::pat,
                  arm_id: ast::node_id, alt_id: ast::node_id) {

        // Here, `cmt` is the categorization for the value being
        // matched and pat is the pattern it is being matched against.
        //
        // In general, the way that this works is that we walk down
        // the pattern, constructing a cmt that represents the path
        // that will be taken to reach the value being matched.
        //
        // When we encounter named bindings, we take the cmt that has
        // been built up and pass it off to guarantee_valid() so that
        // we can be sure that the binding will remain valid for the
        // duration of the arm.
        //
        // The correspondence between the id in the cmt and which
        // pattern is being referred to is somewhat...subtle.  In
        // general, the id of the cmt is the id of the node that
        // produces the value.  For patterns, that's actually the
        // *subpattern*, generally speaking.
        //
        // To see what I mean about ids etc, consider:
        //
        //     let x = @@3;
        //     alt x {
        //       @@y { ... }
        //     }
        //
        // Here the cmt for `y` would be something like
        //
        //     local(x)->@->@
        //
        // where the id of `local(x)` is the id of the `x` that appears
        // in the alt, the id of `local(x)->@` is the `@y` pattern,
        // and the id of `local(x)->@->@` is the id of the `y` pattern.

        #debug["gather_pat: id=%d pat=%s cmt=%s arm_id=%d alt_id=%d",
               pat.id, pprust::pat_to_str(pat),
               self.bccx.cmt_to_repr(cmt), arm_id, alt_id];
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
                let subcmt = self.bccx.cat_variant(subpat, cmt);
                self.gather_pat(subcmt, subpat, arm_id, alt_id);
            }
          }

          ast::pat_ident(_, none) if self.pat_is_variant(pat) {
            // nullary variant
            #debug["nullary variant"];
          }
          ast::pat_ident(id, o_pat) {
            // x or x @ p --- `x` must remain valid for the scope of the alt
            #debug["defines identifier %s", pprust::path_to_str(id)];

            // Note: there is a discussion of the function of
            // cat_discr in the method preserve():
            let cmt1 = self.bccx.cat_discr(cmt, alt_id);
            let arm_scope = ty::re_scope(arm_id);
            self.guarantee_valid(cmt1, m_const, arm_scope);

            for o_pat.each { |p|
                self.gather_pat(cmt, p, arm_id, alt_id);
            }
          }

          ast::pat_rec(field_pats, _) {
            // {f1: p1, ..., fN: pN}
            for field_pats.each { |fp|
                let cmt_field = self.bccx.cat_field(fp.pat, cmt, fp.ident);
                self.gather_pat(cmt_field, fp.pat, arm_id, alt_id);
            }
          }

          ast::pat_tup(subpats) {
            // (p1, ..., pN)
            for subpats.each { |subpat|
                let subcmt = self.bccx.cat_tuple_elt(subpat, cmt);
                self.gather_pat(subcmt, subpat, arm_id, alt_id);
            }
          }

          ast::pat_box(subpat) | ast::pat_uniq(subpat) {
            // @p1, ~p1
            alt self.bccx.cat_deref(subpat, cmt, 0u, true) {
              some(subcmt) {
                self.gather_pat(subcmt, subpat, arm_id, alt_id);
              }
              none {
                tcx.sess.span_bug(pat.span, "Non derefable type");
              }
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
    req_loan_map: req_loan_map,

    // Keep track of whether we're inside a ctor, so as to
    // allow mutating immutable fields in the same class if
    // we are in a ctor, we track the self id
    mut in_ctor: bool,

    mut is_pure: bool
};

fn check_loans(bccx: borrowck_ctxt,
               req_loan_map: req_loan_map,
               crate: @ast::crate) {
    let clcx = check_loan_ctxt(@{bccx: bccx,
                                 req_loan_map: req_loan_map,
                                 mut in_ctor: false,
                                 mut is_pure: false});
    let vt = visit::mk_vt(@{visit_expr: check_loans_in_expr,
                            visit_block: check_loans_in_block,
                            visit_fn: check_loans_in_fn
                            with *visit::default_visitor()});
    visit::visit_crate(*crate, clcx, vt);
}

enum assignment_type {
    at_straight_up,
    at_swap,
    at_mutbl_ref,
}

impl methods for assignment_type {
    fn ing_form(desc: str) -> str {
        alt self {
          at_straight_up { "assigning to " + desc }
          at_swap { "swapping to and from " + desc }
          at_mutbl_ref { "taking mut reference to " + desc }
        }
    }
}

impl methods for check_loan_ctxt {
    fn tcx() -> ty::ctxt { self.bccx.tcx }

    fn walk_loans(scope_id: ast::node_id,
                  f: fn(loan) -> bool) {
        let mut scope_id = scope_id;
        let region_map = self.tcx().region_map;
        let req_loan_map = self.req_loan_map;

        loop {
            for req_loan_map.find(scope_id).each { |loanss|
                for (*loanss).each { |loans|
                    for (*loans).each { |loan|
                        if !f(loan) { ret; }
                    }
                }
            }

            alt region_map.find(scope_id) {
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

    // when we are in a pure context, we check each call to ensure
    // that the function which is invoked is itself pure.
    fn check_pure(expr: @ast::expr) {
        let tcx = self.tcx();
        alt ty::get(tcx.ty(expr)).struct {
          ty::ty_fn(_) {
            // Extract purity or unsafety based on what kind of callee
            // we've got.  This would be cleaner if we just admitted
            // that we have an effect system and carried the purity
            // etc around in the type.

            // First, check the def_map---if expr.id is present then
            // expr must be a path (at least I think that's the idea---NDM)
            let callee_purity = alt tcx.def_map.find(expr.id) {
              some(ast::def_fn(_, p)) { p }
              some(ast::def_variant(_, _)) { ast::pure_fn }
              _ {
                // otherwise it may be a method call that we can trace
                // to the def'n site:
                alt self.bccx.method_map.find(expr.id) {
                  some(typeck::method_static(did)) {
                    if did.crate == ast::local_crate {
                        alt tcx.items.get(did.node) {
                          ast_map::node_method(m, _, _) { m.decl.purity }
                          _ { tcx.sess.span_bug(expr.span,
                                                "Node not bound \
                                                 to a method") }
                        }
                    } else {
                        metadata::csearch::lookup_method_purity(
                            tcx.sess.cstore,
                            did)
                    }
                  }
                  some(typeck::method_param(iid, n_m, _, _)) |
                  some(typeck::method_iface(iid, n_m)) {
                    ty::iface_methods(tcx, iid)[n_m].purity
                  }
                  none {
                    // otherwise it's just some dang thing.  We know
                    // it cannot be unsafe because we do not allow
                    // unsafe functions to be used as values (or,
                    // rather, we only allow that inside an unsafe
                    // block, and then it's up to the user to keep
                    // things confined).
                    ast::impure_fn
                  }
                }
              }
            };

            alt callee_purity {
              ast::crust_fn | ast::pure_fn { /*ok*/ }
              ast::impure_fn {
                self.bccx.span_err(
                    expr.span,
                    "pure function calls function \
                     not known to be pure");
              }
              ast::unsafe_fn {
                self.bccx.span_err(
                    expr.span,
                    "pure function calls unsafe function");
              }
            }
          }
          _ { /* not a fn, ok */ }
        }
    }

    fn check_for_conflicting_loans(scope_id: ast::node_id) {
        let new_loanss = alt self.req_loan_map.find(scope_id) {
            none { ret; }
            some(loanss) { loanss }
        };

        let par_scope_id = self.tcx().region_map.get(scope_id);
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

    fn is_self_field(cmt: cmt) -> bool {
        alt cmt.cat {
          cat_comp(cmt_base, comp_field(_)) {
            alt cmt_base.cat {
              cat_special(sk_self) { true }
              _ { false }
            }
          }
          _ { false }
        }
    }

    fn check_assignment(at: assignment_type, ex: @ast::expr) {
        let cmt = self.bccx.cat_expr(ex);

        #debug["check_assignment(cmt=%s)",
               self.bccx.cmt_to_repr(cmt)];

        // check that the lvalue `ex` is assignable, but be careful
        // because assigning to self.foo in a ctor is always allowed.
        if !self.in_ctor || !self.is_self_field(cmt) {
            alt cmt.mutbl {
              m_mutbl { /*ok*/ }
              m_const | m_imm {
                self.bccx.span_err(
                    ex.span,
                    at.ing_form(self.bccx.cmt_to_str(cmt)));
                ret;
              }
            }
        }

        // if this is a pure function, only loan-able state can be
        // assigned, because it is uniquely tied to this function and
        // is not visible from the outside
        if self.is_pure && cmt.lp.is_none() {
            self.bccx.span_err(
                ex.span,
                #fmt["%s prohibited in pure functions",
                     at.ing_form(self.bccx.cmt_to_str(cmt))]);
        }

        // check for a conflicting loan as well, except in the case of
        // taking a mutable ref.  that will create a loan of its own
        // which will be checked for compat separately in
        // check_for_conflicting_loans()
        if at != at_mutbl_ref {
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
                        #fmt["%s prohibited due to outstanding loan",
                             at.ing_form(self.bccx.cmt_to_str(cmt))]);
                    self.bccx.span_note(
                        loan.cmt.span,
                        #fmt["loan of %s granted here",
                             self.bccx.cmt_to_str(loan.cmt)]);
                    ret;
                  }
                }
            }
        }

        self.bccx.add_to_mutbl_map(cmt);
    }

    fn check_move_out(ex: @ast::expr) {
        let cmt = self.bccx.cat_expr(ex);
        self.check_move_out_from_cmt(cmt);
    }

    fn check_move_out_from_cmt(cmt: cmt) {
        #debug["check_move_out_from_cmt(cmt=%s)",
               self.bccx.cmt_to_repr(cmt)];

        alt cmt.cat {
          // Rvalues and locals can be moved:
          cat_rvalue | cat_local(_) { }

          // Owned arguments can be moved:
          cat_arg(_) if cmt.mutbl == m_mutbl { }

          // We allow moving out of static items because the old code
          // did.  This seems consistent with permitting moves out of
          // rvalues, I guess.
          cat_special(sk_static_item) { }

          // Nothing else.
          _ {
            self.bccx.span_err(
                cmt.span,
                #fmt["moving out of %s", self.bccx.cmt_to_str(cmt)]);
            ret;
          }
        }

        self.bccx.add_to_mutbl_map(cmt);

        // check for a conflicting loan:
        let lp = alt cmt.lp {
          none { ret; }
          some(lp) { lp }
        };
        for self.walk_loans_of(cmt.id, lp) { |loan|
            self.bccx.span_err(
                cmt.span,
                #fmt["moving out of %s prohibited due to outstanding loan",
                     self.bccx.cmt_to_str(cmt)]);
            self.bccx.span_note(
                loan.cmt.span,
                #fmt["loan of %s granted here",
                     self.bccx.cmt_to_str(loan.cmt)]);
            ret;
        }
    }
}

fn check_loans_in_fn(fk: visit::fn_kind, decl: ast::fn_decl, body: ast::blk,
                     sp: span, id: ast::node_id, &&self: check_loan_ctxt,
                     visitor: visit::vt<check_loan_ctxt>) {

    save_and_restore(self.in_ctor) {||
        save_and_restore(self.is_pure) {||
            // In principle, we could consider fk_anon(*) or
            // fk_fn_block(*) to be in a ctor, I suppose, but the
            // purpose of the in_ctor flag is to allow modifications
            // of otherwise immutable fields and typestate wouldn't be
            // able to "see" into those functions anyway, so it
            // wouldn't be very helpful.
            alt fk {
              visit::fk_ctor(*) { self.in_ctor = true; }
              _ { self.in_ctor = false; }
            };

            alt decl.purity {
              ast::pure_fn { self.is_pure = true; }
              _ { }
            }

            visit::visit_fn(fk, decl, body, sp, id, self, visitor);
        }
    }
}

fn check_loans_in_expr(expr: @ast::expr,
                       &&self: check_loan_ctxt,
                       vt: visit::vt<check_loan_ctxt>) {
    self.check_for_conflicting_loans(expr.id);
    alt expr.node {
      ast::expr_swap(l, r) {
        self.check_assignment(at_swap, l);
        self.check_assignment(at_swap, r);
      }
      ast::expr_move(dest, src) {
        self.check_assignment(at_straight_up, dest);
        self.check_move_out(src);
      }
      ast::expr_assign(dest, _) |
      ast::expr_assign_op(_, dest, _) {
        self.check_assignment(at_straight_up, dest);
      }
      ast::expr_fn(_, _, _, cap_clause) |
      ast::expr_fn_block(_, _, cap_clause) {
        for (*cap_clause).each { |cap_item|
            if cap_item.is_move {
                let def = self.tcx().def_map.get(cap_item.id);

                // Hack: the type that is used in the cmt doesn't actually
                // matter here, so just subst nil instead of looking up
                // the type of the def that is referred to
                let cmt = self.bccx.cat_def(cap_item.id, cap_item.span,
                                            ty::mk_nil(self.tcx()), def);
                self.check_move_out_from_cmt(cmt);
            }
        }
      }
      ast::expr_addr_of(mutbl, base) {
        alt mutbl {
          m_const { /*all memory is const*/ }
          m_mutbl {
            // If we are taking an &mut ptr, make sure the memory
            // being pointed at is assignable in the first place:
            self.check_assignment(at_mutbl_ref, base);
          }
          m_imm {
            // XXX explain why no check is req'd here
          }
        }
      }
      ast::expr_call(f, args, _) {
        if self.is_pure {
            self.check_pure(f);
            for args.each { |arg| self.check_pure(arg) }
        }
        let arg_tys = ty::ty_fn_args(ty::expr_ty(self.tcx(), f));
        vec::iter2(args, arg_tys) { |arg, arg_ty|
            alt ty::resolved_mode(self.tcx(), arg_ty.mode) {
              ast::by_move {
                self.check_move_out(arg);
              }
              ast::by_mutbl_ref {
                self.check_assignment(at_mutbl_ref, arg);
              }
              ast::by_ref | ast::by_copy | ast::by_val {
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
    save_and_restore(self.is_pure) {||
        self.check_for_conflicting_loans(blk.node.id);

        alt blk.node.rules {
          ast::default_blk {
          }
          ast::unchecked_blk |
          ast::unsafe_blk { self.is_pure = false; }
        }

        visit::visit_block(blk, self, vt);
    }
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
            self.cat_deref(expr, cmt, 0u, true).get()
          }

          _ {
            self.tcx.sess.span_bug(
                expr.span,
                #fmt["Borrowing of non-derefable type `%s`",
                     ty_to_str(self.tcx, expr_ty)]);
          }
        }
    }

    fn cat_method_ref(expr: @ast::expr, expr_ty: ty::t) -> cmt {
        @{id:expr.id, span:expr.span,
          cat:cat_special(sk_method), lp:none,
          mutbl:m_imm, ty:expr_ty}
    }

    fn cat_rvalue(expr: @ast::expr, expr_ty: ty::t) -> cmt {
        @{id:expr.id, span:expr.span,
          cat:cat_rvalue, lp:none,
          mutbl:m_imm, ty:expr_ty}
    }

    fn cat_expr(expr: @ast::expr) -> cmt {
        #debug["cat_expr: id=%d expr=%s",
               expr.id, pprust::expr_to_str(expr)];

        let tcx = self.tcx;
        let expr_ty = tcx.ty(expr);
        alt expr.node {
          ast::expr_unary(ast::deref, e_base) {
            if self.method_map.contains_key(expr.id) {
                ret self.cat_rvalue(expr, expr_ty);
            }

            let base_cmt = self.cat_expr(e_base);
            alt self.cat_deref(expr, base_cmt, 0u, true) {
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
            if self.method_map.contains_key(expr.id) {
                ret self.cat_method_ref(expr, expr_ty);
            }

            let base_cmt = self.cat_autoderef(base);
            self.cat_field(expr, base_cmt, f_name)
          }

          ast::expr_index(base, _) {
            if self.method_map.contains_key(expr.id) {
                ret self.cat_rvalue(expr, expr_ty);
            }

            self.cat_index(expr, base)
          }

          ast::expr_path(_) {
            let def = self.tcx.def_map.get(expr.id);
            self.cat_def(expr.id, expr.span, expr_ty, def)
          }

          ast::expr_addr_of(*) | ast::expr_call(*) | ast::expr_bind(*) |
          ast::expr_swap(*) | ast::expr_move(*) | ast::expr_assign(*) |
          ast::expr_assign_op(*) | ast::expr_fn(*) | ast::expr_fn_block(*) |
          ast::expr_assert(*) | ast::expr_check(*) | ast::expr_ret(*) |
          ast::expr_loop_body(*) | ast::expr_unary(*) |
          ast::expr_copy(*) | ast::expr_cast(*) | ast::expr_fail(*) |
          ast::expr_vstore(*) | ast::expr_vec(*) | ast::expr_tup(*) |
          ast::expr_if_check(*) | ast::expr_if(*) | ast::expr_log(*) |
          ast::expr_new(*) | ast::expr_binary(*) | ast::expr_while(*) |
          ast::expr_block(*) | ast::expr_loop(*) | ast::expr_alt(*) |
          ast::expr_lit(*) | ast::expr_break | ast::expr_mac(*) |
          ast::expr_cont | ast::expr_rec(*) {
            ret self.cat_rvalue(expr, expr_ty);
          }
        }
    }

    fn cat_discr(cmt: cmt, alt_id: ast::node_id) -> cmt {
        ret @{cat:cat_discr(cmt, alt_id) with *cmt};
    }

    fn cat_field<N:ast_node>(node: N, base_cmt: cmt, f_name: str) -> cmt {
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
          mutbl: m, ty: self.tcx.ty(node)}
    }

    fn cat_deref<N:ast_node>(node: N, base_cmt: cmt, derefs: uint,
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
                  cat:cat_deref(base_cmt, derefs, ptr), lp:lp,
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

    fn cat_autoderef(base: @ast::expr) -> cmt {
        // Creates a string of implicit derefences so long as base is
        // dereferencable.  n.b., it is important that these dereferences are
        // associated with the field/index that caused the autoderef (expr).
        // This is used later to adjust ref counts and so forth in trans.

        // Given something like base.f where base has type @m1 @m2 T, we want
        // to yield the equivalent categories to (**base).f.
        let mut cmt = self.cat_expr(base);
        let mut ctr = 0u;
        loop {
            ctr += 1u;
            alt self.cat_deref(base, cmt, ctr, false) {
              none { ret cmt; }
              some(cmt1) { cmt = cmt1; }
            }
        }
    }

    fn cat_index(expr: @ast::expr, base: @ast::expr) -> cmt {
        let base_cmt = self.cat_autoderef(base);

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
                          cat:cat_deref(base_cmt, 0u, ptr), lp:deref_lp,
                          mutbl:mt.mutbl, ty:mt.ty};
        let comp = comp_index(base_cmt.ty);
        let index_lp = deref_lp.map { |lp| @lp_comp(lp, comp) };
        @{id:expr.id, span:expr.span,
          cat:cat_comp(deref_cmt, comp), lp:index_lp,
          mutbl:mt.mutbl, ty:mt.ty}
    }

    fn cat_variant<N: ast_node>(arg: N, cmt: cmt) -> cmt {
        @{id: arg.id(), span: arg.span(),
          cat: cat_comp(cmt, comp_variant),
          lp: cmt.lp.map { |l| @lp_comp(l, comp_variant) },
          mutbl: cmt.mutbl, // imm iff in an immutable context
          ty: self.tcx.ty(arg)}
    }

    fn cat_tuple_elt<N: ast_node>(elt: N, cmt: cmt) -> cmt {
        @{id: elt.id(), span: elt.span(),
          cat: cat_comp(cmt, comp_tuple),
          lp: cmt.lp.map { |l| @lp_comp(l, comp_tuple) },
          mutbl: cmt.mutbl, // imm iff in an immutable context
          ty: self.tcx.ty(elt)}
    }

    fn cat_def(id: ast::node_id,
               span: span,
               expr_ty: ty::t,
               def: ast::def) -> cmt {
        alt def {
          ast::def_fn(_, _) | ast::def_mod(_) |
          ast::def_native_mod(_) | ast::def_const(_) |
          ast::def_use(_) | ast::def_variant(_, _) |
          ast::def_ty(_) | ast::def_prim_ty(_) |
          ast::def_ty_param(_, _) | ast::def_class(_) |
          ast::def_region(_) {
            @{id:id, span:span,
              cat:cat_special(sk_static_item), lp:none,
              mutbl:m_imm, ty:expr_ty}
          }

          ast::def_arg(vid, mode) {
            // Idea: make this could be rewritten to model by-ref
            // stuff as `&const` and `&mut`?

            // m: mutability of the argument
            // lp: loan path, must be none for aliasable things
            let {m,lp} = alt ty::resolved_mode(self.tcx, mode) {
              ast::by_mutbl_ref {
                {m: m_mutbl,
                 lp: none}
              }
              ast::by_move | ast::by_copy {
                {m: m_mutbl,
                 lp: some(@lp_arg(vid))}
              }
              ast::by_ref {
                {m: if TREAT_CONST_AS_IMM {m_imm} else {m_const},
                 lp: none}
              }
              ast::by_val {
                // by-value is this hybrid mode where we have a
                // pointer but we do not own it.  This is not
                // considered loanable because, for example, a by-ref
                // and and by-val argument might both actually contain
                // the same unique ptr.
                {m: m_imm,
                 lp: none}
              }
            };
            @{id:id, span:span,
              cat:cat_arg(vid), lp:lp,
              mutbl:m, ty:expr_ty}
          }

          ast::def_self(_) {
            @{id:id, span:span,
              cat:cat_special(sk_self), lp:none,
              mutbl:m_imm, ty:expr_ty}
          }

          ast::def_upvar(upvid, inner, fn_node_id) {
            let ty = ty::node_id_to_type(self.tcx, fn_node_id);
            let proto = ty::ty_fn_proto(ty);
            alt proto {
              ast::proto_any | ast::proto_block {
                let upcmt = self.cat_def(id, span, expr_ty, *inner);
                @{id:id, span:span,
                  cat:cat_stack_upvar(upcmt), lp:upcmt.lp,
                  mutbl:upcmt.mutbl, ty:upcmt.ty}
              }
              ast::proto_bare | ast::proto_uniq | ast::proto_box {
                // FIXME #2152 allow mutation of moved upvars
                @{id:id, span:span,
                  cat:cat_special(sk_heap_upvar), lp:none,
                  mutbl:m_imm, ty:expr_ty}
              }
            }
          }

          ast::def_local(vid, mutbl) {
            let m = if mutbl {m_mutbl} else {m_imm};
            @{id:id, span:span,
              cat:cat_local(vid), lp:some(@lp_local(vid)),
              mutbl:m, ty:expr_ty}
          }

          ast::def_binding(vid) {
            // no difference between a binding and any other local variable
            // from out point of view, except that they are always immutable
            @{id:id, span:span,
              cat:cat_local(vid), lp:some(@lp_local(vid)),
              mutbl:m_imm, ty:expr_ty}
          }
        }
    }

    fn cat_to_repr(cat: categorization) -> str {
        alt cat {
          cat_special(sk_method) { "method" }
          cat_special(sk_static_item) { "static_item" }
          cat_special(sk_self) { "self" }
          cat_special(sk_heap_upvar) { "heap-upvar" }
          cat_stack_upvar(_) { "stack-upvar" }
          cat_rvalue { "rvalue" }
          cat_local(node_id) { #fmt["local(%d)", node_id] }
          cat_arg(node_id) { #fmt["arg(%d)", node_id] }
          cat_deref(cmt, derefs, ptr) {
            #fmt["%s->(%s, %u)", self.cat_to_repr(cmt.cat),
                 self.ptr_sigil(ptr), derefs]
          }
          cat_comp(cmt, comp) {
            #fmt["%s.%s", self.cat_to_repr(cmt.cat), self.comp_to_repr(comp)]
          }
          cat_discr(cmt, _) { self.cat_to_repr(cmt.cat) }
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
          comp_index(_) { "[]" }
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
        alt cmt.cat {
          cat_special(sk_method) { "method" }
          cat_special(sk_static_item) { "static item" }
          cat_special(sk_self) { "self reference" }
          cat_special(sk_heap_upvar) { "upvar" }
          cat_rvalue { "non-lvalue" }
          cat_local(_) { mut_str + " local variable" }
          cat_arg(_) { mut_str + " argument" }
          cat_deref(*) { "dereference of " + mut_str + " pointer" }
          cat_stack_upvar(_) { mut_str + " upvar" }
          cat_comp(_, comp_field(_)) { mut_str + " field" }
          cat_comp(_, comp_tuple) { "tuple content" }
          cat_comp(_, comp_res) { "resource content" }
          cat_comp(_, comp_variant) { "enum content" }
          cat_comp(_, comp_index(t)) {
            alt ty::get(t).struct {
              ty::ty_vec(*) | ty::ty_evec(*) {
                mut_str + " vec content"
              }

              ty::ty_str | ty::ty_estr(*) {
                mut_str + " str content"
              }

              _ { mut_str + " indexed content" }
            }
          }
          cat_discr(cmt, _) {
            self.cmt_to_str(cmt)
          }
        }
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

    fn add_to_mutbl_map(cmt: cmt) {
        alt cmt.cat {
          cat_local(id) | cat_arg(id) {
            self.mutbl_map.insert(id, ());
          }
          cat_stack_upvar(cmt) {
            self.add_to_mutbl_map(cmt);
          }
          _ {}
        }
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

impl preserve_methods for borrowck_ctxt {
    fn preserve(cmt: cmt, opt_scope_id: option<ast::node_id>) -> bckres<()> {
        #debug["preserve(%s)", self.cmt_to_repr(cmt)];
        let _i = indenter();

        alt cmt.cat {
          cat_rvalue | cat_special(_) {
            ok(())
          }
          cat_stack_upvar(cmt) {
            self.preserve(cmt, opt_scope_id)
          }
          cat_local(_) {
            // This should never happen.  Local variables are always lendable,
            // so either `loan()` should be called or there must be some
            // intermediate @ or &---they are not lendable but do not recurse.
            self.tcx.sess.span_bug(
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
          cat_comp(cmt_base, comp_index(_)) |
          cat_comp(cmt_base, comp_tuple) |
          cat_comp(cmt_base, comp_res) {
            // Most embedded components: if the base is stable, the
            // type never changes.
            self.preserve(cmt_base, opt_scope_id)
          }
          cat_comp(cmt1, comp_variant) {
            self.require_imm(cmt, cmt1, opt_scope_id, err_mut_variant)
          }
          cat_deref(cmt1, _, uniq_ptr) {
            self.require_imm(cmt, cmt1, opt_scope_id, err_mut_uniq)
          }
          cat_deref(_, _, region_ptr) {
            // References are always "stable" by induction (when the
            // reference of type &MT was created, the memory must have
            // been stable)
            ok(())
          }
          cat_deref(_, _, unsafe_ptr) {
            // Unsafe pointers are the user's problem
            ok(())
          }
          cat_deref(base, derefs, gc_ptr) {
            // GC'd pointers of type @MT: always stable because we can
            // inc the ref count or keep a GC root as necessary.  We
            // need to insert this id into the root_map, however.
            alt opt_scope_id {
              some(scope_id) {
                #debug["Inserting root map entry for %s: \
                        node %d:%u -> scope %d",
                       self.cmt_to_repr(cmt), base.id,
                       derefs, scope_id];

                let rk = {id: base.id, derefs: derefs};
                self.root_map.insert(rk, scope_id);
                ok(())
              }
              none {
                err({cmt:cmt, code:err_preserve_gc})
              }
            }
          }
          cat_discr(base, alt_id) {
            // Subtle: in an alt, we must ensure that each binding
            // variable remains valid for the duration of the arm in
            // which it appears, presuming that this arm is taken.
            // But it is inconvenient in trans to root something just
            // for one arm.  Therefore, we insert a cat_discr(),
            // basically a special kind of category that says "if this
            // value must be dynamically rooted, root it for the scope
            // `alt_id`.
            //
            // As an example, consider this scenario:
            //
            //    let mut x = @some(3);
            //    alt *x { some(y) {...} none {...} }
            //
            // Technically, the value `x` need only be rooted
            // in the `some` arm.  However, we evaluate `x` in trans
            // before we know what arm will be taken, so we just
            // always root it for the duration of the alt.
            //
            // As a second example, consider *this* scenario:
            //
            //    let x = @mut @some(3);
            //    alt x { @@some(y) {...} @@none {...} }
            //
            // Here again, `x` need only be rooted in the `some` arm.
            // In this case, the value which needs to be rooted is
            // found only when checking which pattern matches: but
            // this check is done before entering the arm.  Therefore,
            // even in this case we just choose to keep the value
            // rooted for the entire alt.  This means the value will be
            // rooted even if the none arm is taken.  Oh well.
            //
            // At first, I tried to optimize the second case to only
            // root in one arm, but the result was suboptimal: first,
            // it interfered with the construction of phi nodes in the
            // arm, as we were adding code to root values before the
            // phi nodes were added.  This could have been addressed
            // with a second basic block.  However, the naive approach
            // also yielded suboptimal results for patterns like:
            //
            //    let x = @mut @...;
            //    alt x { @@some_variant(y) | @@some_other_variant(y) {...} }
            //
            // The reason is that we would root the value once for
            // each pattern and not once per arm.  This is also easily
            // fixed, but it's yet more code for what is really quite
            // the corner case.
            //
            // Nonetheless, if you decide to optimize this case in the
            // future, you need only adjust where the cat_discr()
            // node appears to draw the line between what will be rooted
            // in the *arm* vs the *alt*.

            // current scope must be the arm, which is always a child of alt:
            assert self.tcx.region_map.get(opt_scope_id.get()) == alt_id;

            self.preserve(base, some(alt_id))
          }
        }
    }

    fn require_imm(cmt: cmt,
                   cmt1: cmt,
                   opt_scope_id: option<ast::node_id>,
                   code: bckerr_code) -> bckres<()> {
        // Variant contents and unique pointers: must be immutably
        // rooted to a preserved address.
        alt cmt1.mutbl {
          m_mutbl | m_const { err({cmt:cmt, code:code}) }
          m_imm { self.preserve(cmt1, opt_scope_id) }
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
          cat_rvalue | cat_special(_) {
            // should never be loanable
            self.bccx.tcx.sess.span_bug(
                cmt.span,
                "rvalue with a non-none lp");
          }
          cat_local(_) | cat_arg(_) | cat_stack_upvar(_) {
            self.ok_with_loan_of(cmt, req_mutbl)
          }
          cat_discr(base, _) {
            self.loan(base, req_mutbl)
          }
          cat_comp(cmt_base, comp_field(_)) |
          cat_comp(cmt_base, comp_index(_)) |
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
          cat_deref(cmt1, _, uniq_ptr) {
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
          cat_deref(cmt1, _, unsafe_ptr) |
          cat_deref(cmt1, _, gc_ptr) |
          cat_deref(cmt1, _, region_ptr) {
            // Aliased data is simply not lendable.
            self.bccx.tcx.sess.span_bug(
                cmt.span,
                "aliased ptr with a non-none lp");
          }
        }
    }
}
