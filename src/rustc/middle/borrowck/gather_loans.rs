// ----------------------------------------------------------------------
// Gathering loans
//
// The borrow check proceeds in two phases. In phase one, we gather the full
// set of loans that are required at any point.  These are sorted according to
// their associated scopes.  In phase two, checking loans, we will then make
// sure that all of these loans are honored.

import categorization::{public_methods, opt_deref_kind};
import loan::public_methods;
import preserve::{public_methods, preserve_condition, pc_ok, pc_if_pure};

export gather_loans;

/// Context used while gathering loans:
///
/// - `bccx`: the the borrow check context
/// - `req_maps`: the maps computed by `gather_loans()`, see def'n of the
///   type `req_maps` for more info
/// - `item_ub`: the id of the block for the enclosing fn/method item
/// - `root_ub`: the id of the outermost block for which we can root
///   an `@T`.  This is the id of the innermost enclosing
///   loop or function body.
///
/// The role of `root_ub` is to prevent us from having to accumulate
/// vectors of rooted items at runtime.  Consider this case:
///
///     fn foo(...) -> int {
///         let mut ptr: &int;
///         while some_cond {
///             let x: @int = ...;
///             ptr = &*x;
///         }
///         *ptr
///     }
///
/// If we are not careful here, we would infer the scope of the borrow `&*x`
/// to be the body of the function `foo()` as a whole.  We would then
/// have root each `@int` that is produced, which is an unbounded number.
/// No good.  Instead what will happen is that `root_ub` will be set to the
/// body of the while loop and we will refuse to root the pointer `&*x`
/// because it would have to be rooted for a region greater than `root_ub`.
enum gather_loan_ctxt = @{bccx: borrowck_ctxt,
                          req_maps: req_maps,
                          mut item_ub: ast::node_id,
                          mut root_ub: ast::node_id};

fn gather_loans(bccx: borrowck_ctxt, crate: @ast::crate) -> req_maps {
    let glcx = gather_loan_ctxt(@{bccx: bccx,
                                  req_maps: {req_loan_map: int_hash(),
                                             pure_map: int_hash()},
                                  mut item_ub: 0,
                                  mut root_ub: 0});
    let v = visit::mk_vt(@{visit_expr: req_loans_in_expr,
                           visit_fn: req_loans_in_fn,
                           with *visit::default_visitor()});
    visit::visit_crate(*crate, glcx, v);
    ret glcx.req_maps;
}

fn req_loans_in_fn(fk: visit::fn_kind,
                   decl: ast::fn_decl,
                   body: ast::blk,
                   sp: span,
                   id: ast::node_id,
                   &&self: gather_loan_ctxt,
                   v: visit::vt<gather_loan_ctxt>) {
    // see explanation attached to the `root_ub` field:
    let old_item_id = self.item_ub;
    let old_root_ub = self.root_ub;
    self.root_ub = body.node.id;

    alt fk {
      visit::fk_anon(*) | visit::fk_fn_block(*) {}
      visit::fk_item_fn(*) | visit::fk_method(*) |
      visit::fk_ctor(*) | visit::fk_dtor(*) {
        self.item_ub = body.node.id;
      }
    }

    visit::visit_fn(fk, decl, body, sp, id, self, v);
    self.root_ub = old_root_ub;
    self.item_ub = old_item_id;
}

fn req_loans_in_expr(ex: @ast::expr,
                     &&self: gather_loan_ctxt,
                     vt: visit::vt<gather_loan_ctxt>) {
    let bccx = self.bccx;
    let tcx = bccx.tcx;
    let old_root_ub = self.root_ub;

    debug!{"req_loans_in_expr(ex=%s)", pprust::expr_to_str(ex)};

    // If this expression is borrowed, have to ensure it remains valid:
    for tcx.borrowings.find(ex.id).each |borrow| {
        let cmt = self.bccx.cat_borrow_of_expr(ex);
        self.guarantee_valid(cmt, borrow.mutbl, borrow.region);
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
        visit::visit_expr(ex, self, vt);
      }

      ast::expr_call(f, args, _) {
        let arg_tys = ty::ty_fn_args(ty::expr_ty(self.tcx(), f));
        let scope_r = ty::re_scope(ex.id);
        do vec::iter2(args, arg_tys) |arg, arg_ty| {
            alt ty::resolved_mode(self.tcx(), arg_ty.mode) {
              ast::by_mutbl_ref {
                let arg_cmt = self.bccx.cat_expr(arg);
                self.guarantee_valid(arg_cmt, m_mutbl, scope_r);
              }
              ast::by_ref {
                let arg_cmt = self.bccx.cat_expr(arg);
                self.guarantee_valid(arg_cmt, m_imm,  scope_r);
              }
              ast::by_val {
                // Rust's by-val does not actually give ownership to
                // the callee.  This means that if a pointer type is
                // passed, it is effectively a borrow, and so the
                // caller must guarantee that the data remains valid.
                //
                // Subtle: we only guarantee that the pointer is valid
                // and const.  Technically, we ought to pass in the
                // mutability that the caller expects (e.g., if the
                // formal argument has type @mut, we should guarantee
                // validity and mutability, not validity and const).
                // However, the type system already guarantees that
                // the caller's mutability is compatible with the
                // callee, so this is not necessary.  (Note that with
                // actual borrows, typeck is more liberal and allows
                // the pointer to be borrowed as immutable even if it
                // is mutable in the caller's frame, thus effectively
                // passing the buck onto us to enforce this)
                //
                // FIXME (#2493): this handling is not really adequate.
                // For example, if there is a type like, {f: ~[int]}, we
                // will ignore it, but we ought to be requiring it to be
                // immutable (whereas something like {f:int} would be
                // fine).
                //
                alt opt_deref_kind(arg_ty.ty) {
                  some(deref_ptr(region_ptr(_))) |
                  some(deref_ptr(unsafe_ptr)) {
                    /* region pointers are (by induction) guaranteed */
                    /* unsafe pointers are the user's problem */
                  }
                  some(deref_comp(_)) |
                  none {
                    /* not a pointer, no worries */
                  }
                  some(deref_ptr(_)) {
                    let arg_cmt = self.bccx.cat_borrow_of_expr(arg);
                    self.guarantee_valid(arg_cmt, m_const, scope_r);
                  }
                }
              }
              ast::by_move | ast::by_copy {}
            }
        }
        visit::visit_expr(ex, self, vt);
      }

      ast::expr_alt(ex_v, arms, _) {
        let cmt = self.bccx.cat_expr(ex_v);
        for arms.each |arm| {
            for arm.pats.each |pat| {
                self.gather_pat(cmt, pat, arm.body.node.id, ex.id);
            }
        }
        visit::visit_expr(ex, self, vt);
      }

      ast::expr_index(rcvr, _) |
      ast::expr_binary(_, rcvr, _) |
      ast::expr_unary(_, rcvr) if self.bccx.method_map.contains_key(ex.id) {
        // Receivers in method calls are always passed by ref.
        //
        // Here, in an overloaded operator, the call is this expression,
        // and hence the scope of the borrow is this call.
        //
        // FIX? / NOT REALLY---technically we should check the other
        // argument and consider the argument mode.  But how annoying.
        // And this problem when goes away when argument modes are
        // phased out.  So I elect to leave this undone.
        let scope_r = ty::re_scope(ex.id);
        let rcvr_cmt = self.bccx.cat_expr(rcvr);
        self.guarantee_valid(rcvr_cmt, m_imm, scope_r);
        visit::visit_expr(ex, self, vt);
      }

      ast::expr_field(rcvr, _, _)
      if self.bccx.method_map.contains_key(ex.id) {
        // Receivers in method calls are always passed by ref.
        //
        // Here, the field a.b is in fact a closure.  Eventually, this
        // should be an fn&, but for now it's an fn@.  In any case,
        // the enclosing scope is either the call where it is a rcvr
        // (if used like `a.b(...)`), the call where it's an argument
        // (if used like `x(a.b)`), or the block (if used like `let x
        // = a.b`).
        let scope_r = ty::re_scope(self.tcx().region_map.get(ex.id));
        let rcvr_cmt = self.bccx.cat_expr(rcvr);
        self.guarantee_valid(rcvr_cmt, m_imm, scope_r);
        visit::visit_expr(ex, self, vt);
      }

      // see explanation attached to the `root_ub` field:
      ast::expr_while(cond, body) {
        // during the condition, can only root for the condition
        self.root_ub = cond.id;
        vt.visit_expr(cond, self, vt);

        // during body, can only root for the body
        self.root_ub = body.node.id;
        vt.visit_block(body, self, vt);
      }

      // see explanation attached to the `root_ub` field:
      ast::expr_loop(body) {
        self.root_ub = body.node.id;
        visit::visit_expr(ex, self, vt);
      }

      _ => {
        visit::visit_expr(ex, self, vt);
      }
    }

    // Check any contained expressions:

    self.root_ub = old_root_ub;
}

impl methods for gather_loan_ctxt {
    fn tcx() -> ty::ctxt { self.bccx.tcx }

    // guarantees that addr_of(cmt) will be valid for the duration of
    // `static_scope_r`, or reports an error.  This may entail taking
    // out loans, which will be added to the `req_loan_map`.  This can
    // also entail "rooting" GC'd pointers, which means ensuring
    // dynamically that they are not freed.
    fn guarantee_valid(cmt: cmt,
                       req_mutbl: ast::mutability,
                       scope_r: ty::region) {

        self.bccx.guaranteed_paths += 1;

        debug!{"guarantee_valid(cmt=%s, req_mutbl=%s, scope_r=%s)",
               self.bccx.cmt_to_repr(cmt),
               self.bccx.mut_to_str(req_mutbl),
               region_to_str(self.tcx(), scope_r)};
        let _i = indenter();

        alt cmt.lp {
          // If this expression is a loanable path, we MUST take out a
          // loan.  This is somewhat non-obvious.  You might think,
          // for example, that if we have an immutable local variable
          // `x` whose value is being borrowed, we could rely on `x`
          // not to change.  This is not so, however, because even
          // immutable locals can be moved.  So we take out a loan on
          // `x`, guaranteeing that it remains immutable for the
          // duration of the reference: if there is an attempt to move
          // it within that scope, the loan will be detected and an
          // error will be reported.
          some(_) {
            alt self.bccx.loan(cmt, scope_r, req_mutbl) {
              err(e) => { self.bccx.report(e); }
              ok(loans) if loans.len() == 0 => {}
              ok(loans) => {
                alt scope_r {
                  ty::re_scope(scope_id) => {
                    self.add_loans(scope_id, loans);

                    if req_mutbl == m_imm && cmt.mutbl != m_imm {
                        self.bccx.loaned_paths_imm += 1;

                        if self.tcx().sess.borrowck_note_loan() {
                            self.bccx.span_note(
                                cmt.span,
                                fmt!{"immutable loan required"});
                        }
                    } else {
                        self.bccx.loaned_paths_same += 1;
                    }
                  }
                  _ => {
                    self.bccx.tcx.sess.span_bug(
                        cmt.span,
                        #fmt["loans required but scope is scope_region is %s",
                             region_to_str(self.tcx(), scope_r)]);
                  }
                }
              }
            }
          }

          // The path is not loanable: in that case, we must try and
          // preserve it dynamically (or see that it is preserved by
          // virtue of being rooted in some immutable path).  We must
          // also check that the mutability of the desired pointer
          // matches with the actual mutability (but if an immutable
          // pointer is desired, that is ok as long as we are pure)
          none {
            let result: bckres<preserve_condition> = {
                do self.check_mutbl(req_mutbl, cmt).chain |pc1| {
                    do self.bccx.preserve(cmt, scope_r,
                                          self.item_ub,
                                          self.root_ub).chain |pc2| {
                        ok(pc1.combine(pc2))
                    }
                }
            };

            alt result {
              ok(pc_ok) {
                // we were able guarantee the validity of the ptr,
                // perhaps by rooting or because it is immutably
                // rooted.  good.
                self.bccx.stable_paths += 1;
              }
              ok(pc_if_pure(e)) {
                // we are only able to guarantee the validity if
                // the scope is pure
                alt scope_r {
                  ty::re_scope(pure_id) => {
                    // if the scope is some block/expr in the fn,
                    // then just require that this scope be pure
                    self.req_maps.pure_map.insert(pure_id, e);
                    self.bccx.req_pure_paths += 1;

                    if self.tcx().sess.borrowck_note_pure() {
                        self.bccx.span_note(
                            cmt.span,
                            fmt!{"purity required"});
                    }
                  }
                  _ => {
                    // otherwise, we can't enforce purity for that
                    // scope, so give up and report an error
                    self.bccx.report(e);
                  }
                }
              }
              err(e) => {
                // we cannot guarantee the validity of this pointer
                self.bccx.report(e);
              }
            }
          }
        }
    }

    // Check that the pat `cmt` is compatible with the required
    // mutability, presuming that it can be preserved to stay alive
    // long enough.
    //
    // For example, if you have an expression like `&x.f` where `x`
    // has type `@mut{f:int}`, this check might fail because `&x.f`
    // reqires an immutable pointer, but `f` lives in (aliased)
    // mutable memory.
    fn check_mutbl(req_mutbl: ast::mutability,
                   cmt: cmt) -> bckres<preserve_condition> {
        alt (req_mutbl, cmt.mutbl) {
          (m_const, _) |
          (m_imm, m_imm) |
          (m_mutbl, m_mutbl) => {
            ok(pc_ok)
          }

          (_, m_const) |
          (m_imm, m_mutbl) |
          (m_mutbl, m_imm) => {
            let e = {cmt: cmt,
                     code: err_mutbl(req_mutbl, cmt.mutbl)};
            if req_mutbl == m_imm {
                // you can treat mutable things as imm if you are pure
                ok(pc_if_pure(e))
            } else {
                err(e)
            }
          }
        }
    }

    fn add_loans(scope_id: ast::node_id, loans: @dvec<loan>) {
        alt self.req_maps.req_loan_map.find(scope_id) {
          some(l) {
            (*l).push(loans);
          }
          none {
            self.req_maps.req_loan_map.insert(
                scope_id, @dvec::from_vec(~[mut loans]));
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

        debug!{"gather_pat: id=%d pat=%s cmt=%s arm_id=%d alt_id=%d",
               pat.id, pprust::pat_to_str(pat),
               self.bccx.cmt_to_repr(cmt), arm_id, alt_id};
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
            let enum_did = alt self.bccx.tcx.def_map
.find(pat.id) {
              some(ast::def_variant(enum_did, _)) {enum_did}
              e {tcx.sess.span_bug(pat.span,
                                   fmt!{"resolved to %?, \
                                         not variant", e})}
            };

            for subpats.each |subpat| {
                let subcmt = self.bccx.cat_variant(subpat, enum_did, cmt);
                self.gather_pat(subcmt, subpat, arm_id, alt_id);
            }
          }

          ast::pat_ident(_, none) if self.pat_is_variant(pat) {
            // nullary variant
            debug!{"nullary variant"};
          }
          ast::pat_ident(id, o_pat) {
            // x or x @ p --- `x` must remain valid for the scope of the alt
            debug!{"defines identifier %s", pprust::path_to_str(id)};

            // Note: there is a discussion of the function of
            // cat_discr in the method preserve():
            let cmt1 = self.bccx.cat_discr(cmt, alt_id);
            let arm_scope = ty::re_scope(arm_id);

            // Remember the mutability of the location that this
            // binding refers to.  This will be used later when
            // categorizing the binding.  This is a bit of a hack that
            // would be better fixed by #2329; in that case we could
            // allow the user to specify if they want an imm, const,
            // or mut binding, or else just reflect the mutability
            // through the type of the region pointer.
            self.bccx.binding_map.insert(pat.id, cmt1.mutbl);

            self.guarantee_valid(cmt1, m_const, arm_scope);

            for o_pat.each |p| {
                self.gather_pat(cmt, p, arm_id, alt_id);
            }
          }

          ast::pat_rec(field_pats, _) {
            // {f1: p1, ..., fN: pN}
            for field_pats.each |fp| {
                let cmt_field = self.bccx.cat_field(fp.pat, cmt, fp.ident);
                self.gather_pat(cmt_field, fp.pat, arm_id, alt_id);
            }
          }

          ast::pat_tup(subpats) {
            // (p1, ..., pN)
            for subpats.each |subpat| {
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
                tcx.sess.span_bug(pat.span, ~"Non derefable type");
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

