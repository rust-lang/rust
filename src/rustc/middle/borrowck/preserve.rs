// ----------------------------------------------------------------------
// Preserve(Ex, S) holds if ToAddr(Ex) will remain valid for the entirety of
// the scope S.
export public_methods;

impl public_methods for borrowck_ctxt {
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
            // Normally, local variables are lendable, and so this
            // case should never trigger.  However, if we are
            // preserving an expression like a.b where the field `b`
            // has @ type, then it will recurse to ensure that the `a`
            // is stable to try and avoid rooting the value `a.b`.  In
            // this case, opt_scope_id will be none.
            if opt_scope_id.is_some() {
                self.tcx.sess.span_bug(
                    cmt.span,
                    "preserve() called with local and non-none opt_scope_id");
            }
            ok(())
          }
          cat_arg(_) {
            // This can happen as not all args are lendable (e.g., &&
            // modes).  In that case, the caller guarantees stability.
            // This is basically a deref of a region ptr.
            ok(())
          }
          cat_comp(cmt_base, comp_field(*)) |
          cat_comp(cmt_base, comp_index(*)) |
          cat_comp(cmt_base, comp_tuple) |
          cat_comp(cmt_base, comp_res) {
            // Most embedded components: if the base is stable, the
            // type never changes.
            self.preserve(cmt_base, opt_scope_id)
          }
          cat_comp(cmt_base, comp_variant(enum_did)) {
            if ty::enum_is_univariant(self.tcx, enum_did) {
                self.preserve(cmt_base, opt_scope_id)
            } else {
                // If there are multiple variants: overwriting the
                // base could cause the type of this memory to change,
                // so require imm.
                self.require_imm(cmt, cmt_base, opt_scope_id, err_mut_variant)
            }
          }
          cat_deref(cmt_base, _, uniq_ptr) {
            // Overwriting the base could cause this memory to be
            // freed, so require imm.
            self.require_imm(cmt, cmt_base, opt_scope_id, err_mut_uniq)
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
            // GC'd pointers of type @MT: if this pointer lives in
            // immutable, stable memory, then everything is fine.  But
            // otherwise we have no guarantee the pointer will stay
            // live, so we must root the pointer (i.e., inc the ref
            // count) for the duration of the loan.
            if base.mutbl == m_imm {
                alt self.preserve(base, none) {
                  ok(()) {ok(())}
                  err(_) {self.attempt_root(cmt, opt_scope_id, base, derefs)}
                }
            } else {
                self.attempt_root(cmt, opt_scope_id, base, derefs)
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
}

impl private_methods for borrowck_ctxt {
    fn require_imm(cmt: cmt,
                   cmt_base: cmt,
                   opt_scope_id: option<ast::node_id>,
                   code: bckerr_code) -> bckres<()> {
        // Variant contents and unique pointers: must be immutably
        // rooted to a preserved address.
        alt cmt_base.mutbl {
          m_mutbl | m_const { err({cmt:cmt, code:code}) }
          m_imm { self.preserve(cmt_base, opt_scope_id) }
        }
    }

    fn attempt_root(cmt: cmt, opt_scope_id: option<ast::node_id>,
                    base: cmt, derefs: uint) -> bckres<()> {
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
}
