#[doc = "

# Categorization

The job of the categorization module is to analyze an expression to
determine what kind of memory is used in evaluating it (for example,
where dereferences occur and what kind of pointer is dereferenced;
whether the memory is mutable; etc)

Categorization effectively transforms all of our expressions into
expressions of the following forms (the actual enum has many more
possibilities, naturally, but they are all variants of these base
forms):

    E = rvalue    // some computed rvalue
      | x         // address of a local variable, arg, or upvar
      | *E        // deref of a ptr
      | E.comp    // access to an interior component

Imagine a routine ToAddr(Expr) that evaluates an expression and returns an
address where the result is to be found.  If Expr is an lvalue, then this
is the address of the lvalue.  If Expr is an rvalue, this is the address of
some temporary spot in memory where the result is stored.

Now, cat_expr() classies the expression Expr and the address A=ToAddr(Expr)
as follows:

- cat: what kind of expression was this?  This is a subset of the
  full expression forms which only includes those that we care about
  for the purpose of the analysis.
- mutbl: mutability of the address A
- ty: the type of data found at the address A

The resulting categorization tree differs somewhat from the expressions
themselves.  For example, auto-derefs are explicit.  Also, an index a[b] is
decomposed into two operations: a derefence to reach the array data and
then an index to jump forward to the relevant item.
"];

export public_methods;
export opt_deref_kind;

// Categorizes a derefable type.  Note that we include vectors and strings as
// derefable (we model an index as the combination of a deref and then a
// pointer adjustment).
fn opt_deref_kind(t: ty::t) -> option<deref_kind> {
    alt ty::get(t).struct {
      ty::ty_uniq(*) | ty::ty_vec(*) | ty::ty_str |
      ty::ty_evec(_, ty::vstore_uniq) |
      ty::ty_estr(ty::vstore_uniq) {
        some(deref_ptr(uniq_ptr))
      }

      ty::ty_rptr(*) |
      ty::ty_evec(_, ty::vstore_slice(_)) |
      ty::ty_estr(ty::vstore_slice(_)) {
        some(deref_ptr(region_ptr))
      }

      ty::ty_box(*) |
      ty::ty_evec(_, ty::vstore_box) |
      ty::ty_estr(ty::vstore_box) {
        some(deref_ptr(gc_ptr))
      }

      ty::ty_ptr(*) {
        some(deref_ptr(unsafe_ptr))
      }

      ty::ty_enum(did, _) {
        some(deref_comp(comp_variant(did)))
      }

      ty::ty_res(*) {
        some(deref_comp(comp_res))
      }

      ty::ty_evec(mt, ty::vstore_fixed(_)) {
        some(deref_comp(comp_index(t, mt.mutbl)))
      }

      ty::ty_estr(ty::vstore_fixed(_)) {
        some(deref_comp(comp_index(t, m_imm)))
      }

      _ {
        none
      }
    }
}

fn deref_kind(tcx: ty::ctxt, t: ty::t) -> deref_kind {
    alt opt_deref_kind(t) {
      some(k) {k}
      none {
        tcx.sess.bug(
            #fmt["deref_cat() invoked on non-derefable type %s",
                 ty_to_str(tcx, t)]);
      }
    }
}

impl public_methods for borrowck_ctxt {
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
          ast::expr_loop_body(*) | ast::expr_do_body(*) | ast::expr_unary(*) |
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
                {m: m_mutbl, lp: none}
              }
              ast::by_move | ast::by_copy {
                {m: m_imm, lp: some(@lp_arg(vid))}
              }
              ast::by_ref {
                {m: m_imm, lp: none}
              }
              ast::by_val {
                // by-value is this hybrid mode where we have a
                // pointer but we do not own it.  This is not
                // considered loanable because, for example, a by-ref
                // and and by-val argument might both actually contain
                // the same unique ptr.
                {m: m_imm, lp: none}
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

    fn cat_variant<N: ast_node>(arg: N,
                                enum_did: ast::def_id,
                                cmt: cmt) -> cmt {
        @{id: arg.id(), span: arg.span(),
          cat: cat_comp(cmt, comp_variant(enum_did)),
          lp: cmt.lp.map { |l| @lp_comp(l, comp_variant(enum_did)) },
          mutbl: cmt.mutbl, // imm iff in an immutable context
          ty: self.tcx.ty(arg)}
    }

    fn cat_rvalue(expr: @ast::expr, expr_ty: ty::t) -> cmt {
        @{id:expr.id, span:expr.span,
          cat:cat_rvalue, lp:none,
          mutbl:m_imm, ty:expr_ty}
    }

    fn cat_discr(cmt: cmt, alt_id: ast::node_id) -> cmt {
        ret @{cat:cat_discr(cmt, alt_id) with *cmt};
    }

    fn cat_field<N:ast_node>(node: N, base_cmt: cmt,
                             f_name: ast::ident) -> cmt {
        let f_mutbl = alt field_mutbl(self.tcx, base_cmt.ty, f_name) {
          some(f_mutbl) { f_mutbl }
          none {
            self.tcx.sess.span_bug(
                node.span(),
                #fmt["Cannot find field `%s` in type `%s`",
                     *f_name, ty_to_str(self.tcx, base_cmt.ty)]);
          }
        };
        let m = alt f_mutbl {
          m_imm { base_cmt.mutbl } // imm: as mutable as the container
          m_mutbl | m_const { f_mutbl }
        };
        let f_comp = comp_field(f_name, f_mutbl);
        let lp = base_cmt.lp.map { |lp|
            @lp_comp(lp, f_comp)
        };
        @{id: node.id(), span: node.span(),
          cat: cat_comp(base_cmt, f_comp), lp:lp,
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

        ret alt deref_kind(self.tcx, base_cmt.ty) {
          deref_ptr(ptr) {
            // make deref of vectors explicit, as explained in the comment at
            // the head of this section
            let deref_lp = base_cmt.lp.map { |lp| @lp_deref(lp, ptr) };
            let deref_cmt = @{id:expr.id, span:expr.span,
                              cat:cat_deref(base_cmt, 0u, ptr), lp:deref_lp,
                              mutbl:m_imm, ty:mt.ty};
            comp(expr, deref_cmt, base_cmt.ty, mt)
          }

          deref_comp(_) {
            // fixed-length vectors have no deref
            comp(expr, base_cmt, base_cmt.ty, mt)
          }
        };

        fn comp(expr: @ast::expr, of_cmt: cmt,
                vect: ty::t, mt: ty::mt) -> cmt {
            let comp = comp_index(vect, mt.mutbl);
            let index_lp = of_cmt.lp.map { |lp| @lp_comp(lp, comp) };
            @{id:expr.id, span:expr.span,
              cat:cat_comp(of_cmt, comp), lp:index_lp,
              mutbl:mt.mutbl, ty:mt.ty}
        }
    }

    fn cat_tuple_elt<N: ast_node>(elt: N, cmt: cmt) -> cmt {
        @{id: elt.id(), span: elt.span(),
          cat: cat_comp(cmt, comp_tuple),
          lp: cmt.lp.map { |l| @lp_comp(l, comp_tuple) },
          mutbl: cmt.mutbl, // imm iff in an immutable context
          ty: self.tcx.ty(elt)}
    }
}

impl private_methods for borrowck_ctxt {
    fn cat_method_ref(expr: @ast::expr, expr_ty: ty::t) -> cmt {
        @{id:expr.id, span:expr.span,
          cat:cat_special(sk_method), lp:none,
          mutbl:m_imm, ty:expr_ty}
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
}

fn field_mutbl(tcx: ty::ctxt,
               base_ty: ty::t,
               f_name: ast::ident) -> option<ast::mutability> {
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
