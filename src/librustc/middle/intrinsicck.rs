// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use metadata::csearch;
use middle::def::DefFn;
use middle::subst::{Subst, Substs, EnumeratedItems};
use middle::ty::{TransmuteRestriction, ctxt, ty_bare_fn};
use middle::ty::{mod, Ty};
use util::ppaux::Repr;

use syntax::abi::RustIntrinsic;
use syntax::ast::DefId;
use syntax::ast;
use syntax::ast_map::NodeForeignItem;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::visit::Visitor;
use syntax::visit;

pub fn check_crate(tcx: &ctxt) {
    let mut visitor = IntrinsicCheckingVisitor {
        tcx: tcx,
        param_envs: Vec::new(),
        dummy_sized_ty: tcx.types.int,
        dummy_unsized_ty: ty::mk_vec(tcx, tcx.types.int, None),
    };
    visit::walk_crate(&mut visitor, tcx.map.krate());
}

struct IntrinsicCheckingVisitor<'a, 'tcx: 'a> {
    tcx: &'a ctxt<'tcx>,

    // As we traverse the AST, we keep a stack of the parameter
    // environments for each function we encounter. When we find a
    // call to `transmute`, we can check it in the context of the top
    // of the stack (which ought not to be empty).
    param_envs: Vec<ty::ParameterEnvironment<'a,'tcx>>,

    // Dummy sized/unsized types that use to substitute for type
    // parameters in order to estimate how big a type will be for any
    // possible instantiation of the type parameters in scope.  See
    // `check_transmute` for more details.
    dummy_sized_ty: Ty<'tcx>,
    dummy_unsized_ty: Ty<'tcx>,
}

impl<'a, 'tcx> IntrinsicCheckingVisitor<'a, 'tcx> {
    fn def_id_is_transmute(&self, def_id: DefId) -> bool {
        let intrinsic = match ty::lookup_item_type(self.tcx, def_id).ty.sty {
            ty::ty_bare_fn(_, ref bfty) => bfty.abi == RustIntrinsic,
            _ => return false
        };
        if def_id.krate == ast::LOCAL_CRATE {
            match self.tcx.map.get(def_id.node) {
                NodeForeignItem(ref item) if intrinsic => {
                    token::get_ident(item.ident) ==
                        token::intern_and_get_ident("transmute")
                }
                _ => false,
            }
        } else {
            match csearch::get_item_path(self.tcx, def_id).last() {
                Some(ref last) if intrinsic => {
                    token::get_name(last.name()) ==
                        token::intern_and_get_ident("transmute")
                }
                _ => false,
            }
        }
    }

    fn check_transmute(&self, span: Span, from: Ty<'tcx>, to: Ty<'tcx>, id: ast::NodeId) {
        // Find the parameter environment for the most recent function that
        // we entered.

        let param_env = match self.param_envs.last() {
            Some(p) => p,
            None => {
                self.tcx.sess.span_bug(
                    span,
                    "transmute encountered outside of any fn");
            }
        };

        // Simple case: no type parameters involved.
        if
            !ty::type_has_params(from) && !ty::type_has_self(from) &&
            !ty::type_has_params(to) && !ty::type_has_self(to)
        {
            let restriction = TransmuteRestriction {
                span: span,
                original_from: from,
                original_to: to,
                substituted_from: from,
                substituted_to: to,
                id: id,
            };
            self.push_transmute_restriction(restriction);
            return;
        }

        // The rules around type parameters are a bit subtle. We are
        // checking these rules before monomorphization, so there may
        // be unsubstituted type parameters present in the
        // types. Obviously we cannot create LLVM types for those.
        // However, if a type parameter appears only indirectly (i.e.,
        // through a pointer), it does not necessarily affect the
        // size, so that should be allowed. The only catch is that we
        // DO want to be careful around unsized type parameters, since
        // fat pointers have a different size than a thin pointer, and
        // hence `&T` and `&U` have different sizes if `T : Sized` but
        // `U : Sized` does not hold.
        //
        // However, it's not as simple as checking whether `T :
        // Sized`, because even if `T : Sized` does not hold, that
        // just means that `T` *may* not be sized.  After all, even a
        // type parameter `Sized? T` could be bound to a sized
        // type. (Issue #20116)
        //
        // To handle this, we first check for "interior" type
        // parameters, which are always illegal. If there are none of
        // those, then we know that the only way that all type
        // parameters `T` are referenced indirectly, e.g. via a
        // pointer type like `&T`. In that case, we only care whether
        // `T` is sized or not, because that influences whether `&T`
        // is a thin or fat pointer.
        //
        // One could imagine establishing a sophisticated constraint
        // system to ensure that the transmute is legal, but instead
        // we do something brutally dumb. We just substitute dummy
        // sized or unsized types for every type parameter in scope,
        // exhaustively checking all possible combinations. Here are some examples:
        //
        // ```
        // fn foo<T,U>() {
        //     // T=int, U=int
        // }
        //
        // fn bar<Sized? T,U>() {
        //     // T=int, U=int
        //     // T=[int], U=int
        // }
        //
        // fn baz<Sized? T, Sized?U>() {
        //     // T=int, U=int
        //     // T=[int], U=int
        //     // T=int, U=[int]
        //     // T=[int], U=[int]
        // }
        // ```
        //
        // In all cases, we keep the original unsubstituted types
        // around for error reporting.

        let from_tc = ty::type_contents(self.tcx, from);
        let to_tc = ty::type_contents(self.tcx, to);
        if from_tc.interior_param() || to_tc.interior_param() {
            span_err!(self.tcx.sess, span, E0139,
                      "cannot transmute to or from a type that contains \
                       type parameters in its interior");
            return;
        }

        let mut substs = param_env.free_substs.clone();
        self.with_each_combination(
            span,
            param_env,
            param_env.free_substs.types.iter_enumerated(),
            &mut substs,
            &mut |substs| {
                let restriction = TransmuteRestriction {
                    span: span,
                    original_from: from,
                    original_to: to,
                    substituted_from: from.subst(self.tcx, substs),
                    substituted_to: to.subst(self.tcx, substs),
                    id: id,
                };
                self.push_transmute_restriction(restriction);
            });
    }

    fn with_each_combination(&self,
                             span: Span,
                             param_env: &ty::ParameterEnvironment<'a,'tcx>,
                             mut types_in_scope: EnumeratedItems<Ty<'tcx>>,
                             substs: &mut Substs<'tcx>,
                             callback: &mut FnMut(&Substs<'tcx>))
    {
        // This parameter invokes `callback` many times with different
        // substitutions that replace all the parameters in scope with
        // either `int` or `[int]`, depending on whether the type
        // parameter is known to be sized. See big comment above for
        // an explanation of why this is a reasonable thing to do.

        match types_in_scope.next() {
            None => {
                debug!("with_each_combination(substs={})",
                       substs.repr(self.tcx));

                callback.call_mut((substs,));
            }

            Some((space, index, &param_ty)) => {
                debug!("with_each_combination: space={}, index={}, param_ty={}",
                       space, index, param_ty.repr(self.tcx));

                if !ty::type_is_sized(param_env, span, param_ty) {
                    debug!("with_each_combination: param_ty is not known to be sized");

                    substs.types.get_mut_slice(space)[index] = self.dummy_unsized_ty;
                    self.with_each_combination(span, param_env, types_in_scope.clone(),
                                               substs, callback);
                }

                substs.types.get_mut_slice(space)[index] = self.dummy_sized_ty;
                self.with_each_combination(span, param_env, types_in_scope,
                                           substs, callback);
            }
        }
    }

    fn push_transmute_restriction(&self, restriction: TransmuteRestriction<'tcx>) {
        debug!("Pushing transmute restriction: {}", restriction.repr(self.tcx));
        self.tcx.transmute_restrictions.borrow_mut().push(restriction);
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for IntrinsicCheckingVisitor<'a, 'tcx> {
    fn visit_fn(&mut self, fk: visit::FnKind<'v>, fd: &'v ast::FnDecl,
                b: &'v ast::Block, s: Span, id: ast::NodeId) {
        match fk {
            visit::FkItemFn(..) | visit::FkMethod(..) => {
                let param_env = ty::ParameterEnvironment::for_item(self.tcx, id);
                self.param_envs.push(param_env);
                visit::walk_fn(self, fk, fd, b, s);
                self.param_envs.pop();
            }
            visit::FkFnBlock(..) => {
                visit::walk_fn(self, fk, fd, b, s);
            }
        }

    }

    fn visit_expr(&mut self, expr: &ast::Expr) {
        if let ast::ExprPath(..) = expr.node {
            match ty::resolve_expr(self.tcx, expr) {
                DefFn(did, _) if self.def_id_is_transmute(did) => {
                    let typ = ty::node_id_to_type(self.tcx, expr.id);
                    match typ.sty {
                        ty_bare_fn(_, ref bare_fn_ty) if bare_fn_ty.abi == RustIntrinsic => {
                            if let ty::FnConverging(to) = bare_fn_ty.sig.0.output {
                                let from = bare_fn_ty.sig.0.inputs[0];
                                self.check_transmute(expr.span, from, to, expr.id);
                            }
                        }
                        _ => {
                            self.tcx
                                .sess
                                .span_bug(expr.span, "transmute wasn't a bare fn?!");
                        }
                    }
                }
                _ => {}
            }
        }

        visit::walk_expr(self, expr);
    }
}

impl<'tcx> Repr<'tcx> for TransmuteRestriction<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("TransmuteRestriction(id={}, original=({},{}), substituted=({},{}))",
                self.id,
                self.original_from.repr(tcx),
                self.original_to.repr(tcx),
                self.substituted_from.repr(tcx),
                self.substituted_to.repr(tcx))
    }
}
