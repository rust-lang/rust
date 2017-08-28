// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::{self, TyCtxt};
use rustc_errors::DiagnosticBuilder;
use syntax_pos::{MultiSpan, Span};

use std::fmt;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Origin { Ast, Mir }

impl fmt::Display for Origin {
    fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Origin::Mir => write!(w, " (Mir)"),
            Origin::Ast => ty::tls::with_opt(|opt_tcx| {
                // If user passed `-Z borrowck-mir`, then include an
                // AST origin as part of the error report
                if let Some(tcx) = opt_tcx {
                    if tcx.sess.opts.debugging_opts.borrowck_mir {
                        return write!(w, " (Ast)");
                    }
                }
                // otherwise, do not include the origin (i.e., print
                // nothing at all)
                Ok(())
            }),
        }
    }
}

pub trait BorrowckErrors {
    fn struct_span_err_with_code<'a, S: Into<MultiSpan>>(&'a self,
                                                         sp: S,
                                                         msg: &str,
                                                         code: &str)
                                                         -> DiagnosticBuilder<'a>;

    fn struct_span_err<'a, S: Into<MultiSpan>>(&'a self,
                                               sp: S,
                                               msg: &str)
                                               -> DiagnosticBuilder<'a>;

    fn cannot_move_when_borrowed(&self, span: Span, desc: &str, o: Origin)
                                 -> DiagnosticBuilder
    {
        struct_span_err!(self, span, E0505,
                         "cannot move out of `{}` because it is borrowed{OGN}",
                         desc, OGN=o)
    }

    fn cannot_use_when_mutably_borrowed(&self, span: Span, desc: &str, o: Origin)
                                        -> DiagnosticBuilder
    {
        struct_span_err!(self, span, E0503,
                         "cannot use `{}` because it was mutably borrowed{OGN}",
                         desc, OGN=o)
    }

    fn cannot_act_on_uninitialized_variable(&self,
                                            span: Span,
                                            verb: &str,
                                            desc: &str,
                                            o: Origin)
                                            -> DiagnosticBuilder
    {
        struct_span_err!(self, span, E0381,
                         "{} of possibly uninitialized variable: `{}`{OGN}",
                         verb, desc, OGN=o)
    }

    fn cannot_mutably_borrow_multiply(&self,
                                      span: Span,
                                      desc: &str,
                                      opt_via: &str,
                                      o: Origin)
                                      -> DiagnosticBuilder
    {
        struct_span_err!(self, span, E0499,
                         "cannot borrow `{}`{} as mutable more than once at a time{OGN}",
                         desc, opt_via, OGN=o)
    }

    fn cannot_uniquely_borrow_by_two_closures(&self, span: Span, desc: &str, o: Origin)
                                              -> DiagnosticBuilder
    {
        struct_span_err!(self, span, E0524,
                         "two closures require unique access to `{}` at the same time{OGN}",
                         desc, OGN=o)
    }

    fn cannot_uniquely_borrow_by_one_closure(&self,
                                             span: Span,
                                             desc_new: &str,
                                             noun_old: &str,
                                             msg_old: &str,
                                             o: Origin)
                                             -> DiagnosticBuilder
    {
        struct_span_err!(self, span, E0500,
                         "closure requires unique access to `{}` but {} is already borrowed{}{OGN}",
                         desc_new, noun_old, msg_old, OGN=o)
    }

    fn cannot_reborrow_already_uniquely_borrowed(&self,
                                                 span: Span,
                                                 desc_new: &str,
                                                 msg_new: &str,
                                                 kind_new: &str,
                                                 o: Origin)
                                                 -> DiagnosticBuilder
    {
        struct_span_err!(self, span, E0501,
                         "cannot borrow `{}`{} as {} because previous closure \
                          requires unique access{OGN}",
                         desc_new, msg_new, kind_new, OGN=o)
    }

    fn cannot_reborrow_already_borrowed(&self,
                                        span: Span,
                                        desc_new: &str,
                                        msg_new: &str,
                                        kind_new: &str,
                                        noun_old: &str,
                                        kind_old: &str,
                                        msg_old: &str,
                                        o: Origin)
                                        -> DiagnosticBuilder
    {
        struct_span_err!(self, span, E0502,
                         "cannot borrow `{}`{} as {} because {} is also borrowed as {}{}{OGN}",
                         desc_new, msg_new, kind_new, noun_old, kind_old, msg_old, OGN=o)
    }

    fn cannot_assign_to_borrowed(&self, span: Span, desc: &str, o: Origin)
                                 -> DiagnosticBuilder
    {
        struct_span_err!(self, span, E0506,
                         "cannot assign to `{}` because it is borrowed{OGN}",
                         desc, OGN=o)
    }

    fn cannot_move_into_closure(&self, span: Span, desc: &str, o: Origin)
                                -> DiagnosticBuilder
    {
        struct_span_err!(self, span, E0504,
                         "cannot move `{}` into closure because it is borrowed{OGN}",
                         desc, OGN=o)
    }

    fn cannot_reassign_immutable(&self, span: Span, desc: &str, o: Origin)
                                 -> DiagnosticBuilder
    {
        struct_span_err!(self, span, E0384,
                         "re-assignment of immutable variable `{}`{OGN}",
                         desc, OGN=o)
    }

    fn cannot_assign_static(&self, span: Span, desc: &str, o: Origin)
                            -> DiagnosticBuilder
    {
        self.struct_span_err(span, &format!("cannot assign to immutable static item {}{OGN}",
                                            desc, OGN=o))
    }
}

impl<'b, 'tcx, 'gcx> BorrowckErrors for TyCtxt<'b, 'tcx, 'gcx> {
    fn struct_span_err_with_code<'a, S: Into<MultiSpan>>(&'a self,
                                                         sp: S,
                                                         msg: &str,
                                                         code: &str)
                                                         -> DiagnosticBuilder<'a>
    {
        self.sess.struct_span_err_with_code(sp, msg, code)
    }

    fn struct_span_err<'a, S: Into<MultiSpan>>(&'a self,
                                               sp: S,
                                               msg: &str)
                                               -> DiagnosticBuilder<'a>
    {
        self.sess.struct_span_err(sp, msg)
    }
}
