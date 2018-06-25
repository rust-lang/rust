// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::def_id::DefId;
use ty;
use ty::subst::Substs;
use ty::query::TyCtxtAt;
use mir::interpret::ConstValue;
use errors::DiagnosticBuilder;

use graphviz::IntoCow;
use syntax_pos::Span;
use syntax::ast;

use std::borrow::Cow;
use rustc_data_structures::sync::Lrc;

pub type EvalResult<'tcx> = Result<&'tcx ty::Const<'tcx>, ConstEvalErr<'tcx>>;

#[derive(Copy, Clone, Debug, Hash, RustcEncodable, RustcDecodable, Eq, PartialEq, Ord, PartialOrd)]
pub enum ConstVal<'tcx> {
    Unevaluated(DefId, &'tcx Substs<'tcx>),
    Value(ConstValue<'tcx>),
}

#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub struct ConstEvalErr<'tcx> {
    pub span: Span,
    pub data: Lrc<(::mir::interpret::EvalError<'tcx>, Vec<FrameInfo>)>,
}

#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub struct FrameInfo {
    pub span: Span,
    pub location: String,
    pub lint_root: Option<ast::NodeId>,
}

#[derive(Clone, Debug)]
pub enum ConstEvalErrDescription<'a, 'tcx: 'a> {
    Simple(Cow<'a, str>),
    Backtrace(&'a ::mir::interpret::EvalError<'tcx>, &'a [FrameInfo]),
}

impl<'a, 'tcx> ConstEvalErrDescription<'a, 'tcx> {
    /// Return a one-line description of the error, for lints and such
    pub fn into_oneline(self) -> Cow<'a, str> {
        match self {
            ConstEvalErrDescription::Simple(simple) => simple,
            ConstEvalErrDescription::Backtrace(miri, _) => format!("{}", miri).into_cow(),
        }
    }
}

impl<'a, 'gcx, 'tcx> ConstEvalErr<'tcx> {
    pub fn struct_error(&self,
        tcx: TyCtxtAt<'a, 'gcx, 'tcx>,
        message: &str)
        -> Option<DiagnosticBuilder<'tcx>>
    {
        self.struct_generic(tcx, message, None)
    }

    pub fn report_as_error(&self,
        tcx: TyCtxtAt<'a, 'gcx, 'tcx>,
        message: &str
    ) {
        let err = self.struct_generic(tcx, message, None);
        if let Some(mut err) = err {
            err.emit();
        }
    }

    pub fn report_as_lint(&self,
        tcx: TyCtxtAt<'a, 'gcx, 'tcx>,
        message: &str,
        lint_root: ast::NodeId,
    ) {
        let lint = self.struct_generic(
            tcx,
            message,
            Some(lint_root),
        );
        if let Some(mut lint) = lint {
            lint.emit();
        }
    }

    fn struct_generic(
        &self,
        tcx: TyCtxtAt<'a, 'gcx, 'tcx>,
        message: &str,
        lint_root: Option<ast::NodeId>,
    ) -> Option<DiagnosticBuilder<'tcx>> {
        match self.data.0.kind {
            ::mir::interpret::EvalErrorKind::TypeckError |
            ::mir::interpret::EvalErrorKind::TooGeneric |
            ::mir::interpret::EvalErrorKind::CheckMatchError |
            ::mir::interpret::EvalErrorKind::Layout(_) => return None,
            ::mir::interpret::EvalErrorKind::ReferencedConstant(ref inner) => {
                inner.struct_generic(tcx, "referenced constant", lint_root)?.emit();
            },
            _ => {},
        }
        trace!("reporting const eval failure at {:?}", self.span);
        let mut err = if let Some(lint_root) = lint_root {
            let node_id = self.data.1
                .iter()
                .rev()
                .filter_map(|frame| frame.lint_root)
                .next()
                .unwrap_or(lint_root);
            tcx.struct_span_lint_node(
                ::rustc::lint::builtin::CONST_ERR,
                node_id,
                tcx.span,
                message,
            )
        } else {
            struct_error(tcx, message)
        };
        err.span_label(self.span, self.data.0.to_string());
        for FrameInfo { span, location, .. } in &self.data.1 {
            err.span_label(*span, format!("inside call to `{}`", location));
        }
        Some(err)
    }
}

pub fn struct_error<'a, 'gcx, 'tcx>(
    tcx: TyCtxtAt<'a, 'gcx, 'tcx>,
    msg: &str,
) -> DiagnosticBuilder<'tcx> {
    struct_span_err!(tcx.sess, tcx.span, E0080, "{}", msg)
}
