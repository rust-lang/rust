// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use rustc_const_math::ConstInt;

use hir::def_id::DefId;
use ty::{self, TyCtxt, layout};
use ty::subst::Substs;
use rustc_const_math::*;
use mir::interpret::{Value, PrimVal};
use errors::DiagnosticBuilder;

use graphviz::IntoCow;
use serialize;
use syntax_pos::Span;

use std::borrow::Cow;

pub type EvalResult<'tcx> = Result<&'tcx ty::Const<'tcx>, ConstEvalErr<'tcx>>;

#[derive(Copy, Clone, Debug, Hash, RustcEncodable, RustcDecodable, Eq, PartialEq)]
pub enum ConstVal<'tcx> {
    Unevaluated(DefId, &'tcx Substs<'tcx>),
    Value(Value),
}

#[derive(Copy, Clone, Debug, Hash, RustcEncodable, Eq, PartialEq)]
pub struct ByteArray<'tcx> {
    pub data: &'tcx [u8],
}

impl<'tcx> serialize::UseSpecializedDecodable for ByteArray<'tcx> {}

impl<'tcx> ConstVal<'tcx> {
    pub fn to_u128(&self) -> Option<u128> {
        match *self {
            ConstVal::Value(Value::ByVal(PrimVal::Bytes(b))) => {
                Some(b)
            },
            _ => None,
        }
    }
    pub fn unwrap_u64(&self) -> u64 {
        match self.to_u128() {
            Some(val) => {
                assert_eq!(val as u64 as u128, val);
                val as u64
            },
            None => bug!("expected constant u64, got {:#?}", self),
        }
    }
    pub fn unwrap_usize<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> ConstUsize {
        match *self {
            ConstVal::Value(Value::ByVal(PrimVal::Bytes(b))) => {
                assert_eq!(b as u64 as u128, b);
                match ConstUsize::new(b as u64, tcx.sess.target.usize_ty) {
                    Ok(val) => val,
                    Err(e) => bug!("{:#?} is not a usize {:?}", self, e),
                }
            },
            _ => bug!("expected constant u64, got {:#?}", self),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConstEvalErr<'tcx> {
    pub span: Span,
    pub kind: ErrKind<'tcx>,
}

#[derive(Clone, Debug)]
pub enum ErrKind<'tcx> {
    CannotCast,
    MissingStructField,

    NonConstPath,
    UnimplementedConstVal(&'static str),
    ExpectedConstTuple,
    ExpectedConstStruct,
    IndexedNonVec,
    IndexNotUsize,
    IndexOutOfBounds { len: u64, index: u64 },

    MiscBinaryOp,
    MiscCatchAll,

    IndexOpFeatureGated,
    Math(ConstMathErr),
    LayoutError(layout::LayoutError<'tcx>),

    ErroneousReferencedConstant(Box<ConstEvalErr<'tcx>>),

    TypeckError,
    CheckMatchError,
    Miri(::mir::interpret::EvalError<'tcx>),
}

impl<'tcx> From<::mir::interpret::EvalError<'tcx>> for ErrKind<'tcx> {
    fn from(err: ::mir::interpret::EvalError<'tcx>) -> ErrKind<'tcx> {
        ErrKind::Miri(err)
    }
}

impl<'tcx> From<ConstMathErr> for ErrKind<'tcx> {
    fn from(err: ConstMathErr) -> ErrKind<'tcx> {
        match err {
            ConstMathErr::UnsignedNegation => ErrKind::TypeckError,
            _ => ErrKind::Math(err)
        }
    }
}

#[derive(Clone, Debug)]
pub enum ConstEvalErrDescription<'a> {
    Simple(Cow<'a, str>),
}

impl<'a> ConstEvalErrDescription<'a> {
    /// Return a one-line description of the error, for lints and such
    pub fn into_oneline(self) -> Cow<'a, str> {
        match self {
            ConstEvalErrDescription::Simple(simple) => simple,
        }
    }
}

impl<'a, 'gcx, 'tcx> ConstEvalErr<'tcx> {
    pub fn description(&self) -> ConstEvalErrDescription {
        use self::ErrKind::*;
        use self::ConstEvalErrDescription::*;

        macro_rules! simple {
            ($msg:expr) => ({ Simple($msg.into_cow()) });
            ($fmt:expr, $($arg:tt)+) => ({
                Simple(format!($fmt, $($arg)+).into_cow())
            })
        }

        match self.kind {
            CannotCast => simple!("can't cast this type"),
            MissingStructField  => simple!("nonexistent struct field"),
            NonConstPath        => simple!("non-constant path in constant expression"),
            UnimplementedConstVal(what) =>
                simple!("unimplemented constant expression: {}", what),
            ExpectedConstTuple => simple!("expected constant tuple"),
            ExpectedConstStruct => simple!("expected constant struct"),
            IndexedNonVec => simple!("indexing is only supported for arrays"),
            IndexNotUsize => simple!("indices must be of type `usize`"),
            IndexOutOfBounds { len, index } => {
                simple!("index out of bounds: the len is {} but the index is {}",
                        len, index)
            }

            MiscBinaryOp => simple!("bad operands for binary"),
            MiscCatchAll => simple!("unsupported constant expr"),
            IndexOpFeatureGated => simple!("the index operation on const values is unstable"),
            Math(ref err) => Simple(err.description().into_cow()),
            LayoutError(ref err) => Simple(err.to_string().into_cow()),

            ErroneousReferencedConstant(_) => simple!("could not evaluate referenced constant"),

            TypeckError => simple!("type-checking failed"),
            CheckMatchError => simple!("match-checking failed"),
            // FIXME: report a full backtrace
            Miri(ref err) => simple!("miri failed: {}", err),
        }
    }

    pub fn struct_error(&self,
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        primary_span: Span,
        primary_kind: &str)
        -> DiagnosticBuilder<'gcx>
    {
        let mut err = self;
        while let &ConstEvalErr {
            kind: ErrKind::ErroneousReferencedConstant(box ref i_err), ..
        } = err {
            err = i_err;
        }

        let mut diag = struct_error(tcx, err.span, "constant evaluation error");
        err.note(tcx, primary_span, primary_kind, &mut diag);
        diag
    }

    pub fn note(&self,
        _tcx: TyCtxt<'a, 'gcx, 'tcx>,
        primary_span: Span,
        primary_kind: &str,
        diag: &mut DiagnosticBuilder)
    {
        match self.description() {
            ConstEvalErrDescription::Simple(message) => {
                diag.span_label(self.span, message);
            }
        }

        if !primary_span.contains(self.span) {
            diag.span_note(primary_span,
                        &format!("for {} here", primary_kind));
        }
    }

    pub fn report(&self,
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        primary_span: Span,
        primary_kind: &str)
    {
        match self.kind {
            ErrKind::TypeckError | ErrKind::CheckMatchError => return,
            _ => {}
        }
        self.struct_error(tcx, primary_span, primary_kind).emit();
    }
}

pub fn struct_error<'a, 'gcx, 'tcx>(
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    span: Span,
    msg: &str,
) -> DiagnosticBuilder<'gcx> {
    struct_span_err!(tcx.sess, span, E0080, "{}", msg)
}
