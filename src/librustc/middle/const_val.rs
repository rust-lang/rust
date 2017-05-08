// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::ConstVal::*;
pub use rustc_const_math::ConstInt;

use hir;
use hir::def::Def;
use hir::def_id::DefId;
use ty::TyCtxt;
use ty::subst::Substs;
use util::common::ErrorReported;
use rustc_const_math::*;

use graphviz::IntoCow;
use errors::DiagnosticBuilder;
use syntax::symbol::InternedString;
use syntax::ast;
use syntax_pos::Span;

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::rc::Rc;

pub type EvalResult<'tcx> = Result<ConstVal<'tcx>, ConstEvalErr<'tcx>>;

#[derive(Clone, Debug, Hash, RustcEncodable, RustcDecodable, Eq, PartialEq)]
pub enum ConstVal<'tcx> {
    Float(ConstFloat),
    Integral(ConstInt),
    Str(InternedString),
    ByteStr(Rc<Vec<u8>>),
    Bool(bool),
    Char(char),
    Variant(DefId),
    Function(DefId, &'tcx Substs<'tcx>),
    Struct(BTreeMap<ast::Name, ConstVal<'tcx>>),
    Tuple(Vec<ConstVal<'tcx>>),
    Array(Vec<ConstVal<'tcx>>),
    Repeat(Box<ConstVal<'tcx>>, u64),
}

impl<'tcx> ConstVal<'tcx> {
    pub fn description(&self) -> &'static str {
        match *self {
            Float(f) => f.description(),
            Integral(i) => i.description(),
            Str(_) => "string literal",
            ByteStr(_) => "byte string literal",
            Bool(_) => "boolean",
            Char(..) => "char",
            Variant(_) => "enum variant",
            Struct(_) => "struct",
            Tuple(_) => "tuple",
            Function(..) => "function definition",
            Array(..) => "array",
            Repeat(..) => "repeat",
        }
    }

    pub fn to_const_int(&self) -> Option<ConstInt> {
        match *self {
            ConstVal::Integral(i) => Some(i),
            ConstVal::Bool(b) => Some(ConstInt::U8(b as u8)),
            ConstVal::Char(ch) => Some(ConstInt::U32(ch as u32)),
            _ => None
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
    NegateOn(ConstVal<'tcx>),
    NotOn(ConstVal<'tcx>),

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

    ErroneousReferencedConstant(Box<ConstEvalErr<'tcx>>),

    TypeckError
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
            NegateOn(ref const_val) => simple!("negate on {}", const_val.description()),
            NotOn(ref const_val) => simple!("not on {}", const_val.description()),

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

            ErroneousReferencedConstant(_) => simple!("could not evaluate referenced constant"),

            TypeckError => simple!("type-checking failed"),
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

        let mut diag = struct_span_err!(tcx.sess, err.span, E0080, "constant evaluation error");
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
        if let ErrKind::TypeckError = self.kind {
            return;
        }
        self.struct_error(tcx, primary_span, primary_kind).emit();
    }
}

/// Returns the value of the length-valued expression
pub fn eval_length(tcx: TyCtxt,
                   count: hir::BodyId,
                   reason: &str)
                   -> Result<usize, ErrorReported>
{
    let count_expr = &tcx.hir.body(count).value;
    let count_def_id = tcx.hir.body_owner_def_id(count);
    let substs = Substs::empty();
    match tcx.at(count_expr.span).const_eval((count_def_id, substs)) {
        Ok(Integral(Usize(count))) => {
            let val = count.as_u64(tcx.sess.target.uint_type);
            assert_eq!(val as usize as u64, val);
            Ok(val as usize)
        },
        Ok(_) |
        Err(ConstEvalErr { kind: ErrKind::TypeckError, .. }) => Err(ErrorReported),
        Err(err) => {
            let mut diag = err.struct_error(tcx, count_expr.span, reason);

            if let hir::ExprPath(hir::QPath::Resolved(None, ref path)) = count_expr.node {
                if let Def::Local(..) = path.def {
                    diag.note(&format!("`{}` is a variable",
                                       tcx.hir.node_to_pretty_string(count_expr.id)));
                }
            }

            diag.emit();
            Err(ErrorReported)
        }
    }
}
