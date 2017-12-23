// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;
use syntax::codemap::Span;

use macros::MacroArg;
use utils::{mk_sp, outer_attributes};

/// Spanned returns a span including attributes, if available.
pub trait Spanned {
    fn span(&self) -> Span;
}

macro_rules! span_with_attrs_lo_hi {
    ($this:ident, $lo:expr, $hi:expr) => {
        {
            let attrs = outer_attributes(&$this.attrs);
            if attrs.is_empty() {
                mk_sp($lo, $hi)
            } else {
                mk_sp(attrs[0].span.lo(), $hi)
            }
        }
    }
}

macro_rules! span_with_attrs {
    ($this:ident) => {
        span_with_attrs_lo_hi!($this, $this.span.lo(), $this.span.hi())
    }
}

macro_rules! implement_spanned {
    ($this:ty) => {
        impl Spanned for $this {
            fn span(&self) -> Span {
                span_with_attrs!(self)
            }
        }
    }
}

// Implement `Spanned` for structs with `attrs` field.
implement_spanned!(ast::Expr);
implement_spanned!(ast::Field);
implement_spanned!(ast::ForeignItem);
implement_spanned!(ast::Item);
implement_spanned!(ast::Local);
implement_spanned!(ast::TraitItem);
implement_spanned!(ast::ImplItem);

impl Spanned for ast::Stmt {
    fn span(&self) -> Span {
        match self.node {
            ast::StmtKind::Local(ref local) => mk_sp(local.span().lo(), self.span.hi()),
            ast::StmtKind::Item(ref item) => mk_sp(item.span().lo(), self.span.hi()),
            ast::StmtKind::Expr(ref expr) | ast::StmtKind::Semi(ref expr) => {
                mk_sp(expr.span().lo(), self.span.hi())
            }
            ast::StmtKind::Mac(ref mac) => {
                let (_, _, ref attrs) = **mac;
                if attrs.is_empty() {
                    self.span
                } else {
                    mk_sp(attrs[0].span.lo(), self.span.hi())
                }
            }
        }
    }
}

impl Spanned for ast::Pat {
    fn span(&self) -> Span {
        self.span
    }
}

impl Spanned for ast::Ty {
    fn span(&self) -> Span {
        self.span
    }
}

impl Spanned for ast::Arm {
    fn span(&self) -> Span {
        span_with_attrs_lo_hi!(self, self.pats[0].span.lo(), self.body.span.hi())
    }
}

impl Spanned for ast::Arg {
    fn span(&self) -> Span {
        if ::items::is_named_arg(self) {
            mk_sp(self.pat.span.lo(), self.ty.span.hi())
        } else {
            self.ty.span
        }
    }
}

impl Spanned for ast::GenericParam {
    fn span(&self) -> Span {
        match *self {
            ast::GenericParam::Lifetime(ref lifetime_def) => lifetime_def.span(),
            ast::GenericParam::Type(ref ty) => ty.span(),
        }
    }
}

impl Spanned for ast::StructField {
    fn span(&self) -> Span {
        span_with_attrs_lo_hi!(self, self.span.lo(), self.ty.span.hi())
    }
}

impl Spanned for ast::WherePredicate {
    fn span(&self) -> Span {
        match *self {
            ast::WherePredicate::BoundPredicate(ref p) => p.span,
            ast::WherePredicate::RegionPredicate(ref p) => p.span,
            ast::WherePredicate::EqPredicate(ref p) => p.span,
        }
    }
}

impl Spanned for ast::FunctionRetTy {
    fn span(&self) -> Span {
        match *self {
            ast::FunctionRetTy::Default(span) => span,
            ast::FunctionRetTy::Ty(ref ty) => ty.span,
        }
    }
}

impl Spanned for ast::TyParam {
    fn span(&self) -> Span {
        // Note that ty.span is the span for ty.ident, not the whole item.
        let lo = if self.attrs.is_empty() {
            self.span.lo()
        } else {
            self.attrs[0].span.lo()
        };
        if let Some(ref def) = self.default {
            return mk_sp(lo, def.span.hi());
        }
        if self.bounds.is_empty() {
            return mk_sp(lo, self.span.hi());
        }
        let hi = self.bounds[self.bounds.len() - 1].span().hi();
        mk_sp(lo, hi)
    }
}

impl Spanned for ast::TyParamBound {
    fn span(&self) -> Span {
        match *self {
            ast::TyParamBound::TraitTyParamBound(ref ptr, _) => ptr.span,
            ast::TyParamBound::RegionTyParamBound(ref l) => l.span,
        }
    }
}

impl Spanned for ast::LifetimeDef {
    fn span(&self) -> Span {
        let hi = if self.bounds.is_empty() {
            self.lifetime.span.hi()
        } else {
            self.bounds[self.bounds.len() - 1].span.hi()
        };
        mk_sp(self.lifetime.span.lo(), hi)
    }
}

impl Spanned for MacroArg {
    fn span(&self) -> Span {
        match *self {
            MacroArg::Expr(ref expr) => expr.span(),
            MacroArg::Ty(ref ty) => ty.span(),
            MacroArg::Pat(ref pat) => pat.span(),
        }
    }
}
