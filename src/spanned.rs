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
use syntax::source_map::Span;

use macros::MacroArg;
use utils::{mk_sp, outer_attributes};

use std::cmp::max;

/// Spanned returns a span including attributes, if available.
pub trait Spanned {
    fn span(&self) -> Span;
}

macro_rules! span_with_attrs_lo_hi {
    ($this:ident, $lo:expr, $hi:expr) => {{
        let attrs = outer_attributes(&$this.attrs);
        if attrs.is_empty() {
            mk_sp($lo, $hi)
        } else {
            mk_sp(attrs[0].span.lo(), $hi)
        }
    }};
}

macro_rules! span_with_attrs {
    ($this:ident) => {
        span_with_attrs_lo_hi!($this, $this.span.lo(), $this.span.hi())
    };
}

macro_rules! implement_spanned {
    ($this:ty) => {
        impl Spanned for $this {
            fn span(&self) -> Span {
                span_with_attrs!(self)
            }
        }
    };
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
        let lo = if self.attrs.is_empty() {
            self.pats[0].span.lo()
        } else {
            self.attrs[0].span.lo()
        };
        span_with_attrs_lo_hi!(self, lo, self.body.span.hi())
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
        let lo = if self.attrs.is_empty() {
            self.ident.span.lo()
        } else {
            self.attrs[0].span.lo()
        };
        let hi = if self.bounds.is_empty() {
            self.ident.span.hi()
        } else {
            self.bounds.last().unwrap().span().hi()
        };
        let ty_hi = if let ast::GenericParamKind::Type {
            default: Some(ref ty),
        } = self.kind
        {
            ty.span().hi()
        } else {
            hi
        };
        mk_sp(lo, max(hi, ty_hi))
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

impl Spanned for ast::GenericArg {
    fn span(&self) -> Span {
        match *self {
            ast::GenericArg::Lifetime(ref lt) => lt.ident.span,
            ast::GenericArg::Type(ref ty) => ty.span(),
        }
    }
}

impl Spanned for ast::GenericBound {
    fn span(&self) -> Span {
        match *self {
            ast::GenericBound::Trait(ref ptr, _) => ptr.span,
            ast::GenericBound::Outlives(ref l) => l.ident.span,
        }
    }
}

impl Spanned for MacroArg {
    fn span(&self) -> Span {
        match *self {
            MacroArg::Expr(ref expr) => expr.span(),
            MacroArg::Ty(ref ty) => ty.span(),
            MacroArg::Pat(ref pat) => pat.span(),
            MacroArg::Item(ref item) => item.span(),
        }
    }
}
