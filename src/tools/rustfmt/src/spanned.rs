use std::cmp::max;

use rustc_ast::ast;
use rustc_span::{Span, source_map};

use crate::macros::MacroArg;
use crate::patterns::RangeOperand;
use crate::utils::{mk_sp, outer_attributes};

/// Spanned returns a span including attributes, if available.
pub(crate) trait Spanned {
    fn span(&self) -> Span;
}

impl<T: Spanned> Spanned for Box<T> {
    fn span(&self) -> Span {
        (**self).span()
    }
}

impl<T> Spanned for source_map::Spanned<T> {
    fn span(&self) -> Span {
        self.span
    }
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
implement_spanned!(ast::AssocItem);
implement_spanned!(ast::Expr);
implement_spanned!(ast::ExprField);
implement_spanned!(ast::ForeignItem);
implement_spanned!(ast::Item);
implement_spanned!(ast::Local);
implement_spanned!(ast::WherePredicate);

impl Spanned for ast::Stmt {
    fn span(&self) -> Span {
        match self.kind {
            ast::StmtKind::Let(ref local) => mk_sp(local.span().lo(), self.span.hi()),
            ast::StmtKind::Item(ref item) => mk_sp(item.span().lo(), self.span.hi()),
            ast::StmtKind::Expr(ref expr) | ast::StmtKind::Semi(ref expr) => {
                mk_sp(expr.span().lo(), self.span.hi())
            }
            ast::StmtKind::MacCall(ref mac_stmt) => {
                if mac_stmt.attrs.is_empty() {
                    self.span
                } else {
                    mk_sp(mac_stmt.attrs[0].span.lo(), self.span.hi())
                }
            }
            ast::StmtKind::Empty => self.span,
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
            self.pat.span.lo()
        } else {
            self.attrs[0].span.lo()
        };
        let hi = if let Some(body) = &self.body {
            body.span.hi()
        } else {
            self.pat.span.hi()
        };
        span_with_attrs_lo_hi!(self, lo, hi)
    }
}

impl Spanned for ast::Param {
    fn span(&self) -> Span {
        if crate::items::is_named_param(self) {
            mk_sp(crate::items::span_lo_for_param(self), self.ty.span.hi())
        } else {
            self.ty.span
        }
    }
}

impl Spanned for ast::GenericParam {
    fn span(&self) -> Span {
        let lo = match self.kind {
            _ if !self.attrs.is_empty() => self.attrs[0].span.lo(),
            ast::GenericParamKind::Const { span, .. } => span.lo(),
            _ => self.ident.span.lo(),
        };
        let hi = if self.bounds.is_empty() {
            self.ident.span.hi()
        } else {
            self.bounds.last().unwrap().span().hi()
        };
        let ty_hi = if let ast::GenericParamKind::Type {
            default: Some(ref ty),
        }
        | ast::GenericParamKind::Const { ref ty, .. } = self.kind
        {
            ty.span().hi()
        } else {
            hi
        };
        mk_sp(lo, max(hi, ty_hi))
    }
}

impl Spanned for ast::FieldDef {
    fn span(&self) -> Span {
        // FIXME(default_field_values): This needs to be adjusted.
        span_with_attrs_lo_hi!(self, self.span.lo(), self.ty.span.hi())
    }
}

impl Spanned for ast::FnRetTy {
    fn span(&self) -> Span {
        match *self {
            ast::FnRetTy::Default(span) => span,
            ast::FnRetTy::Ty(ref ty) => ty.span,
        }
    }
}

impl Spanned for ast::GenericArg {
    fn span(&self) -> Span {
        match *self {
            ast::GenericArg::Lifetime(ref lt) => lt.ident.span,
            ast::GenericArg::Type(ref ty) => ty.span(),
            ast::GenericArg::Const(ref _const) => _const.value.span(),
        }
    }
}

impl Spanned for ast::GenericBound {
    fn span(&self) -> Span {
        match *self {
            ast::GenericBound::Trait(ref ptr) => ptr.span,
            ast::GenericBound::Outlives(ref l) => l.ident.span,
            ast::GenericBound::Use(_, span) => span,
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
            MacroArg::Keyword(_, span) => span,
        }
    }
}

impl Spanned for ast::MetaItemInner {
    fn span(&self) -> Span {
        self.span()
    }
}

impl Spanned for ast::PreciseCapturingArg {
    fn span(&self) -> Span {
        match self {
            ast::PreciseCapturingArg::Lifetime(lt) => lt.ident.span,
            ast::PreciseCapturingArg::Arg(path, _) => path.span,
        }
    }
}

impl<'a, T> Spanned for RangeOperand<'a, T> {
    fn span(&self) -> Span {
        self.span
    }
}
