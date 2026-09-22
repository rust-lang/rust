//! A mini version of ast::Ty, which is easier to use, and features an explicit `Self` type to use
//! when specifying impls to be derived.

use std::iter::once;

pub(crate) use Ty::*;
use rustc_ast::{self as ast, GenericArg};
use rustc_expand::base::ExtCtxt;
use rustc_span::{Ident, Span, Symbol, kw};
use thin_vec::ThinVec;

pub(crate) fn new_path(cx: &ExtCtxt<'_>, span: Span, path: &[Symbol], params: &[Ty]) -> ast::Path {
    let idents = path.iter().map(|s| Ident::new(*s, span));
    let tys = params.iter().map(|t| t.to_ty(cx, span));
    let params = tys.map(GenericArg::Type).collect();

    let idents = once(Ident::new(kw::DollarCrate, span)).chain(idents).collect();
    cx.path_all(span, false, idents, params)
}

/// A type. Supports pointers, Self, literals, unit or an arbitrary AST path.
#[derive(Clone)]
pub(crate) enum Ty {
    Self_,
    /// A reference.
    Ref(Box<Ty>, ast::Mutability),
    /// `mod::mod::Type<[lifetime], [Params...]>`, including a plain type
    /// parameter, and things like `i32`
    Path(ast::Path),
    /// For () return types.
    Unit,
    /// An arbitrary type.
    AstTy(Box<ast::Ty>),
}

pub(crate) fn self_ref() -> Ty {
    Ref(Box::new(Self_), ast::Mutability::Not)
}

impl Ty {
    pub(crate) fn to_ty(&self, cx: &ExtCtxt<'_>, span: Span) -> Box<ast::Ty> {
        match self {
            Ref(ty, mutbl) => {
                let raw_ty = ty.to_ty(cx, span);
                cx.ty_ref(span, raw_ty, None, *mutbl)
            }
            Path(p) => cx.ty_path(p.clone()),
            Self_ => cx.ty_path(cx.path_ident(span, Ident::new(kw::SelfUpper, span))),
            Unit => cx.ty(span, ast::TyKind::Tup(ThinVec::new())),
            AstTy(ty) => ty.clone(),
        }
    }
}
