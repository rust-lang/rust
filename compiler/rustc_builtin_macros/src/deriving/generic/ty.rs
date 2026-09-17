//! A mini version of ast::Ty, which is easier to use, and features an explicit `Self` type to use
//! when specifying impls to be derived.

use std::iter::once;

pub(crate) use Ty::*;
use rustc_ast::{self as ast, GenericArg};
use rustc_expand::base::ExtCtxt;
use rustc_span::{DUMMY_SP, Ident, Span, Symbol, kw};
use thin_vec::ThinVec;

/// A path, e.g., `::std::option::Option::<i32>` (global). Has support
/// for type parameters.
#[derive(Clone)]
pub(crate) struct Path {
    path: Vec<Symbol>,
    params: Vec<Box<Ty>>,
}

impl Path {
    pub(crate) fn new(path: Vec<Symbol>) -> Path {
        Path::new_(path, Vec::new())
    }
    pub(crate) fn new_(path: Vec<Symbol>, params: Vec<Box<Ty>>) -> Path {
        Path { path, params }
    }

    pub(crate) fn to_path(&self, cx: &ExtCtxt<'_>, span: Span) -> ast::Path {
        let idents = self.path.iter().map(|s| Ident::new(*s, span));
        let tys = self.params.iter().map(|t| t.to_ty(cx, span));
        let params = tys.map(GenericArg::Type).collect();

        let def_site = cx.with_def_site_ctxt(DUMMY_SP);
        let idents = once(Ident::new(kw::DollarCrate, def_site)).chain(idents).collect();
        cx.path_all(span, false, idents, params)
    }
}

/// A type. Supports pointers, Self, literals, unit or an arbitrary AST path.
#[derive(Clone)]
pub(crate) enum Ty {
    Self_,
    /// A reference.
    Ref(Box<Ty>, ast::Mutability),
    /// `mod::mod::Type<[lifetime], [Params...]>`, including a plain type
    /// parameter, and things like `i32`
    Path(Path),
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
            Path(p) => cx.ty_path(p.to_path(cx, span)),
            Self_ => cx.ty_path(cx.path_ident(span, Ident::new(kw::SelfUpper, span))),
            Unit => cx.ty(span, ast::TyKind::Tup(ThinVec::new())),
            AstTy(ty) => ty.clone(),
        }
    }
}
