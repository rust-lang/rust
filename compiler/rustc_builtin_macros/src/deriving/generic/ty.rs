//! A mini version of ast::Ty, which is easier to use, and features an explicit `Self` type to use
//! when specifying impls to be derived.

pub use Ty::*;

use rustc_ast::ptr::P;
use rustc_ast::{self as ast, Expr, GenericArg, GenericParamKind, Generics, SelfKind};
use rustc_expand::base::ExtCtxt;
use rustc_span::source_map::{respan, DUMMY_SP};
use rustc_span::symbol::{kw, Ident, Symbol};
use rustc_span::Span;

/// A path, e.g., `::std::option::Option::<i32>` (global). Has support
/// for type parameters.
#[derive(Clone)]
pub struct Path {
    path: Vec<Symbol>,
    params: Vec<Box<Ty>>,
    kind: PathKind,
}

#[derive(Clone)]
pub enum PathKind {
    Local,
    Global,
    Std,
}

impl Path {
    pub fn new(path: Vec<Symbol>) -> Path {
        Path::new_(path, Vec::new(), PathKind::Std)
    }
    pub fn new_local(path: Symbol) -> Path {
        Path::new_(vec![path], Vec::new(), PathKind::Local)
    }
    pub fn new_(path: Vec<Symbol>, params: Vec<Box<Ty>>, kind: PathKind) -> Path {
        Path { path, params, kind }
    }

    pub fn to_ty(
        &self,
        cx: &ExtCtxt<'_>,
        span: Span,
        self_ty: Ident,
        self_generics: &Generics,
    ) -> P<ast::Ty> {
        cx.ty_path(self.to_path(cx, span, self_ty, self_generics))
    }
    pub fn to_path(
        &self,
        cx: &ExtCtxt<'_>,
        span: Span,
        self_ty: Ident,
        self_generics: &Generics,
    ) -> ast::Path {
        let mut idents = self.path.iter().map(|s| Ident::new(*s, span)).collect();
        let tys = self.params.iter().map(|t| t.to_ty(cx, span, self_ty, self_generics));
        let params = tys.map(GenericArg::Type).collect();

        match self.kind {
            PathKind::Global => cx.path_all(span, true, idents, params),
            PathKind::Local => cx.path_all(span, false, idents, params),
            PathKind::Std => {
                let def_site = cx.with_def_site_ctxt(DUMMY_SP);
                idents.insert(0, Ident::new(kw::DollarCrate, def_site));
                cx.path_all(span, false, idents, params)
            }
        }
    }
}

/// A type. Supports pointers, Self, and literals.
#[derive(Clone)]
pub enum Ty {
    Self_,
    /// A reference.
    Ref(Box<Ty>, ast::Mutability),
    /// `mod::mod::Type<[lifetime], [Params...]>`, including a plain type
    /// parameter, and things like `i32`
    Path(Path),
    /// For () return types.
    Unit,
}

pub fn self_ref() -> Ty {
    Ref(Box::new(Self_), ast::Mutability::Not)
}

impl Ty {
    pub fn to_ty(
        &self,
        cx: &ExtCtxt<'_>,
        span: Span,
        self_ty: Ident,
        self_generics: &Generics,
    ) -> P<ast::Ty> {
        match self {
            Ref(ty, mutbl) => {
                let raw_ty = ty.to_ty(cx, span, self_ty, self_generics);
                cx.ty_ref(span, raw_ty, None, *mutbl)
            }
            Path(p) => p.to_ty(cx, span, self_ty, self_generics),
            Self_ => cx.ty_path(self.to_path(cx, span, self_ty, self_generics)),
            Unit => {
                let ty = ast::TyKind::Tup(vec![]);
                cx.ty(span, ty)
            }
        }
    }

    pub fn to_path(
        &self,
        cx: &ExtCtxt<'_>,
        span: Span,
        self_ty: Ident,
        generics: &Generics,
    ) -> ast::Path {
        match self {
            Self_ => {
                let params: Vec<_> = generics
                    .params
                    .iter()
                    .map(|param| match param.kind {
                        GenericParamKind::Lifetime { .. } => {
                            GenericArg::Lifetime(ast::Lifetime { id: param.id, ident: param.ident })
                        }
                        GenericParamKind::Type { .. } => {
                            GenericArg::Type(cx.ty_ident(span, param.ident))
                        }
                        GenericParamKind::Const { .. } => {
                            GenericArg::Const(cx.const_ident(span, param.ident))
                        }
                    })
                    .collect();

                cx.path_all(span, false, vec![self_ty], params)
            }
            Path(p) => p.to_path(cx, span, self_ty, generics),
            Ref(..) => cx.span_bug(span, "ref in a path in generic `derive`"),
            Unit => cx.span_bug(span, "unit in a path in generic `derive`"),
        }
    }
}

fn mk_ty_param(
    cx: &ExtCtxt<'_>,
    span: Span,
    name: Symbol,
    bounds: &[Path],
    self_ident: Ident,
    self_generics: &Generics,
) -> ast::GenericParam {
    let bounds = bounds
        .iter()
        .map(|b| {
            let path = b.to_path(cx, span, self_ident, self_generics);
            cx.trait_bound(path, false)
        })
        .collect();
    cx.typaram(span, Ident::new(name, span), bounds, None)
}

/// Bounds on type parameters.
#[derive(Clone)]
pub struct Bounds {
    pub bounds: Vec<(Symbol, Vec<Path>)>,
}

impl Bounds {
    pub fn empty() -> Bounds {
        Bounds { bounds: Vec::new() }
    }
    pub fn to_generics(
        &self,
        cx: &ExtCtxt<'_>,
        span: Span,
        self_ty: Ident,
        self_generics: &Generics,
    ) -> Generics {
        let params = self
            .bounds
            .iter()
            .map(|&(name, ref bounds)| mk_ty_param(cx, span, name, &bounds, self_ty, self_generics))
            .collect();

        Generics {
            params,
            where_clause: ast::WhereClause { has_where_token: false, predicates: Vec::new(), span },
            span,
        }
    }
}

pub fn get_explicit_self(cx: &ExtCtxt<'_>, span: Span) -> (P<Expr>, ast::ExplicitSelf) {
    // This constructs a fresh `self` path.
    let self_path = cx.expr_self(span);
    let self_ty = respan(span, SelfKind::Region(None, ast::Mutability::Not));
    (self_path, self_ty)
}
