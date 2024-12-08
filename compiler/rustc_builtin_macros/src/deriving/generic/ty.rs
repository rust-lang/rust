//! A mini version of ast::Ty, which is easier to use, and features an explicit `Self` type to use
//! when specifying impls to be derived.

pub(crate) use Ty::*;
use rustc_ast::ptr::P;
use rustc_ast::{self as ast, Expr, GenericArg, GenericParamKind, Generics, SelfKind};
use rustc_expand::base::ExtCtxt;
use rustc_span::source_map::respan;
use rustc_span::symbol::{Ident, Symbol, kw};
use rustc_span::{DUMMY_SP, Span};
use thin_vec::ThinVec;

/// A path, e.g., `::std::option::Option::<i32>` (global). Has support
/// for type parameters.
#[derive(Clone)]
pub(crate) struct Path {
    path: Vec<Symbol>,
    params: Vec<Box<Ty>>,
    kind: PathKind,
}

#[derive(Clone)]
pub(crate) enum PathKind {
    Local,
    Global,
    Std,
}

impl Path {
    pub(crate) fn new(path: Vec<Symbol>) -> Path {
        Path::new_(path, Vec::new(), PathKind::Std)
    }
    pub(crate) fn new_local(path: Symbol) -> Path {
        Path::new_(vec![path], Vec::new(), PathKind::Local)
    }
    pub(crate) fn new_(path: Vec<Symbol>, params: Vec<Box<Ty>>, kind: PathKind) -> Path {
        Path { path, params, kind }
    }

    pub(crate) fn to_ty(
        &self,
        cx: &ExtCtxt<'_>,
        span: Span,
        self_ty: Ident,
        self_generics: &Generics,
    ) -> P<ast::Ty> {
        cx.ty_path(self.to_path(cx, span, self_ty, self_generics))
    }
    pub(crate) fn to_path(
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
pub(crate) enum Ty {
    Self_,
    /// A reference.
    Ref(Box<Ty>, ast::Mutability),
    /// `mod::mod::Type<[lifetime], [Params...]>`, including a plain type
    /// parameter, and things like `i32`
    Path(Path),
    /// For () return types.
    Unit,
}

pub(crate) fn self_ref() -> Ty {
    Ref(Box::new(Self_), ast::Mutability::Not)
}

impl Ty {
    pub(crate) fn to_ty(
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
                let ty = ast::TyKind::Tup(ThinVec::new());
                cx.ty(span, ty)
            }
        }
    }

    pub(crate) fn to_path(
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
            Ref(..) => cx.dcx().span_bug(span, "ref in a path in generic `derive`"),
            Unit => cx.dcx().span_bug(span, "unit in a path in generic `derive`"),
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
pub(crate) struct Bounds {
    pub bounds: Vec<(Symbol, Vec<Path>)>,
}

impl Bounds {
    pub(crate) fn empty() -> Bounds {
        Bounds { bounds: Vec::new() }
    }
    pub(crate) fn to_generics(
        &self,
        cx: &ExtCtxt<'_>,
        span: Span,
        self_ty: Ident,
        self_generics: &Generics,
    ) -> Generics {
        let params = self
            .bounds
            .iter()
            .map(|&(name, ref bounds)| mk_ty_param(cx, span, name, bounds, self_ty, self_generics))
            .collect();

        Generics {
            params,
            where_clause: ast::WhereClause {
                has_where_token: false,
                predicates: ThinVec::new(),
                span,
            },
            span,
        }
    }
}

pub(crate) fn get_explicit_self(cx: &ExtCtxt<'_>, span: Span) -> (P<Expr>, ast::ExplicitSelf) {
    // This constructs a fresh `self` path.
    let self_path = cx.expr_self(span);
    let self_ty = respan(span, SelfKind::Region(None, ast::Mutability::Not));
    (self_path, self_ty)
}
