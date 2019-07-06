//! A mini version of ast::Ty, which is easier to use, and features an explicit `Self` type to use
//! when specifying impls to be derived.

pub use PtrTy::*;
pub use Ty::*;

use syntax::ast::{self, Expr, GenericParamKind, Generics, Ident, SelfKind, GenericArg};
use syntax::ext::base::ExtCtxt;
use syntax::ext::build::AstBuilder;
use syntax::source_map::{respan, DUMMY_SP};
use syntax::ptr::P;
use syntax_pos::Span;
use syntax_pos::symbol::kw;

/// The types of pointers
#[derive(Clone)]
pub enum PtrTy<'a> {
    /// &'lifetime mut
    Borrowed(Option<&'a str>, ast::Mutability),
    /// *mut
    Raw(ast::Mutability),
}

/// A path, e.g., `::std::option::Option::<i32>` (global). Has support
/// for type parameters and a lifetime.
#[derive(Clone)]
pub struct Path<'a> {
    path: Vec<&'a str>,
    lifetime: Option<&'a str>,
    params: Vec<Box<Ty<'a>>>,
    kind: PathKind,
}

#[derive(Clone)]
pub enum PathKind {
    Local,
    Global,
    Std,
}

impl<'a> Path<'a> {
    pub fn new(path: Vec<&str>) -> Path<'_> {
        Path::new_(path, None, Vec::new(), PathKind::Std)
    }
    pub fn new_local(path: &str) -> Path<'_> {
        Path::new_(vec![path], None, Vec::new(), PathKind::Local)
    }
    pub fn new_<'r>(path: Vec<&'r str>,
                    lifetime: Option<&'r str>,
                    params: Vec<Box<Ty<'r>>>,
                    kind: PathKind)
                    -> Path<'r> {
        Path {
            path,
            lifetime,
            params,
            kind,
        }
    }

    pub fn to_ty(&self,
                 cx: &ExtCtxt<'_>,
                 span: Span,
                 self_ty: Ident,
                 self_generics: &Generics)
                 -> P<ast::Ty> {
        cx.ty_path(self.to_path(cx, span, self_ty, self_generics))
    }
    pub fn to_path(&self,
                   cx: &ExtCtxt<'_>,
                   span: Span,
                   self_ty: Ident,
                   self_generics: &Generics)
                   -> ast::Path {
        let mut idents = self.path.iter().map(|s| cx.ident_of(*s)).collect();
        let lt = mk_lifetimes(cx, span, &self.lifetime);
        let tys: Vec<P<ast::Ty>> =
            self.params.iter().map(|t| t.to_ty(cx, span, self_ty, self_generics)).collect();
        let params = lt.into_iter()
                       .map(|lt| GenericArg::Lifetime(lt))
                       .chain(tys.into_iter().map(|ty| GenericArg::Type(ty)))
                       .collect();

        match self.kind {
            PathKind::Global => cx.path_all(span, true, idents, params, Vec::new()),
            PathKind::Local => cx.path_all(span, false, idents, params, Vec::new()),
            PathKind::Std => {
                let def_site = DUMMY_SP.apply_mark(cx.current_expansion.mark);
                idents.insert(0, Ident::new(kw::DollarCrate, def_site));
                cx.path_all(span, false, idents, params, Vec::new())
            }
        }

    }
}

/// A type. Supports pointers, Self, and literals.
#[derive(Clone)]
pub enum Ty<'a> {
    Self_,
    /// &/Box/ Ty
    Ptr(Box<Ty<'a>>, PtrTy<'a>),
    /// mod::mod::Type<[lifetime], [Params...]>, including a plain type
    /// parameter, and things like `i32`
    Literal(Path<'a>),
    /// includes unit
    Tuple(Vec<Ty<'a>>),
}

/// A const expression. Supports literals and blocks.
#[derive(Clone, Eq, PartialEq)]
pub enum Const {
    Literal,
    Block,
}

pub fn borrowed_ptrty<'r>() -> PtrTy<'r> {
    Borrowed(None, ast::Mutability::Immutable)
}
pub fn borrowed(ty: Box<Ty<'_>>) -> Ty<'_> {
    Ptr(ty, borrowed_ptrty())
}

pub fn borrowed_explicit_self<'r>() -> Option<Option<PtrTy<'r>>> {
    Some(Some(borrowed_ptrty()))
}

pub fn borrowed_self<'r>() -> Ty<'r> {
    borrowed(Box::new(Self_))
}

pub fn nil_ty<'r>() -> Ty<'r> {
    Tuple(Vec::new())
}

fn mk_lifetime(cx: &ExtCtxt<'_>, span: Span, lt: &Option<&str>) -> Option<ast::Lifetime> {
    lt.map(|s|
        cx.lifetime(span, Ident::from_str(s))
    )
}

fn mk_lifetimes(cx: &ExtCtxt<'_>, span: Span, lt: &Option<&str>) -> Vec<ast::Lifetime> {
    mk_lifetime(cx, span, lt).into_iter().collect()
}

impl<'a> Ty<'a> {
    pub fn to_ty(&self,
                 cx: &ExtCtxt<'_>,
                 span: Span,
                 self_ty: Ident,
                 self_generics: &Generics)
                 -> P<ast::Ty> {
        match *self {
            Ptr(ref ty, ref ptr) => {
                let raw_ty = ty.to_ty(cx, span, self_ty, self_generics);
                match *ptr {
                    Borrowed(ref lt, mutbl) => {
                        let lt = mk_lifetime(cx, span, lt);
                        cx.ty_rptr(span, raw_ty, lt, mutbl)
                    }
                    Raw(mutbl) => cx.ty_ptr(span, raw_ty, mutbl),
                }
            }
            Literal(ref p) => p.to_ty(cx, span, self_ty, self_generics),
            Self_ => cx.ty_path(self.to_path(cx, span, self_ty, self_generics)),
            Tuple(ref fields) => {
                let ty = ast::TyKind::Tup(fields.iter()
                    .map(|f| f.to_ty(cx, span, self_ty, self_generics))
                    .collect());
                cx.ty(span, ty)
            }
        }
    }

    pub fn to_path(&self,
                   cx: &ExtCtxt<'_>,
                   span: Span,
                   self_ty: Ident,
                   generics: &Generics)
                   -> ast::Path {
        match *self {
            Self_ => {
                let params: Vec<_> = generics.params.iter().map(|param| match param.kind {
                    GenericParamKind::Lifetime { .. } => {
                        GenericArg::Lifetime(ast::Lifetime { id: param.id, ident: param.ident })
                    }
                    GenericParamKind::Type { .. } => {
                        GenericArg::Type(cx.ty_ident(span, param.ident))
                    }
                    GenericParamKind::Const { .. } => {
                        GenericArg::Const(cx.const_ident(span, param.ident))
                    }
                }).collect();

                cx.path_all(span, false, vec![self_ty], params, vec![])
            }
            Literal(ref p) => p.to_path(cx, span, self_ty, generics),
            Ptr(..) => cx.span_bug(span, "pointer in a path in generic `derive`"),
            Tuple(..) => cx.span_bug(span, "tuple in a path in generic `derive`"),
        }
    }
}


fn mk_ty_param(cx: &ExtCtxt<'_>,
               span: Span,
               name: &str,
               attrs: &[ast::Attribute],
               bounds: &[Path<'_>],
               self_ident: Ident,
               self_generics: &Generics)
               -> ast::GenericParam {
    let bounds = bounds.iter()
        .map(|b| {
            let path = b.to_path(cx, span, self_ident, self_generics);
            cx.trait_bound(path)
        })
        .collect();
    cx.typaram(span, cx.ident_of(name), attrs.to_owned(), bounds, None)
}

fn mk_generics(params: Vec<ast::GenericParam>, span: Span) -> Generics {
    Generics {
        params,
        where_clause: ast::WhereClause {
            predicates: Vec::new(),
            span,
        },
        span,
    }
}

/// Lifetimes and bounds on type parameters
#[derive(Clone)]
pub struct LifetimeBounds<'a> {
    pub lifetimes: Vec<(&'a str, Vec<&'a str>)>,
    pub bounds: Vec<(&'a str, Vec<Path<'a>>)>,
}

impl<'a> LifetimeBounds<'a> {
    pub fn empty() -> LifetimeBounds<'a> {
        LifetimeBounds {
            lifetimes: Vec::new(),
            bounds: Vec::new(),
        }
    }
    pub fn to_generics(&self,
                       cx: &ExtCtxt<'_>,
                       span: Span,
                       self_ty: Ident,
                       self_generics: &Generics)
                       -> Generics {
        let generic_params = self.lifetimes
            .iter()
            .map(|&(lt, ref bounds)| {
                let bounds = bounds.iter()
                    .map(|b| ast::GenericBound::Outlives(cx.lifetime(span, Ident::from_str(b))));
                cx.lifetime_def(span, Ident::from_str(lt), vec![], bounds.collect())
            })
            .chain(self.bounds
                .iter()
                .map(|t| {
                    let (name, ref bounds) = *t;
                    mk_ty_param(cx, span, name, &[], &bounds, self_ty, self_generics)
                })
            )
            .collect();

        mk_generics(generic_params, span)
    }
}

pub fn get_explicit_self(cx: &ExtCtxt<'_>,
                         span: Span,
                         self_ptr: &Option<PtrTy<'_>>)
                         -> (P<Expr>, ast::ExplicitSelf) {
    // this constructs a fresh `self` path
    let self_path = cx.expr_self(span);
    match *self_ptr {
        None => (self_path, respan(span, SelfKind::Value(ast::Mutability::Immutable))),
        Some(ref ptr) => {
            let self_ty =
                respan(span,
                       match *ptr {
                           Borrowed(ref lt, mutbl) => {
                               let lt = lt.map(|s| cx.lifetime(span, Ident::from_str(s)));
                               SelfKind::Region(lt, mutbl)
                           }
                           Raw(_) => {
                               cx.span_bug(span, "attempted to use *self in deriving definition")
                           }
                       });
            let self_expr = cx.expr_deref(span, self_path);
            (self_expr, self_ty)
        }
    }
}
