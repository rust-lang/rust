use rustc::hir;
use rustc::ty;

// Whether an item exists in the type or value namespace.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Namespace {
    Type,
    Value,
}

impl From<ty::AssocKind> for Namespace {
    fn from(a_kind: ty::AssocKind) -> Self {
        match a_kind {
            ty::AssocKind::Existential |
            ty::AssocKind::Type => Namespace::Type,
            ty::AssocKind::Const |
            ty::AssocKind::Method => Namespace::Value,
        }
    }
}

impl<'a> From <&'a hir::ImplItemKind> for Namespace {
    fn from(impl_kind: &'a hir::ImplItemKind) -> Self {
        match *impl_kind {
            hir::ImplItemKind::Existential(..) |
            hir::ImplItemKind::Type(..) => Namespace::Type,
            hir::ImplItemKind::Const(..) |
            hir::ImplItemKind::Method(..) => Namespace::Value,
        }
    }
}
