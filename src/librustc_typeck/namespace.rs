use rustc::hir;
use rustc::ty;

// Whether an item exists in the type or value namespace.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Namespace {
    Type,
    Value,
}

impl From<ty::AssociatedKind> for Namespace {
    fn from(a_kind: ty::AssociatedKind) -> Self {
        match a_kind {
            ty::AssociatedKind::Existential |
            ty::AssociatedKind::Type => Namespace::Type,
            ty::AssociatedKind::Const |
            ty::AssociatedKind::Method => Namespace::Value,
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
