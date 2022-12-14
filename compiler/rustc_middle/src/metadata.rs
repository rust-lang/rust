use crate::ty;

use rustc_hir::def::Res;
use rustc_macros::HashStable;
use rustc_span::def_id::DefId;
use rustc_span::symbol::Ident;
use rustc_span::Span;

/// This structure is supposed to keep enough data to re-create `NameBinding`s for other crates
/// during name resolution. Right now the bindings are not recreated entirely precisely so we may
/// need to add more data in the future to correctly support macros 2.0, for example.
/// Module child can be either a proper item or a reexport (including private imports).
/// In case of reexport all the fields describe the reexport item itself, not what it refers to.
#[derive(Copy, Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub struct ModChild {
    /// Name of the item.
    pub ident: Ident,
    /// Resolution result corresponding to the item.
    /// Local variables cannot be exported, so this `Res` doesn't need the ID parameter.
    pub res: Res<!>,
    /// Visibility of the item.
    pub vis: ty::Visibility<DefId>,
    /// Span of the item.
    pub span: Span,
    /// A proper `macro_rules` item (not a reexport).
    pub macro_rules: bool,
}
