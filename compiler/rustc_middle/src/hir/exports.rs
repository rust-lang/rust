use crate::ty;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::Res;
use rustc_hir::def_id::LocalDefId;
use rustc_macros::HashStable;
use rustc_span::symbol::Ident;
use rustc_span::Span;

use std::fmt::Debug;

/// This is the replacement export map. It maps a module to all of the exports
/// within.
pub type ExportMap = FxHashMap<LocalDefId, Vec<Export>>;

#[derive(Copy, Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub struct Export {
    /// The name of the target.
    pub ident: Ident,
    /// The resolution of the target.
    /// Local variables cannot be exported, so this `Res` doesn't need the ID parameter.
    pub res: Res<!>,
    /// The span of the target.
    pub span: Span,
    /// The visibility of the export.
    /// We include non-`pub` exports for hygienic macros that get used from extern crates.
    pub vis: ty::Visibility,
}
