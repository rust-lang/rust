use crate::ty;

use rustc_hir::{def::Res};
use rustc_macros::HashStable;
use rustc_span::symbol::Ident;
use rustc_span::Span;
use rustc_hir::def_id::DefId;

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
    pub vis: ty::Visibility,
    /// Span of the item.
    pub span: Span,
    /// A proper `macro_rules` item (not a reexport).
    pub macro_rules: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, HashStable)]
pub enum DiffMode {
    Forward,
    Reverse,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Copy, HashStable)]
pub enum DiffActivity {
    Active,
    Const,
    OnlyGrad,
}

#[allow(dead_code)]
#[derive(Clone, Debug, HashStable)]
pub struct DiffItem {
    pub source: DefId,
    pub target: Ident,
    pub mode: DiffMode,
    pub respect_to: Vec<DiffActivity>,
}

//impl Default for DiffItem {
//    fn default() -> Self {
//        Self {
//            source: Ident::empty(),
//            target: Ident::empty(),
//            mode: DiffMode::Forward,
//            respect_to: Vec::new(),
//        }
//    }
//}

//impl<CTX: rustc_span::HashStableContext> HashStable<CTX> for DiffItem {
//    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
//        self.source.hash_stable(ctx, hasher);
//        self.respect_to.hash_stable(ctx, hasher);
//        self.target.hash_stable(ctx, hasher);
//        match self.mode {
//            DiffMode::Forward => 0.hash_stable(ctx, hasher),
//            DiffMode::Reverse => 1.hash_stable(ctx, hasher)
//        }
//    }
//}

pub type DiffItems = Vec<DiffItem>;
