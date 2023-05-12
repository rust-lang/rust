use rustc_hir::def_id::CrateNum;

/// Contains information needed to resolve types and (in the future) look up
/// the types of AST nodes.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct CReaderCacheKey {
    pub cnum: Option<CrateNum>,
    pub pos: usize,
}
