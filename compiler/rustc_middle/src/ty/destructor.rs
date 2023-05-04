use rustc_hir as hir;
use rustc_hir::def_id::DefId;

#[derive(Copy, Clone, Debug, HashStable, Encodable, Decodable)]
pub struct Destructor {
    /// The `DefId` of the destructor method
    pub did: DefId,
    /// The constness of the destructor method
    pub constness: hir::Constness,
}
