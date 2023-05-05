use crate::ty::UniverseIndex;

/// The "placeholder index" fully defines a placeholder region, type, or const. Placeholders are
/// identified by both a universe, as well as a name residing within that universe. Distinct bound
/// regions/types/consts within the same universe simply have an unknown relationship to one
/// another.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[derive(HashStable, TyEncodable, TyDecodable)]
pub struct Placeholder<T> {
    pub universe: UniverseIndex,
    pub bound: T,
}
