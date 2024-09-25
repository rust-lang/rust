//! Dataflow analyses are built upon some interpretation of the
//! bitvectors attached to each basic block, represented via a
//! zero-sized structure.

mod borrowed_locals;
mod initialized;
mod liveness;
mod storage_liveness;

pub use self::borrowed_locals::{MaybeBorrowedLocals, borrowed_locals};
pub use self::initialized::{
    DefinitelyInitializedPlaces, EverInitializedPlaces, MaybeInitializedPlaces,
    MaybeUninitializedPlaces,
};
pub use self::liveness::{
    MaybeLiveLocals, MaybeTransitiveLiveLocals, TransferFunction as LivenessTransferFunction,
};
pub use self::storage_liveness::{MaybeRequiresStorage, MaybeStorageDead, MaybeStorageLive};
