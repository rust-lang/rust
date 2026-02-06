use rustc_data_structures::fingerprint::Fingerprint;

use crate::ich::StableHashingContext;

// njn: what about this?
pub type HashResult<V> = Option<fn(&mut StableHashingContext<'_>, &V) -> Fingerprint>;
