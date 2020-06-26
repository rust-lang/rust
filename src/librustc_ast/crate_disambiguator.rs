// This is here because `rustc_session` wants to refer to it,
// and so does `rustc_hir`, but `rustc_hir` shouldn't refer to `rustc_session`.

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::{base_n, impl_stable_hash_via_hash};

use std::fmt;

/// Hash value constructed out of all the `-C metadata` arguments passed to the
/// compiler. Together with the crate-name forms a unique global identifier for
/// the crate.
#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone, Copy, RustcEncodable, RustcDecodable)]
pub struct CrateDisambiguator(Fingerprint);

impl CrateDisambiguator {
    pub fn to_fingerprint(self) -> Fingerprint {
        self.0
    }
}

impl fmt::Display for CrateDisambiguator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let (a, b) = self.0.as_value();
        let as_u128 = a as u128 | ((b as u128) << 64);
        f.write_str(&base_n::encode(as_u128, base_n::CASE_INSENSITIVE))
    }
}

impl From<Fingerprint> for CrateDisambiguator {
    fn from(fingerprint: Fingerprint) -> CrateDisambiguator {
        CrateDisambiguator(fingerprint)
    }
}

impl_stable_hash_via_hash!(CrateDisambiguator);
