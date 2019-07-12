use crate::ich::StableHashingContext;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher,
                                           StableHasherResult};

#[derive(Copy, Clone, Debug, RustcEncodable, RustcDecodable)]
pub enum SignalledError { SawSomeError, NoErrorsSeen }

impl Default for SignalledError {
    fn default() -> SignalledError {
        SignalledError::NoErrorsSeen
    }
}

impl_stable_hash_for!(enum self::SignalledError { SawSomeError, NoErrorsSeen });

#[derive(Debug, Default, RustcEncodable, RustcDecodable)]
pub struct BorrowCheckResult {
    pub signalled_any_error: SignalledError,
}

impl<'a> HashStable<StableHashingContext<'a>> for BorrowCheckResult {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
        let BorrowCheckResult {
            ref signalled_any_error,
        } = *self;
        signalled_any_error.hash_stable(hcx, hasher);
    }
}
