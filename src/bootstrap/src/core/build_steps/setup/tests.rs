use super::{RUST_ANALYZER_SETTINGS, SETTINGS_HASHES};
use crate::utils::helpers::hex_encode;
use sha2::Digest;

#[test]
fn check_matching_settings_hash() {
    let mut hasher = sha2::Sha256::new();
    hasher.update(&RUST_ANALYZER_SETTINGS);
    let hash = hex_encode(hasher.finalize().as_slice());
    assert_eq!(
        &hash,
        SETTINGS_HASHES.last().unwrap(),
        "Update `SETTINGS_HASHES` with the new hash of `src/etc/rust_analyzer_settings.json`"
    );
}
