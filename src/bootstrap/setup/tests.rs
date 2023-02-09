use super::{SETTINGS_HASHES, VSCODE_SETTINGS};
use sha2::Digest;

#[test]
fn check_matching_settings_hash() {
    let mut hasher = sha2::Sha256::new();
    hasher.update(&VSCODE_SETTINGS);
    let hash = hex::encode(hasher.finalize().as_slice());
    assert_eq!(
        &hash,
        SETTINGS_HASHES.last().unwrap(),
        "Update `SETTINGS_HASHES` with the new hash of `src/etc/vscode_settings.json`"
    );
}
