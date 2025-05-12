use sha2::Digest;

use super::EditorKind;
use crate::utils::helpers::hex_encode;

#[test]
fn check_matching_settings_hash() {
    for editor in EditorKind::ALL {
        let mut hasher = sha2::Sha256::new();
        hasher.update(&editor.settings_template());
        let hash = hex_encode(hasher.finalize().as_slice());
        assert_eq!(
            &hash,
            editor.hashes().last().unwrap(),
            "Update `EditorKind::hashes()` with the new hash of `{}` for `EditorKind::{:?}`",
            editor.settings_template(),
            editor,
        );
    }
}
