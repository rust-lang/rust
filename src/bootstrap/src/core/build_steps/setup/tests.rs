use sha2::Digest;

use super::EditorKind;
use crate::utils::helpers::hex_encode;

#[test]
fn check_matching_settings_hash() {
    for editor in EditorKind::ALL {
        let mut hasher = sha2::Sha256::new();
        hasher.update(&editor.settings_template());
        let actual = hex_encode(hasher.finalize().as_slice());
        let expected = *editor.hashes().last().unwrap();
        assert_eq!(
            expected, actual,
            "Update `setup/hashes.json` with the new hash of `{actual}` for `EditorKind::{editor:?}`",
        );
    }
}
