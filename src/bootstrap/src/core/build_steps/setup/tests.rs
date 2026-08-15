use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use sha2::Digest;

use super::EditorKind;
use crate::utils::helpers::hex_encode;

#[test]
fn check_matching_settings_hash() {
    // Needs to be a btree so we serialize in a deterministic order.
    let mut mismatched = BTreeMap::new();

    for editor in EditorKind::ALL {
        let mut hasher = sha2::Sha256::new();
        hasher.update(&editor.settings_template());
        let actual = hex_encode(hasher.finalize().as_slice());
        let expected = *editor.hashes().last().unwrap();

        if expected != actual {
            mismatched.insert(editor, (expected, actual));
        }
    }

    if mismatched.is_empty() {
        return;
    }

    if option_env!("INSTA_UPDATE").is_some_and(|s| s != "0") {
        let mut updated = super::PARSED_HASHES.clone();
        for (editor, (_, actual)) in &mismatched {
            *updated.get_mut(editor).unwrap().last_mut().unwrap() = actual;
        }
        let hash_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/core/build_steps/setup/hashes.json");
        let mut hash_file = File::create(hash_path).unwrap();
        serde_json::to_writer_pretty(&mut hash_file, &updated).unwrap();
        hash_file.write_all(b"\n").unwrap();
    } else {
        for (editor, (expected, actual)) in &mismatched {
            eprintln!("recorded hash did not match actual hash: {expected} != {actual}");
            eprintln!(
                "Run `x test --bless -- hash`, or manually update `setup/hashes.json` with the new hash of `{actual}` for `EditorKind::{editor:?}`"
            );
        }
        panic!("mismatched hashes");
    }
}
