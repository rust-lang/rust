use rustc_data_structures::temp_dir::MaybeTempDir;
use rustc_session::Session;

use std::fs;
use std::path::PathBuf;

// FIXME(eddyb) maybe include the crate name in this?
pub const METADATA_FILENAME: &str = "lib.rmeta";

/// We use a temp directory here to avoid races between concurrent rustc processes,
/// such as builds in the same directory using the same filename for metadata while
/// building an `.rlib` (stomping over one another), or writing an `.rmeta` into a
/// directory being searched for `extern crate` (observing an incomplete file).
/// The returned path is the temporary file containing the complete metadata.
pub fn emit_metadata(sess: &Session, metadata: &[u8], tmpdir: &MaybeTempDir) -> PathBuf {
    let out_filename = tmpdir.as_ref().join(METADATA_FILENAME);
    let result = fs::write(&out_filename, metadata);

    if let Err(e) = result {
        sess.fatal(&format!("failed to write {}: {}", out_filename.display(), e));
    }

    out_filename
}
