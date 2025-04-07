//@ build-pass

// FIXME: Remove this test once `doc_cfg` feature is completely removed.

#[doc(cfg(unix))]
fn main() {}
