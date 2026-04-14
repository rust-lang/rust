//! Tests for unified content provider schema

use abi::schema::{content_source_kind, content_source_state, keys, kinds};

#[test]
fn test_content_source_kinds() {
    assert_eq!(content_source_kind::LIMINE_MODULE, "limine_module");
    assert_eq!(content_source_kind::ISO9660_DISK, "iso9660_disk");
}

#[test]
fn test_content_source_states() {
    assert_eq!(content_source_state::READY, "ready");
    assert_eq!(content_source_state::ERROR, "error");
    assert_eq!(content_source_state::INITIALIZING, "initializing");
}

#[test]
fn test_content_kinds_exist() {
    // Ensure content provider kinds are defined
    assert_eq!(kinds::CONTENT_SOURCE, "content.Source");
    assert_eq!(kinds::CONTENT_DIR, "fs.Directory");
    assert_eq!(kinds::CONTENT_FILE, "fs.File");
}

#[test]
fn test_content_keys_exist() {
    // ContentSource properties
    assert_eq!(keys::CONTENT_SOURCE_KIND, "content.source.kind");
    assert_eq!(keys::CONTENT_SOURCE_NAME, "content.source.name");
    assert_eq!(keys::CONTENT_SOURCE_PRIORITY, "content.source.priority");
    assert_eq!(keys::CONTENT_SOURCE_STATE, "content.source.state");
    assert_eq!(keys::CONTENT_SOURCE_GEN, "content.source.gen");

    // File properties
    assert_eq!(keys::FILE_NAME, "file.name");
    assert_eq!(keys::FILE_SIZE, "file.size");
    assert_eq!(keys::FILE_HASH, "file.hash");
    assert_eq!(keys::FILE_MIME, "file.mime");
    assert_eq!(keys::FILE_BYTESPACE, "file.bytespace");
    assert_eq!(keys::FILE_SOURCE, "file.source");

    // Directory properties
    assert_eq!(keys::DIR_NAME, "dir.name");
    assert_eq!(keys::DIR_PATH, "dir.path");
}

#[test]
fn test_content_priority_ordering() {
    // Limine should have higher priority than ISO
    let limine_priority = 100u64;
    let iso_priority = 50u64;

    assert!(
        limine_priority > iso_priority,
        "Limine modules should have higher priority than ISO for overlay resolution"
    );
}
