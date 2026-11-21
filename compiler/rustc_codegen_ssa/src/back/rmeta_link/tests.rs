use rustc_serialize::Encodable;
use rustc_serialize::opaque::MAGIC_END_BYTES;
use rustc_serialize::opaque::mem_encoder::MemEncoder;

use super::RmetaLink;

fn roundtrip(link: &RmetaLink) -> RmetaLink {
    RmetaLink::decode(&link.encode()).expect("freshly encoded rmeta-link must decode")
}

fn assert_same(a: &RmetaLink, b: &RmetaLink) {
    assert_eq!(a.rust_object_files, b.rust_object_files);
    assert_eq!(a.native_lib_filenames, b.native_lib_filenames);
    assert_eq!(a.native_lib_members, b.native_lib_members);
}

#[test]
fn empty_mapping_roundtrips() {
    let link = RmetaLink {
        rust_object_files: vec![],
        native_lib_filenames: vec![],
        native_lib_members: vec![],
    };
    assert_same(&link, &roundtrip(&link));
}

#[test]
fn libraries_without_members_stay_distinct_from_no_libraries() {
    let link = RmetaLink {
        rust_object_files: vec!["a.rcgu.o".to_string()],
        native_lib_filenames: vec![None, None, None],
        native_lib_members: vec![vec![], vec![], vec![]],
    };
    let decoded = roundtrip(&link);
    assert_eq!(decoded.native_lib_members.len(), 3);
    assert!(decoded.native_lib_members.iter().all(Vec::is_empty));
    assert_same(&link, &decoded);
}

#[test]
fn multiple_libraries_with_overlapping_ownership() {
    let link = RmetaLink {
        rust_object_files: vec!["a.rcgu.o".to_string(), "b.rcgu.o".to_string()],
        native_lib_filenames: vec![None, Some("libpacked.a".to_string()), None],
        native_lib_members: vec![vec![3, 4], vec![4, 5], vec![]],
    };
    let decoded = roundtrip(&link);
    assert_eq!(decoded.native_lib_members, vec![vec![3, 4], vec![4, 5], vec![]]);
    assert_same(&link, &decoded);
}

#[test]
fn exact_positions_are_preserved() {
    let positions = vec![0, 1, 127, 128, 255, 256, 16_383, 16_384, u32::MAX - 1, u32::MAX];
    let link = RmetaLink {
        rust_object_files: vec![],
        native_lib_filenames: vec![None, None],
        native_lib_members: vec![positions.clone(), positions.iter().rev().copied().collect()],
    };
    let decoded = roundtrip(&link);
    assert_eq!(decoded.native_lib_members[0], positions);
    assert_eq!(decoded.native_lib_members[1], positions.iter().rev().copied().collect::<Vec<_>>());
}

#[test]
fn data_without_ownership_map_is_rejected() {
    let mut encoder = MemEncoder::new();
    vec!["a.rcgu.o".to_string()].encode(&mut encoder);
    vec![None::<String>].encode(&mut encoder);
    let mut data = encoder.finish();
    data.extend_from_slice(MAGIC_END_BYTES);
    assert!(RmetaLink::decode(&data).is_none());
}

#[test]
fn trailing_data_is_rejected() {
    let link = RmetaLink {
        rust_object_files: vec![],
        native_lib_filenames: vec![None],
        native_lib_members: vec![vec![1]],
    };
    let mut data = link.encode();
    data.truncate(data.len() - MAGIC_END_BYTES.len());
    data.push(0);
    data.extend_from_slice(MAGIC_END_BYTES);
    assert!(RmetaLink::decode(&data).is_none());
}

#[test]
fn missing_end_marker_is_rejected() {
    let link = RmetaLink {
        rust_object_files: vec![],
        native_lib_filenames: vec![],
        native_lib_members: vec![],
    };
    let mut data = link.encode();
    data.truncate(data.len() - MAGIC_END_BYTES.len());
    assert!(RmetaLink::decode(&data).is_none());
}
