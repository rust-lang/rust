use super::*;

#[test]
fn test_all_except_most_recent() {
    let input: UnordMap<_, Option<flock::Lock>> = UnordMap::from_iter([
        ((UNIX_EPOCH + Duration::new(4, 0), PathBuf::from("4")), None),
        ((UNIX_EPOCH + Duration::new(1, 0), PathBuf::from("1")), None),
        ((UNIX_EPOCH + Duration::new(5, 0), PathBuf::from("5")), None),
        ((UNIX_EPOCH + Duration::new(3, 0), PathBuf::from("3")), None),
        ((UNIX_EPOCH + Duration::new(2, 0), PathBuf::from("2")), None),
    ]);
    assert_eq!(
        all_except_most_recent(input).into_items().map(|(path, _)| path).into_sorted_stable_ord(),
        vec![PathBuf::from("1"), PathBuf::from("2"), PathBuf::from("3"), PathBuf::from("4")]
    );

    assert!(all_except_most_recent(UnordMap::default()).is_empty());
}

#[test]
fn test_timestamp_serialization() {
    for i in 0..1_000u64 {
        let time = UNIX_EPOCH + Duration::new(i * 1_434_578, (i as u32) * 239_000);
        let s = timestamp_to_string(time);
        assert_eq!(Ok(time), string_to_timestamp(&s));
    }
}

#[test]
fn test_find_source_directory_in_iter() {
    let already_visited = FxHashSet::default();

    // Find newest
    assert_eq!(
        find_source_directory_in_iter(
            [
                PathBuf::from("crate-dir/s-3234-0000-svh"),
                PathBuf::from("crate-dir/s-2234-0000-svh"),
                PathBuf::from("crate-dir/s-1234-0000-svh")
            ]
            .into_iter(),
            &already_visited
        ),
        Some(PathBuf::from("crate-dir/s-3234-0000-svh"))
    );

    // Filter out "-working"
    assert_eq!(
        find_source_directory_in_iter(
            [
                PathBuf::from("crate-dir/s-3234-0000-working"),
                PathBuf::from("crate-dir/s-2234-0000-svh"),
                PathBuf::from("crate-dir/s-1234-0000-svh")
            ]
            .into_iter(),
            &already_visited
        ),
        Some(PathBuf::from("crate-dir/s-2234-0000-svh"))
    );

    // Handle empty
    assert_eq!(find_source_directory_in_iter([].into_iter(), &already_visited), None);

    // Handle only working
    assert_eq!(
        find_source_directory_in_iter(
            [
                PathBuf::from("crate-dir/s-3234-0000-working"),
                PathBuf::from("crate-dir/s-2234-0000-working"),
                PathBuf::from("crate-dir/s-1234-0000-working")
            ]
            .into_iter(),
            &already_visited
        ),
        None
    );
}

#[test]
fn test_replace_hard_link_with_copy_unshares_hard_link() {
    let dir = rustc_fs_util::TempDirBuilder::new().tempdir_in(std::env::temp_dir()).unwrap();
    let file = dir.path().join("file");
    let link = dir.path().join("link");
    std_fs::write(&file, b"original").unwrap();
    std_fs::hard_link(&file, &link).unwrap();

    replace_hard_link_with_copy(&link).unwrap();

    std_fs::write(&file, b"changed").unwrap();
    assert_eq!(std_fs::read(&link).unwrap(), b"original");
    assert!(!link.with_added_extension("tmp").exists());
}
