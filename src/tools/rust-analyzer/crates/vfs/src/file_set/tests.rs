use super::*;

#[test]
fn path_prefix() {
    let mut file_set = FileSetConfig::builder();
    file_set.add_file_set(vec![VfsPath::new_virtual_path("/foo".into())]);
    file_set.add_file_set(vec![VfsPath::new_virtual_path("/foo/bar/baz".into())]);
    let file_set = file_set.build();

    let mut vfs = Vfs::default();
    vfs.set_file_contents(VfsPath::new_virtual_path("/foo/src/lib.rs".into()), Some(Vec::new()));
    vfs.set_file_contents(
        VfsPath::new_virtual_path("/foo/src/bar/baz/lib.rs".into()),
        Some(Vec::new()),
    );
    vfs.set_file_contents(
        VfsPath::new_virtual_path("/foo/bar/baz/lib.rs".into()),
        Some(Vec::new()),
    );
    vfs.set_file_contents(VfsPath::new_virtual_path("/quux/lib.rs".into()), Some(Vec::new()));

    let partition = file_set.partition(&vfs).into_iter().map(|it| it.len()).collect::<Vec<_>>();
    assert_eq!(partition, vec![2, 1, 1]);
}

#[test]
fn name_prefix() {
    let mut file_set = FileSetConfig::builder();
    file_set.add_file_set(vec![VfsPath::new_virtual_path("/foo".into())]);
    file_set.add_file_set(vec![VfsPath::new_virtual_path("/foo-things".into())]);
    let file_set = file_set.build();

    let mut vfs = Vfs::default();
    vfs.set_file_contents(VfsPath::new_virtual_path("/foo/src/lib.rs".into()), Some(Vec::new()));
    vfs.set_file_contents(
        VfsPath::new_virtual_path("/foo-things/src/lib.rs".into()),
        Some(Vec::new()),
    );

    let partition = file_set.partition(&vfs).into_iter().map(|it| it.len()).collect::<Vec<_>>();
    assert_eq!(partition, vec![1, 1, 0]);
}

/// Ensure that we don't consider `/foo/bar_baz.rs` to be in the
/// `/foo/bar/` root.
#[test]
fn name_prefix_partially_matches() {
    let mut file_set = FileSetConfig::builder();
    file_set.add_file_set(vec![VfsPath::new_virtual_path("/foo".into())]);
    file_set.add_file_set(vec![VfsPath::new_virtual_path("/foo/bar".into())]);
    let file_set = file_set.build();

    let mut vfs = Vfs::default();

    // These two are both in /foo.
    vfs.set_file_contents(VfsPath::new_virtual_path("/foo/lib.rs".into()), Some(Vec::new()));
    vfs.set_file_contents(VfsPath::new_virtual_path("/foo/bar_baz.rs".into()), Some(Vec::new()));

    // Only this file is in /foo/bar.
    vfs.set_file_contents(VfsPath::new_virtual_path("/foo/bar/biz.rs".into()), Some(Vec::new()));

    let partition = file_set.partition(&vfs).into_iter().map(|it| it.len()).collect::<Vec<_>>();

    assert_eq!(partition, vec![2, 1, 0]);
}
