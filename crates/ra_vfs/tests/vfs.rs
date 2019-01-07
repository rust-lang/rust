use std::{collections::HashSet, fs};

use flexi_logger::Logger;
use ra_vfs::{Vfs, VfsChange};
use tempfile::tempdir;

#[test]
fn test_vfs_works() -> std::io::Result<()> {
    Logger::with_str("debug").start().unwrap();

    let files = [
        ("a/foo.rs", "hello"),
        ("a/bar.rs", "world"),
        ("a/b/baz.rs", "nested hello"),
    ];

    let dir = tempdir()?;
    for (path, text) in files.iter() {
        let file_path = dir.path().join(path);
        fs::create_dir_all(file_path.parent().unwrap())?;
        fs::write(file_path, text)?
    }

    let a_root = dir.path().join("a");
    let b_root = dir.path().join("a/b");

    let (mut vfs, _) = Vfs::new(vec![a_root, b_root]);
    for _ in 0..2 {
        let task = vfs.task_receiver().recv().unwrap();
        vfs.handle_task(task);
    }
    {
        let files = vfs
            .commit_changes()
            .into_iter()
            .flat_map(|change| {
                let files = match change {
                    VfsChange::AddRoot { files, .. } => files,
                    _ => panic!("unexpected change"),
                };
                files.into_iter().map(|(_id, path, text)| {
                    let text: String = (&*text).clone();
                    (format!("{}", path.display()), text)
                })
            })
            .collect::<HashSet<_>>();

        let expected_files = [
            ("foo.rs", "hello"),
            ("bar.rs", "world"),
            ("baz.rs", "nested hello"),
        ]
        .iter()
        .map(|(path, text)| (path.to_string(), text.to_string()))
        .collect::<HashSet<_>>();

        assert_eq!(files, expected_files);
    }

    // on disk change
    fs::write(&dir.path().join("a/b/baz.rs"), "quux").unwrap();
    let task = vfs.task_receiver().recv().unwrap();
    vfs.handle_task(task);
    match vfs.commit_changes().as_slice() {
        [VfsChange::ChangeFile { text, .. }] => assert_eq!(text.as_str(), "quux"),
        _ => panic!("unexpected changes"),
    }

    // in memory change
    vfs.change_file_overlay(&dir.path().join("a/b/baz.rs"), "m".to_string());
    match vfs.commit_changes().as_slice() {
        [VfsChange::ChangeFile { text, .. }] => assert_eq!(text.as_str(), "m"),
        _ => panic!("unexpected changes"),
    }

    // in memory remove, restores data on disk
    vfs.remove_file_overlay(&dir.path().join("a/b/baz.rs"));
    match vfs.commit_changes().as_slice() {
        [VfsChange::ChangeFile { text, .. }] => assert_eq!(text.as_str(), "quux"),
        _ => panic!("unexpected changes"),
    }

    // in memory add
    vfs.add_file_overlay(&dir.path().join("a/b/spam.rs"), "spam".to_string());
    match vfs.commit_changes().as_slice() {
        [VfsChange::AddFile { text, path, .. }] => {
            assert_eq!(text.as_str(), "spam");
            assert_eq!(path, "spam.rs");
        }
        _ => panic!("unexpected changes"),
    }

    // in memory remove
    vfs.remove_file_overlay(&dir.path().join("a/b/spam.rs"));
    match vfs.commit_changes().as_slice() {
        [VfsChange::RemoveFile { path, .. }] => assert_eq!(path, "spam.rs"),
        _ => panic!("unexpected changes"),
    }

    // on disk add
    fs::write(&dir.path().join("a/new.rs"), "new hello").unwrap();
    let task = vfs.task_receiver().recv().unwrap();
    vfs.handle_task(task);
    match vfs.commit_changes().as_slice() {
        [VfsChange::AddFile { text, path, .. }] => {
            assert_eq!(text.as_str(), "new hello");
            assert_eq!(path, "new.rs");
        }
        _ => panic!("unexpected changes"),
    }

    // on disk rename
    fs::rename(&dir.path().join("a/new.rs"), &dir.path().join("a/new1.rs")).unwrap();
    let task = vfs.task_receiver().recv().unwrap();
    vfs.handle_task(task);
    match vfs.commit_changes().as_slice() {
        [VfsChange::RemoveFile {
            path: removed_path, ..
        }, VfsChange::AddFile {
            text,
            path: added_path,
            ..
        }] => {
            assert_eq!(removed_path, "new.rs");
            assert_eq!(added_path, "new1.rs");
            assert_eq!(text.as_str(), "new hello");
        }
        _ => panic!("unexpected changes"),
    }

    // on disk remove
    fs::remove_file(&dir.path().join("a/new1.rs")).unwrap();
    let task = vfs.task_receiver().recv().unwrap();
    vfs.handle_task(task);
    match vfs.commit_changes().as_slice() {
        [VfsChange::RemoveFile { path, .. }] => assert_eq!(path, "new1.rs"),
        _ => panic!("unexpected changes"),
    }

    match vfs.task_receiver().try_recv() {
        Err(crossbeam_channel::TryRecvError::Empty) => (),
        res => panic!("unexpected {:?}", res),
    }

    vfs.shutdown().unwrap();
    Ok(())
}
