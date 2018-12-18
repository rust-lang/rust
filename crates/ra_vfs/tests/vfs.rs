use std::{
    fs,
    collections::HashSet,
};

use tempfile::tempdir;

use ra_vfs::{Vfs, VfsChange};

#[test]
fn test_vfs_works() -> std::io::Result<()> {
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

    let mut vfs = Vfs::new(vec![a_root, b_root]);
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

    vfs.add_file_overlay(&dir.path().join("a/b/baz.rs"), "quux".to_string());
    let change = vfs.commit_changes().pop().unwrap();
    match change {
        VfsChange::ChangeFile { text, .. } => assert_eq!(&*text, "quux"),
        _ => panic!("unexpected change"),
    }

    vfs.change_file_overlay(&dir.path().join("a/b/baz.rs"), "m".to_string());
    let change = vfs.commit_changes().pop().unwrap();
    match change {
        VfsChange::ChangeFile { text, .. } => assert_eq!(&*text, "m"),
        _ => panic!("unexpected change"),
    }

    vfs.remove_file_overlay(&dir.path().join("a/b/baz.rs"));
    let change = vfs.commit_changes().pop().unwrap();
    match change {
        VfsChange::ChangeFile { text, .. } => assert_eq!(&*text, "nested hello"),
        _ => panic!("unexpected change"),
    }

    vfs.add_file_overlay(&dir.path().join("a/b/spam.rs"), "spam".to_string());
    let change = vfs.commit_changes().pop().unwrap();
    match change {
        VfsChange::AddFile { text, path, .. } => {
            assert_eq!(&*text, "spam");
            assert_eq!(path, "spam.rs");
        }
        _ => panic!("unexpected change"),
    }

    vfs.remove_file_overlay(&dir.path().join("a/b/spam.rs"));
    let change = vfs.commit_changes().pop().unwrap();
    match change {
        VfsChange::RemoveFile { .. } => (),
        _ => panic!("unexpected change"),
    }

    vfs.shutdown().unwrap();
    Ok(())
}
