use std::{collections::HashSet, fs, time::Duration};

// use flexi_logger::Logger;
use crossbeam_channel::RecvTimeoutError;
use ra_vfs::{Vfs, VfsChange};
use tempfile::tempdir;

/// Processes exactly `num_tasks` events waiting in the `vfs` message queue.
///
/// Panics if there are not exactly that many tasks enqueued for processing.
fn process_tasks(vfs: &mut Vfs, num_tasks: u32) {
    process_tasks_in_range(vfs, num_tasks, num_tasks);
}

/// Processes up to `max_count` events waiting in the `vfs` message queue.
///
/// Panics if it cannot process at least `min_count` events.
/// Panics if more than `max_count` events are enqueued for processing.
fn process_tasks_in_range(vfs: &mut Vfs, min_count: u32, max_count: u32) {
    for i in 0..max_count {
        let task = match vfs.task_receiver().recv_timeout(Duration::from_secs(3)) {
            Err(RecvTimeoutError::Timeout) if i >= min_count => return,
            otherwise => otherwise.unwrap(),
        };
        log::debug!("{:?}", task);
        vfs.handle_task(task);
    }
    assert!(vfs.task_receiver().is_empty());
}

macro_rules! assert_match {
    ($x:expr, $pat:pat) => {
        assert_match!($x, $pat, ())
    };
    ($x:expr, $pat:pat, $assert:expr) => {
        match $x {
            $pat => $assert,
            x => assert!(false, "Expected {}, got {:?}", stringify!($pat), x),
        };
    };
}

#[test]
fn test_vfs_works() -> std::io::Result<()> {
    // Logger::with_str("vfs=debug,ra_vfs=debug").start().unwrap();

    let files = [("a/foo.rs", "hello"), ("a/bar.rs", "world"), ("a/b/baz.rs", "nested hello")];

    let dir = tempdir().unwrap();
    for (path, text) in files.iter() {
        let file_path = dir.path().join(path);
        fs::create_dir_all(file_path.parent().unwrap()).unwrap();
        fs::write(file_path, text)?
    }

    let a_root = dir.path().join("a");
    let b_root = dir.path().join("a/b");

    let (mut vfs, _) = Vfs::new(vec![a_root, b_root]);
    process_tasks(&mut vfs, 2);
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

        let expected_files = [("foo.rs", "hello"), ("bar.rs", "world"), ("baz.rs", "nested hello")]
            .iter()
            .map(|(path, text)| (path.to_string(), text.to_string()))
            .collect::<HashSet<_>>();

        assert_eq!(files, expected_files);
    }

    // rust-analyzer#734: fsevents has a bunch of events still sitting around.
    process_tasks_in_range(&mut vfs, 0, 7);
    match vfs.commit_changes().as_slice() {
        [] => {}

        // This arises on fsevents (unless we wait 30 seconds before
        // calling `Vfs::new` above). We need to churn through these
        // events so that we can focus on the event that arises from
        // the `fs::write` below.
        [VfsChange::ChangeFile { .. }, // hello
         VfsChange::ChangeFile { .. }, // world
         VfsChange::AddFile { .. }, // b/baz.rs, nested hello
         VfsChange::ChangeFile { .. }, // hello
         VfsChange::ChangeFile { .. }, // world
         VfsChange::ChangeFile { .. }, // nested hello
         VfsChange::ChangeFile { .. }, // nested hello
        ] => {}

        changes => panic!("Expected events for setting up initial files, got: {GOT:?}",
                          GOT=changes),
    }

    fs::write(&dir.path().join("a/b/baz.rs"), "quux").unwrap();
    process_tasks(&mut vfs, 1);
    assert_match!(
        vfs.commit_changes().as_slice(),
        [VfsChange::ChangeFile { text, .. }],
        assert_eq!(text.as_str(), "quux")
    );

    vfs.add_file_overlay(&dir.path().join("a/b/baz.rs"), "m".to_string());
    assert_match!(
        vfs.commit_changes().as_slice(),
        [VfsChange::ChangeFile { text, .. }],
        assert_eq!(text.as_str(), "m")
    );

    // changing file on disk while overlayed doesn't generate a VfsChange
    fs::write(&dir.path().join("a/b/baz.rs"), "corge").unwrap();
    process_tasks(&mut vfs, 1);
    assert_match!(vfs.commit_changes().as_slice(), []);

    // removing overlay restores data on disk
    vfs.remove_file_overlay(&dir.path().join("a/b/baz.rs"));
    assert_match!(
        vfs.commit_changes().as_slice(),
        [VfsChange::ChangeFile { text, .. }],
        assert_eq!(text.as_str(), "corge")
    );

    vfs.add_file_overlay(&dir.path().join("a/b/spam.rs"), "spam".to_string());
    assert_match!(vfs.commit_changes().as_slice(), [VfsChange::AddFile { text, path, .. }], {
        assert_eq!(text.as_str(), "spam");
        assert_eq!(path, "spam.rs");
    });

    vfs.remove_file_overlay(&dir.path().join("a/b/spam.rs"));
    assert_match!(
        vfs.commit_changes().as_slice(),
        [VfsChange::RemoveFile { path, .. }],
        assert_eq!(path, "spam.rs")
    );

    fs::create_dir_all(dir.path().join("a/sub1/sub2")).unwrap();
    fs::write(dir.path().join("a/sub1/sub2/new.rs"), "new hello").unwrap();
    process_tasks(&mut vfs, 1);
    assert_match!(vfs.commit_changes().as_slice(), [VfsChange::AddFile { text, path, .. }], {
        assert_eq!(text.as_str(), "new hello");
        assert_eq!(path, "sub1/sub2/new.rs");
    });

    fs::rename(&dir.path().join("a/sub1/sub2/new.rs"), &dir.path().join("a/sub1/sub2/new1.rs"))
        .unwrap();

    // rust-analyzer#734: For testing purposes, work-around
    // passcod/notify#181 by processing either 1 or 2 events. (In
    // particular, Mac can hand back either 1 or 2 events in a
    // timing-dependent fashion.)
    //
    // rust-analyzer#827: Windows generates extra `Write` events when
    // renaming? meaning we have extra tasks to process.
    process_tasks_in_range(&mut vfs, 1, if cfg!(windows) { 4 } else { 2 });
    match vfs.commit_changes().as_slice() {
        [VfsChange::RemoveFile { path: removed_path, .. }, VfsChange::AddFile { text, path: added_path, .. }] =>
        {
            assert_eq!(removed_path, "sub1/sub2/new.rs");
            assert_eq!(added_path, "sub1/sub2/new1.rs");
            assert_eq!(text.as_str(), "new hello");
        }

        // Hopefully passcod/notify#181 will be addressed in some
        // manner that will reliably emit an event mentioning
        // `sub1/sub2/new.rs`. But until then, must accept that
        // debouncing loses information unrecoverably.
        [VfsChange::AddFile { text, path: added_path, .. }] => {
            assert_eq!(added_path, "sub1/sub2/new1.rs");
            assert_eq!(text.as_str(), "new hello");
        }

        changes => panic!(
            "Expected events for rename of {OLD} to {NEW}, got: {GOT:?}",
            OLD = "sub1/sub2/new.rs",
            NEW = "sub1/sub2/new1.rs",
            GOT = changes
        ),
    }

    fs::remove_file(&dir.path().join("a/sub1/sub2/new1.rs")).unwrap();
    process_tasks(&mut vfs, 1);
    assert_match!(
        vfs.commit_changes().as_slice(),
        [VfsChange::RemoveFile { path, .. }],
        assert_eq!(path, "sub1/sub2/new1.rs")
    );

    {
        vfs.add_file_overlay(&dir.path().join("a/memfile.rs"), "memfile".to_string());
        assert_match!(
            vfs.commit_changes().as_slice(),
            [VfsChange::AddFile { text, .. }],
            assert_eq!(text.as_str(), "memfile")
        );
        fs::write(&dir.path().join("a/memfile.rs"), "ignore me").unwrap();
        process_tasks(&mut vfs, 1);
        assert_match!(vfs.commit_changes().as_slice(), []);
    }

    // should be ignored
    fs::create_dir_all(dir.path().join("a/target")).unwrap();
    fs::write(&dir.path().join("a/target/new.rs"), "ignore me").unwrap();

    assert_match!(
        vfs.task_receiver().recv_timeout(Duration::from_millis(300)), // slightly more than watcher debounce delay
        Err(RecvTimeoutError::Timeout)
    );

    Ok(())
}
