//@ ignore-windows

// This test attempts to make sure that running `remove_dir_all`
// doesn't result in a NotFound error if one of the files it
// is deleting is deleted concurrently.
//
// The windows implementation for `remove_dir_all` is significantly
// more complicated, and has not yet been brought up to par with
// the implementation on other platforms, so this test is marked as
// `ignore-windows` until someone more experienced with windows can
// sort that out.

use std::fs::remove_dir_all;
use std::path::Path;
use std::thread;
use std::time::Duration;

use run_make_support::rfs::{create_dir, write};
use run_make_support::run_in_tmpdir;

fn main() {
    let mut race_happened = false;
    run_in_tmpdir(|| {
        for i in 0..150 {
            create_dir("outer");
            create_dir("outer/inner");
            write("outer/inner.txt", b"sometext");

            thread::scope(|scope| {
                scope.spawn(|| {
                    thread::sleep(Duration::from_nanos(i));
                    remove_dir_all("outer").unwrap();
                });

                scope.spawn(|| {
                    let r1 = remove_dir_all("outer/inner");
                    let r2 = remove_dir_all("outer/inner.txt");
                    if r1.is_ok() && r2.is_err() {
                        race_happened = true;
                    }
                });
            });

            assert!(!Path::new("outer").exists());

            // trying to remove a nonexistent top-level directory should
            // still result in an error.
            let Err(err) = remove_dir_all("outer") else {
                panic!("removing nonexistent dir did not result in an error");
            };
            assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        }
    });
    if !race_happened {
        eprintln!(
            "WARNING: multithreaded deletion never raced, \
                   try increasing the number of attempts or \
                   adjusting the sleep timing"
        );
    }
}
