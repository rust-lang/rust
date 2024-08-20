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
                let t1 = scope.spawn(|| {
                    thread::sleep(Duration::from_nanos(i));
                    remove_dir_all("outer").unwrap();
                });

                let race_happened_ref = &race_happened;
                let t2 = scope.spawn(|| {
                    let r1 = remove_dir_all("outer/inner");
                    let r2 = remove_dir_all("outer/inner.txt");
                    if r1.is_ok() && r2.is_err() {
                        race_happened = true;
                    }
                });
            });

            assert!(!Path::new("outer").exists());

            // trying to remove the top-level directory should
            // still result in an error
            let Err(err) = remove_dir_all("outer") else {
                panic!("removing nonexistant dir did not result in an error");
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
