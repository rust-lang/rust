use run_make_support::{
    fs_wrapper::{create_dir, write},
    run_in_tmpdir,
};
use std::thread;
use std::time::Duration;
use std::fs::remove_dir_all;

fn main() {
    run_in_tmpdir(|| {
        for i in 0..15 {
            create_dir("outer");
            create_dir("outer/inner");
            write("outer/inner.txt", b"sometext");

            let t1 = thread::spawn(move || {
                thread::sleep(Duration::from_nanos(i));
                remove_dir_all("outer").unwrap();
            });

            let t2 = thread::spawn(move || {
				let _ = remove_dir_all("outer/inner");
				let _ = remove_dir_all("outer/inner.txt");
			});

            t1.join().unwrap();
            t2.join().unwrap();

			// trying to remove the top-level directory should
			// still result in an error
			let Err(err) = remove_dir_all("outer") else {
				panic!("removing nonexistant dir did not result in an error");
			};
			assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        }
    })
}
