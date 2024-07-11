use run_make_support::{
    fs_wrapper::{create_dir, remove_dir_all, write},
    run_in_tmpdir,
};
use std::thread;
use std::time::Duration;

fn main() {
    run_in_tmpdir(|| {
        for i in 0..10 {
            create_dir("outer");
            create_dir("outer/inner");
            write("outer/inner.txt", b"sometext");

            let t1 = thread::spawn(move || {
                thread::sleep(Duration::from_millis(i * 10));
                remove_dir_all("outer")
            });

            let t2 = thread::spawn(move || remove_dir_all("outer"));

            t1.join().unwrap();
            t2.join().unwrap();
        }
    })
}
