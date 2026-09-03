use std::thread;

fn _main() {
    let _t1 = thread::spawn(|| {
        for _ in 0..100 {
            println!("test");
        }
    });
}
