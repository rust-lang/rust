// program should terminate when std::process::exit is called from any thread

//@ run-pass
//@ needs-threads

use std::{process, thread};

fn main() {
    let h = thread::spawn(|| {
        process::exit(0);
    });
    let _ = h.join();
}
