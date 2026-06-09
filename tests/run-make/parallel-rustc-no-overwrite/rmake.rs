//@ needs-target-std
//
// When two instances of rustc are invoked in parallel, they
// can conflict on their temporary files and overwrite each others',
// leading to unsuccessful compilation. The -Z temps-dir flag adds
// separate designated directories for each rustc invocation, preventing
// conflicts. This test uses this flag and checks for successful compilation.
// See https://github.com/rust-lang/rust/pull/83846

use std::sync::{Arc, Barrier};
use std::thread;

use run_make_support::{rfs, rustc};

fn main() {
    rfs::create_file("lib.rs");
    let barrier = Arc::new(Barrier::new(2));
    let handle = {
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();
            rustc().crate_type("lib").arg("-Ztemps-dir=temp1").input("lib.rs").run();
        })
    };
    barrier.wait();
    rustc().crate_type("staticlib").arg("-Ztemps-dir=temp2").input("lib.rs").run();
    handle.join().expect("lib thread panicked");
}
