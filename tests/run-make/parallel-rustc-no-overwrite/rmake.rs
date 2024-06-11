// When two instances of rustc are invoked in parallel, they
// can conflict on their temporary files and overwrite each others',
// leading to unsuccessful compilation. The -Z temps-dir flag adds
// separate designated directories for each rustc invocation, preventing
// conflicts. This test uses this flag and checks for successful compilation.
// See https://github.com/rust-lang/rust/pull/83846

use run_make_support::{fs_wrapper, rustc};
use std::thread;

fn main() {
    fs_wrapper::create_file("lib.rs");
    let handle1 = thread::spawn(move || {
        rustc().crate_type("lib").arg("-Ztemps-dir=temp1").input("lib.rs");
    });

    let handle2 = thread::spawn(move || {
        rustc().crate_type("staticlib").arg("-Ztemps-dir=temp2").input("lib.rs");
    });
    handle1.join().expect("lib thread panicked");
    handle2.join().expect("staticlib thread panicked");
}
