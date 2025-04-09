//! Test that using `__builtin_available` in C (`@available` in Objective-C)
//! successfully links (because `std` provides the required symbols).

//@ only-apple __builtin_available is (mostly) specific to Apple platforms.

use run_make_support::{cc, rustc, target};

fn main() {
    // Invoke the C compiler to generate an object file.
    //
    // (We cheat a bit here, and use the `rustc` target tuple directly as the
    // Clang tuple, though that might not work for older Clang versions).
    cc().arg("-target").arg(target()).arg("-c").input("foo.c").output("foo.o").run();

    // Link the object file together with a Rust program.
    rustc().target(target()).input("main.rs").link_arg("foo.o").run();
}
