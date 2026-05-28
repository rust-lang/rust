// In Rust programs where the standard library is unavailable (#![no_std]), we may be interested
// in customizing how panics are handled. Here, the provider specifies that panics should be handled
// by entering an infinite loop. This test checks that this panic implementation can be transitively
// provided by an external crate.
// --emit=llvm-ir is used to avoid running the linker, as linking will fail due to the lack of main
// function in the crate.
// See https://github.com/rust-lang/rust/pull/50338

use run_make_support::rustc;

fn main() {
    rustc().input("panic-impl-provider.rs").run();
    rustc().input("panic-impl-consumer.rs").panic("abort").emit("llvm-ir").run();
}
