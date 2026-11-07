//@compile-flags: --crate-name=build_script_build
//@ check-pass

#![warn(clippy::print_stdout)]

fn main() {
    // Fix #6041
    //
    // The `print_stdout` lint shouldn't emit in `build.rs`
    // as these methods are used for the build script.
    println!("Hello");
    print!("Hello");
}
