// Ensure that Cranelift can be used to compile a simple program with `x86_64-unknown-linux-gnu`
// dist artifacts.

//@ only-dist
//@ only-nightly (cranelift is not stable yet)
//@ only-x86_64-unknown-linux-gnu
//@ compile-flags: -Z codegen-backend=cranelift
//@ run-pass

fn main() {
    println!("Hello world!");
}
