//@ revisions: incremental threads
//@ dont-check-compiler-stderr
//
//@ [threads] compile-flags: -Zfuel=a=1 -Zthreads=2
//@ [threads] error-pattern:optimization fuel is incompatible with multiple threads
//
//@ [incremental] incremental
//@ [incremental] compile-flags: -Zprint-fuel=a
//@ [incremental] error-pattern:optimization fuel is incompatible with incremental compilation

fn main() {}
