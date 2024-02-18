//@ check-pass
//@ only-x86_64

// Checks that the compiler does not actually try to allocate 4 TB during compilation and OOM crash.

fn main() {
    [0; 4 * 1024 * 1024 * 1024 * 1024];
}
