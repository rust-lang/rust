// check-pass

// Checks that the compiler does not actually try to allocate 4 TB during compilation and OOM crash.

fn foo() -> [u8; 4 * 1024 * 1024 * 1024 * 1024] {
    unimplemented!()
}

fn main() {
    foo();
}
