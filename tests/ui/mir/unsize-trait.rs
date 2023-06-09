// Check that the interpreter does not ICE when trying to unsize `B` to `[u8]`.
// This is a `build` test to ensure that const-prop-lint runs.
// build-pass

#![feature(unsize)]

fn foo<B>(buffer: &mut [B; 2])
    where B: std::marker::Unsize<[u8]>,
{
    let buffer: &[u8] = &buffer[0];
}

fn main() {
    foo(&mut [[0], [5]]);
}
