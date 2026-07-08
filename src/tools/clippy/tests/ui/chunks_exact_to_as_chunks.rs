#![warn(clippy::chunks_exact_to_as_chunks)]
#![allow(unused)]

fn main() {
    let slice = [1, 2, 3, 4, 5, 6, 7, 8];

    // Should trigger lint - literal constant
    let mut it = slice.chunks_exact(4);
    //~^ chunks_exact_to_as_chunks
    for chunk in it {}

    // Should trigger lint - const value
    const CHUNK_SIZE: usize = 4;
    let mut it = slice.chunks_exact(CHUNK_SIZE);
    //~^ chunks_exact_to_as_chunks
    for chunk in it {}

    // Should NOT trigger - runtime value
    let size = 4;
    let mut it = slice.chunks_exact(size);
    for chunk in it {}

    // Should trigger lint - with remainder
    let mut it = slice.chunks_exact(3);
    //~^ chunks_exact_to_as_chunks
    for chunk in &mut it {}
    for e in it.remainder() {}

    // Should trigger - mutable variant
    let mut arr = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut it = arr.chunks_exact_mut(4);
    //~^ chunks_exact_to_as_chunks
    for chunk in it {}

    // Should NOT trigger - type must unify with another branch
    let condition = true;
    let y = 3;
    let _ = if condition {
        slice.chunks_exact(5)
    } else {
        slice.chunks_exact(y)
    };
}
