#![warn(clippy::chunks_exact_to_as_chunks)]
#![allow(unused, clippy::deref_addrof)]

fn main() {
    let mut arr = [1, 2, 3, 4, 5, 6, 7, 8];

    for _ in arr.chunks_exact(4) {}
    //~^ chunks_exact_to_as_chunks
    for _ in arr.chunks_exact_mut(4) {}
    //~^ chunks_exact_to_as_chunks
    for chunk in arr.chunks_exact_mut(4).take(2) {
        //~^ chunks_exact_to_as_chunks
        chunk[0] += 1; // mutate chunks
    }

    // All const expressions should produce valid Rust code
    for _ in arr.chunks_exact(1 + 1) {}
    //~^ chunks_exact_to_as_chunks
    const CHUNK_SIZE: usize = 4;
    for _ in arr.chunks_exact(CHUNK_SIZE + 1) {}
    //~^ chunks_exact_to_as_chunks
    for _ in arr.chunks_exact(size_of::<u32>()) {}
    //~^ chunks_exact_to_as_chunks
    for _ in arr.chunks_exact(const { 1 }) {}
    //~^ chunks_exact_to_as_chunks
    for _ in arr.chunks_exact(unsafe { 1 }) {}
    //~^ chunks_exact_to_as_chunks
    struct A;
    impl A {
        const A: usize = 4;
    }
    for _ in arr.chunks_exact(A::A) {}
    //~^ chunks_exact_to_as_chunks
}
