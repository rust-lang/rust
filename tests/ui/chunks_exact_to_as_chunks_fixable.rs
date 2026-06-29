#![warn(clippy::chunks_exact_to_as_chunks)]
#![allow(unused)]

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
}
