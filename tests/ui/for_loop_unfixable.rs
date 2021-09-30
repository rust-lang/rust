// Tests from for_loop.rs that don't have suggestions

#[warn(
    clippy::needless_range_loop,
    clippy::explicit_iter_loop,
    clippy::explicit_into_iter_loop,
    clippy::iter_next_loop,
    clippy::for_kv_map
)]
#[allow(clippy::linkedlist, clippy::unnecessary_mut_passed, clippy::similar_names)]
fn main() {
    let vec = vec![1, 2, 3, 4];

    for _v in vec.iter().next() {}
}
