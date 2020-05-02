// Tests from for_loop.rs that don't have suggestions

#[warn(
    clippy::needless_range_loop,
    clippy::explicit_iter_loop,
    clippy::explicit_into_iter_loop,
    clippy::iter_next_loop,
    clippy::reverse_range_loop,
    clippy::for_kv_map
)]
#[allow(
    clippy::linkedlist,
    clippy::shadow_unrelated,
    clippy::unnecessary_mut_passed,
    clippy::similar_names,
    unused,
    dead_code
)]
#[allow(clippy::many_single_char_names, unused_variables)]
fn main() {
    for i in 5..5 {
        println!("{}", i);
    }

    let vec = vec![1, 2, 3, 4];

    for _v in vec.iter().next() {}

    for i in (5 + 2)..(8 - 1) {
        println!("{}", i);
    }

    const ZERO: usize = 0;

    for i in ZERO..vec.len() {
        if f(&vec[i], &vec[i]) {
            panic!("at the disco");
        }
    }
}
