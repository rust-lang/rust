//@ check-pass

#![allow(clippy::explicit_counter_loop)]

fn main() {
    let v = vec![1, 2, 3];
    let mut i = 0;
    let max_storage_size = [0; 128 * 1024];
    for item in &v {
        bar(i, *item);
        i += 1;
    }
}

fn bar(_: usize, _: u32) {}
