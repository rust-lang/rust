use std::{collections::VecDeque, time::Instant};

const VECDEQUE_LEN: i32 = 100000;
const WARMUP_N: usize = 100;
const BENCH_N: usize = 1000;

fn main() {
    let a: VecDeque<i32> = (0..VECDEQUE_LEN).collect();
    let b: VecDeque<i32> = (0..VECDEQUE_LEN).collect();

    for _ in 0..WARMUP_N {
        let mut c = a.clone();
        let mut d = b.clone();
        c.append(&mut d);
    }

    let mut durations = Vec::with_capacity(BENCH_N);

    for _ in 0..BENCH_N {
        let mut c = a.clone();
        let mut d = b.clone();
        let before = Instant::now();
        c.append(&mut d);
        let after = Instant::now();
        durations.push(after.duration_since(before));
    }

    let l = durations.len();
    durations.sort();

    assert!(BENCH_N % 2 == 0);
    let median = (durations[(l / 2) - 1] + durations[l / 2]) / 2;
    println!(
        "\ncustom-bench vec_deque_append {:?} ns/iter\n",
        median.as_nanos()
    );
}
