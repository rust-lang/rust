//@ compile-flags: -Cinstrument-coverage=presence-only -Zunstable-options -Copt-level=0

use std::hint::black_box;

#[inline(never)]
fn repeated_hits() {
    for value in 0..100 {
        black_box(value);
    }
}

#[inline(never)]
fn selected_branch(take_true: bool) -> u32 {
    if take_true { 1 } else { 2 }
}

#[inline(never)]
fn concurrent_hit() {
    black_box(());
}

fn main() {
    repeated_hits();
    assert_eq!(selected_branch(true), 1);

    let threads = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                for _ in 0..100 {
                    concurrent_hit()
                }
            })
        })
        .collect::<Vec<_>>();
    for thread in threads {
        thread.join().unwrap();
    }
}
