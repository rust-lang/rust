// Ensure we appropriately error instead of overflowing a calculation when creating a new Alloc
// Layout

//@ run-fail
//@ compile-flags: -C opt-level=3
//@ error-pattern: index out of bounds: the len is 0 but the index is 16777216

fn do_test(x: usize) {
    let arr = vec![vec![0u8; 3]];

    let mut z = Vec::new();
    for arr_ref in arr {
        for y in 0..x {
            for _ in 0..1 {
                z.extend(std::iter::repeat(0).take(x));
                let a = y * x;
                let b = (y + 1) * x - 1;
                let slice = &arr_ref[a..b];
                eprintln!("{} {} {} {}", a, b, arr_ref.len(), slice.len());
                eprintln!("{:?}", slice[1 << 24]);
            }
        }
    }
}

fn main() {
    do_test(1);
    do_test(2);
}
