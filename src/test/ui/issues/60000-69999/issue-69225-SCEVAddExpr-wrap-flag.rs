// run-fail
// compile-flags: -C opt-level=3
// error-pattern: index out of bounds: the len is 0 but the index is 16777216
// ignore-wasm no panic or subprocess support
// ignore-emscripten no panic or subprocess support

fn do_test(x: usize) {
    let mut arr = vec![vec![0u8; 3]];

    let mut z = vec![0];
    for arr_ref in arr.iter_mut() {
        for y in 0..x {
            for _ in 0..1 {
                z.reserve_exact(x);
                let iterator = std::iter::repeat(0).take(x);
                let mut cnt = 0;
                iterator.for_each(|_| {
                    z[0] = 0;
                    cnt += 1;
                });
                let a = y * x;
                let b = (y + 1) * x - 1;
                let slice = &mut arr_ref[a..b];
                slice[1 << 24] += 1;
            }
        }
    }
}

fn main() {
    do_test(1);
    do_test(2);
}
