use std::collections::{VecDeque, vec_deque};
use std::mem;

use test::{Bencher, black_box};

#[bench]
fn bench_new(b: &mut Bencher) {
    b.iter(|| {
        let ring: VecDeque<i32> = VecDeque::new();
        black_box(ring);
    })
}

#[bench]
fn bench_grow_1025(b: &mut Bencher) {
    b.iter(|| {
        let mut deq = VecDeque::new();
        for i in 0..1025 {
            deq.push_front(i);
        }
        black_box(deq);
    })
}

#[bench]
fn bench_iter_1000(b: &mut Bencher) {
    let ring: VecDeque<_> = (0..1000).collect();

    b.iter(|| {
        let mut sum = 0;
        for &i in &ring {
            sum += i;
        }
        black_box(sum);
    })
}

#[bench]
fn bench_mut_iter_1000(b: &mut Bencher) {
    let mut ring: VecDeque<_> = (0..1000).collect();

    b.iter(|| {
        let mut sum = 0;
        for i in &mut ring {
            sum += *i;
        }
        black_box(sum);
    })
}

#[bench]
fn bench_try_fold(b: &mut Bencher) {
    let ring: VecDeque<_> = (0..1000).collect();

    b.iter(|| black_box(ring.iter().try_fold(0, |a, b| Some(a + b))))
}

/// does the memory bookkeeping to reuse the buffer of the Vec between iterations.
/// `setup` must not modify its argument's length or capacity. `g` must not move out of its argument.
fn into_iter_helper<
    T: Copy,
    F: FnOnce(&mut VecDeque<T>),
    G: FnOnce(&mut vec_deque::IntoIter<T>),
>(
    v: &mut Vec<T>,
    setup: F,
    g: G,
) {
    let ptr = v.as_mut_ptr();
    let len = v.len();
    // ensure that the vec is full, to make sure that any wrapping from the deque doesn't
    // access uninitialized memory.
    assert_eq!(v.len(), v.capacity());

    let mut deque = VecDeque::from(mem::take(v));
    setup(&mut deque);

    let mut it = deque.into_iter();
    g(&mut it);

    mem::forget(it);

    // SAFETY: the provided functions are not allowed to modify the allocation, so the buffer is still alive.
    // len and capacity are accurate due to the above assertion.
    // All the elements in the buffer are still valid, because of `T: Copy` which implies `T: !Drop`.
    mem::forget(mem::replace(v, unsafe { Vec::from_raw_parts(ptr, len, len) }));
}

#[bench]
fn bench_into_iter(b: &mut Bencher) {
    let len = 1024;
    // we reuse this allocation for every run
    let mut vec: Vec<usize> = (0..len).collect();
    vec.shrink_to_fit();

    b.iter(|| {
        let mut sum = 0;
        into_iter_helper(
            &mut vec,
            |_| {},
            |it| {
                for i in it {
                    sum += i;
                }
            },
        );
        black_box(sum);

        let mut sum = 0;
        // rotating a full deque doesn't move any memory.
        into_iter_helper(
            &mut vec,
            |d| d.rotate_left(len / 2),
            |it| {
                for i in it {
                    sum += i;
                }
            },
        );
        black_box(sum);
    });
}

#[bench]
fn bench_into_iter_fold(b: &mut Bencher) {
    let len = 1024;

    // because `fold` takes ownership of the iterator,
    // we can't prevent it from dropping the memory,
    // so we have to bite the bullet and reallocate
    // for every iteration.
    b.iter(|| {
        let deque: VecDeque<usize> = (0..len).collect();
        assert_eq!(deque.len(), deque.capacity());
        let sum = deque.into_iter().fold(0, |a, b| a + b);
        black_box(sum);

        // rotating a full deque doesn't move any memory.
        let mut deque: VecDeque<usize> = (0..len).collect();
        assert_eq!(deque.len(), deque.capacity());
        deque.rotate_left(len / 2);
        let sum = deque.into_iter().fold(0, |a, b| a + b);
        black_box(sum);
    });
}

#[bench]
fn bench_into_iter_try_fold(b: &mut Bencher) {
    let len = 1024;
    // we reuse this allocation for every run
    let mut vec: Vec<usize> = (0..len).collect();
    vec.shrink_to_fit();

    // Iterator::any uses Iterator::try_fold under the hood
    b.iter(|| {
        let mut b = false;
        into_iter_helper(&mut vec, |_| {}, |it| b = it.any(|i| i == len - 1));
        black_box(b);

        into_iter_helper(&mut vec, |d| d.rotate_left(len / 2), |it| b = it.any(|i| i == len - 1));
        black_box(b);
    });
}

#[bench]
fn bench_into_iter_next_chunk(b: &mut Bencher) {
    let len = 1024;
    // we reuse this allocation for every run
    let mut vec: Vec<usize> = (0..len).collect();
    vec.shrink_to_fit();

    b.iter(|| {
        let mut buf = [0; 64];
        into_iter_helper(
            &mut vec,
            |_| {},
            |it| {
                while let Ok(a) = it.next_chunk() {
                    buf = a;
                }
            },
        );
        black_box(buf);

        into_iter_helper(
            &mut vec,
            |d| d.rotate_left(len / 2),
            |it| {
                while let Ok(a) = it.next_chunk() {
                    buf = a;
                }
            },
        );
        black_box(buf);
    });
}

#[bench]
fn bench_from_array_1000(b: &mut Bencher) {
    const N: usize = 1000;
    let mut array: [usize; N] = [0; N];

    for i in 0..N {
        array[i] = i;
    }

    b.iter(|| {
        let deq: VecDeque<_> = array.into();
        black_box(deq);
    })
}

#[bench]
fn bench_extend_bytes(b: &mut Bencher) {
    let mut ring: VecDeque<u8> = VecDeque::with_capacity(1000);
    let input: &[u8] = &[128; 512];

    b.iter(|| {
        ring.clear();
        ring.extend(black_box(input));
    });
}

#[bench]
fn bench_extend_vec(b: &mut Bencher) {
    let mut ring: VecDeque<u8> = VecDeque::with_capacity(1000);
    let input = vec![128; 512];

    b.iter(|| {
        ring.clear();

        let input = input.clone();
        ring.extend(black_box(input));
    });
}

#[bench]
fn bench_extend_trustedlen(b: &mut Bencher) {
    let mut ring: VecDeque<u16> = VecDeque::with_capacity(1000);

    b.iter(|| {
        ring.clear();
        ring.extend(black_box(0..512));
    });
}

#[bench]
fn bench_extend_chained_trustedlen(b: &mut Bencher) {
    let mut ring: VecDeque<u16> = VecDeque::with_capacity(1000);

    b.iter(|| {
        ring.clear();
        ring.extend(black_box((0..256).chain(768..1024)));
    });
}

#[bench]
fn bench_extend_chained_bytes(b: &mut Bencher) {
    let mut ring: VecDeque<u16> = VecDeque::with_capacity(1000);
    let input1: &[u16] = &[128; 256];
    let input2: &[u16] = &[255; 256];

    b.iter(|| {
        ring.clear();
        ring.extend(black_box(input1.iter().chain(input2.iter())));
    });
}

fn retain_rand(i: usize) -> u32 {
    let mut x = (i as u32).wrapping_add(0x9e37_79b9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn fill_retain_deque(v: &mut VecDeque<u32>, src: &[u32], wrapped: bool) {
    v.clear();
    if wrapped {
        let (front, back) = src.split_at(src.len() / 2);
        v.extend(back);
        v.prepend(front.iter().copied());
    } else {
        v.extend(src);
    }
    assert_eq!(v.as_slices().1.is_empty(), !wrapped);
}

macro_rules! retain_layout_bench {
    ($name:ident, $len:literal, $wrapped:literal, $src:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let src: Vec<u32> = (0..$len).map($src).collect();
            let mut v = VecDeque::with_capacity($len);
            b.iter(|| {
                fill_retain_deque(&mut v, black_box(&src), $wrapped);
                v.retain(|x| *x == 0);
            });
        }
    };
}

macro_rules! retain_layout_benches {
    (@layout $len:literal, $suffix:literal, $layout:ident, $wrapped:literal) => {
        retain_layout_bench!(
            ${concat(bench_retain_whole_, $layout, _, $suffix)},
            $len, $wrapped, |_| 0
        );
        retain_layout_bench!(
            ${concat(bench_retain_odd_, $layout, _, $suffix)},
            $len, $wrapped, |i| (i & 1) as u32
        );
        retain_layout_bench!(
            ${concat(bench_retain_half_, $layout, _, $suffix)},
            $len, $wrapped, |i| (i < $len / 2) as u32
        );
        retain_layout_bench!(
            ${concat(bench_retain_rand50_, $layout, _, $suffix)},
            $len, $wrapped, |i| retain_rand(i) & 1
        );
    };
    ($($len:literal => $suffix:literal;)*) => {
        $(
            retain_layout_benches!(@layout $len, $suffix, contiguous, false);
            retain_layout_benches!(@layout $len, $suffix, wrapped, true);
        )*
    };
}

retain_layout_benches! {
    4 => "000004";
    8 => "000008";
    16 => "000016";
    32 => "000032";
    64 => "000064";
    128 => "000128";
    256 => "000256";
    512 => "000512";
    1000 => "001000";
    10000 => "010000";
    100000 => "100000";
}
