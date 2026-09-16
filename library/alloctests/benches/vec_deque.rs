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

/// first_len is the length of the first slice returned by as_slices
fn clone_fixture<T: Clone>(value: &T, (len, first_len): (usize, usize)) -> VecDeque<T> {
    let mut deque = VecDeque::with_capacity(len);
    let push_front = if first_len == len { 0 } else { first_len };
    deque.resize_with(len - push_front, || value.clone());
    for _ in 0..push_front {
        deque.push_front(value.clone());
    }
    let (first, second) = deque.as_slices();
    assert_eq!((first.len(), second.len()), (first_len, len - first_len));
    deque
}

/// times allocation and drop as well as cloning
fn do_bench_clone<T: Clone>(b: &mut Bencher, value: T, layout: (usize, usize)) {
    let src = clone_fixture(&value, layout);

    b.iter(|| black_box(black_box(&src).clone()));
}

fn do_bench_clone_batch_32<T: Clone>(b: &mut Bencher, value: T, layout: (usize, usize)) {
    let src = clone_fixture(&value, layout);

    b.iter(|| {
        // keep all clones alive so the allocator can't reuse the same buffer for each clone
        let clones: [VecDeque<T>; 32] = std::array::from_fn(|_| black_box(&src).clone());
        black_box(&clones);
    });
}

/// clone_from may make the destination contiguous even when the source is wrapped
fn do_bench_clone_from<T: Clone>(b: &mut Bencher, value: T, layout: (usize, usize)) {
    let src = clone_fixture(&value, layout);
    let mut dst = clone_fixture(&value, layout);

    b.iter(|| {
        dst.clone_from(black_box(&src));
        black_box(&dst);
    });
}

/// shrink and grow through clone_from so the timing doesn't include a separate reset
fn do_bench_clone_from_alternating<T: Clone>(
    b: &mut Bencher,
    value: T,
    long: (usize, usize),
    short: (usize, usize),
) {
    let long_src = clone_fixture(&value, long);
    let short_src = clone_fixture(&value, short);
    let mut dst = clone_fixture(&value, long);

    b.iter(|| {
        dst.clone_from(black_box(&short_src));
        dst.clone_from(black_box(&long_src));
        black_box(&dst);
    });
}

fn do_bench_clone_from_empty<T: Clone>(b: &mut Bencher, value: T, layout: (usize, usize)) {
    let src = clone_fixture(&value, layout);

    b.iter(|| {
        let mut dst = VecDeque::new();
        dst.clone_from(black_box(&src));
        black_box(dst);
    });
}

macro_rules! clone_benches {
    ($($name:ident, $value:expr, $layout:expr;)*) => {
        $(
            #[bench]
            fn ${concat(bench_clone_, $name)}(b: &mut Bencher) {
                do_bench_clone(b, $value, $layout);
            }

            #[bench]
            fn ${concat(bench_clone_from_, $name)}(b: &mut Bencher) {
                do_bench_clone_from(b, $value, $layout);
            }
        )*
    };
}

clone_benches! {
    u64_empty, 42u64, (0, 0);
    u64_one, 42u64, (1, 1);
    u64_small, 42u64, (16, 16);
    u64_small_wrapped, 42u64, (16, 5);
    u64_contiguous, 42u64, (1024, 1024);
    u64_wrapped, 42u64, (1024, 384);
    string_contiguous, "abcdefgh".repeat(15), (1024, 1024);
    string_wrapped, "abcdefgh".repeat(15), (1024, 384);
    zst, (), (1024, 1024);
}

#[bench]
fn bench_clone_u64_small_batch_32(b: &mut Bencher) {
    do_bench_clone_batch_32(b, 42u64, (16, 16));
}

#[bench]
fn bench_clone_u64_small_wrapped_batch_32(b: &mut Bencher) {
    do_bench_clone_batch_32(b, 42u64, (16, 5));
}

macro_rules! clone_from_benches {
    ($($name:ident, $value:expr, $long:expr, $short:expr;)*) => {
        $(
            #[bench]
            fn ${concat(bench_clone_from_, $name)}(b: &mut Bencher) {
                do_bench_clone_from_alternating(b, $value, $long, $short);
            }
        )*
    };
}

clone_from_benches! {
    u64_alternating_contiguous, 42u64, (1024, 1024), (512, 512);
    u64_alternating_wrapped, 42u64, (1024, 384), (512, 192);
    string_alternating_contiguous, "abcdefgh".repeat(15), (1024, 1024), (512, 512);
    string_alternating_wrapped, "abcdefgh".repeat(15), (1024, 384), (512, 192);
}

#[bench]
fn bench_clone_from_u64_from_empty(b: &mut Bencher) {
    do_bench_clone_from_empty(b, 42u64, (1024, 1024));
}

#[bench]
fn bench_clone_from_string_from_empty(b: &mut Bencher) {
    do_bench_clone_from_empty(b, "abcdefgh".repeat(15), (1024, 1024));
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
