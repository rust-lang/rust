use core::borrow::Borrow;
use core::iter::*;
use core::mem;
use core::num::Wrapping;
use core::ops::Range;
use test::{black_box, Bencher};

#[bench]
fn bench_rposition(b: &mut Bencher) {
    let it: Vec<usize> = (0..300).collect();
    b.iter(|| {
        it.iter().rposition(|&x| x <= 150);
    });
}

#[bench]
fn bench_skip_while(b: &mut Bencher) {
    b.iter(|| {
        let it = 0..100;
        let mut sum = 0;
        it.skip_while(|&x| {
            sum += x;
            sum < 4000
        })
        .all(|_| true);
    });
}

#[bench]
fn bench_multiple_take(b: &mut Bencher) {
    let mut it = (0..42).cycle();
    b.iter(|| {
        let n = it.next().unwrap();
        for _ in 0..n {
            it.clone().take(it.next().unwrap()).all(|_| true);
        }
    });
}

fn scatter(x: i32) -> i32 {
    (x * 31) % 127
}

#[bench]
fn bench_max_by_key(b: &mut Bencher) {
    b.iter(|| {
        let it = 0..100;
        it.map(black_box).max_by_key(|&x| scatter(x))
    })
}

// https://www.reddit.com/r/rust/comments/31syce/using_iterators_to_find_the_index_of_the_min_or/
#[bench]
fn bench_max_by_key2(b: &mut Bencher) {
    fn max_index_iter(array: &[i32]) -> usize {
        array.iter().enumerate().max_by_key(|&(_, item)| item).unwrap().0
    }

    let mut data = vec![0; 1638];
    data[514] = 9999;

    b.iter(|| max_index_iter(&data));
}

#[bench]
fn bench_max(b: &mut Bencher) {
    b.iter(|| {
        let it = 0..100;
        it.map(black_box).map(scatter).max()
    })
}

#[bench]
fn bench_range_step_by_sum_reducible(b: &mut Bencher) {
    let r = 0u32..1024;
    b.iter(|| {
        let r = black_box(r.clone()).step_by(8);

        let mut sum: u32 = 0;
        for i in r {
            sum += i;
        }

        sum
    })
}

#[bench]
fn bench_range_step_by_loop_u32(b: &mut Bencher) {
    let r = 0..(u16::MAX as u32);
    b.iter(|| {
        let r = black_box(r.clone()).step_by(64);

        let mut sum: u32 = 0;
        for i in r {
            let i = i ^ i.wrapping_sub(1);
            sum = sum.wrapping_add(i);
        }

        sum
    })
}

#[bench]
fn bench_range_step_by_fold_usize(b: &mut Bencher) {
    let r: Range<usize> = 0..(u16::MAX as usize);
    b.iter(|| {
        let r = black_box(r.clone());
        r.step_by(64)
            .map(|x: usize| x ^ (x.wrapping_sub(1)))
            .fold(0usize, |acc, i| acc.wrapping_add(i))
    })
}

#[bench]
fn bench_range_step_by_fold_u16(b: &mut Bencher) {
    let r: Range<u16> = 0..u16::MAX;
    b.iter(|| {
        let r = black_box(r.clone());
        r.step_by(64).map(|x: u16| x ^ (x.wrapping_sub(1))).fold(0u16, |acc, i| acc.wrapping_add(i))
    })
}

pub fn copy_zip(xs: &[u8], ys: &mut [u8]) {
    for (a, b) in ys.iter_mut().zip(xs) {
        *a = *b;
    }
}

pub fn add_zip(xs: &[f32], ys: &mut [f32]) {
    for (a, b) in ys.iter_mut().zip(xs) {
        *a += *b;
    }
}

#[bench]
fn bench_zip_copy(b: &mut Bencher) {
    let source = vec![0u8; 16 * 1024];
    let mut dst = black_box(vec![0u8; 16 * 1024]);
    b.iter(|| copy_zip(&source, &mut dst))
}

#[bench]
fn bench_zip_add(b: &mut Bencher) {
    let source = vec![1.; 16 * 1024];
    let mut dst = vec![0.; 16 * 1024];
    b.iter(|| add_zip(&source, &mut dst));
}

/// `Iterator::for_each` implemented as a plain loop.
fn for_each_loop<I, F>(iter: I, mut f: F)
where
    I: Iterator,
    F: FnMut(I::Item),
{
    for item in iter {
        f(item);
    }
}

/// `Iterator::for_each` implemented with `fold` for internal iteration.
/// (except when `by_ref()` effectively disables that optimization.)
fn for_each_fold<I, F>(iter: I, mut f: F)
where
    I: Iterator,
    F: FnMut(I::Item),
{
    iter.fold((), move |(), item| f(item));
}

#[bench]
fn bench_for_each_chain_loop(b: &mut Bencher) {
    b.iter(|| {
        let mut acc = 0;
        let iter = (0i64..1000000).chain(0..1000000).map(black_box);
        for_each_loop(iter, |x| acc += x);
        acc
    });
}

#[bench]
fn bench_for_each_chain_fold(b: &mut Bencher) {
    b.iter(|| {
        let mut acc = 0;
        let iter = (0i64..1000000).chain(0..1000000).map(black_box);
        for_each_fold(iter, |x| acc += x);
        acc
    });
}

#[bench]
fn bench_for_each_chain_ref_fold(b: &mut Bencher) {
    b.iter(|| {
        let mut acc = 0;
        let mut iter = (0i64..1000000).chain(0..1000000).map(black_box);
        for_each_fold(iter.by_ref(), |x| acc += x);
        acc
    });
}

/// Helper to benchmark `sum` for iterators taken by value which
/// can optimize `fold`, and by reference which cannot.
macro_rules! bench_sums {
    ($bench_sum:ident, $bench_ref_sum:ident, $iter:expr) => {
        #[bench]
        fn $bench_sum(b: &mut Bencher) {
            b.iter(|| -> i64 { $iter.map(black_box).sum() });
        }

        #[bench]
        fn $bench_ref_sum(b: &mut Bencher) {
            b.iter(|| -> i64 { $iter.map(black_box).by_ref().sum() });
        }
    };
}

bench_sums! {
    bench_flat_map_sum,
    bench_flat_map_ref_sum,
    (0i64..1000).flat_map(|x| x..x+1000)
}

bench_sums! {
    bench_flat_map_chain_sum,
    bench_flat_map_chain_ref_sum,
    (0i64..1000000).flat_map(|x| once(x).chain(once(x)))
}

bench_sums! {
    bench_enumerate_sum,
    bench_enumerate_ref_sum,
    (0i64..1000000).enumerate().map(|(i, x)| x * i as i64)
}

bench_sums! {
    bench_enumerate_chain_sum,
    bench_enumerate_chain_ref_sum,
    (0i64..1000000).chain(0..1000000).enumerate().map(|(i, x)| x * i as i64)
}

bench_sums! {
    bench_filter_sum,
    bench_filter_ref_sum,
    (0i64..1000000).filter(|x| x % 3 == 0)
}

bench_sums! {
    bench_filter_chain_sum,
    bench_filter_chain_ref_sum,
    (0i64..1000000).chain(0..1000000).filter(|x| x % 3 == 0)
}

bench_sums! {
    bench_filter_map_sum,
    bench_filter_map_ref_sum,
    (0i64..1000000).filter_map(|x| x.checked_mul(x))
}

bench_sums! {
    bench_filter_map_chain_sum,
    bench_filter_map_chain_ref_sum,
    (0i64..1000000).chain(0..1000000).filter_map(|x| x.checked_mul(x))
}

bench_sums! {
    bench_fuse_sum,
    bench_fuse_ref_sum,
    (0i64..1000000).fuse()
}

bench_sums! {
    bench_fuse_chain_sum,
    bench_fuse_chain_ref_sum,
    (0i64..1000000).chain(0..1000000).fuse()
}

bench_sums! {
    bench_inspect_sum,
    bench_inspect_ref_sum,
    (0i64..1000000).inspect(|_| {})
}

bench_sums! {
    bench_inspect_chain_sum,
    bench_inspect_chain_ref_sum,
    (0i64..1000000).chain(0..1000000).inspect(|_| {})
}

bench_sums! {
    bench_peekable_sum,
    bench_peekable_ref_sum,
    (0i64..1000000).peekable()
}

bench_sums! {
    bench_peekable_chain_sum,
    bench_peekable_chain_ref_sum,
    (0i64..1000000).chain(0..1000000).peekable()
}

bench_sums! {
    bench_skip_sum,
    bench_skip_ref_sum,
    (0i64..1000000).skip(1000)
}

bench_sums! {
    bench_skip_chain_sum,
    bench_skip_chain_ref_sum,
    (0i64..1000000).chain(0..1000000).skip(1000)
}

bench_sums! {
    bench_skip_while_sum,
    bench_skip_while_ref_sum,
    (0i64..1000000).skip_while(|&x| x < 1000)
}

bench_sums! {
    bench_skip_while_chain_sum,
    bench_skip_while_chain_ref_sum,
    (0i64..1000000).chain(0..1000000).skip_while(|&x| x < 1000)
}

bench_sums! {
    bench_take_while_chain_sum,
    bench_take_while_chain_ref_sum,
    (0i64..1000000).chain(1000000..).take_while(|&x| x < 1111111)
}

bench_sums! {
    bench_cycle_take_sum,
    bench_cycle_take_ref_sum,
    (0..10000).cycle().take(1000000)
}

bench_sums! {
    bench_cycle_skip_take_sum,
    bench_cycle_skip_take_ref_sum,
    (0..100000).cycle().skip(1000000).take(1000000)
}

bench_sums! {
    bench_cycle_take_skip_sum,
    bench_cycle_take_skip_ref_sum,
    (0..100000).cycle().take(1000000).skip(100000)
}

bench_sums! {
    bench_skip_cycle_skip_zip_add_sum,
    bench_skip_cycle_skip_zip_add_ref_sum,
    (0..100000).skip(100).cycle().skip(100)
      .zip((0..100000).cycle().skip(10))
      .map(|(a,b)| a+b)
      .skip(100000)
      .take(1000000)
}

// Checks whether Skip<Zip<A,B>> is as fast as Zip<Skip<A>, Skip<B>>, from
// https://users.rust-lang.org/t/performance-difference-between-iterator-zip-and-skip-order/15743
#[bench]
fn bench_zip_then_skip(b: &mut Bencher) {
    let v: Vec<_> = (0..100_000).collect();
    let t: Vec<_> = (0..100_000).collect();

    b.iter(|| {
        let s = v
            .iter()
            .zip(t.iter())
            .skip(10000)
            .take_while(|t| *t.0 < 10100)
            .map(|(a, b)| *a + *b)
            .sum::<u64>();
        assert_eq!(s, 2009900);
    });
}
#[bench]
fn bench_skip_then_zip(b: &mut Bencher) {
    let v: Vec<_> = (0..100_000).collect();
    let t: Vec<_> = (0..100_000).collect();

    b.iter(|| {
        let s = v
            .iter()
            .skip(10000)
            .zip(t.iter().skip(10000))
            .take_while(|t| *t.0 < 10100)
            .map(|(a, b)| *a + *b)
            .sum::<u64>();
        assert_eq!(s, 2009900);
    });
}

#[bench]
fn bench_filter_count(b: &mut Bencher) {
    b.iter(|| (0i64..1000000).map(black_box).filter(|x| x % 3 == 0).count())
}

#[bench]
fn bench_filter_ref_count(b: &mut Bencher) {
    b.iter(|| (0i64..1000000).map(black_box).by_ref().filter(|x| x % 3 == 0).count())
}

#[bench]
fn bench_filter_chain_count(b: &mut Bencher) {
    b.iter(|| (0i64..1000000).chain(0..1000000).map(black_box).filter(|x| x % 3 == 0).count())
}

#[bench]
fn bench_filter_chain_ref_count(b: &mut Bencher) {
    b.iter(|| {
        (0i64..1000000).chain(0..1000000).map(black_box).by_ref().filter(|x| x % 3 == 0).count()
    })
}

#[bench]
fn bench_partial_cmp(b: &mut Bencher) {
    b.iter(|| (0..100000).map(black_box).partial_cmp((0..100000).map(black_box)))
}

#[bench]
fn bench_chain_partial_cmp(b: &mut Bencher) {
    b.iter(|| {
        (0..50000).chain(50000..100000).map(black_box).partial_cmp((0..100000).map(black_box))
    })
}

#[bench]
fn bench_lt(b: &mut Bencher) {
    b.iter(|| (0..100000).map(black_box).lt((0..100000).map(black_box)))
}

#[bench]
fn bench_trusted_random_access_adapters(b: &mut Bencher) {
    let vec1: Vec<_> = (0usize..100000).collect();
    let vec2 = black_box(vec1.clone());
    b.iter(|| {
        let mut iter = vec1
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, e)| idx.wrapping_add(e))
            .zip(vec2.iter().copied())
            .map(|(a, b)| a.wrapping_add(b))
            .fuse();
        let mut acc: usize = 0;
        let size = iter.size();
        for i in 0..size {
            // SAFETY: TRA requirements are satisfied by 0..size iteration and then dropping the
            // iterator.
            acc = acc.wrapping_add(unsafe { iter.__iterator_get_unchecked(i) });
        }
        acc
    })
}

/// Exercises the iter::Copied specialization for slice::Iter
#[bench]
fn bench_next_chunk_copied(b: &mut Bencher) {
    let v = vec![1u8; 1024];

    b.iter(|| {
        let mut iter = black_box(&v).iter().copied();
        let mut acc = Wrapping(0);
        // This uses a while-let loop to side-step the TRA specialization in ArrayChunks
        while let Ok(chunk) = iter.next_chunk::<{ mem::size_of::<u64>() }>() {
            let d = u64::from_ne_bytes(chunk);
            acc += Wrapping(d.rotate_left(7).wrapping_add(1));
        }
        acc
    })
}

/// Exercises the TrustedRandomAccess specialization in ArrayChunks
#[bench]
fn bench_next_chunk_trusted_random_access(b: &mut Bencher) {
    let v = vec![1u8; 1024];

    b.iter(|| {
        black_box(&v)
            .iter()
            // this shows that we're not relying on the slice::Iter specialization in Copied
            .map(|b| *b.borrow())
            .array_chunks::<{ mem::size_of::<u64>() }>()
            .map(|ary| {
                let d = u64::from_ne_bytes(ary);
                Wrapping(d.rotate_left(7).wrapping_add(1))
            })
            .sum::<Wrapping<u64>>()
    })
}

#[bench]
fn bench_next_chunk_filter_even(b: &mut Bencher) {
    let a = (0..1024).next_chunk::<1024>().unwrap();

    b.iter(|| black_box(&a).iter().filter(|&&i| i % 2 == 0).next_chunk::<32>())
}

#[bench]
fn bench_next_chunk_filter_predictably_true(b: &mut Bencher) {
    let a = (0..1024).next_chunk::<1024>().unwrap();

    b.iter(|| black_box(&a).iter().filter(|&&i| i < 100).next_chunk::<32>())
}

#[bench]
fn bench_next_chunk_filter_mostly_false(b: &mut Bencher) {
    let a = (0..1024).next_chunk::<1024>().unwrap();

    b.iter(|| black_box(&a).iter().filter(|&&i| i > 900).next_chunk::<32>())
}

#[bench]
fn bench_next_chunk_filter_map_even(b: &mut Bencher) {
    let a = (0..1024).next_chunk::<1024>().unwrap();

    b.iter(|| black_box(&a).iter().filter_map(|&i| (i % 2 == 0).then(|| i)).next_chunk::<32>())
}

#[bench]
fn bench_next_chunk_filter_map_predictably_true(b: &mut Bencher) {
    let a = (0..1024).next_chunk::<1024>().unwrap();

    b.iter(|| black_box(&a).iter().filter_map(|&i| (i < 100).then(|| i)).next_chunk::<32>())
}

#[bench]
fn bench_next_chunk_filter_map_mostly_false(b: &mut Bencher) {
    let a = (0..1024).next_chunk::<1024>().unwrap();

    b.iter(|| black_box(&a).iter().filter_map(|&i| (i > 900).then(|| i)).next_chunk::<32>())
}
