use core::iter::*;
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
fn bench_lt(b: &mut Bencher) {
    b.iter(|| (0..100000).map(black_box).lt((0..100000).map(black_box)))
}
