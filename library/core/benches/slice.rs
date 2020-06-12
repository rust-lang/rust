use test::black_box;
use test::Bencher;

enum Cache {
    L1,
    L2,
    L3,
}

fn binary_search<F>(b: &mut Bencher, cache: Cache, mapper: F)
where
    F: Fn(usize) -> usize,
{
    let size = match cache {
        Cache::L1 => 1000,      // 8kb
        Cache::L2 => 10_000,    // 80kb
        Cache::L3 => 1_000_000, // 8Mb
    };
    let v = (0..size).map(&mapper).collect::<Vec<_>>();
    let mut r = 0usize;
    b.iter(move || {
        // LCG constants from https://en.wikipedia.org/wiki/Numerical_Recipes.
        r = r.wrapping_mul(1664525).wrapping_add(1013904223);
        // Lookup the whole range to get 50% hits and 50% misses.
        let i = mapper(r % size);
        black_box(v.binary_search(&i).is_ok());
    })
}

#[bench]
fn binary_search_l1(b: &mut Bencher) {
    binary_search(b, Cache::L1, |i| i * 2);
}

#[bench]
fn binary_search_l2(b: &mut Bencher) {
    binary_search(b, Cache::L2, |i| i * 2);
}

#[bench]
fn binary_search_l3(b: &mut Bencher) {
    binary_search(b, Cache::L3, |i| i * 2);
}

#[bench]
fn binary_search_l1_with_dups(b: &mut Bencher) {
    binary_search(b, Cache::L1, |i| i / 16 * 16);
}

#[bench]
fn binary_search_l2_with_dups(b: &mut Bencher) {
    binary_search(b, Cache::L2, |i| i / 16 * 16);
}

#[bench]
fn binary_search_l3_with_dups(b: &mut Bencher) {
    binary_search(b, Cache::L3, |i| i / 16 * 16);
}

macro_rules! rotate {
    ($fn:ident, $n:expr, $mapper:expr) => {
        #[bench]
        fn $fn(b: &mut Bencher) {
            let mut x = (0usize..$n).map(&$mapper).collect::<Vec<_>>();
            b.iter(|| {
                for s in 0..x.len() {
                    x[..].rotate_right(s);
                }
                black_box(x[0].clone())
            })
        }
    };
}

#[derive(Clone)]
struct Rgb(u8, u8, u8);

rotate!(rotate_u8, 32, |i| i as u8);
rotate!(rotate_rgb, 32, |i| Rgb(i as u8, (i as u8).wrapping_add(7), (i as u8).wrapping_add(42)));
rotate!(rotate_usize, 32, |i| i);
rotate!(rotate_16_usize_4, 16, |i| [i; 4]);
rotate!(rotate_16_usize_5, 16, |i| [i; 5]);
rotate!(rotate_64_usize_4, 64, |i| [i; 4]);
rotate!(rotate_64_usize_5, 64, |i| [i; 5]);
