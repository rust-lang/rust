#![cfg(not(target_arch = "wasm32"))]

use std::i16;
use std::str;

use core::num::flt2dec::MAX_SIG_DIGITS;
use core::num::flt2dec::strategy::grisu::format_exact_opt;
use core::num::flt2dec::strategy::grisu::format_shortest_opt;
use core::num::flt2dec::{decode, DecodableFloat, FullDecoded, Decoded};

use rand::FromEntropy;
use rand::rngs::SmallRng;
use rand::distributions::{Distribution, Uniform};

pub fn decode_finite<T: DecodableFloat>(v: T) -> Decoded {
    match decode(v).1 {
        FullDecoded::Finite(decoded) => decoded,
        full_decoded => panic!("expected finite, got {:?} instead", full_decoded)
    }
}


fn iterate<F, G, V>(func: &str, k: usize, n: usize, mut f: F, mut g: G, mut v: V) -> (usize, usize)
        where F: FnMut(&Decoded, &mut [u8]) -> Option<(usize, i16)>,
              G: FnMut(&Decoded, &mut [u8]) -> (usize, i16),
              V: FnMut(usize) -> Decoded {
    assert!(k <= 1024);

    let mut npassed = 0; // f(x) = Some(g(x))
    let mut nignored = 0; // f(x) = None

    for i in 0..n {
        if (i & 0xfffff) == 0 {
            println!("in progress, {:x}/{:x} (ignored={} passed={} failed={})",
                     i, n, nignored, npassed, i - nignored - npassed);
        }

        let decoded = v(i);
        let mut buf1 = [0; 1024];
        if let Some((len1, e1)) = f(&decoded, &mut buf1[..k]) {
            let mut buf2 = [0; 1024];
            let (len2, e2) = g(&decoded, &mut buf2[..k]);
            if e1 == e2 && &buf1[..len1] == &buf2[..len2] {
                npassed += 1;
            } else {
                println!("equivalence test failed, {:x}/{:x}: {:?} f(i)={}e{} g(i)={}e{}",
                         i, n, decoded, str::from_utf8(&buf1[..len1]).unwrap(), e1,
                                        str::from_utf8(&buf2[..len2]).unwrap(), e2);
            }
        } else {
            nignored += 1;
        }
    }
    println!("{}({}): done, ignored={} passed={} failed={}",
             func, k, nignored, npassed, n - nignored - npassed);
    assert!(nignored + npassed == n,
            "{}({}): {} out of {} values returns an incorrect value!",
            func, k, n - nignored - npassed, n);
    (npassed, nignored)
}

pub fn f32_random_equivalence_test<F, G>(f: F, g: G, k: usize, n: usize)
        where F: FnMut(&Decoded, &mut [u8]) -> Option<(usize, i16)>,
              G: FnMut(&Decoded, &mut [u8]) -> (usize, i16) {
    if cfg!(target_os = "emscripten") {
        return // using rng pulls in i128 support, which doesn't work
    }
    let mut rng = SmallRng::from_entropy();
    let f32_range = Uniform::new(0x0000_0001u32, 0x7f80_0000);
    iterate("f32_random_equivalence_test", k, n, f, g, |_| {
        let x = f32::from_bits(f32_range.sample(&mut rng));
        decode_finite(x)
    });
}

pub fn f64_random_equivalence_test<F, G>(f: F, g: G, k: usize, n: usize)
        where F: FnMut(&Decoded, &mut [u8]) -> Option<(usize, i16)>,
              G: FnMut(&Decoded, &mut [u8]) -> (usize, i16) {
    if cfg!(target_os = "emscripten") {
        return // using rng pulls in i128 support, which doesn't work
    }
    let mut rng = SmallRng::from_entropy();
    let f64_range = Uniform::new(0x0000_0000_0000_0001u64, 0x7ff0_0000_0000_0000);
    iterate("f64_random_equivalence_test", k, n, f, g, |_| {
        let x = f64::from_bits(f64_range.sample(&mut rng));
        decode_finite(x)
    });
}

pub fn f32_exhaustive_equivalence_test<F, G>(f: F, g: G, k: usize)
        where F: FnMut(&Decoded, &mut [u8]) -> Option<(usize, i16)>,
              G: FnMut(&Decoded, &mut [u8]) -> (usize, i16) {
    // we have only 2^23 * (2^8 - 1) - 1 = 2,139,095,039 positive finite f32 values,
    // so why not simply testing all of them?
    //
    // this is of course very stressful (and thus should be behind an `#[ignore]` attribute),
    // but with `-C opt-level=3 -C lto` this only takes about an hour or so.

    // iterate from 0x0000_0001 to 0x7f7f_ffff, i.e., all finite ranges
    let (npassed, nignored) = iterate("f32_exhaustive_equivalence_test",
                                      k, 0x7f7f_ffff, f, g, |i: usize| {

        let x = f32::from_bits(i as u32 + 1);
        decode_finite(x)
    });
    assert_eq!((npassed, nignored), (2121451881, 17643158));
}

#[test]
fn shortest_random_equivalence_test() {
    use core::num::flt2dec::strategy::dragon::format_shortest as fallback;
    f64_random_equivalence_test(format_shortest_opt, fallback, MAX_SIG_DIGITS, 10_000);
    f32_random_equivalence_test(format_shortest_opt, fallback, MAX_SIG_DIGITS, 10_000);
}

#[test] #[ignore] // it is too expensive
fn shortest_f32_exhaustive_equivalence_test() {
    // it is hard to directly test the optimality of the output, but we can at least test if
    // two different algorithms agree to each other.
    //
    // this reports the progress and the number of f32 values returned `None`.
    // with `--nocapture` (and plenty of time and appropriate rustc flags), this should print:
    // `done, ignored=17643158 passed=2121451881 failed=0`.

    use core::num::flt2dec::strategy::dragon::format_shortest as fallback;
    f32_exhaustive_equivalence_test(format_shortest_opt, fallback, MAX_SIG_DIGITS);
}

#[test] #[ignore] // it is too expensive
fn shortest_f64_hard_random_equivalence_test() {
    // this again probably has to use appropriate rustc flags.

    use core::num::flt2dec::strategy::dragon::format_shortest as fallback;
    f64_random_equivalence_test(format_shortest_opt, fallback,
                                         MAX_SIG_DIGITS, 100_000_000);
}

#[test]
fn exact_f32_random_equivalence_test() {
    use core::num::flt2dec::strategy::dragon::format_exact as fallback;
    for k in 1..21 {
        f32_random_equivalence_test(|d, buf| format_exact_opt(d, buf, i16::MIN),
                                             |d, buf| fallback(d, buf, i16::MIN), k, 1_000);
    }
}

#[test]
fn exact_f64_random_equivalence_test() {
    use core::num::flt2dec::strategy::dragon::format_exact as fallback;
    for k in 1..21 {
        f64_random_equivalence_test(|d, buf| format_exact_opt(d, buf, i16::MIN),
                                             |d, buf| fallback(d, buf, i16::MIN), k, 1_000);
    }
}
