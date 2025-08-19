//@aux-build: proc_macros.rs
#![warn(clippy::manual_is_multiple_of)]

fn main() {}

#[clippy::msrv = "1.87"]
fn f(a: u64, b: u64) {
    let _ = a % b == 0; //~ manual_is_multiple_of
    let _ = (a + 1) % (b + 1) == 0; //~ manual_is_multiple_of
    let _ = a % b != 0; //~ manual_is_multiple_of
    let _ = (a + 1) % (b + 1) != 0; //~ manual_is_multiple_of

    let _ = a % b > 0; //~ manual_is_multiple_of
    let _ = 0 < a % b; //~ manual_is_multiple_of

    proc_macros::external! {
        let a: u64 = 23424;
        let _ = a % 4096 == 0;
    }
}

#[clippy::msrv = "1.86"]
fn g(a: u64, b: u64) {
    let _ = a % b == 0;
}

fn needs_deref(a: &u64, b: &u64) {
    let _ = a % b == 0; //~ manual_is_multiple_of
}

fn closures(a: u64, b: u64) {
    // Do not lint, types are ambiguous at this point
    let cl = |a, b| a % b == 0;
    let _ = cl(a, b);

    // Do not lint, types are ambiguous at this point
    let cl = |a: _, b: _| a % b == 0;
    let _ = cl(a, b);

    // Type of `a` is enough
    let cl = |a: u64, b| a % b == 0; //~ manual_is_multiple_of
    let _ = cl(a, b);

    // Type of `a` is enough
    let cl = |a: &u64, b| a % b == 0; //~ manual_is_multiple_of
    let _ = cl(&a, b);

    // Type of `b` is not enough
    let cl = |a, b: u64| a % b == 0;
    let _ = cl(&a, b);
}

fn any_rem<T: std::ops::Rem<Output = u32>>(a: T, b: T) {
    // An arbitrary `Rem` implementation should not lint
    let _ = a % b == 0;
}

mod issue15103 {
    fn foo() -> Option<u64> {
        let mut n: u64 = 150_000_000;

        (2..).find(|p| {
            while n % p == 0 {
                //~^ manual_is_multiple_of
                n /= p;
            }
            n <= 1
        })
    }

    const fn generate_primes<const N: usize>() -> [u64; N] {
        let mut result = [0; N];
        if N == 0 {
            return result;
        }
        result[0] = 2;
        if N == 1 {
            return result;
        }
        let mut idx = 1;
        let mut p = 3;
        while idx < N {
            let mut j = 0;
            while j < idx && p % result[j] != 0 {
                j += 1;
            }
            if j == idx {
                result[idx] = p;
                idx += 1;
            }
            p += 1;
        }
        result
    }

    fn bar() -> u32 {
        let d = |n: u32| -> u32 { (1..=n / 2).filter(|i| n % i == 0).sum() };
        //~^ manual_is_multiple_of

        let d = |n| (1..=n / 2).filter(|i| n % i == 0).sum();
        (1..1_000).filter(|&i| i == d(d(i)) && i != d(i)).sum()
    }
}
