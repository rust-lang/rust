// run-pass

#![allow(dead_code)]
// Check that functions can modify local state.

// pretty-expanded FIXME #23616

fn sums_to(v: Vec<isize> , sum: isize) -> bool {
    let mut i = 0;
    let mut sum0 = 0;
    while i < v.len() {
        sum0 += v[i];
        i += 1;
    }
    return sum0 == sum;
}

fn sums_to_using_uniq(v: Vec<isize> , sum: isize) -> bool {
    let mut i = 0;
    let mut sum0: Box<_> = 0.into();
    while i < v.len() {
        *sum0 += v[i];
        i += 1;
    }
    return *sum0 == sum;
}

fn sums_to_using_rec(v: Vec<isize> , sum: isize) -> bool {
    let mut i = 0;
    let mut sum0 = F {f: 0};
    while i < v.len() {
        sum0.f += v[i];
        i += 1;
    }
    return sum0.f == sum;
}

struct F<T> { f: T }

fn sums_to_using_uniq_rec(v: Vec<isize> , sum: isize) -> bool {
    let mut i = 0;
    let mut sum0 = F::<Box<_>> {f: 0.into() };
    while i < v.len() {
        *sum0.f += v[i];
        i += 1;
    }
    return *sum0.f == sum;
}

pub fn main() {
}
