//@ compile-flags: -Zinline-mir

use std::fmt::Display;

fn main() {
    permutations(&['a', 'b', 'c']);
}

#[inline(always)]
fn permutations<T: Copy + Display>(xs: &[T]) {
    let mut ys = xs.to_owned();
    permutate(&mut ys, 0);
}

fn permutate<T: Copy + Display>(xs: &mut [T], k: usize) {
    let n = length(xs);
    if k == n {
        display(xs);
    } else if k < n {
        for i in k..n {
            swap(xs, i, k);
            permutate(xs, k + 1);
            swap(xs, i, k);
        }
    } else {
        error();
    }
}

fn length<T>(xs: &[T]) -> usize {
    xs.len()
}

#[inline]
fn swap<T: Copy>(xs: &mut [T], i: usize, j: usize) {
    let t = xs[i];
    xs[i] = xs[j];
    xs[j] = t;
}

fn display<T: Display>(xs: &[T]) {
    for x in xs {
        print!("{}", x);
    }
    println!();
}

#[inline(always)]
fn error() {
    panic!("error");
}
