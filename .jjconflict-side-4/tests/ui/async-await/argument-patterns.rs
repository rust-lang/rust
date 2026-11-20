//@ edition:2018
//@ check-pass

#![deny(unused_mut)]

type A = Vec<u32>;

async fn a(n: u32, mut vec: A) {
    vec.push(n);
}

async fn b(n: u32, ref mut vec: A) {
    vec.push(n);
}

async fn c(ref vec: A) {
    vec.contains(&0);
}

async fn d((a, mut b): (A, A)) {
    b.push(1);
}

async fn f((ref mut a, ref b): (A, A)) {}

async fn g(((ref a, ref mut b), (ref mut c, ref d)): ((A, A), (A, A))) {}

fn main() {}
