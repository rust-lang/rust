// This test comprehensively checks the passing static and dynamic semantics
// of subslice patterns `..`, `x @ ..`, `ref x @ ..`, and `ref mut @ ..`
// in slice patterns `[$($pat), $(,)?]` .

// run-pass

#![allow(unreachable_patterns)]

use std::convert::identity;

#[derive(PartialEq, Debug, Clone)]
struct N(u8);

macro_rules! n {
    ($($e:expr),* $(,)?) => {
        [$(N($e)),*]
    }
}

macro_rules! c {
    ($inp:expr, $typ:ty, $out:expr $(,)?) => {
        assert_eq!($out, identity::<$typ>($inp));
    }
}

macro_rules! m {
    ($e:expr, $p:pat => $b:expr) => {
        match $e {
            $p => $b,
            _ => panic!(),
        }
    }
}

fn main() {
    slices();
    arrays();
}

fn slices() {
    // Matching slices using `ref` patterns:
    let mut v = vec![N(0), N(1), N(2), N(3), N(4)];
    let mut vc = (0..=4).collect::<Vec<u8>>();

    let [..] = v[..]; // Always matches.
    m!(v[..], [N(0), ref sub @ .., N(4)] => c!(sub, &[N], n![1, 2, 3]));
    m!(v[..], [N(0), ref sub @ ..] => c!(sub, &[N], n![1, 2, 3, 4]));
    m!(v[..], [ref sub @ .., N(4)] => c!(sub, &[N], n![0, 1, 2, 3]));
    m!(v[..], [ref sub @ .., _, _, _, _, _] => c!(sub, &[N], &n![] as &[N]));
    m!(v[..], [_, _, _, _, _, ref sub @ ..] => c!(sub, &[N], &n![] as &[N]));
    m!(vc[..], [x, .., y] => c!((x, y), (u8, u8), (0, 4)));

    // Matching slices using `ref mut` patterns:
    let [..] = v[..]; // Always matches.
    m!(v[..], [N(0), ref mut sub @ .., N(4)] => c!(sub, &mut [N], n![1, 2, 3]));
    m!(v[..], [N(0), ref mut sub @ ..] => c!(sub, &mut [N], n![1, 2, 3, 4]));
    m!(v[..], [ref mut sub @ .., N(4)] => c!(sub, &mut [N], n![0, 1, 2, 3]));
    m!(v[..], [ref mut sub @ .., _, _, _, _, _] => c!(sub, &mut [N], &mut n![] as &mut [N]));
    m!(v[..], [_, _, _, _, _, ref mut sub @ ..] => c!(sub, &mut [N], &mut n![] as &mut [N]));
    m!(vc[..], [x, .., y] => c!((x, y), (u8, u8), (0, 4)));

    // Matching slices using default binding modes (&):
    let [..] = &v[..]; // Always matches.
    m!(&v[..], [N(0), sub @ .., N(4)] => c!(sub, &[N], n![1, 2, 3]));
    m!(&v[..], [N(0), sub @ ..] => c!(sub, &[N], n![1, 2, 3, 4]));
    m!(&v[..], [sub @ .., N(4)] => c!(sub, &[N], n![0, 1, 2, 3]));
    m!(&v[..], [sub @ .., _, _, _, _, _] => c!(sub, &[N], &n![] as &[N]));
    m!(&v[..], [_, _, _, _, _, sub @ ..] => c!(sub, &[N], &n![] as &[N]));
    m!(&vc[..], [x, .., y] => c!((x, y), (&u8, &u8), (&0, &4)));

    // Matching slices using default binding modes (&mut):
    let [..] = &mut v[..]; // Always matches.
    m!(&mut v[..], [N(0), sub @ .., N(4)] => c!(sub, &mut [N], n![1, 2, 3]));
    m!(&mut v[..], [N(0), sub @ ..] => c!(sub, &mut [N], n![1, 2, 3, 4]));
    m!(&mut v[..], [sub @ .., N(4)] => c!(sub, &mut [N], n![0, 1, 2, 3]));
    m!(&mut v[..], [sub @ .., _, _, _, _, _] => c!(sub, &mut [N], &mut n![] as &mut [N]));
    m!(&mut v[..], [_, _, _, _, _, sub @ ..] => c!(sub, &mut [N], &mut n![] as &mut [N]));
    m!(&mut vc[..], [x, .., y] => c!((x, y), (&mut u8, &mut u8), (&mut 0, &mut 4)));
}

fn arrays() {
    let mut v = n![0, 1, 2, 3, 4];
    let vc = [0, 1, 2, 3, 4];

    // Matching arrays by value:
    m!(v.clone(), [N(0), sub @ .., N(4)] => c!(sub, [N; 3], n![1, 2, 3]));
    m!(v.clone(), [N(0), sub @ ..] => c!(sub, [N; 4], n![1, 2, 3, 4]));
    m!(v.clone(), [sub @ .., N(4)] => c!(sub, [N; 4], n![0, 1, 2, 3]));
    m!(v.clone(), [sub @ .., _, _, _, _, _] => c!(sub, [N; 0], n![] as [N; 0]));
    m!(v.clone(), [_, _, _, _, _, sub @ ..] => c!(sub, [N; 0], n![] as [N; 0]));
    m!(v.clone(), [x, .., y] => c!((x, y), (N, N), (N(0), N(4))));
    m!(v.clone(), [..] => ());

    // Matching arrays by ref patterns:
    m!(v, [N(0), ref sub @ .., N(4)] => c!(sub, &[N; 3], &n![1, 2, 3]));
    m!(v, [N(0), ref sub @ ..] => c!(sub, &[N; 4], &n![1, 2, 3, 4]));
    m!(v, [ref sub @ .., N(4)] => c!(sub, &[N; 4], &n![0, 1, 2, 3]));
    m!(v, [ref sub @ .., _, _, _, _, _] => c!(sub, &[N; 0], &n![] as &[N; 0]));
    m!(v, [_, _, _, _, _, ref sub @ ..] => c!(sub, &[N; 0], &n![] as &[N; 0]));
    m!(vc, [x, .., y] => c!((x, y), (u8, u8), (0, 4)));

    // Matching arrays by ref mut patterns:
    m!(v, [N(0), ref mut sub @ .., N(4)] => c!(sub, &mut [N; 3], &mut n![1, 2, 3]));
    m!(v, [N(0), ref mut sub @ ..] => c!(sub, &mut [N; 4], &mut n![1, 2, 3, 4]));
    m!(v, [ref mut sub @ .., N(4)] => c!(sub, &mut [N; 4], &mut n![0, 1, 2, 3]));
    m!(v, [ref mut sub @ .., _, _, _, _, _] => c!(sub, &mut [N; 0], &mut n![] as &mut [N; 0]));
    m!(v, [_, _, _, _, _, ref mut sub @ ..] => c!(sub, &mut [N; 0], &mut n![] as &mut [N; 0]));

    // Matching arrays by default binding modes (&):
    m!(&v, [N(0), sub @ .., N(4)] => c!(sub, &[N; 3], &n![1, 2, 3]));
    m!(&v, [N(0), sub @ ..] => c!(sub, &[N; 4], &n![1, 2, 3, 4]));
    m!(&v, [sub @ .., N(4)] => c!(sub, &[N; 4], &n![0, 1, 2, 3]));
    m!(&v, [sub @ .., _, _, _, _, _] => c!(sub, &[N; 0], &n![] as &[N; 0]));
    m!(&v, [_, _, _, _, _, sub @ ..] => c!(sub, &[N; 0], &n![] as &[N; 0]));
    m!(&v, [..] => ());
    m!(&v, [x, .., y] => c!((x, y), (&N, &N), (&N(0), &N(4))));

    // Matching arrays by default binding modes (&mut):
    m!(&mut v, [N(0), sub @ .., N(4)] => c!(sub, &mut [N; 3], &mut n![1, 2, 3]));
    m!(&mut v, [N(0), sub @ ..] => c!(sub, &mut [N; 4], &mut n![1, 2, 3, 4]));
    m!(&mut v, [sub @ .., N(4)] => c!(sub, &mut [N; 4], &mut n![0, 1, 2, 3]));
    m!(&mut v, [sub @ .., _, _, _, _, _] => c!(sub, &mut [N; 0], &mut n![] as &[N; 0]));
    m!(&mut v, [_, _, _, _, _, sub @ ..] => c!(sub, &mut [N; 0], &mut n![] as &[N; 0]));
    m!(&mut v, [..] => ());
    m!(&mut v, [x, .., y] => c!((x, y), (&mut N, &mut N), (&mut N(0), &mut N(4))));
}
