//@ revisions: explicit implicit
//@ run-pass
#![feature(deref_patterns)]
#![allow(incomplete_features)]

#[cfg(explicit)]
fn simple_vec(vec: Vec<u32>) -> u32 {
    match vec {
        deref!([]) => 100,
        deref!([x]) if x == 4 => x + 4,
        deref!([x]) => x,
        deref!([1, x]) => x + 200,
        deref!(ref slice) => slice.iter().sum(),
        _ => 2000,
    }
}

#[cfg(implicit)]
fn simple_vec(vec: Vec<u32>) -> u32 {
    match vec {
        [] => 100,
        [x] if x == 4 => x + 4,
        [x] => x,
        [1, x] => x + 200,
        deref!(ref slice) => slice.iter().sum(),
        _ => 2000,
    }
}

#[cfg(explicit)]
fn nested_vec(vecvec: Vec<Vec<u32>>) -> u32 {
    match vecvec {
        deref!([]) => 0,
        deref!([deref!([x])]) => x,
        deref!([deref!([0, x]) | deref!([1, x])]) => x,
        deref!([ref x]) => x.iter().sum(),
        deref!([deref!([]), deref!([1, x, y])]) => y - x,
        _ => 2000,
    }
}

#[cfg(implicit)]
fn nested_vec(vecvec: Vec<Vec<u32>>) -> u32 {
    match vecvec {
        [] => 0,
        [[x]] => x,
        [[0, x] | [1, x]] => x,
        [ref x] => x.iter().sum(),
        [[], [1, x, y]] => y - x,
        _ => 2000,
    }
}

#[cfg(explicit)]
fn ref_mut(val: u32) -> u32 {
    let mut b = Box::new(0u32);
    match &mut b {
        deref!(_x) if false => unreachable!(),
        deref!(x) => {
            *x = val;
        }
        _ => unreachable!(),
    }
    let deref!(x) = &b else { unreachable!() };
    *x
}

#[cfg(implicit)]
fn ref_mut(val: u32) -> u32 {
    let mut b = Box::new((0u32,));
    match &mut b {
        (_x,) if false => unreachable!(),
        (x,) => {
            *x = val;
        }
        _ => unreachable!(),
    }
    let (x,) = &b else { unreachable!() };
    *x
}

#[cfg(explicit)]
#[rustfmt::skip]
fn or_and_guard(tuple: (u32, u32)) -> u32 {
    let mut sum = 0;
    let b = Box::new(tuple);
    match b {
        deref!((x, _) | (_, x)) if { sum += x; false } => {},
        _ => {},
    }
    sum
}

#[cfg(implicit)]
#[rustfmt::skip]
fn or_and_guard(tuple: (u32, u32)) -> u32 {
    let mut sum = 0;
    let b = Box::new(tuple);
    match b {
        (x, _) | (_, x) if { sum += x; false } => {},
        _ => {},
    }
    sum
}

fn main() {
    assert_eq!(simple_vec(vec![1]), 1);
    assert_eq!(simple_vec(vec![1, 2]), 202);
    assert_eq!(simple_vec(vec![1, 2, 3]), 6);
    assert_eq!(simple_vec(vec![4]), 8);

    assert_eq!(nested_vec(vec![vec![0, 42]]), 42);
    assert_eq!(nested_vec(vec![vec![1, 42]]), 42);
    assert_eq!(nested_vec(vec![vec![1, 2, 3]]), 6);
    assert_eq!(nested_vec(vec![vec![], vec![1, 2, 3]]), 1);

    assert_eq!(ref_mut(42), 42);
    assert_eq!(or_and_guard((10, 32)), 42);
}
