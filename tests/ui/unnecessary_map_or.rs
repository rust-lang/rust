//@aux-build:proc_macros.rs
#![warn(clippy::unnecessary_map_or)]
#![allow(clippy::no_effect)]
#![allow(clippy::eq_op)]
#![allow(clippy::unnecessary_lazy_evaluations)]
#[clippy::msrv = "1.70.0"]
#[macro_use]
extern crate proc_macros;

fn main() {
    // should trigger
    let _ = Some(5).map_or(false, |n| n == 5);
    let _ = Some(5).map_or(true, |n| n != 5);
    let _ = Some(5).map_or(false, |n| {
        let _ = 1;
        n == 5
    });
    let _ = Some(5).map_or(false, |n| {
        let _ = n;
        6 >= 5
    });
    let _ = Some(vec![5]).map_or(false, |n| n == [5]);
    let _ = Some(vec![1]).map_or(false, |n| vec![2] == n);
    let _ = Some(5).map_or(false, |n| n == n);
    let _ = Some(5).map_or(false, |n| n == if 2 > 1 { n } else { 0 });
    let _ = Ok::<Vec<i32>, i32>(vec![5]).map_or(false, |n| n == [5]);
    let _ = Ok::<i32, i32>(5).map_or(false, |n| n == 5);
    let _ = Some(5).map_or(false, |n| n == 5).then(|| 1);
    let _ = Some(5).map_or(true, |n| n == 5);
    let _ = Some(5).map_or(true, |n| 5 == n);

    macro_rules! x {
        () => {
            Some(1)
        };
    }
    // methods lints dont fire on macros
    let _ = x!().map_or(false, |n| n == 1);
    let _ = x!().map_or(false, |n| n == vec![1][0]);

    msrv_1_69();

    external! {
        let _ = Some(5).map_or(false, |n| n == 5);
    }

    with_span! {
        let _ = Some(5).map_or(false, |n| n == 5);
    }

    // check for presence of PartialEq, and alter suggestion to use `is_ok_and` if absent
    struct S;
    let r: Result<i32, S> = Ok(3);
    let _ = r.map_or(false, |x| x == 7);

    // lint constructs that are not comparaisons as well
    let func = |_x| true;
    let r: Result<i32, S> = Ok(3);
    let _ = r.map_or(false, func);
    let _ = Some(5).map_or(false, func);
    let _ = Some(5).map_or(true, func);

    #[derive(PartialEq)]
    struct S2;
    let r: Result<i32, S2> = Ok(4);
    let _ = r.map_or(false, |x| x == 8);

    // do not lint `Result::map_or(true, …)`
    let r: Result<i32, S2> = Ok(4);
    let _ = r.map_or(true, |x| x == 8);
}

#[clippy::msrv = "1.69.0"]
fn msrv_1_69() {
    // is_some_and added in 1.70.0
    let _ = Some(5).map_or(false, |n| n == if 2 > 1 { n } else { 0 });
}

#[clippy::msrv = "1.81.0"]
fn msrv_1_81() {
    // is_none_or added in 1.82.0
    let _ = Some(5).map_or(true, |n| n == if 2 > 1 { n } else { 0 });
}
