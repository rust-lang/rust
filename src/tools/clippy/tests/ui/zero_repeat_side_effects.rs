#![warn(clippy::zero_repeat_side_effects)]
#![allow(clippy::unnecessary_operation)]
#![allow(clippy::useless_vec)]
#![allow(clippy::needless_late_init)]

fn f() -> i32 {
    println!("side effect");
    10
}

fn main() {
    const N: usize = 0;
    const M: usize = 1;

    // should trigger

    // on arrays
    let a = [f(); 0];
    let a = [f(); N];
    let mut b;
    b = [f(); 0];
    b = [f(); N];

    // on vecs
    // vecs dont support infering value of consts
    let c = vec![f(); 0];
    let d;
    d = vec![f(); 0];

    // for macros
    let e = [println!("side effect"); 0];

    // for nested calls
    let g = [{ f() }; 0];

    // as function param
    drop(vec![f(); 0]);

    // when singled out/not part of assignment/local
    vec![f(); 0];
    [f(); 0];
    [f(); N];

    // should not trigger

    // on arrays with > 0 repeat
    let a = [f(); 1];
    let a = [f(); M];
    let mut b;
    b = [f(); 1];
    b = [f(); M];

    // on vecs with > 0 repeat
    let c = vec![f(); 1];
    let d;
    d = vec![f(); 1];

    // as function param
    drop(vec![f(); 1]);
}
