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
    //~^ zero_repeat_side_effects
    let mut b;
    b = [f(); 0];
    //~^ zero_repeat_side_effects

    // on vecs
    // vecs dont support inferring value of consts
    let c = vec![f(); 0];
    //~^ zero_repeat_side_effects
    let d;
    d = vec![f(); 0];
    //~^ zero_repeat_side_effects

    // for macros
    let e = [println!("side effect"); 0];
    //~^ zero_repeat_side_effects

    // for nested calls
    let g = [{ f() }; 0];
    //~^ zero_repeat_side_effects

    // as function param
    drop(vec![f(); 0]);
    //~^ zero_repeat_side_effects

    // when singled out/not part of assignment/local
    vec![f(); 0];
    //~^ zero_repeat_side_effects
    [f(); 0];
    //~^ zero_repeat_side_effects

    // should not trigger
    let a = [f(); N];
    b = [f(); N];
    [f(); N];

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

macro_rules! LEN {
    () => {
        0
    };
}

fn issue_13110() {
    let _data = [f(); LEN!()];
    const LENGTH: usize = LEN!();
    let _data = [f(); LENGTH];
}
