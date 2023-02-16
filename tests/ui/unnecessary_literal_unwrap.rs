//@run-rustfix
#![warn(clippy::unnecessary_literal_unwrap)]
#![allow(unreachable_code)]
#![allow(clippy::unnecessary_lazy_evaluations)]

fn unwrap_option_some() {
    let _val = Some(1).unwrap();
    let _val = Some(1).expect("this never happens");
}

fn unwrap_option_none() {
    None::<usize>.unwrap();
    None::<usize>.expect("this always happens");
}

fn unwrap_result_ok() {
    let _val = Ok::<usize, ()>(1).unwrap();
    let _val = Ok::<usize, ()>(1).expect("this never happens");
    Ok::<usize, ()>(1).unwrap_err();
    Ok::<usize, ()>(1).expect_err("this always happens");
}

fn unwrap_result_err() {
    let _val = Err::<(), usize>(1).unwrap_err();
    let _val = Err::<(), usize>(1).expect_err("this never happens");
    Err::<(), usize>(1).unwrap();
    Err::<(), usize>(1).expect("this always happens");
}

fn unwrap_methods_option() {
    let _val = Some(1).unwrap_or(2);
    let _val = Some(1).unwrap_or_default();
    let _val = Some(1).unwrap_or_else(|| 2);
}

fn unwrap_methods_result() {
    let _val = Ok::<usize, ()>(1).unwrap_or(2);
    let _val = Ok::<usize, ()>(1).unwrap_or_default();
    let _val = Ok::<usize, ()>(1).unwrap_or_else(|_| 2);
}

fn main() {
    unwrap_option_some();
    unwrap_option_none();
    unwrap_result_ok();
    unwrap_result_err();
    unwrap_methods_option();
    unwrap_methods_result();
}
