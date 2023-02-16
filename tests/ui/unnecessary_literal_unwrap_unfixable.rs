#![warn(clippy::unnecessary_literal_unwrap)]
#![allow(clippy::unnecessary_lazy_evaluations)]
#![allow(unreachable_code)]

fn unwrap_option_some() {
    let val = Some(1);
    let _val2 = val.unwrap();
    let _val2 = val.expect("this never happens");
}

fn unwrap_option_none() {
    let val = None::<usize>;
    val.unwrap();
    val.expect("this always happens");
}

fn unwrap_result_ok() {
    let val = Ok::<usize, ()>(1);
    let _val2 = val.unwrap();
    let _val2 = val.expect("this never happens");
    val.unwrap_err();
    val.expect_err("this always happens");
}

fn unwrap_result_err() {
    let val = Err::<(), usize>(1);
    let _val2 = val.unwrap_err();
    let _val2 = val.expect_err("this never happens");
    val.unwrap();
    val.expect("this always happens");
}

fn unwrap_methods_option() {
    let val = Some(1);
    let _val2 = val.unwrap_or(2);
    let _val2 = val.unwrap_or_default();
    let _val2 = val.unwrap_or_else(|| 2);
}

fn unwrap_methods_result() {
    let val = Ok::<usize, ()>(1);
    let _val2 = val.unwrap_or(2);
    let _val2 = val.unwrap_or_default();
    let _val2 = val.unwrap_or_else(|_| 2);
}

fn main() {
    unwrap_option_some();
    unwrap_option_none();
    unwrap_result_ok();
    unwrap_result_err();
    unwrap_methods_option();
    unwrap_methods_result();
}
