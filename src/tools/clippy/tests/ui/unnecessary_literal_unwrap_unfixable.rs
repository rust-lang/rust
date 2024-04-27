#![warn(clippy::unnecessary_literal_unwrap)]
#![allow(unreachable_code)]
#![allow(clippy::unnecessary_lazy_evaluations, clippy::let_unit_value)]
//@no-rustfix
fn unwrap_option_some() {
    let val = Some(1);
    let _val2 = val.unwrap();
    //~^ ERROR: used `unwrap()` on `Some` value
    let _val2 = val.expect("this never happens");
    //~^ ERROR: used `expect()` on `Some` value
}

fn unwrap_option_some_context() {
    let _val = Some::<usize>([1, 2, 3].iter().sum()).unwrap();
    //~^ ERROR: used `unwrap()` on `Some` value
    let _val = Some::<usize>([1, 2, 3].iter().sum()).expect("this never happens");
    //~^ ERROR: used `expect()` on `Some` value

    let val = Some::<usize>([1, 2, 3].iter().sum());
    let _val2 = val.unwrap();
    //~^ ERROR: used `unwrap()` on `Some` value
    let _val2 = val.expect("this never happens");
    //~^ ERROR: used `expect()` on `Some` value
}

fn unwrap_option_none() {
    let val = None::<()>;
    let _val2 = val.unwrap();
    //~^ ERROR: used `unwrap()` on `None` value
    let _val2 = val.expect("this always happens");
    //~^ ERROR: used `expect()` on `None` value
    let _val3: u8 = None.unwrap_or_default();
    //~^ ERROR: used `unwrap_or_default()` on `None` value
    None::<()>.unwrap_or_default();
    //~^ ERROR: used `unwrap_or_default()` on `None` value
}

fn unwrap_result_ok() {
    let val = Ok::<_, ()>(1);
    let _val2 = val.unwrap();
    //~^ ERROR: used `unwrap()` on `Ok` value
    let _val2 = val.expect("this never happens");
    //~^ ERROR: used `expect()` on `Ok` value
    let _val2 = val.unwrap_err();
    //~^ ERROR: used `unwrap_err()` on `Ok` value
    let _val2 = val.expect_err("this always happens");
    //~^ ERROR: used `expect_err()` on `Ok` value
}

fn unwrap_result_ok_context() {
    let _val = Ok::<usize, ()>([1, 2, 3].iter().sum()).unwrap();
    //~^ ERROR: used `unwrap()` on `Ok` value
    let _val = Ok::<usize, ()>([1, 2, 3].iter().sum()).expect("this never happens");
    //~^ ERROR: used `expect()` on `Ok` value
    let _val = Ok::<usize, ()>([1, 2, 3].iter().sum()).unwrap_err();
    //~^ ERROR: used `unwrap_err()` on `Ok` value
    let _val = Ok::<usize, ()>([1, 2, 3].iter().sum()).expect_err("this always happens");
    //~^ ERROR: used `expect_err()` on `Ok` value

    let val = Ok::<usize, ()>([1, 2, 3].iter().sum());
    let _val2 = val.unwrap();
    //~^ ERROR: used `unwrap()` on `Ok` value
    let _val2 = val.expect("this never happens");
    //~^ ERROR: used `expect()` on `Ok` value
    let _val2 = val.unwrap_err();
    //~^ ERROR: used `unwrap_err()` on `Ok` value
    let _val2 = val.expect_err("this always happens");
    //~^ ERROR: used `expect_err()` on `Ok` value
}

fn unwrap_result_err() {
    let val = Err::<(), _>(1);
    let _val2 = val.unwrap_err();
    //~^ ERROR: used `unwrap_err()` on `Err` value
    let _val2 = val.expect_err("this never happens");
    //~^ ERROR: used `expect_err()` on `Err` value
    let _val2 = val.unwrap();
    //~^ ERROR: used `unwrap()` on `Err` value
    let _val2 = val.expect("this always happens");
    //~^ ERROR: used `expect()` on `Err` value
}

fn unwrap_result_err_context() {
    let _val = Err::<(), usize>([1, 2, 3].iter().sum()).unwrap_err();
    //~^ ERROR: used `unwrap_err()` on `Err` value
    let _val = Err::<(), usize>([1, 2, 3].iter().sum()).expect_err("this never happens");
    //~^ ERROR: used `expect_err()` on `Err` value
    let _val = Err::<(), usize>([1, 2, 3].iter().sum()).unwrap();
    //~^ ERROR: used `unwrap()` on `Err` value
    let _val = Err::<(), usize>([1, 2, 3].iter().sum()).expect("this always happens");
    //~^ ERROR: used `expect()` on `Err` value

    let val = Err::<(), usize>([1, 2, 3].iter().sum());
    let _val2 = val.unwrap_err();
    //~^ ERROR: used `unwrap_err()` on `Err` value
    let _val2 = val.expect_err("this never happens");
    //~^ ERROR: used `expect_err()` on `Err` value
    let _val2 = val.unwrap();
    //~^ ERROR: used `unwrap()` on `Err` value
    let _val2 = val.expect("this always happens");
    //~^ ERROR: used `expect()` on `Err` value
}

fn unwrap_methods_option() {
    let val = Some(1);
    let _val2 = val.unwrap_or(2);
    //~^ ERROR: used `unwrap_or()` on `Some` value
    let _val2 = val.unwrap_or_default();
    //~^ ERROR: used `unwrap_or_default()` on `Some` value
    let _val2 = val.unwrap_or_else(|| 2);
    //~^ ERROR: used `unwrap_or_else()` on `Some` value
}

fn unwrap_methods_option_context() {
    let _val = Some::<usize>([1, 2, 3].iter().sum()).unwrap_or(2);
    //~^ ERROR: used `unwrap_or()` on `Some` value
    let _val = Some::<usize>([1, 2, 3].iter().sum()).unwrap_or_default();
    //~^ ERROR: used `unwrap_or_default()` on `Some` value
    let _val = Some::<usize>([1, 2, 3].iter().sum()).unwrap_or_else(|| 2);
    //~^ ERROR: used `unwrap_or_else()` on `Some` value

    let val = Some::<usize>([1, 2, 3].iter().sum());
    let _val2 = val.unwrap_or(2);
    //~^ ERROR: used `unwrap_or()` on `Some` value
    let _val2 = val.unwrap_or_default();
    //~^ ERROR: used `unwrap_or_default()` on `Some` value
    let _val2 = val.unwrap_or_else(|| 2);
    //~^ ERROR: used `unwrap_or_else()` on `Some` value
}

fn unwrap_methods_result() {
    let val = Ok::<_, ()>(1);
    let _val2 = val.unwrap_or(2);
    //~^ ERROR: used `unwrap_or()` on `Ok` value
    let _val2 = val.unwrap_or_default();
    //~^ ERROR: used `unwrap_or_default()` on `Ok` value
    let _val2 = val.unwrap_or_else(|_| 2);
    //~^ ERROR: used `unwrap_or_else()` on `Ok` value
}

fn unwrap_methods_result_context() {
    let _val = Ok::<usize, ()>([1, 2, 3].iter().sum()).unwrap_or(2);
    //~^ ERROR: used `unwrap_or()` on `Ok` value
    let _val = Ok::<usize, ()>([1, 2, 3].iter().sum()).unwrap_or_default();
    //~^ ERROR: used `unwrap_or_default()` on `Ok` value
    let _val = Ok::<usize, ()>([1, 2, 3].iter().sum()).unwrap_or_else(|_| 2);
    //~^ ERROR: used `unwrap_or_else()` on `Ok` value

    let val = Ok::<usize, ()>([1, 2, 3].iter().sum());
    let _val2 = val.unwrap_or(2);
    //~^ ERROR: used `unwrap_or()` on `Ok` value
    let _val2 = val.unwrap_or_default();
    //~^ ERROR: used `unwrap_or_default()` on `Ok` value
    let _val2 = val.unwrap_or_else(|_| 2);
    //~^ ERROR: used `unwrap_or_else()` on `Ok` value
}

fn main() {
    unwrap_option_some();
    unwrap_option_some_context();
    unwrap_option_none();
    unwrap_result_ok();
    unwrap_result_ok_context();
    unwrap_result_err();
    unwrap_result_err_context();
    unwrap_methods_option();
    unwrap_methods_option_context();
    unwrap_methods_result();
    unwrap_methods_result_context();
}
