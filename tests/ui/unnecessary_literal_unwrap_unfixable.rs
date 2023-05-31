#![warn(clippy::unnecessary_literal_unwrap)]
#![allow(unreachable_code)]
#![allow(clippy::unnecessary_lazy_evaluations, clippy::let_unit_value)]

fn unwrap_option_some() {
    let val = Some(1);
    let _val2 = val.unwrap();
    let _val2 = val.expect("this never happens");
}

fn unwrap_option_some_context() {
    let _val = Some::<usize>([1, 2, 3].iter().sum()).unwrap();
    let _val = Some::<usize>([1, 2, 3].iter().sum()).expect("this never happens");

    let val = Some::<usize>([1, 2, 3].iter().sum());
    let _val2 = val.unwrap();
    let _val2 = val.expect("this never happens");
}

fn unwrap_option_none() {
    let val = None::<()>;
    let _val2 = val.unwrap();
    let _val2 = val.expect("this always happens");
}

fn unwrap_result_ok() {
    let val = Ok::<_, ()>(1);
    let _val2 = val.unwrap();
    let _val2 = val.expect("this never happens");
    let _val2 = val.unwrap_err();
    let _val2 = val.expect_err("this always happens");
}

fn unwrap_result_ok_context() {
    let _val = Ok::<usize, ()>([1, 2, 3].iter().sum()).unwrap();
    let _val = Ok::<usize, ()>([1, 2, 3].iter().sum()).expect("this never happens");
    let _val = Ok::<usize, ()>([1, 2, 3].iter().sum()).unwrap_err();
    let _val = Ok::<usize, ()>([1, 2, 3].iter().sum()).expect_err("this always happens");

    let val = Ok::<usize, ()>([1, 2, 3].iter().sum());
    let _val2 = val.unwrap();
    let _val2 = val.expect("this never happens");
    let _val2 = val.unwrap_err();
    let _val2 = val.expect_err("this always happens");
}

fn unwrap_result_err() {
    let val = Err::<(), _>(1);
    let _val2 = val.unwrap_err();
    let _val2 = val.expect_err("this never happens");
    let _val2 = val.unwrap();
    let _val2 = val.expect("this always happens");
}

fn unwrap_result_err_context() {
    let _val = Err::<(), usize>([1, 2, 3].iter().sum()).unwrap_err();
    let _val = Err::<(), usize>([1, 2, 3].iter().sum()).expect_err("this never happens");
    let _val = Err::<(), usize>([1, 2, 3].iter().sum()).unwrap();
    let _val = Err::<(), usize>([1, 2, 3].iter().sum()).expect("this always happens");

    let val = Err::<(), usize>([1, 2, 3].iter().sum());
    let _val2 = val.unwrap_err();
    let _val2 = val.expect_err("this never happens");
    let _val2 = val.unwrap();
    let _val2 = val.expect("this always happens");
}

fn unwrap_methods_option() {
    let val = Some(1);
    let _val2 = val.unwrap_or(2);
    let _val2 = val.unwrap_or_default();
    let _val2 = val.unwrap_or_else(|| 2);
}

fn unwrap_methods_option_context() {
    let _val = Some::<usize>([1, 2, 3].iter().sum()).unwrap_or(2);
    let _val = Some::<usize>([1, 2, 3].iter().sum()).unwrap_or_default();
    let _val = Some::<usize>([1, 2, 3].iter().sum()).unwrap_or_else(|| 2);

    let val = Some::<usize>([1, 2, 3].iter().sum());
    let _val2 = val.unwrap_or(2);
    let _val2 = val.unwrap_or_default();
    let _val2 = val.unwrap_or_else(|| 2);
}

fn unwrap_methods_result() {
    let val = Ok::<_, ()>(1);
    let _val2 = val.unwrap_or(2);
    let _val2 = val.unwrap_or_default();
    let _val2 = val.unwrap_or_else(|_| 2);
}

fn unwrap_methods_result_context() {
    let _val = Ok::<usize, ()>([1, 2, 3].iter().sum()).unwrap_or(2);
    let _val = Ok::<usize, ()>([1, 2, 3].iter().sum()).unwrap_or_default();
    let _val = Ok::<usize, ()>([1, 2, 3].iter().sum()).unwrap_or_else(|_| 2);

    let val = Ok::<usize, ()>([1, 2, 3].iter().sum());
    let _val2 = val.unwrap_or(2);
    let _val2 = val.unwrap_or_default();
    let _val2 = val.unwrap_or_else(|_| 2);
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
