//@run-rustfix
#![warn(clippy::unnecessary_literal_unwrap)]
#![allow(unreachable_code)]
#![allow(
    clippy::unnecessary_lazy_evaluations,
    clippy::diverging_sub_expression,
    clippy::let_unit_value,
    clippy::no_effect
)]

fn unwrap_option_some() {
    let _val = Some(1).unwrap();
    let _val = Some(1).expect("this never happens");

    Some(1).unwrap();
    Some(1).expect("this never happens");
}

fn unwrap_option_none() {
    let _val = None::<()>.unwrap();
    let _val = None::<()>.expect("this always happens");

    None::<()>.unwrap();
    None::<()>.expect("this always happens");
}

fn unwrap_result_ok() {
    let _val = Ok::<_, ()>(1).unwrap();
    let _val = Ok::<_, ()>(1).expect("this never happens");
    let _val = Ok::<_, ()>(1).unwrap_err();
    let _val = Ok::<_, ()>(1).expect_err("this always happens");

    Ok::<_, ()>(1).unwrap();
    Ok::<_, ()>(1).expect("this never happens");
    Ok::<_, ()>(1).unwrap_err();
    Ok::<_, ()>(1).expect_err("this always happens");
}

fn unwrap_result_err() {
    let _val = Err::<(), _>(1).unwrap_err();
    let _val = Err::<(), _>(1).expect_err("this never happens");
    let _val = Err::<(), _>(1).unwrap();
    let _val = Err::<(), _>(1).expect("this always happens");

    Err::<(), _>(1).unwrap_err();
    Err::<(), _>(1).expect_err("this never happens");
    Err::<(), _>(1).unwrap();
    Err::<(), _>(1).expect("this always happens");
}

fn unwrap_methods_option() {
    let _val = Some(1).unwrap_or(2);
    let _val = Some(1).unwrap_or_default();
    let _val = Some(1).unwrap_or_else(|| 2);

    Some(1).unwrap_or(2);
    Some(1).unwrap_or_default();
    Some(1).unwrap_or_else(|| 2);
}

fn unwrap_methods_result() {
    let _val = Ok::<_, ()>(1).unwrap_or(2);
    let _val = Ok::<_, ()>(1).unwrap_or_default();
    let _val = Ok::<_, ()>(1).unwrap_or_else(|_| 2);

    Ok::<_, ()>(1).unwrap_or(2);
    Ok::<_, ()>(1).unwrap_or_default();
    Ok::<_, ()>(1).unwrap_or_else(|_| 2);
}

fn main() {
    unwrap_option_some();
    unwrap_option_none();
    unwrap_result_ok();
    unwrap_result_err();
    unwrap_methods_option();
    unwrap_methods_result();
}
