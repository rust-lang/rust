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
    //~^ unnecessary_literal_unwrap
    let _val = Some(1).expect("this never happens");
    //~^ unnecessary_literal_unwrap

    Some(1).unwrap();
    //~^ unnecessary_literal_unwrap
    Some(1).expect("this never happens");
    //~^ unnecessary_literal_unwrap
}

#[rustfmt::skip] // force rustfmt not to remove braces in `|| { 234 }`
fn unwrap_option_none() {
    let _val = None::<()>.unwrap();
    //~^ unnecessary_literal_unwrap
    let _val = None::<()>.expect("this always happens");
    //~^ unnecessary_literal_unwrap
    let _val: String = None.unwrap_or_default();
    //~^ unnecessary_literal_unwrap
    let _val: u16 = None.unwrap_or(234);
    //~^ unnecessary_literal_unwrap
    let _val: u16 = None.unwrap_or_else(|| 234);
    //~^ unnecessary_literal_unwrap
    let _val: u16 = None.unwrap_or_else(|| { 234 });
    //~^ unnecessary_literal_unwrap
    let _val: u16 = None.unwrap_or_else(|| -> u16 { 234 });
    //~^ unnecessary_literal_unwrap

    None::<()>.unwrap();
    //~^ unnecessary_literal_unwrap
    None::<()>.expect("this always happens");
    //~^ unnecessary_literal_unwrap
    None::<String>.unwrap_or_default();
    //~^ unnecessary_literal_unwrap
    None::<u16>.unwrap_or(234);
    //~^ unnecessary_literal_unwrap
    None::<u16>.unwrap_or_else(|| 234);
    //~^ unnecessary_literal_unwrap
    None::<u16>.unwrap_or_else(|| { 234 });
    //~^ unnecessary_literal_unwrap
    None::<u16>.unwrap_or_else(|| -> u16 { 234 });
    //~^ unnecessary_literal_unwrap
}

fn unwrap_result_ok() {
    let _val = Ok::<_, ()>(1).unwrap();
    //~^ unnecessary_literal_unwrap
    let _val = Ok::<_, ()>(1).expect("this never happens");
    //~^ unnecessary_literal_unwrap
    let _val = Ok::<_, ()>(1).unwrap_err();
    //~^ unnecessary_literal_unwrap
    let _val = Ok::<_, ()>(1).expect_err("this always happens");
    //~^ unnecessary_literal_unwrap

    Ok::<_, ()>(1).unwrap();
    //~^ unnecessary_literal_unwrap
    Ok::<_, ()>(1).expect("this never happens");
    //~^ unnecessary_literal_unwrap
    Ok::<_, ()>(1).unwrap_err();
    //~^ unnecessary_literal_unwrap
    Ok::<_, ()>(1).expect_err("this always happens");
    //~^ unnecessary_literal_unwrap
}

fn unwrap_result_err() {
    let _val = Err::<(), _>(1).unwrap_err();
    //~^ unnecessary_literal_unwrap
    let _val = Err::<(), _>(1).expect_err("this never happens");
    //~^ unnecessary_literal_unwrap
    let _val = Err::<(), _>(1).unwrap();
    //~^ unnecessary_literal_unwrap
    let _val = Err::<(), _>(1).expect("this always happens");
    //~^ unnecessary_literal_unwrap

    Err::<(), _>(1).unwrap_err();
    //~^ unnecessary_literal_unwrap
    Err::<(), _>(1).expect_err("this never happens");
    //~^ unnecessary_literal_unwrap
    Err::<(), _>(1).unwrap();
    //~^ unnecessary_literal_unwrap
    Err::<(), _>(1).expect("this always happens");
    //~^ unnecessary_literal_unwrap
}

fn unwrap_methods_option() {
    let _val = Some(1).unwrap_or(2);
    //~^ unnecessary_literal_unwrap
    let _val = Some(1).unwrap_or_default();
    //~^ unnecessary_literal_unwrap
    let _val = Some(1).unwrap_or_else(|| 2);
    //~^ unnecessary_literal_unwrap

    Some(1).unwrap_or(2);
    //~^ unnecessary_literal_unwrap
    Some(1).unwrap_or_default();
    //~^ unnecessary_literal_unwrap
    Some(1).unwrap_or_else(|| 2);
    //~^ unnecessary_literal_unwrap
}

fn unwrap_methods_result() {
    let _val = Ok::<_, ()>(1).unwrap_or(2);
    //~^ unnecessary_literal_unwrap
    let _val = Ok::<_, ()>(1).unwrap_or_default();
    //~^ unnecessary_literal_unwrap
    let _val = Ok::<_, ()>(1).unwrap_or_else(|_| 2);
    //~^ unnecessary_literal_unwrap

    Ok::<_, ()>(1).unwrap_or(2);
    //~^ unnecessary_literal_unwrap
    Ok::<_, ()>(1).unwrap_or_default();
    //~^ unnecessary_literal_unwrap
    Ok::<_, ()>(1).unwrap_or_else(|_| 2);
    //~^ unnecessary_literal_unwrap
}

fn unwrap_from_binding() {
    macro_rules! from_macro {
        () => {
            Some("")
        };
    }
    let val = from_macro!();
    let _ = val.unwrap_or("");
}

fn unwrap_unchecked() {
    let _ = unsafe { Some(1).unwrap_unchecked() };
    //~^ unnecessary_literal_unwrap
    let _ = unsafe { Some(1).unwrap_unchecked() + *(&1 as *const i32) }; // needs to keep the unsafe block
    //
    //~^^ unnecessary_literal_unwrap
    let _ = unsafe { Some(1).unwrap_unchecked() } + 1;
    //~^ unnecessary_literal_unwrap
    let _ = unsafe { Ok::<_, ()>(1).unwrap_unchecked() };
    //~^ unnecessary_literal_unwrap
    let _ = unsafe { Ok::<_, ()>(1).unwrap_unchecked() + *(&1 as *const i32) };
    //~^ unnecessary_literal_unwrap
    let _ = unsafe { Ok::<_, ()>(1).unwrap_unchecked() } + 1;
    //~^ unnecessary_literal_unwrap
    let _ = unsafe { Err::<(), i32>(123).unwrap_err_unchecked() };
    //~^ unnecessary_literal_unwrap
}

fn main() {
    unwrap_option_some();
    unwrap_option_none();
    unwrap_result_ok();
    unwrap_result_err();
    unwrap_methods_option();
    unwrap_methods_result();
    unwrap_unchecked();
}
