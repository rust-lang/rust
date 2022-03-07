// run-rustfix
#![warn(clippy::nop_match)]
#![allow(clippy::manual_map)]
#![allow(clippy::question_mark)]
#![allow(dead_code)]

fn func_ret_err<T>(err: T) -> Result<(), T> {
    Err(err)
}

enum SampleEnum {
    A,
    B,
    C,
}

fn useless_prim_type_match(x: i32) -> i32 {
    match x {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => x,
    }
}

fn useless_custom_type_match(se: SampleEnum) -> SampleEnum {
    match se {
        SampleEnum::A => SampleEnum::A,
        SampleEnum::B => SampleEnum::B,
        SampleEnum::C => SampleEnum::C,
    }
}

// Don't trigger
fn mingled_custom_type(se: SampleEnum) -> SampleEnum {
    match se {
        SampleEnum::A => SampleEnum::B,
        SampleEnum::B => SampleEnum::C,
        SampleEnum::C => SampleEnum::A,
    }
}

fn option_match() -> Option<i32> {
    match Some(1) {
        Some(a) => Some(a),
        None => None,
    }
}

fn result_match() -> Result<i32, i32> {
    match Ok(1) {
        Ok(a) => Ok(a),
        Err(err) => Err(err),
    }
}

fn result_match_func_call() {
    let _ = match func_ret_err(0_i32) {
        Ok(a) => Ok(a),
        Err(err) => Err(err),
    };
}

fn option_check() -> Option<i32> {
    if let Some(a) = Some(1) { Some(a) } else { None }
}

fn option_check_no_else() -> Option<i32> {
    if let Some(a) = Some(1) {
        return Some(a);
    }
    None
}

fn result_check_no_else() -> Result<(), i32> {
    if let Err(e) = func_ret_err(0_i32) {
        return Err(e);
    }
    Ok(())
}

fn result_check_a() -> Result<(), i32> {
    if let Err(e) = func_ret_err(0_i32) {
        Err(e)
    } else {
        Ok(())
    }
}

// Don't trigger
fn result_check_b() -> Result<(), i32> {
    if let Err(e) = Ok(1) { Err(e) } else { Ok(()) }
}

fn result_check_c() -> Result<(), i32> {
    let example = Ok(());
    if let Err(e) = example { Err(e) } else { example }
}

// Don't trigger
fn result_check_d() -> Result<(), i32> {
    let example = Ok(1);
    if let Err(e) = example { Err(e) } else { Ok(()) }
}

fn main() {}
