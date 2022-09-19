// will suggest better error when the expression type is the same as `Ok`
fn parse_num(num: String) -> i32 {
    result_i32_error() // specific Result error message
    //~^ ERROR mismatched types [E0308]
}

fn result_i32_error() -> Result<i32, std::num::ParseIntError> {
    Ok(42)
}

// will continue using generic error when the expression type is different than `Ok`
fn foo(num: String) -> String {
    result_i32_error() // generic error message
    //~^ ERROR mismatched types [E0308]
}


fn main(){}
