//! Regression test for <https://github.com/rust-lang/rust/issues/57664>.
//! Checks that compiler doesn't get confused by `?` operator and complex
//! return types when reporting type mismatches.

fn unrelated() -> Result<(), std::string::ParseError> {
    let x = 0;

    match x {
        1 => {
            let property_value_as_string = "a".parse()?;
        }
        2 => {
            let value: &bool = unsafe { &42 };
            //~^ ERROR mismatched types
        }
    };

    Ok(())
}

fn main() {}
