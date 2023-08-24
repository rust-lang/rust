#![warn(clippy::from_str_radix_10)]

mod some_mod {
    // fake function that shouldn't trigger the lint
    pub fn from_str_radix(_: &str, _: u32) -> Result<(), std::num::ParseIntError> {
        unimplemented!()
    }
}

// fake function that shouldn't trigger the lint
fn from_str_radix(_: &str, _: u32) -> Result<(), std::num::ParseIntError> {
    unimplemented!()
}

// to test parenthesis addition
struct Test;

impl std::ops::Add<Test> for Test {
    type Output = &'static str;

    fn add(self, _: Self) -> Self::Output {
        "304"
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // all of these should trigger the lint
    u32::from_str_radix("30", 10)?;
    //~^ ERROR: this call to `from_str_radix` can be replaced with a call to `str::parse`
    //~| NOTE: `-D clippy::from-str-radix-10` implied by `-D warnings`
    i64::from_str_radix("24", 10)?;
    //~^ ERROR: this call to `from_str_radix` can be replaced with a call to `str::parse`
    isize::from_str_radix("100", 10)?;
    //~^ ERROR: this call to `from_str_radix` can be replaced with a call to `str::parse`
    u8::from_str_radix("7", 10)?;
    //~^ ERROR: this call to `from_str_radix` can be replaced with a call to `str::parse`
    u16::from_str_radix(&("10".to_owned() + "5"), 10)?;
    //~^ ERROR: this call to `from_str_radix` can be replaced with a call to `str::parse`
    i128::from_str_radix(Test + Test, 10)?;
    //~^ ERROR: this call to `from_str_radix` can be replaced with a call to `str::parse`

    let string = "300";
    i32::from_str_radix(string, 10)?;
    //~^ ERROR: this call to `from_str_radix` can be replaced with a call to `str::parse`

    let stringier = "400".to_string();
    i32::from_str_radix(&stringier, 10)?;
    //~^ ERROR: this call to `from_str_radix` can be replaced with a call to `str::parse`

    // none of these should trigger the lint
    u16::from_str_radix("20", 3)?;
    i32::from_str_radix("45", 12)?;
    usize::from_str_radix("10", 16)?;
    i128::from_str_radix("10", 13)?;
    some_mod::from_str_radix("50", 10)?;
    some_mod::from_str_radix("50", 6)?;
    from_str_radix("50", 10)?;
    from_str_radix("50", 6)?;

    Ok(())
}
