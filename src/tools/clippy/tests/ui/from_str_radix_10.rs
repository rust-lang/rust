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
    //~^ from_str_radix_10

    i64::from_str_radix("24", 10)?;
    //~^ from_str_radix_10

    isize::from_str_radix("100", 10)?;
    //~^ from_str_radix_10

    u8::from_str_radix("7", 10)?;
    //~^ from_str_radix_10

    u16::from_str_radix(&("10".to_owned() + "5"), 10)?;
    //~^ from_str_radix_10

    i128::from_str_radix(Test + Test, 10)?;
    //~^ from_str_radix_10

    let string = "300";
    i32::from_str_radix(string, 10)?;
    //~^ from_str_radix_10

    let stringier = "400".to_string();
    i32::from_str_radix(&stringier, 10)?;
    //~^ from_str_radix_10

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

// https://github.com/rust-lang/rust-clippy/issues/12731
fn issue_12731() {
    const A: Result<u32, std::num::ParseIntError> = u32::from_str_radix("123", 10);
    const B: () = {
        let _ = u32::from_str_radix("123", 10);
    };
    const fn foo() {
        let _ = u32::from_str_radix("123", 10);
    }
}

fn fix_str_ref_check() {
    #![allow(clippy::needless_borrow)]
    let s = "1";
    let _ = u32::from_str_radix(&s, 10).unwrap();
    //~^ from_str_radix_10
    let s_ref = &s;
    let _ = u32::from_str_radix(&s_ref, 10).unwrap();
    //~^ from_str_radix_10
}
