//@compile-flags: --test
//@no-rustfix
#![warn(clippy::expect_used)]
#![allow(clippy::unnecessary_literal_unwrap)]

fn expect_option() {
    let opt = Some(0);
    let _ = opt.expect("");
    //~^ expect_used
}

fn expect_result() {
    let res: Result<u8, ()> = Ok(0);
    let _ = res.expect("");
    //~^ expect_used
}

fn main() {
    expect_option();
    expect_result();

    const SOME: Option<i32> = Some(3);
    const UNWRAPPED: i32 = SOME.expect("Not three?");
    //~^ expect_used
    const {
        SOME.expect("Still not three?");
        //~^ expect_used
    }
}

#[test]
fn test_expect_option() {
    let opt = Some(0);
    let _ = opt.expect("");
}

#[test]
fn test_expect_result() {
    let res: Result<u8, ()> = Ok(0);
    let _ = res.expect("");
}

#[cfg(test)]
mod issue9612 {
    // should not lint in `#[cfg(test)]` modules
    #[test]
    fn test_fn() {
        let _a: u8 = 2.try_into().unwrap();
        let _a: u8 = 3.try_into().expect("");

        util();
    }

    fn util() {
        let _a: u8 = 4.try_into().unwrap();
        let _a: u8 = 5.try_into().expect("");
    }
}
