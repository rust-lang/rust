//@compile-flags: --test
#![warn(clippy::expect_used)]
#![allow(clippy::unnecessary_literal_unwrap)]

fn expect_option() {
    let opt = Some(0);
    let _ = opt.expect("");
}

fn expect_result() {
    let res: Result<u8, ()> = Ok(0);
    let _ = res.expect("");
}

fn main() {
    expect_option();
    expect_result();
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
