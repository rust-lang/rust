#![warn(clippy::pattern_type_mismatch)]
#![allow(clippy::single_match)]

fn main() {}

fn should_lint() {
    let value = &Some(23);
    match value {
        Some(_) => (),
        //~^ pattern_type_mismatch
        _ => (),
    }

    let value = &mut Some(23);
    match value {
        Some(_) => (),
        //~^ pattern_type_mismatch
        _ => (),
    }
}

fn should_not_lint() {
    let value = &Some(23);
    match value {
        &Some(_) => (),
        _ => (),
    }
    match *value {
        Some(_) => (),
        _ => (),
    }

    let value = &mut Some(23);
    match value {
        &mut Some(_) => (),
        _ => (),
    }
    match *value {
        Some(_) => (),
        _ => (),
    }

    const FOO: &str = "foo";

    fn foo(s: &str) -> i32 {
        match s {
            FOO => 1,
            _ => 0,
        }
    }
}
